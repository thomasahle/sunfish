#!/usr/bin/env python3
"""Credit game results to the opening-book nodes that produced them.

Walks each game of one or more PGNs through a polyglot ``.bin`` and, for every
ply that the book could have played, credits the ``(position-key, move)`` node
with the game's result **from the mover's point of view**.  The output is a
deterministic JSON file that ``rebuild.py`` turns back into a reweighted book.

Semantics, chosen to match the deployed bot rather than to be convenient:

* **Horizon.**  ``--max-depth`` is in FULL MOVES, exactly as lichess-bot's
  ``config.yml`` spells it; ``engine_wrapper.py`` turns that into
  ``max_depth * 2 - 1`` plies.  The deployed value is 8, i.e. **15 plies**.
* **Exit and re-entry.**  lichess-bot re-probes the book on every move and does
  not latch an "out of book" flag, so a game that transposes back into the book
  is played from the book again.  This walker does the same by default.
  ``--latch-exit`` gives the stricter reading (a side that has once left the
  book is never credited again) for comparison.
* **Both sides.**  Credit is per mover, so a game in which both engines were in
  book credits both.  ``--player`` restricts credit to named players, which is
  what you want for a lichess export where only one side is ours.
* **Transpositions** need no special handling: a polyglot key IS the position,
  so two move orders reaching the same position land on the same node.

A node's key determines the side to move, so "W/D/L by colour" is recorded as
``stm`` on the node plus mover-relative ``w``/``d``/``l`` on each move.  Each move
is written three ways: ``uci`` (the real move), ``raw`` (polyglot's own
king-takes-rook spelling, which is what ``rebuild.py`` joins on) and ``san``.

Usage::

    tools/book/attribute.py --book book3k.bin --out stats.json games.pgn [...]
    tools/book/attribute.py --book book3k.bin --player sunfish-engine lichess.pgn
"""

import argparse
import hashlib
import io
import json
import sys
from collections import Counter

import chess
import chess.pgn
import chess.polyglot

VERSION = 1
PROMO = " nbrq"
SQUARES = ["abcdefgh"[s & 7] + "12345678"[s >> 3] for s in range(64)]
# Result tag -> score for White.  Anything else ("*", "?") is unscored.
SCORES = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def raw_move_uci(raw):
    """Polyglot's 16-bit move -> UCI in the book's OWN spelling (castling is king-takes-rook)."""
    to_sq, from_sq, promo = raw & 0x3F, (raw >> 6) & 0x3F, (raw >> 12) & 0x7
    return SQUARES[from_sq] + SQUARES[to_sq] + (PROMO[promo] if promo else "")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Stats:
    """Accumulates per-(key, move) credit.  Deterministic: insertion-ordered."""

    def __init__(self):
        self.nodes = {}          # key -> {"stm", "first_ply", "fen", "moves": {uci: [n, w, d, l]}}
        self.exits = Counter()   # (colour, ply) -> games that left the book there
        self.credited = 0

    def credit(self, board, entry, score):
        """Credit the book `entry` played in `board` with `score`, already mover-relative."""
        key = entry.key
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = {
                "stm": "w" if board.turn == chess.WHITE else "b",
                "first_ply": board.ply(),
                "fen": board.fen(),
                "moves": {},
            }
        # Keyed on polyglot's raw spelling: e1h1, not e1g1.  Castling credit is
        # silently lost if the join key is the normalised move.
        rec = node["moves"].get(entry.raw_move)
        if rec is None:
            rec = node["moves"][entry.raw_move] = {
                "uci": entry.move.uci(), "raw": raw_move_uci(entry.raw_move),
                "san": board.san(entry.move), "n": [0, 0, 0, 0]}
        rec["n"][0] += 1
        rec["n"][1 if score == 1.0 else 3 if score == 0.0 else 2] += 1
        self.credited += 1

    def to_json(self, meta):
        nodes = []
        for key, node in self.nodes.items():
            moves = []
            for rec in node["moves"].values():
                n, w, d, lo = rec["n"]
                moves.append({"uci": rec["uci"], "raw": rec["raw"], "san": rec["san"],
                              "games": n, "w": w, "d": d, "l": lo, "score": w + 0.5 * d})
            # Sorted so the file is a function of the data, never of dict order.
            moves.sort(key=lambda m: m["raw"])
            nodes.append({
                "key": "%016x" % key,
                "stm": node["stm"],
                "first_ply": node["first_ply"],
                "fen": node["fen"],
                "games": sum(m["games"] for m in moves),
                "moves": moves,
            })
        nodes.sort(key=lambda n: n["key"])
        exits = {"%s%d" % (c, p): n for (c, p), n in sorted(self.exits.items())}
        return {"meta": meta, "exits": exits, "nodes": nodes}


def walk_game(game, reader, stats, max_plies, players=None, latch_exit=False):
    """Walk one game's mainline through the book.  Returns True if it was scored."""
    score_white = SCORES.get(game.headers.get("Result", "*"))
    if score_white is None: return False
    names = {chess.WHITE: game.headers.get("White", ""), chess.BLACK: game.headers.get("Black", "")}
    board = game.board()
    # A non-standard start position has no book to speak of; skip rather than lie.
    if board.fen() != chess.STARTING_FEN: return False
    out_of_book = {chess.WHITE: False, chess.BLACK: False}
    for move in game.mainline_moves():
        if board.ply() >= max_plies: break
        mover = board.turn
        wanted = players is None or names[mover] in players
        if wanted and not (latch_exit and out_of_book[mover]):
            book = {e.move: e for e in reader.find_all(board)}
            if book:
                if move in book:
                    stats.credit(board, book[move], score_white if mover == chess.WHITE else 1.0 - score_white)
                elif not out_of_book[mover]:
                    out_of_book[mover] = True
                    stats.exits[("w" if mover == chess.WHITE else "b", board.ply())] += 1
        board.push(move)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", nargs="+", help="PGN files ('-' for stdin)")
    ap.add_argument("--book", required=True, help="polyglot .bin the games are walked through")
    ap.add_argument("--out", help="write JSON here (default: stdout)")
    ap.add_argument("--max-depth", type=int, default=8,
                    help="book horizon in FULL MOVES, as lichess-bot spells it: "
                         "max_depth*2-1 plies (default 8 = the deployed 15)")
    ap.add_argument("--max-plies", type=int, help="override the horizon in plies directly")
    ap.add_argument("--player", action="append", default=[],
                    help="only credit plies played by this player (repeatable; default: both sides)")
    ap.add_argument("--latch-exit", action="store_true",
                    help="never credit a side again once it has left the book "
                         "(default: re-probe every ply, which is what lichess-bot does)")
    args = ap.parse_args(argv)

    max_plies = args.max_plies if args.max_plies is not None else args.max_depth * 2 - 1
    players = set(args.player) or None
    stats = Stats()
    games = scored = 0
    results = Counter()

    with chess.polyglot.open_reader(args.book) as reader:
        book_entries = len(reader)
        for path in args.pgn:
            handle = io.StringIO(sys.stdin.read()) if path == "-" else open(path, encoding="utf-8", errors="replace")
            with handle as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None: break
                    games += 1
                    results[game.headers.get("Result", "*")] += 1
                    if walk_game(game, reader, stats, max_plies, players, args.latch_exit): scored += 1

    meta = {
        "tool": "attribute.py", "version": VERSION,
        "book": args.book, "book_sha256": sha256(args.book), "book_entries": book_entries,
        "max_plies": max_plies, "latch_exit": args.latch_exit,
        "players": sorted(players) if players else None,
        "pgns": [{"path": p, "sha256": sha256(p) if p != "-" else None} for p in args.pgn],
        "games": games, "games_scored": scored, "results": dict(sorted(results.items())),
        "credited_plies": stats.credited, "nodes": len(stats.nodes),
    }
    text = json.dumps(stats.to_json(meta), indent=1, sort_keys=False) + "\n"
    if args.out:
        with open(args.out, "w") as f: f.write(text)
        print("%s: %d games (%d scored), %d nodes, %d credited plies"
              % (args.out, games, scored, len(stats.nodes), stats.credited), file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
