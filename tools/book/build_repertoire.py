"""Build a REPERTOIRE opening book from the CC0 lichess database export.

TWO BOOKS, TWO JOBS. `book3k.bin` is a MEASUREMENT INSTRUMENT: uniform weights
over 3000 test-variety lines, which is exactly right for an arena (every line
equally sampled) and exactly wrong in a rated game, where uniform weights
surface the test variety as if it were preparation. This book is the other one.
It plays. So its weights are REAL-GAME FREQUENCIES and its tail is PRUNED --
the mirror image of the measurement book, whose registered rule gives every
entry a 2% exploration FLOOR so it can never stop learning about a line.

Source: https://database.lichess.org/ -- "Database exports are released under
the Creative Commons CC0 license." A byte-prefix of one monthly export is
streamed, filtered to rated standard games where BOTH players are >= MIN_ELO,
and every (position, move) inside the deployed 15-ply horizon is counted.

Weights are counts, rescaled per node so the most popular move gets TOP_WEIGHT.
Two prunes make it a repertoire rather than a database dump:
  * a move below MIN_SHARE of its node's traffic is DROPPED (the "no weird
    lines" rule -- this is what keeps 1.Na3 and 1...Na6 out);
  * a move with fewer than MIN_GAMES observations is DROPPED (noise).
Reads a decompressed PGN stream on stdin; writes the book and a provenance JSON.
Configured by environment variable so the whole build is one reproducible pipe::

    URL=https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst
    curl -s -r 0-629145599 -o prefix.zst "$URL"      # a byte prefix; sha it for provenance
    zstd -dc prefix.zst | MIN_ELO=2000 MIN_SHARE=0.01 MIN_GAMES=20 \
        OUT=repertoire.bin python3 tools/book/build_repertoire.py

A byte prefix is a TIME slice (the first days of the month), which is unbiased
for opening frequency; record its range and sha256 and the build is reproducible
byte-for-byte rather than merely repeatable.
"""
import collections
import hashlib
import json
import os
import re
import sys

import chess
import chess.polyglot

MAX_PLIES = 15          # lichess-bot max_depth 8 -> max_depth*2-1
MIN_ELO = int(os.environ.get("MIN_ELO", 2000))
MIN_SHARE = float(os.environ.get("MIN_SHARE", 0.01))
MIN_GAMES = int(os.environ.get("MIN_GAMES", 20))
TOP_WEIGHT = 2000
TAG = re.compile(r'^\[(\w+) "(.*)"\]')
TOKEN = re.compile(r"[a-hKQRBNO][^\s{}]*")


def raw_move(board, move):
    if board.is_castling(move):
        rook = chess.square(7 if board.is_kingside_castling(move) else 0,
                            chess.square_rank(move.from_square))
        move = chess.Move(move.from_square, rook)
    promo = 0 if move.promotion is None else (move.promotion - 1) << 12
    return move.to_square | (move.from_square << 6) | promo


def main():
    counts = collections.defaultdict(collections.Counter)   # key -> {raw_move: games}
    seen = kept = 0
    hdr, moves_line = {}, None
    for line in sys.stdin:
        m = TAG.match(line)
        if m:
            if moves_line is not None: hdr, moves_line = {}, None
            hdr[m.group(1)] = m.group(2)
            continue
        if not line.strip(): continue
        if line[0] in "123456789*":                       # movetext
            seen += 1
            try:
                we, be = int(hdr.get("WhiteElo", 0)), int(hdr.get("BlackElo", 0))
            except ValueError:
                we = be = 0
            if we >= MIN_ELO and be >= MIN_ELO and hdr.get("Variant", "Standard") == "Standard":
                body = re.sub(r"\{[^}]*\}", "", line)
                board = chess.Board()
                for tok in TOKEN.findall(body):
                    if board.ply() >= MAX_PLIES: break
                    tok = tok.rstrip("!?+#")
                    try:
                        mv = board.parse_san(tok)
                    except Exception:
                        break
                    counts[chess.polyglot.zobrist_hash(board)][raw_move(board, mv)] += 1
                    board.push(mv)
                kept += 1
            hdr, moves_line = {}, None
            if seen % 200000 == 0:
                print("  scanned %d games, kept %d, %d nodes" % (seen, kept, len(counts)), file=sys.stderr)
    build(counts, seen, kept)


def build(counts, seen, kept):
    entries, dropped_share, dropped_n = [], 0, 0
    for key, mvs in counts.items():
        total = sum(mvs.values())
        keep = {m: c for m, c in mvs.items() if c >= MIN_GAMES and c / total >= MIN_SHARE}
        dropped_share += sum(1 for m, c in mvs.items() if c / total < MIN_SHARE)
        dropped_n += sum(1 for m, c in mvs.items() if c < MIN_GAMES and c / total >= MIN_SHARE)
        if not keep: continue
        top = max(keep.values())
        for m, c in keep.items():
            w = max(1, min(65535, round(TOP_WEIGHT * c / top)))
            entries.append((key, m, w, 0))
    entries.sort()
    out = os.environ.get("OUT", "repertoire.bin")
    with open(out, "wb") as f:
        import struct
        for k, m, w, l in entries: f.write(struct.pack(">QHHI", k, m, w, l))
    prov = {"games_scanned": seen, "games_kept": kept, "nodes_raw": len(counts),
            "entries": len(entries), "min_elo": MIN_ELO, "min_share": MIN_SHARE,
            "min_games": MIN_GAMES, "max_plies": MAX_PLIES, "top_weight": TOP_WEIGHT,
            "moves_dropped_below_share": dropped_share, "moves_dropped_below_min_games": dropped_n,
            "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
            "bytes": os.path.getsize(out)}
    json.dump(prov, open(out + ".provenance.json", "w"), indent=1)
    print(json.dumps(prov, indent=1), file=sys.stderr)


main()
