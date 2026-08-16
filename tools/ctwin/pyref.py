#!/usr/bin/env pypy3
"""Reference side of the ctwin differential harness.

Speaks the same line protocol as sunfish.c, but every answer comes from the
REAL classic engine: sunfish.py at the repo root, imported and driven
in-process.  Nothing of the search is reimplemented here -- difftest.py
compares this transcript against the C twin's byte for byte.

Protocol (stdin -> stdout, one line per command):
    reset                       fresh module state: K_MID table, new Searcher
    position startpos [moves m1 m2 ...]      (UCI move strings)
    position fen <FEN> [moves ...]
    push <i> <j> <prom|->      apply a raw move to the current position
    pop                        undo last push
    moves                      list gen_moves() in order: "mv i,j,p val"
    go depth <D>               print every MTD-bi yield up to depth D:
                               "info depth D gamma G score S move i,j,p nodes N"
    quit
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sunfish
from sunfish import Move, Position, Searcher

out = sys.stdout


def emit(*args):
    print(*args, file=out)
    out.flush()


def fmt_move(m):
    if m is None:
        return "-"
    return "%d,%d,%s" % (m.i, m.j, m.prom or "-")


# --- movegen call counter (battery metric, compared in `done` lines) ---
GEN_CALLS = 0
_orig_gen_moves = Position.gen_moves


def _counting_gen_moves(self):
    global GEN_CALLS
    GEN_CALLS += 1
    return _orig_gen_moves(self)


Position.gen_moves = _counting_gen_moves

# --- tp_move replacement-policy battery knobs ------------------------------
# Non-default values (or USE_VARIANT=1) swap the searcher for the drift-
# guarded transcription in variants.py; defaults keep the REAL live-imported
# Searcher, so the ordinary gate never depends on the transcription.
BATTERY = {"EVICT_POLICY": 0, "EVICT_SCAN_K": 4, "KILLER_COUNT": 1, "USE_VARIANT": 0}

# --- harness knobs (no Python-side sunfish attribute) ----------------------
# FEN_HIST: how `position fen` builds the history it hands to search().
#   1 (default) = sunfish_ui/uci.py's construction, which is what MATCHES run:
#       hist = [pos] if white else [pos.rotate(), pos].  The extra ply is a
#       search input -- search() does self.history = set(hist) and bound()
#       scores any non-root repeat of it as a draw -- so a reference that
#       builds one ply is not the engine a match plays.
#   0 = the one-ply construction this file used before, kept so the
#       difference can be measured rather than argued.
HARNESS = {"FEN_HIST": 1}


def make_searcher():
    if any(BATTERY[k] != d for k, d in
           (("EVICT_POLICY", 0), ("EVICT_SCAN_K", 4), ("KILLER_COUNT", 1),
            ("USE_VARIANT", 0))):
        import variants
        variants.EVICT_POLICY = BATTERY["EVICT_POLICY"]
        variants.EVICT_SCAN_K = BATTERY["EVICT_SCAN_K"]
        variants.KILLER_COUNT = BATTERY["KILLER_COUNT"]
        return variants.VariantSearcher()
    return Searcher()


def fen_history(fen_fields):
    """Build the HISTORY a `position fen` starts from, the way the driver every
    match runs builds it (sunfish_ui/uci.py):

        hist = [pos] if get_color(pos) == WHITE else [pos.rotate(), pos]

    For a black-to-move FEN that is TWO plies: the root, preceded by its own
    white-POV mirror.  It is not cosmetic -- Searcher.search does
    `self.history = set(hist)` and bound() returns 0 for a non-root node found
    there, so the mirror scores as a draw from move 1, and the null move lands
    on it exactly whenever ep == kp == 0.  Returning one ply here made the gate
    certify a construction no match ever played."""
    pos, side = from_fen(fen_fields)
    if side == "b" and HARNESS["FEN_HIST"]:
        return [pos.rotate(), pos], side
    return [pos], side


def from_fen(fen_fields):
    """Build a Position from FEN, oriented to the side to move.  The C twin
    implements this construction bit for bit (see setup_fen in sunfish.c)."""
    placement, side, castling, ep = fen_fields[:4]
    board = "         \n" * 2
    for row in placement.split("/"):
        line = " "
        for c in row:
            line += "." * int(c) if c.isdigit() else c
        board += line + "\n"
    board += "         \n" * 2
    assert len(board) == 120, "bad FEN placement"
    wc = ("Q" in castling, "K" in castling)
    bc = ("k" in castling, "q" in castling)
    epi = sunfish.parse(ep) if ep != "-" else 0
    score = sum(sunfish.pst[c][i] for i, c in enumerate(board) if c.isupper())
    score -= sum(sunfish.pst[c.upper()][119 - i] for i, c in enumerate(board) if c.islower())
    pos = Position(board, score, wc, bc, epi, 0)
    if side == "b":
        pos = pos.rotate()
    return pos, side


def apply_uci_moves(hist, moves, side0):
    # Classic main(): flip the parsed squares whenever the mover is black.
    # For startpos this is exactly the ply%2==1 rule in sunfish.py.
    for ply, move in enumerate(moves):
        i, j, prom = sunfish.parse(move[:2]), sunfish.parse(move[2:4]), move[4:].upper()
        black = (ply % 2 == 1) if side0 == "w" else (ply % 2 == 0)
        if black:
            i, j = 119 - i, 119 - j
        hist.append(hist[-1].move(Move(i, j, prom)))


def main():
    searcher = Searcher()
    hist = [Position(sunfish.initial, 0, (True, True), (True, True), 0, 0)]

    for line in sys.stdin:
        args = line.split()
        if not args:
            continue
        cmd = args[0]

        if cmd == "quit":
            break

        elif cmd == "reset":
            sunfish.pst["K"] = sunfish.K_MID
            searcher = make_searcher()
            hist = [Position(sunfish.initial, 0, (True, True), (True, True), 0, 0)]
            emit("ok")

        elif cmd == "set":
            # Shared tuning knobs; the C-only knobs have no Python side.
            if args[1] in (
                    "QS", "QS_A", "LMR", "EVAL_ROUGHNESS", "TABLE_SIZE", "NULL_MARGIN"):
                setattr(sunfish, args[1], int(args[2]))
                # TABLE_SIZE sizes the policy-3 slot table at construction:
                # rebuild so the last set wins (fresh, like the C twin's
                # size-on-first-use; sets arrive right after reset).
                searcher = make_searcher()
                emit("ok")
            elif args[1] in BATTERY:
                BATTERY[args[1]] = int(args[2])
                searcher = make_searcher()
                emit("ok")
            elif args[1] in HARNESS:
                HARNESS[args[1]] = int(args[2])
                emit("ok")
            else:
                emit("err knob")

        elif cmd == "position":
            if args[1] == "startpos":
                hist = [Position(sunfish.initial, 0, (True, True), (True, True), 0, 0)]
                apply_uci_moves(hist, args[3:], "w")
            elif args[1] == "fen":
                fen = args[2:]
                moves = []
                if "moves" in fen:
                    k = fen.index("moves")
                    fen, moves = fen[:k], fen[k + 1:]
                hist, side = fen_history(fen)
                apply_uci_moves(hist, moves, side)
            emit("ok")

        elif cmd == "push":
            prom = "" if args[3] == "-" else args[3]
            hist.append(hist[-1].move(Move(int(args[1]), int(args[2]), prom)))
            emit("ok")

        elif cmd == "pop":
            hist.pop()
            emit("ok")

        elif cmd == "moves":
            pos = hist[-1]
            for m in pos.gen_moves():
                emit("mv", fmt_move(m), pos.value(m))
            emit("end")

        elif cmd == "go" and args[1] == "depth":
            maxd = int(args[2])
            global GEN_CALLS
            GEN_CALLS = 0
            # Consume the REAL search generator; shadow the driver's bracket
            # so we can stop exactly when depth==maxd converges, without
            # pulling (and paying for) the first probe of depth maxd+1.
            lower, upper, last_d, nodes = None, None, 0, 0
            for depth, gamma, score, move in searcher.search(hist):
                if depth != last_d:
                    lower, upper, last_d = 1 - sunfish.MATE_UPPER, sunfish.MATE_UPPER, depth
                emit("info depth", depth, "gamma", gamma, "score", score,
                     "move", fmt_move(move), "nodes", searcher.nodes)
                nodes = searcher.nodes
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                if depth == maxd and not lower < upper - sunfish.EVAL_ROUGHNESS:
                    break
            emit("done nodes", nodes, "gen", GEN_CALLS)

        else:
            emit("err unknown command:", cmd)


if __name__ == "__main__":
    main()
