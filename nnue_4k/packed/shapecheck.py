"""Search-friendliness gate: eval-shape statistics on REAL game positions.

The kbbil collapse (ledger b45e48d) showed quiet-position val is blind to
play: the best-val net lost -99 field Elo. The discriminating instrument,
validated against four nets with known play results, is the CLIP-PEGGING
RATE on real game positions. Calibration ON THIS EXACT position set:

    net    sat600   play (field Elo)
    v2     1.67%    +100
    kb4    2.13%    +84 / +23
    kb8    1.87%    +75 (best of its RR)
    kbbil  4.93%    -99  <- 2.3-3x every good net

(Absolute rates depend on the position distribution -- an earlier ad-hoc
replay of ~19 full games gave 0.13-0.96% with the same separation -- which
is why the set below is FIXED and the gate is calibrated to it. Two
plausible-sounding proxies FAILED validation and are documented so nobody
re-invents them: random-playout saturation does not separate -- every net
saturates 3.5-8% on garbage positions -- and fixed-depth bench node counts
do not either: v2/kb4 play fine at 390k nodes where kb8 runs 259k; small
trees are a virtue kb8 has, not a vice kbbil has.)

Positions: packed/shapecheck_fens.txt, 1500 FENs sampled sparsely (every
7th ply past the book, decisive tails included) from the decision-RR and
generation-RR games -- mixed engines, fixed forever so numbers stay
comparable.

GATE: a candidate's sat600 must not exceed 2.6% on this set (above every
good net's rate, well below the failure); treat a rate 1.5x the current
incumbent's as a warning even under the ceiling.

usage: shapecheck.py ENGINE.py NET.pickle [FENS.txt]
"""
import importlib.util
import os
import re
import sys

ENG, NET = sys.argv[1], sys.argv[2]
FENS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "shapecheck_fens.txt")

os.environ["SF_NET"] = NET
spec = importlib.util.spec_from_file_location("eng", ENG)
eng = importlib.util.module_from_spec(spec)
sys.modules["eng"] = eng
spec.loader.exec_module(eng)


def board120(fen_board):
    b = re.sub(r"\d", lambda m: "." * int(m.group(0)), fen_board)
    b = list(21 * " " + "  ".join(b.split("/")) + 21 * " ")
    b[9::10] = ["\n"] * 12
    return "".join(b)


n = sat = big = 0
worst = []
for line in open(FENS):
    parts = line.split()
    if len(parts) < 4:
        continue
    board, color, castling, enpas = parts[:4]
    wc = ("Q" in castling, "K" in castling)
    bc = ("k" in castling, "q" in castling)
    ep = eng.parse(enpas) if enpas != "-" else 0
    pos = eng.from_board(board120(board), wc, bc, ep, 0)
    if color == "b":
        pos = pos.rotate()
    r = abs(pos.score - pos.ps)
    n += 1
    sat += r >= 599
    big += r >= 300
    worst.append(r)
worst.sort()
print("shapecheck %s over %d game positions:" % (os.path.basename(NET), n))
print("  sat600 %.2f%%   >=300 %.1f%%   p99 %d   p90 %d" % (
    100.0 * sat / n, 100.0 * big / n, worst[int(n * .99)], worst[int(n * .9)]))
ok = sat / n <= 0.026
print("  GATE (sat600 <= 2.6%%): %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
