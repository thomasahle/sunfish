"""Deterministic packed-engine bench: fixed-depth searches over a small
book of replayed move sequences (the engine is TCEC/UCI-driven -- no FEN);
prints nodes, CPU time and us/node per position and in total.

usage: bench.py ENGINE.py NET.pickle [depth]

Timing hygiene: run on the BENCH BOX at nice 19; treat the minimum of
several runs as the number (docs and ledger say why the laptop lies).
"""
import importlib.util, os, sys, time

ENG, NET = sys.argv[1], sys.argv[2]
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 5

os.environ["SF_NET"] = NET
spec = importlib.util.spec_from_file_location("eng", ENG)
eng = importlib.util.module_from_spec(spec)
sys.modules["eng"] = eng
spec.loader.exec_module(eng)

# The same eight openings/middlegames every time: startpos, four short
# openings, three longer middlegame sequences (all legal from startpos).
BOOK = [
    "",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6 e1g1 e8g8 f1e1 a7a6"
    " a2a4 c5a7 h2h3 c6e7 d3d4",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 e2e3 b8d7 f1d3 d5c4 d3c4 b7b5"
    " c4d3 f8d6 e1g1 e8g8 d1c2 a8b8",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6 g1f3 b8d7"
    " h4h5 g6h7 f1d3 h7d3 d1d3 g8f6",
]


def replay(seq):
    pos = eng.from_board(eng.initial)
    for ply, mv in enumerate(seq.split()):
        i, j, prom = eng.parse(mv[:2]), eng.parse(mv[2:4]), mv[4:].upper()
        if ply % 2 == 1:
            i, j = 119 - i, 119 - j
        pos = pos.move(eng.Move(i, j, prom))
    return pos


tot_n = tot_t = 0
for seq in BOOK:
    pos = replay(seq)
    s = eng.Searcher()
    t0 = time.process_time()
    for depth, gamma, score, move in s.search([pos]):
        if depth > DEPTH:
            break
    el = time.process_time() - t0
    tot_n += s.nodes
    tot_t += el
    print("depth %d: nodes %8d  cpu %6.2fs  %6.2f us/node   [%s]"
          % (DEPTH, s.nodes, el, 1e6 * el / s.nodes, seq[:20] or "startpos"))
print("TOTAL   : nodes %8d  cpu %6.2fs  %6.2f us/node"
      % (tot_n, tot_t, 1e6 * tot_t / tot_n))
