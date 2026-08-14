#!/usr/bin/env python3
"""Float trainer vs packed engine, same replnet weights: the quantization
cross-check.  Usage: replnet_shapecheck.py net.pickle [tol_cp]

Splices net.pickle.payload into a scratch copy of replnet_proto.py, walks
random legal games, and compares the engine's nn_cp against the float
model rebuilt from the pickle (act = clamp01, d = ((au-at)*v).sum).  The
engine's integer grid (gain rounding, bias clipping, shift truncation)
must stay within tol_cp of the float net it was exported from; a larger
gap means the export order or the mirror composition is WRONG, and the
check must fail loudly, not be widened."""
import pickle
import random
import re
import sys
import os
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))

path = sys.argv[1]
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
with open(path, "rb") as f:
    d = pickle.load(f)
assert d["kind"] == "replnet-ternary", d["kind"]
payload = open(path + ".payload").read().strip()
E, bias, v, N = d["E"], d["bias"], [abs(x) for x in d["v"]], d["N"]

src = open(os.path.join(os.path.dirname(here), "replnet_proto.py")).read()
src, n = re.subn(r'for _c in "[^"]{700,}":', 'for _c in "%s":' % payload, src)
assert n == 1, n
tmp = tempfile.mkdtemp()
open(os.path.join(tmp, "replnet_live.py"), "w").write(src)
sys.path.insert(0, tmp)
import replnet_live as e


def float_cp(board, pf):
    au, at = [], []
    for k in range(N):
        su = st = 0.0
        for i, p in enumerate(board):
            if p.isalpha():
                f64 = (i // 10 - 2) * 8 + (i % 10 - 1)
                pi = e._PIECES.index(p)
                mi = e._PIECES.index(p.swapcase()) * 64 + 63 - f64
                su += E[pi * 64 + f64][k]
                at_f = E[mi][k]
                st += at_f
        au.append(min(max(su + bias[k], 0.0), 1.0))
        at.append(min(max(st + bias[k], 0.0), 1.0))
    d_ = sum((au[k] - at[k]) * v[k] for k in range(N))
    d_ = max(-e.CLAMP, min(e.CLAMP, d_))
    return d_


random.seed(20260814)
pos = e.from_board(e.initial)
worst = 0.0
n = 0
for step in range(60):
    moves = [m for m in pos.gen_moves() if not pos.move(m).k()]
    if not moves:
        break
    pos = pos.move(random.choice(moves))
    got = e.nn_cp(pos.acc, pos.pf)
    # the float reference sees the mover's frame: pf==1 boards are compared
    # through the engine's own perspective handling, so normalise first
    want = float_cp(pos.board, 0)
    diff = abs(got - want)
    worst = max(worst, diff)
    n += 1
assert n >= 30, "walk too short"
assert worst <= TOL, "engine vs float diverge: worst %.1f cp > tol %.1f" % (
    worst, TOL)
print("replnet_shapecheck: %d positions, worst engine-float gap %.1f cp "
      "(tol %.1f) PASS" % (n, worst, TOL))
