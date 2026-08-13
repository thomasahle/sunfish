"""Texel-tune a TAPERED eval: mg and eg tables, fitted jointly.

Tapering is the biggest known HCE win we lack, and it stays FREE in our
architecture for the same reason a PST does: it is a function of (piece,
square) only, so it remains incrementally updatable -- two accumulators
instead of one, interpolated at read time. Mobility and pawn structure
are NOT free for us the way they are for ice4/4ku, because our eval is
O(1) incremental and theirs is recomputed per node; adding a term that
depends on the whole position would force movegen at every leaf.

The model stays LINEAR in its 768 parameters:

    score = (phase * mg . x + (24 - phase) * eg . x) / 24

so the same closed-form fit works: features are the piece-square counts
scaled by phase/24 and (24-phase)/24 respectively.

usage: texel_taper.py DATA.npz OUT.json
"""
import json
import re
import sys

import chess
import numpy as np

DATA, OUT = sys.argv[1], sys.argv[2]
REPO = "/Users/ahle/repos/sunfish-packed"
PIECES = "PNBRQK"
PHASE = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}   # 4ku's weights

d = np.load(DATA, allow_pickle=True)
X = d["X"].astype(np.float32)
y = d["y"].astype(np.float32)
fens = d["fens"]
n = len(y)

# phase per position, 24 = full material
ph = np.zeros(n, dtype=np.float32)
for i, fen in enumerate(fens):
    b = chess.Board(str(fen))
    t = 0
    for _, pc in b.piece_map().items():
        t += PHASE[pc.symbol().upper()]
    ph[i] = min(t, 24)
print("positions %d, mean phase %.1f" % (n, ph.mean()))

Xmg = X * (ph / 24.0)[:, None]
Xeg = X * (1 - ph / 24.0)[:, None]
XX = np.concatenate([Xmg, Xeg], axis=1)          # (n, 768)

src = open(REPO + "/sunfish.py").read()
piece = eval(re.search(r"^piece = (\{[^}]*\})", src, re.M).group(1))
pst0 = eval(re.search(r"^pst = (\{.*?^\})", src, re.M | re.S).group(1))
w0 = np.zeros(384, dtype=np.float32)
for pi, p in enumerate(PIECES):
    tab = np.array(pst0[p], dtype=np.float32) + piece[p]
    w0[pi * 64:(pi + 1) * 64] = tab.reshape(8, 8)[::-1].reshape(64)
w = np.concatenate([w0, w0]).astype(np.float32)  # warm start: mg = eg = classic

K = 350.0
t = 1 / (1 + np.exp(-y / K))
KINGmg = slice(5 * 64, 6 * 64)
KINGeg = slice(384 + 5 * 64, 384 + 6 * 64)


def loss(v):
    p = 1 / (1 + np.exp(-(XX @ v) / K))
    return float(np.mean((p - t) ** 2))


base = loss(w)
print("start loss %.6f" % base)
lr, best, bw = 8000.0, base, w.copy()
for it in range(6000):
    s = XX @ w
    p = 1 / (1 + np.exp(-s / K))
    w -= lr * (2.0 / n) * (XX.T @ ((p - t) * p * (1 - p) / K))
    w[KINGmg] = w0[5 * 64:6 * 64]
    w[KINGeg] = w0[5 * 64:6 * 64]
    if it % 1000 == 0 or it == 5999:
        L = loss(w)
        if L < best:
            best, bw = L, w.copy()
        print("  iter %4d loss %.6f" % (it, L))
w = bw
print("final %.6f  (%.1f%% better than classic's single table)"
      % (best, 100 * (base - best) / base))

out = {}
for half, tag in ((0, "mg"), (384, "eg")):
    for pi, p in enumerate(PIECES):
        tab = w[half + pi * 64: half + (pi + 1) * 64]
        val = float(np.median(tab))
        out["%s_%s" % (tag, p)] = np.round(tab - val).astype(int).reshape(8, 8)[::-1].reshape(64).tolist()
        out["%s_value_%s" % (tag, p)] = int(round(val))
json.dump(out, open(OUT, "w"))
print("mg values:", {p: out["mg_value_" + p] for p in PIECES})
print("eg values:", {p: out["eg_value_" + p] for p in PIECES})
print("wrote", OUT)
