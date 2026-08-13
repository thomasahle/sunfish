"""Texel-tune classic's piece-square tables. Zero bytes, warm-started.

Classic's eval is EXACTLY linear in the 384 table values (score = sum of
pst[piece][square] over the pieces on the board), so this is a linear fit
with a sigmoid link, not a black-box search: gradients are closed-form
and the whole thing runs in seconds.

Warm start from classic's own tables so the tuner can only improve on
them, and keep the table SHAPE identical -- 6x64 integers -- so the
artifact's byte count is unchanged by construction. Tables are ~2014
vintage and have never been fitted to anything this project measured.

usage: texel_tune.py DATA.npz OUT.json
"""
import json
import pathlib
import re
import sys

import numpy as np

DATA, OUT = sys.argv[1], sys.argv[2]
# Derive the repo from THIS file. The hard-coded path made the fit read
# ANOTHER checkout's sunfish.py for its warm start -- the same defect PR
# #176 fixed in the entry generator, and here it would silently warm-start
# from tables that are not the ones being replaced.
REPO = str(pathlib.Path(__file__).resolve().parents[2])
PIECES = "PNBRQK"

d = np.load(DATA, allow_pickle=True)
X = d["X"].astype(np.float32)          # (n, 384) white-minus-mirrored-black
y = d["y"].astype(np.float32)          # centipawns, white POV
n = len(y)
print("positions %d, features %d" % (n, X.shape[1]))

# ---- warm start: classic's piece values folded into its tables --------------
src = open(REPO + "/sunfish.py").read()
piece = eval(re.search(r"^piece = (\{[^}]*\})", src, re.M).group(1))
pst_txt = re.search(r"^pst = (\{.*?^\})", src, re.M | re.S).group(1)
pst0 = eval(pst_txt)
w0 = np.zeros(384, dtype=np.float32)
for pi, p in enumerate(PIECES):
    tab = np.array(pst0[p], dtype=np.float32) + piece[p]
    # classic's tables are written rank 8 first; our features index chess
    # squares (A1=0), so flip the rank order once, here.
    w0[pi * 64:(pi + 1) * 64] = tab.reshape(8, 8)[::-1].reshape(64)

K = 350.0                                   # cp -> win-prob scale


def loss(w):
    p = 1 / (1 + np.exp(-(X @ w) / K))
    t = 1 / (1 + np.exp(-y / K))
    return float(np.mean((p - t) ** 2))


def grad(w):
    s = X @ w
    p = 1 / (1 + np.exp(-s / K))
    t = 1 / (1 + np.exp(-y / K))
    g = (2.0 / n) * (X.T @ ((p - t) * p * (1 - p) / K))
    return g


w = w0.copy()
print("start loss %.6f" % loss(w))
lr, best, bw = 8000.0, loss(w), w.copy()
KING = slice(5 * 64, 6 * 64)
for it in range(4000):
    g = grad(w)
    w -= lr * g
    # the king's 60000 sentinel is structural, not an evaluation term
    w[KING] = w0[KING]
    if it % 500 == 0 or it == 3999:
        L = loss(w)
        if L < best:
            best, bw = L, w.copy()
        print("  iter %4d loss %.6f" % (it, L))
w = bw
print("final loss %.6f (start %.6f, %.1f%% better)"
      % (best, loss(w0), 100 * (loss(w0) - best) / loss(w0)))

# ---- emit tables in classic's shape: integers, piece value factored out -----
out = {}
for pi, p in enumerate(PIECES):
    tab = w[pi * 64:(pi + 1) * 64]
    if p == "K":
        # UN-FLIP like every other piece. Emitting the king un-flipped was a
        # real bug: it mirrored the most orientation-sensitive table in the
        # engine (castling shelter vs the 8th rank) and cost -67 Elo while
        # the FIT looked 10% better, because the fit never sees the emit.
        kt = (w0[pi * 64:(pi + 1) * 64] - piece[p])
        out[p] = np.round(kt).astype(int).reshape(8, 8)[::-1].reshape(64).tolist()
        continue
    base = float(np.median(tab))                 # new piece value
    rel = np.round(tab - base).astype(int)
    out[p] = rel.reshape(8, 8)[::-1].reshape(64).tolist()   # back to rank-8-first
    out["_value_" + p] = int(round(base))
out["_value_K"] = piece["K"]
json.dump(out, open(OUT, "w"))
print("wrote %s" % OUT)
print("new piece values:", {p: out.get("_value_" + p) for p in PIECES})
print("classic values   :", piece)
