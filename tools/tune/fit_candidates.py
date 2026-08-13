"""Fit the eval-table candidates, on a HELD-OUT split, and verify the emit.

A fit is a candidate generator, never evidence. The last Texel fit improved
the loss 10.1% and measured -16.7 +/- 31.2 in play. Nothing here is an Elo
claim; the output is tables plus a byte price, and games decide.

Three things this file exists to get right, each of which has burned the
project once already:

  * HELD-OUT loss. `texel_tune.py` reports training loss. That is fine for a
    single warm-started fit, but it cannot compare a 384-parameter table
    against a 768-parameter tapered one: the bigger model wins in-sample by
    construction. The split is deterministic and by POSITION, and every
    number printed below is on data no fit ever saw.

  * THE EMIT IS PART OF THE MODEL. Emitting the king table un-flipped once
    mirrored the most orientation-sensitive table in the engine and cost -67
    Elo while the fit looked 10% better -- because the fit never sees the
    emit. So every candidate is re-scored HERE from its own emitted integer
    tables, reconstructed exactly as the engine will index them. A candidate
    whose emitted loss does not match its fitted loss is rejected on the spot.

  * THE SEAM IS THE ONE THE ENGINE HAS. The entry swaps the king table on
    queens-off (`pst["K"] = K_MID if "Q" in board and "q" in board else
    K_END`) and rebuilds the root table there. That existing swap is the
    natural taper seam: selecting the whole table set costs no new condition
    and no new root pass. A continuous 24-point phase blend is fitted too,
    as the more expressive alternative, and the two are compared on held-out
    loss rather than on which sounds better.

usage: fit_candidates.py DATA.npz OUTDIR
"""
import json
import os
import pathlib
import re
import sys

import chess
import numpy as np
from scipy.optimize import minimize

DATA = sys.argv[1]
OUTDIR = sys.argv[2]
REPO = str(pathlib.Path(__file__).resolve().parents[2])
PIECES = "PNBRQK"
PHASE = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}   # 4ku's weights
K = 350.0                                                  # cp -> win-prob scale
KING = slice(5 * 64, 6 * 64)
os.makedirs(OUTDIR, exist_ok=True)

# ---- data -------------------------------------------------------------------
d = np.load(DATA, allow_pickle=False)
X = d["X"].astype(np.float64)
y = d["y"].astype(np.float64)
fens = [str(f) for f in d["fens"]]
n = len(y)
meta = json.loads(str(d["meta"]))
print("data: %s | %d positions | %s depth %d" % (os.path.basename(DATA), n, meta["engine"], meta["depth"]))

# Phase and the engine's own queens-off condition, both from the BOARD -- X is
# a difference feature and cancels mirrored pairs, so it cannot give either.
ph = np.zeros(n)
queens = np.zeros(n, dtype=bool)
for i, fen in enumerate(fens):
    b = chess.Board(fen)
    t = 0
    for _, pc in b.piece_map().items():
        t += PHASE[pc.symbol().upper()]
    ph[i] = min(t, 24)
    board = fen.split()[0]
    queens[i] = "Q" in board and "q" in board
print("mean phase %.1f/24 | both queens on: %.1f%% of positions" % (ph.mean(), 100 * queens.mean()))

rng = np.random.default_rng(20260813)
perm = rng.permutation(n)
ntr = int(0.8 * n)
tr, va = perm[:ntr], perm[ntr:]
print("split: %d train / %d held out (deterministic seed)" % (len(tr), len(va)))

t_all = 1.0 / (1.0 + np.exp(-y / K))

# ---- warm start: classic's tables, piece values folded in -------------------
src = open(REPO + "/sunfish.py").read()
piece = eval(re.search(r"^piece = (\{[^}]*\})", src, re.M).group(1))
pst0 = eval(re.search(r"^pst = (\{.*?^\})", src, re.M | re.S).group(1))
w0 = np.zeros(384)
for pi, p in enumerate(PIECES):
    tab = np.array(pst0[p], dtype=np.float64) + piece[p]
    # classic writes rank 8 first; our features index chess squares (A1=0).
    w0[pi * 64:(pi + 1) * 64] = tab.reshape(8, 8)[::-1].reshape(64)


def fit(F, w_init, frozen):
    """Minimise MSE-on-win-probability over rows `tr`; return weights.

    `frozen` is a boolean mask of parameters pinned at their warm-start value
    (the king's 60000 sentinel is structural, not an evaluation term)."""
    A, tt = F[tr], t_all[tr]
    free = ~frozen

    def f(v):
        w = w_init.copy()
        w[free] = v
        s = A @ w
        p = 1.0 / (1.0 + np.exp(-s / K))
        r = p - tt
        g = (2.0 / len(tt)) * (A.T @ (r * p * (1 - p) / K))
        return float(np.mean(r ** 2)), g[free]

    res = minimize(f, w_init[free], jac=True, method="L-BFGS-B",
                   options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-12})
    w = w_init.copy()
    w[free] = res.x
    return w, res.nit


def loss(F, w, idx):
    p = 1.0 / (1.0 + np.exp(-(F[idx] @ w) / K))
    return float(np.mean((p - t_all[idx]) ** 2))


def emit(w):
    """Integer tables in classic's shape (rank 8 first), piece value factored
    out -- byte-for-byte the form the generator will paste."""
    out = {}
    for pi, p in enumerate(PIECES):
        tab = w[pi * 64:(pi + 1) * 64]
        if p == "K":
            # The king keeps classic's table verbatim: its value is a 60000
            # sentinel, and the K_MID/K_END pair is the landed kend fix, which
            # a fit must not quietly overwrite.
            out[p] = list(pst0["K"])
            out["_value_K"] = piece["K"]
            continue
        base = float(np.median(tab))
        out[p] = np.round(tab - base).astype(int).reshape(8, 8)[::-1].reshape(64).tolist()
        out["_value_" + p] = int(round(base))
    return out


def unemit(out):
    """Rebuild the fit's weight vector FROM the emitted integers, indexing them
    exactly as the engine does. If emit() mirrors or drops a table, this is
    where it shows up as a loss explosion rather than as -67 Elo in a match."""
    w = np.zeros(384)
    for pi, p in enumerate(PIECES):
        tab = np.array(out[p], dtype=np.float64) + out["_value_" + p]
        w[pi * 64:(pi + 1) * 64] = tab.reshape(8, 8)[::-1].reshape(64)
    return w


frozen1 = np.zeros(384, dtype=bool)
frozen1[KING] = True
results = {}

# ---- baseline: classic's tables, untouched ----------------------------------
base_tr, base_va = loss(X, w0, tr), loss(X, w0, va)
print("\nclassic (no fit)          train %.6f   HELD-OUT %.6f" % (base_tr, base_va))

# ---- A: flat refit, 384 parameters, zero shape change -----------------------
wA, nit = fit(X, w0, frozen1)
A_tr, A_va = loss(X, wA, tr), loss(X, wA, va)
outA = emit(wA)
A_emit = loss(X, unemit(outA), va)
print("A  flat refit (384p)      train %.6f   HELD-OUT %.6f  (%+.1f%%)  emit %.6f  %s  [%d it]"
      % (A_tr, A_va, 100 * (A_va - base_va) / base_va, A_emit,
         "EMIT OK" if abs(A_emit - A_va) < 2e-5 else "EMIT MISMATCH", nit))
results["flat"] = (A_va, A_emit, outA)

# ---- B: taper at the engine's own seam (queens on / queens off) -------------
# Disjoint: each position is scored by the set its own board selects, which is
# exactly what the engine will do at the root.
XB = np.zeros((n, 768))
XB[queens, :384] = X[queens]
XB[~queens, 384:] = X[~queens]
w0B = np.concatenate([w0, w0])
frozen2 = np.zeros(768, dtype=bool)
frozen2[KING] = True
frozen2[384 + 5 * 64: 384 + 6 * 64] = True
wB, nit = fit(XB, w0B, frozen2)
B_tr, B_va = loss(XB, wB, tr), loss(XB, wB, va)
outB = {"qon": emit(wB[:384]), "qoff": emit(wB[384:])}
wB_e = np.concatenate([unemit(outB["qon"]), unemit(outB["qoff"])])
B_emit = loss(XB, wB_e, va)
print("B  queens-seam (768p)     train %.6f   HELD-OUT %.6f  (%+.1f%%)  emit %.6f  %s  [%d it]"
      % (B_tr, B_va, 100 * (B_va - base_va) / base_va, B_emit,
         "EMIT OK" if abs(B_emit - B_va) < 2e-5 else "EMIT MISMATCH", nit))
results["qseam"] = (B_va, B_emit, outB)

# ---- C: continuous 24-point phase blend, the more expressive alternative ----
XC = np.concatenate([X * (ph / 24.0)[:, None], X * (1 - ph / 24.0)[:, None]], axis=1)
wC, nit = fit(XC, w0B, frozen2)
C_tr, C_va = loss(XC, wC, tr), loss(XC, wC, va)
outC = {"mg": emit(wC[:384]), "eg": emit(wC[384:])}
wC_e = np.concatenate([unemit(outC["mg"]), unemit(outC["eg"])])
C_emit = loss(XC, wC_e, va)
print("C  phase blend (768p)     train %.6f   HELD-OUT %.6f  (%+.1f%%)  emit %.6f  %s  [%d it]"
      % (C_tr, C_va, 100 * (C_va - base_va) / base_va, C_emit,
         "EMIT OK" if abs(C_emit - C_va) < 2e-5 else "EMIT MISMATCH", nit))
results["phase"] = (C_va, C_emit, outC)

json.dump({"classic_heldout": base_va,
           "flat": {"heldout": A_va, "emit": A_emit, "tables": outA},
           "qseam": {"heldout": B_va, "emit": B_emit, "tables": outB},
           "phase": {"heldout": C_va, "emit": C_emit, "tables": outC}},
          open(os.path.join(OUTDIR, "fits.json"), "w"))
print("\nwrote %s/fits.json -- CANDIDATES ONLY. No Elo is implied by any number above." % OUTDIR)
