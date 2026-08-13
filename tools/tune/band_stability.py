"""Is the phase-band post-mortem a finding, or is it split noise?

C2's failure was explained on the record by a band diagnostic: "the whole
-5.31% lives in the endgame and the middlegame band is slightly worse than
classic". That explanation was read off ONE held-out split, and the band it
turns on holds ~450 positions. Re-splitting moved it from +0.6% to -7.4%.

So this refits the same model on many splits and reports the SPREAD. A band
delta whose sign changes across splits cannot explain an Elo result, however
plausible it sounds -- and this lane has already paid for one mechanism it
believed before measuring (the corrhist sign, the "85% board" inference).

usage: band_stability.py DATA.npz [NSPLITS]
"""
import hashlib
import json
import re
import sys

import chess
import numpy as np
import torch

DATA = sys.argv[1]
NSPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 12
REPO = __file__.rsplit("/tools/", 1)[0]
PIECES = "PNBRQK"
PHASE = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
K = 350.0
KING = slice(5 * 64, 6 * 64)
BANDS = ((0, 5), (6, 11), (12, 17), (18, 24))

torch.manual_seed(20260813)
torch.set_num_threads(1)

d = np.load(DATA, allow_pickle=False)
X = torch.tensor(d["X"].astype(np.float64))
y = d["y"].astype(np.float64)
fens = [str(f) for f in d["fens"]]
n = len(y)
tt = torch.tensor(1.0 / (1.0 + np.exp(-y / K)))
ph = np.array([min(sum(PHASE[p.symbol().upper()] for _, p in chess.Board(f).piece_map().items()), 24)
               for f in fens])

src = open(REPO + "/sunfish.py").read()
piece = eval(re.search(r"^piece = (\{[^}]*\})", src, re.M).group(1))
pst0 = eval(re.search(r"^pst = (\{.*?^\})", src, re.M | re.S).group(1))
w0 = np.zeros(384)
for pi, p in enumerate(PIECES):
    tab = np.array(pst0[p], dtype=np.float64) + piece[p]
    w0[pi * 64:(pi + 1) * 64] = tab.reshape(8, 8)[::-1].reshape(64)
W0 = torch.tensor(w0)
FREE = torch.ones(384, dtype=torch.bool); FREE[KING] = False


def loss(w, idx):
    return ((torch.sigmoid(X[idx] @ w / K) - tt[idx]) ** 2).mean().item()


def fit(tr):
    v = W0[FREE].clone().requires_grad_(True)
    # The tolerances are not decoration. At this loss scale (~0.017) the
    # gradients are ~1e-9, and LBFGS's DEFAULT tolerance_grad of 1e-7 declares
    # convergence before the first step: the "fit" comes back exactly equal to
    # the warm start and every band reads 0.00%, which looks like a result.
    opt = torch.optim.LBFGS([v], max_iter=300, history_size=50, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-16)

    def closure():
        opt.zero_grad()
        l = ((torch.sigmoid(X[tr] @ W0.clone().masked_scatter(FREE, v) / K) - tt[tr]) ** 2).mean()
        l.backward()
        return l
    opt.step(closure)
    return W0.clone().masked_scatter(FREE, v).detach()


rows = {b: [] for b in BANDS}
overall = []
for s in range(NSPLIT):
    h = np.array([int(hashlib.sha256(("%d%s" % (s, f)).encode()).hexdigest()[:8], 16) % 5 for f in fens])
    va, tr = np.where(h == 0)[0], np.where(h != 0)[0]
    w = fit(torch.tensor(tr))
    b, c = loss(W0, torch.tensor(va)), loss(w, torch.tensor(va))
    overall.append(100 * (c - b) / b)
    for lo, hi in BANDS:
        idx = va[(ph[va] >= lo) & (ph[va] <= hi)]
        b, c = loss(W0, torch.tensor(idx)), loss(w, torch.tensor(idx))
        rows[(lo, hi)].append(100 * (c - b) / b)

o = np.array(overall)
print("data %s | %d positions | %d splits" % (DATA.split("/")[-1], n, NSPLIT))
print("\nheld-out loss vs classic, %% (negative = the fit is better)\n")
print("%-10s %8s %8s %8s %8s %s" % ("band", "mean", "sd", "min", "max", "sign"))
print("%-10s %8.2f %8.2f %8.2f %8.2f %s"
      % ("OVERALL", o.mean(), o.std(), o.min(), o.max(),
         "stable" if (o < 0).all() or (o > 0).all() else "FLIPS"))
for (lo, hi), v in rows.items():
    v = np.array(v)
    print("%-10s %8.2f %8.2f %8.2f %8.2f %s"
          % ("phase %d-%d" % (lo, hi), v.mean(), v.std(), v.min(), v.max(),
             "stable" if (v < 0).all() or (v > 0).all() else "FLIPS"))
print("\nA band whose sign FLIPS across splits is not a mechanism. It is noise")
print("with a story attached, and it cannot explain any Elo result.")
