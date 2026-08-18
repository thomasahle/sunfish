"""Positions-per-bucket census for the candidate partitions, on pool10m.

Pre-registration requirement: the ledger killed king-WING buckets on this
exact ground (80.4% of positions have the white king on the king side,
0.3 pos/param).  Any new partition states its balance BEFORE it is trained.
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import features

d = np.load("pool10m.npz", allow_pickle=False)
fens = d["fens"]
n = min(400000, len(fens))
print("corpus n =", len(fens), " sampled =", n)

PH = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}   # classic phase weights, max 24

kb2 = np.zeros((2, 2), dtype=np.int64)     # own x opp rank-band
kb4 = np.zeros((4, 4), dtype=np.int64)
ph  = np.zeros(25, dtype=np.int64)
step = max(1, len(fens) // n)
for s in fens[::step][:n]:
    f = s.decode() if isinstance(s, bytes) else s
    board = features.fen_to_board120(f.split()[0])
    wk, bk = board.index("K"), 119 - board.index("k")
    kb2[features.KBF[4](wk) // 2, features.KBF[4](bk) // 2] += 1
    kb4[features.KBF[4](wk), features.KBF[4](bk)] += 1
    p = sum(PH[c.upper()] for c in board if c.isalpha())
    ph[min(24, p)] += 1

tot = kb2.sum()
print("\n== kb2 (own-king rank band: 0 = advanced, 1 = back two ranks) ==")
print("own marginal :", (kb2.sum(1) / tot).round(4).tolist())
print("opp marginal :", (kb2.sum(0) / tot).round(4).tolist())
print("joint        :", (kb2 / tot).round(4).tolist())
print("\n== kb4 (rank band x queenside/kingside) own marginal ==")
print((kb4.sum(1) / tot).round(4).tolist())
print("worst kb4 own bucket share:", float(kb4.sum(1).min() / tot))

print("\n== phase (0-24, classic weights) ==")
c = ph.cumsum() / tot
for q in (2, 3, 4):
    cuts = [int(np.searchsorted(c, i / q)) for i in range(1, q)]
    edges = [-1] + cuts + [24]
    shares = [float((ph[edges[i] + 1:edges[i + 1] + 1]).sum() / tot) for i in range(q)]
    print("ph%d cuts at phase %s -> shares %s" % (q, cuts, [round(x, 4) for x in shares]))
print("phase hist (share):", (ph / tot).round(4).tolist())
