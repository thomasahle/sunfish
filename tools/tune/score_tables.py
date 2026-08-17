"""Score the tables the ARTIFACT DECODES, not the ones the fit produced.

codec.py's own lesson: "the fit never sees the emit, so the emit gets scored
separately or it is not measured at all". The shipped eg table is the float
solution rounded to a 16 cp grid and reconstructed through the delta decoder,
so its held-out loss is a different number from the fit's, and the shipped one
is the only one that is true of the artifact.

Same held-out rows (the corpus's own val_a), same K, same model.
usage: score_tables.py DECODED.json [N_VAL]
"""
import json, os, sys, time
import numpy as np

CORPUS = "/home/thomas-ahle/sunfish-bench/replnet-20260814/train/pool10m.npz"
DEC = json.load(open(sys.argv[1]))
N_VAL = int(sys.argv[2]) if len(sys.argv) > 2 else 400000
PIECES = "PNBRQK"; PIDX = {p: i for i, p in enumerate(PIECES)}
PHW = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
KSIG = 350.0; SEED = 20260817
piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
CLASSIC = json.load(open(os.environ.get("SF_CLASSIC_TABLES", "classic_tables.json")))
mg = {p: np.array(CLASSIC[p], dtype=np.float64) + piece[p] for p in PIECES}
K_END = np.array([60000 + 70 - 10 * (abs(2 * ((21 + r * 10 + f) // 10) - 11)
                                     + abs(2 * ((21 + r * 10 + f) % 10) - 9))
                  for r in range(8) for f in range(8)], dtype=np.float64)
K_MID = mg["K"]
eg_q = {p: np.array(DEC["eg_q"][p], dtype=np.float64) for p in "PNBRQ"}
eg_f = {p: np.array(json.load(open("taper_fit.json"))["constrained"]["eg"][p], dtype=np.float64)
        for p in "PNBRQ"}
# sanity: the decoder's mg must BE classic's, or the delta was taken elsewhere
for p in "PNBRQ":
    assert np.array_equal(np.array(DEC["mg"][p], dtype=np.float64), mg[p]), \
        "decoded mg is not classic's for %s -- the arm is not what it claims" % p
print("decoded mg == classic's tables for all five pieces")

d = np.load(CORPUS, allow_pickle=False)
cp_all, va_all, fens_all = d["cp"], d["val_a"], d["fens"]
rng = np.random.default_rng(SEED)
va = np.sort(rng.choice(np.flatnonzero(va_all), N_VAL, replace=False))

def feat(idx):
    n = len(idx); X = np.zeros((n, 6, 64), dtype=np.float32)
    ph = np.zeros(n, dtype=np.float32); qoff = np.zeros(n, dtype=bool)
    stm = np.ones(n, dtype=np.float64)
    for row, fen in enumerate(fens_all[idx]):
        parts = fen.split(" "); board = parts[0]
        if len(parts) > 1 and parts[1] == "b": stm[row] = -1.0
        s = p_ = 0
        for c in board:
            if c == "/": continue
            if c.isdigit(): s += int(c); continue
            up = c.upper(); pi = PIDX[up]
            if c.isupper(): X[row, pi, s] += 1.0
            else: X[row, pi, 63 - s] -= 1.0
            p_ += PHW[up]; s += 1
        ph[row] = min(24, p_); qoff[row] = ("Q" not in board) or ("q" not in board)
    return X, ph, qoff, cp_all[idx].astype(np.float64) * stm

def sig(v): return 1.0 / (1.0 + np.exp(-v / KSIG))

tot = {k: 0.0 for k in ("base", "float", "quant")}
n = 0
for c0 in range(0, len(va), 200000):
    ii = va[c0:c0 + 200000]
    X, ph, qoff, cp = feat(ii)
    kterm = np.einsum("ns,ns->n", X[:, PIDX["K"], :],
                      np.where(qoff[:, None], K_END[None, :], K_MID[None, :]))
    base = kterm.copy(); fl = kterm.copy(); qu = kterm.copy()
    w_mg, w_eg = ph / 24.0, (24.0 - ph) / 24.0
    for p in "PNBRQ":
        xp = X[:, PIDX[p], :]
        base += xp @ mg[p]
        fl += (xp @ mg[p]) * w_mg + (xp @ eg_f[p]) * w_eg
        qu += (xp @ mg[p]) * w_mg + (xp @ eg_q[p]) * w_eg
    t = sig(cp)
    tot["base"] += float(((sig(base) - t) ** 2).sum())
    tot["float"] += float(((sig(fl) - t) ** 2).sum())
    tot["quant"] += float(((sig(qu) - t) ** 2).sum())
    n += len(ii)
b = tot["base"] / n
for k in ("base", "float", "quant"):
    v = tot[k] / n
    print("%-6s VAL wp-mse %.6f   %+.2f%% vs base   (n=%d)" % (k, v, 100 * (v - b) / b, n))
