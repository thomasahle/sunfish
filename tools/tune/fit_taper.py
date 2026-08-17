"""Distil a TAPERED endgame table set from the 10M corpus -- eg only, mg PINNED.

WHAT IS AND IS NOT BEING FITTED, because this lane's record is that fitted
tables play worse than their loss promises. This fit deliberately does NOT
refit the midgame: `mg` is classic's table, byte for byte, so the arm differs
from the baseline ONLY by an endgame set and the interpolation that reaches
it. A refit would confound "does a taper help" with "does a refit help", and
the ledger already answered the second question (it does not).

MODEL, from white's point of view, matching the engine exactly:

    eval = sum_K x[K][s] * Kt[s]                      # king: NOT tapered,
                                                      # the landed kend rule
         + sum_{p in PNBRQ} x[p][s]
             * (mg[p][s] * ph + eg[p][s] * (24 - ph)) / 24

  x[p][s] = (# white p on s) - (# black p on 63 - s), the 180-degree mirror
  the engine's `119 - i` performs (a rank flip alone would be the WRONG
  mirror and would fit a table the engine never reads).
  Kt = K_END when either queen is off, else K_MID -- the queens-off rule the
  entry ships. `ph` is the standard 24-point phase, N=B=1 R=2 Q=4, clamped.

Everything except eg is therefore KNOWN, so this is one linear least-squares
solve in 320 unknowns. It is solved through NORMAL EQUATIONS accumulated in
chunks: A is 320x320 whatever the row count, so millions of positions cost
memory that does not grow.

OBJECTIVE. The lane scores candidates in win-probability space with K=350,
not in centipawns; a plain cp least-squares would spend the table on the
+-1000 tail. The weights here are the linearisation of that objective,
w = sigma'(cp/K)^2, which keeps the solve closed-form while putting the
error where the lane measures it. Both losses are reported on a held-out
split, and the split is the corpus's OWN val_a flag -- not a fresh one --
so this number is comparable to every other fit in the ledger.

usage: fit_taper.py OUT.json [N_TRAIN] [N_VAL]
"""
import json, os, sys, time
import numpy as np

CORPUS = "/home/thomas-ahle/sunfish-bench/replnet-20260814/train/pool10m.npz"
OUT = sys.argv[1]
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 2000000
N_VAL = int(sys.argv[3]) if len(sys.argv) > 3 else 400000
PIECES = "PNBRQK"
PHW = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
KSIG = 350.0
SEED = 20260817

# ---- classic's tables, transcribed from sunfish.py values ------------------
piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}
CLASSIC = json.load(open(os.environ.get("SF_CLASSIC_TABLES", "classic_tables.json")))   # {piece: 64 ints, rank-8 first}
mg = {p: np.array(CLASSIC[p], dtype=np.float64) + piece[p] for p in PIECES}
# K_END, classic's centralization gradient, in the same 64-square frame. The
# entry builds it over the padded 120 board; index 21 + r*10 + f is square
# r*8 + f here, and the formula is reproduced rather than imported so this
# script has no dependency on a checkout.
K_END = np.array([60000 + 70 - 10 * (abs(2 * ((21 + r * 10 + f) // 10) - 11)
                                     + abs(2 * ((21 + r * 10 + f) % 10) - 9))
                  for r in range(8) for f in range(8)], dtype=np.float64)
K_MID = mg["K"]

t0 = time.time()
print("loading corpus ...", flush=True)
d = np.load(CORPUS, allow_pickle=False)
cp_all = d["cp"]
va_all = d["val_a"]
n_all = len(cp_all)
rng = np.random.default_rng(SEED)
tr_idx = np.flatnonzero(~va_all)
va_idx = np.flatnonzero(va_all)
tr_idx = rng.choice(tr_idx, min(N_TRAIN, len(tr_idx)), replace=False)
va_idx = rng.choice(va_idx, min(N_VAL, len(va_idx)), replace=False)
tr_idx.sort(); va_idx.sort()
print("corpus %d rows; train %d, val %d (%.1fs)" % (n_all, len(tr_idx), len(va_idx), time.time() - t0), flush=True)

fens_all = d["fens"]

PIDX = {p: i for i, p in enumerate(PIECES)}


def featurize(idx):
    """x (n, 6, 64) int8-ish, phase, queens-off flag, cp."""
    n = len(idx)
    X = np.zeros((n, 6, 64), dtype=np.float32)
    ph = np.zeros(n, dtype=np.float32)
    qoff = np.zeros(n, dtype=bool)
    # THE FRAME, and it bit once. The corpus meta says `"frame": "labels
    # side-to-move"` -- cp is the MOVER's score, and the dump's own evals were
    # negated for black to make it so. These features are WHITE-framed. Feeding
    # the two together sign-scrambles every black-to-move row, half the corpus,
    # and the fit answers by driving the piece values toward zero, which really
    # is the best linear fit to a sign-scrambled target. IT DOES NOT LOOK LIKE A
    # BUG FROM THE LOSS: the first run reported a 24.5% held-out
    # win-probability IMPROVEMENT over classic. The tell was in the tables --
    # the queen came back 1,146 cp below its own midgame value.
    stm = np.ones(n, dtype=np.float64)
    fens = fens_all[idx]
    for row, fen in enumerate(fens):
        parts = fen.split(" ")
        board = parts[0]
        if len(parts) > 1 and parts[1] == "b":
            stm[row] = -1.0
        s = 0
        p_ = 0
        for c in board:
            if c == "/":
                continue
            if c.isdigit():
                s += int(c); continue
            up = c.upper()
            pi = PIDX[up]
            if c.isupper():
                X[row, pi, s] += 1.0
            else:
                X[row, pi, 63 - s] -= 1.0
            p_ += PHW[up]
            s += 1
        ph[row] = min(24, p_)
        qoff[row] = ("Q" not in board) or ("q" not in board)
    return X, ph, qoff, cp_all[idx].astype(np.float64) * stm


def known_part(X, ph, qoff):
    """Everything the fit does NOT move: the king term and the mg half."""
    Kt = np.where(qoff[:, None], K_END[None, :], K_MID[None, :])
    out = np.einsum("ns,ns->n", X[:, PIDX["K"], :], Kt)
    for p in "PNBRQ":
        out += (X[:, PIDX[p], :] @ mg[p]) * (ph / 24.0)
    return out


def design(X, ph):
    """The 320-column eg design: x[p][s] * (24 - ph) / 24, p in PNBRQ."""
    w = ((24.0 - ph) / 24.0)[:, None]
    return np.concatenate([X[:, PIDX[p], :] * w for p in "PNBRQ"], axis=1)


CHUNK = 200000
A = np.zeros((320, 320)); b = np.zeros(320)
for c0 in range(0, len(tr_idx), CHUNK):
    idx = tr_idx[c0:c0 + CHUNK]
    X, ph, qoff, cp = featurize(idx)
    resid = cp - known_part(X, ph, qoff)
    D = design(X, ph).astype(np.float64)
    # sigma'(cp/K)^2 -- the linearised win-probability objective
    sg = 1.0 / (1.0 + np.exp(-cp / KSIG))
    w = (sg * (1 - sg) / KSIG) ** 2
    w /= w.mean()
    Dw = D * w[:, None]
    A += Dw.T @ D
    b += Dw.T @ resid
    print("  chunk %d/%d  (%.0fs)" % (c0 // CHUNK + 1, (len(tr_idx) + CHUNK - 1) // CHUNK,
                                      time.time() - t0), flush=True)

# Ridge, small and stated: 320 parameters against millions of rows needs almost
# none, but squares that never occur (a pawn on rank 1 or 8) are EXACTLY
# singular and would otherwise come back as arbitrary large numbers that then
# cost bytes to store. lam pins those at the mg value and moves nothing else.
lam = 1e-3 * np.trace(A) / 320
prior = np.concatenate([mg[p] for p in "PNBRQ"])
Areg, breg = A + lam * np.eye(320), b + lam * prior
eg_free = np.linalg.solve(Areg, breg)

# ---- THE MEAN CONSTRAINT, and why the free fit is not the answer ----------
# PHASE IS A FUNCTION OF THE MATERIAL, so in `(mg*ph + eg*(24-ph))/24` the
# eg block and the phase are collinear: a lone rook at ph=2 IS what makes the
# phase 2. The free fit exploits that and comes back with a queen worth 635 cp
# less in the endgame table than in the midgame one -- it is re-pricing
# MATERIAL through the taper, which is not what a taper is for, is not what
# this experiment asked, and is expensive twice over (a large mg-to-eg delta
# is exactly what the delta encoding cannot compress).
#
# So the constrained fit pins each piece's eg table to the SAME MEAN as its
# mg table: five linear equalities, solved exactly through the KKT system
# rather than by re-centring afterwards (re-centring a least-squares solution
# gives a different, worse answer than solving the constrained problem). What
# survives is purely the SHAPE difference -- where on the board a piece is
# worth more in an endgame -- which is the only thing a second table set can
# say that the first one cannot.
#
# BOTH are reported. If the constrained fit keeps most of the free fit's
# held-out gain, the taper is carrying positional information; if the gain
# collapses, the free fit's number was material re-pricing wearing a taper's
# clothes, and that is the finding.
C = np.zeros((5, 320))
dvec = np.zeros(5)
for i, p in enumerate("PNBRQ"):
    C[i, i * 64:(i + 1) * 64] = 1.0 / 64
    dvec[i] = mg[p].mean()
KKT = np.zeros((325, 325))
KKT[:320, :320] = Areg
KKT[:320, 320:] = C.T
KKT[320:, :320] = C
rhs = np.concatenate([breg, dvec])
eg = np.linalg.solve(KKT, rhs)[:320]
print("solved (%.0fs)  free and mean-constrained" % (time.time() - t0), flush=True)


def losses(idx, eg_vec):
    tot_cp = tot_wp = 0.0; n = 0
    base_cp = base_wp = 0.0
    for c0 in range(0, len(idx), CHUNK):
        ii = idx[c0:c0 + CHUNK]
        X, ph, qoff, cp = featurize(ii)
        k = known_part(X, ph, qoff)
        pred = k + design(X, ph) @ eg_vec
        # BASELINE: the shipped engine. mg at full weight for PNBRQ (no taper)
        # plus the same king term -- i.e. what the entry evaluates today,
        # minus pend, which is a separate landed term and is not this fit's
        # to claim or to lose.
        bpred = np.einsum("ns,ns->n", X[:, PIDX["K"], :],
                          np.where(qoff[:, None], K_END[None, :], K_MID[None, :]))
        for p in "PNBRQ":
            bpred = bpred + X[:, PIDX[p], :] @ mg[p]
        s = lambda v: 1.0 / (1.0 + np.exp(-v / KSIG))
        tot_cp += float(((pred - cp) ** 2).sum()); base_cp += float(((bpred - cp) ** 2).sum())
        tot_wp += float(((s(pred) - s(cp)) ** 2).sum()); base_wp += float(((s(bpred) - s(cp)) ** 2).sum())
        n += len(ii)
    return tot_cp / n, tot_wp / n, base_cp / n, base_wp / n


for arm, vec in (("free", eg_free), ("constrained", eg)):
    for name, idx in (("train", tr_idx[:N_VAL]), ("VAL", va_idx)):
        c, w, bc, bw = losses(idx, vec)
        print("%-12s %-6s  fit cp-mse %10.1f  wp-mse %.6f   |  base cp-mse %10.1f  "
              "wp-mse %.6f  (wp %+.2f%%)"
              % (arm, name, c, w, bc, bw, 100 * (w - bw) / bw), flush=True)

res = {}
for arm, vec in (("free", eg_free), ("constrained", eg)):
    out = {p: [int(round(v)) for v in vec[i * 64:(i + 1) * 64]] for i, p in enumerate("PNBRQ")}
    delta = {p: [int(o - m) for o, m in zip(out[p], mg[p])] for p in "PNBRQ"}
    print("%s: eg-minus-mg delta per piece" % arm)
    for p in "PNBRQ":
        dd = delta[p]
        print("  %s  min %5d  max %5d  mean %7.1f  |d|>32: %2d/64  span %d" %
              (p, min(dd), max(dd), sum(dd) / 64.0,
               sum(1 for v in dd if abs(v) > 32), max(dd) - min(dd)))
    res[arm] = {"eg": out, "delta": delta}
json.dump(dict(res, n_train=len(tr_idx), n_val=len(va_idx), ksig=KSIG, seed=SEED),
          open(OUT, "w"), indent=1)
print("wrote", OUT, "(%.0fs total)" % (time.time() - t0))
