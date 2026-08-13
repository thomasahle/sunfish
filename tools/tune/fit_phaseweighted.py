"""Phase-REWEIGHTED flat refits: does correcting the set's endgame skew pay?

The labelled set is endgame-heavy on purpose (65.6% at <= 16 pieces), which
makes every uniform fit a fit to endgames. This asks the obvious follow-up --
would weighting the training rows toward the middlegame produce a better
table -- and it asks it with the bar written down first, because a reweighting
has an unfair advantage on whichever metric it was reweighted toward.

THE TRAP THIS FILE IS BUILT AROUND. `X` is a DIFFERENCE feature: a white pawn
on e2 and a black pawn on e7 cancel in the same cell, so |X|.sum() reads 11.1
pieces where the board has 14.3. Phase must come from `fens`, and it does --
counted off the FEN board field, never off X.

PRE-REGISTERED, before any result below was looked at
-----------------------------------------------------
Primary metric M1: unweighted held-out loss on the pinned 80/20 split (seed
20260813) -- the same number the C1/C2 candidates were selected on, so it is
comparable to the ledger.

Secondary metric M2: PHASE-BALANCED held-out loss -- validation rows reweighted
to a flat phase density, with the density estimated on TRAIN rows only. This is
the metric a reweighting is supposed to win, so it cannot be the primary; it is
here to show whether a reweighting buys anything at all.

Bar for producing a THIRD candidate beyond C1/C2:
  * M1: paired-bootstrap 95% interval of (uniform - reweighted) held-out loss
    must be STRICTLY ABOVE ZERO. Same validation rows for both arms, resampled
    together, so the split's own luck cancels.
  * M2: that interval must not be strictly below zero.
A reweighting that wins only M2 is recorded as a negative, not shipped: it
would mean the reweighting only helps on the distribution it assumed.

Every candidate is re-scored twice from tables it did not fit: once from its
own emitted integers, and once from the tables the CODEC actually decodes at
the shipping quantisation (step 8, mirrored, K exact -- C1's encoding). The
mirrored-king lesson is that the fit never sees the emit, so the emit gets
scored separately or it is not measured at all.

usage: fit_phaseweighted.py [DATA.npz]
"""
import json
import os
import pathlib
import re
import sys

import numpy as np
from scipy.optimize import minimize

REPO = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, os.path.join(REPO, "tools/eval4k"))
import codec  # noqa: E402
import splice  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "tools/tune/data/set20260813.npz")
PIECES = "PNBRQK"
PHASE = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
K = 350.0
KING = slice(5 * 64, 6 * 64)
SEED = 20260813

# ---- data, and the phase read off the BOARD ---------------------------------
d = np.load(DATA, allow_pickle=False)
X = d["X"].astype(np.float64)
y = d["y"].astype(np.float64)
fens = [str(f) for f in d["fens"]]
n = len(y)
meta = json.loads(str(d["meta"]))
ph = np.array([min(24, sum(PHASE[c.upper()] for c in f.split()[0] if c.isalpha())) for f in fens],
              dtype=np.float64)
print("data: %s | %d positions | %s depth %d" % (os.path.basename(DATA), n, meta["engine"], meta["depth"]))
print("phase from FENS: mean %.2f/24, %.1f%% below 12" % (ph.mean(), 100 * (ph < 12).mean()))
# The difference-feature control, stated as a number rather than as a warning.
print("control: |X|.sum() reads %.2f pieces/position, the boards hold %.2f -- X cannot give phase"
      % (np.abs(X).sum(1).mean(), np.mean([sum(c.isalpha() for c in f.split()[0]) for f in fens])))

rng = np.random.default_rng(SEED)
perm = rng.permutation(n)
ntr = int(0.8 * n)
tr, va = perm[:ntr], perm[ntr:]
t_all = 1.0 / (1.0 + np.exp(-y / K))

# ---- warm start: classic's tables, piece values folded in -------------------
src = open(REPO + "/sunfish.py").read()
piece0 = eval(re.search(r"^piece = (\{[^}]*\})", src, re.M).group(1))
pst0 = eval(re.search(r"^pst = (\{.*?^\})", src, re.M | re.S).group(1))
w0 = np.zeros(384)
for pi, p in enumerate(PIECES):
    tab = np.array(pst0[p], dtype=np.float64) + piece0[p]
    w0[pi * 64:(pi + 1) * 64] = tab.reshape(8, 8)[::-1].reshape(64)

# ---- the weightings ---------------------------------------------------------
# Density of phase on TRAIN rows only. Estimated once, used for both the
# training weights and the M2 validation weights, so no validation information
# reaches any fit.
BINS = np.arange(0, 26, 2.0)
cnt, _ = np.histogram(ph[tr], bins=BINS)
dens = np.maximum(cnt, 1).astype(np.float64) / cnt.sum()
bin_of = np.clip(np.digitize(ph, BINS) - 1, 0, len(dens) - 1)
inv = 1.0 / dens[bin_of]

WEIGHTS = {
    "uniform": np.ones(n),
    "flatphase": inv,                      # full correction to a flat phase density
    "sqrtflat": np.sqrt(inv),              # half correction, in log space
    "mgtilt": 1.0 + ph / 24.0,             # mild middlegame emphasis, no row zeroed
    "mgonly": ph / 24.0,                   # aggressive: endgames toward zero weight
}
for k in WEIGHTS:
    WEIGHTS[k] = WEIGHTS[k] / WEIGHTS[k][tr].mean()
w_m2 = inv / inv[va].mean()               # M2: phase-balanced validation weights


def fit(w_init, sw):
    """Weighted MSE-on-win-probability over `tr`. The king block stays frozen:
    its 60000 is a structural sentinel and K_MID/K_END is the landed kend fix."""
    A, tt, s = X[tr], t_all[tr], sw[tr]
    frozen = np.zeros(384, dtype=bool)
    frozen[KING] = True
    free = ~frozen
    sm = s.sum()

    def f(v):
        w = w_init.copy()
        w[free] = v
        p = 1.0 / (1.0 + np.exp(-(A @ w) / K))
        r = p - tt
        g = (2.0 / sm) * (A.T @ (s * r * p * (1 - p) / K))
        return float(np.sum(s * r ** 2) / sm), g[free]

    res = minimize(f, w_init[free], jac=True, method="L-BFGS-B",
                   options={"maxiter": 3000, "ftol": 1e-14, "gtol": 1e-12})
    w = w_init.copy()
    w[free] = res.x
    return w, res.nit


def sqerr(w, idx):
    """Per-row squared error -- kept per row so the bootstrap can resample it."""
    p = 1.0 / (1.0 + np.exp(-(X[idx] @ w) / K))
    return (p - t_all[idx]) ** 2


def emit(w):
    out = {}
    for pi, p in enumerate(PIECES):
        tab = w[pi * 64:(pi + 1) * 64]
        if p == "K":
            out[p] = list(pst0["K"])
            out["_value_K"] = piece0["K"]
            continue
        base = float(np.median(tab))
        out[p] = np.round(tab - base).astype(int).reshape(8, 8)[::-1].reshape(64).tolist()
        out["_value_" + p] = int(round(base))
    return out


def unemit(out):
    w = np.zeros(384)
    for pi, p in enumerate(PIECES):
        tab = np.array(out[p], dtype=np.float64) + out["_value_" + p]
        w[pi * 64:(pi + 1) * 64] = tab.reshape(8, 8)[::-1].reshape(64)
    return w


def decoded(out, step=8, half=True, kexact=True):
    """Weights the ARTIFACT would hold: run the codec's own source and read
    `pst` back. Not a model of the rounding -- the rounding itself."""
    raw = {p: out[p] for p in PIECES}
    vals = {p: out["_value_" + p] for p in PIECES}
    ns = {}
    exec(codec.emit(vals, raw, step, half, exact="K" if kexact else ""), ns)
    w = np.zeros(384)
    for pi, p in enumerate(PIECES):
        pad = ns["pst"][p]
        flat = np.array([v for r in range(8) for v in pad[20 + r * 10 + 1: 20 + r * 10 + 9]],
                        dtype=np.float64)
        w[pi * 64:(pi + 1) * 64] = flat.reshape(8, 8)[::-1].reshape(64)
    return w


# ---- fit every weighting ----------------------------------------------------
base_e = sqerr(w0, va)
print("\nclassic, no fit:  M1 %.6f   M2 %.6f\n" % (base_e.mean(), np.average(base_e, weights=w_m2[va])))
print("%-11s %10s %10s %11s %11s %11s" % ("weighting", "M1", "M2", "M1 emit", "M1 decoded", "vs classic"))
res = {}
for name, sw in WEIGHTS.items():
    w, nit = fit(w0, sw)
    out = emit(w)
    e = sqerr(w, va)
    e_emit = sqerr(unemit(out), va)
    e_dec = sqerr(decoded(out), va)
    res[name] = dict(w=w, out=out, e=e, e_emit=e_emit, e_dec=e_dec)
    print("%-11s %10.6f %10.6f %11.6f %11.6f %+10.2f%%  [%d it]"
          % (name, e.mean(), np.average(e, weights=w_m2[va]), e_emit.mean(), e_dec.mean(),
             100 * (e.mean() - base_e.mean()) / base_e.mean(), nit))
    assert abs(e_emit.mean() - e.mean()) < 2e-5, "%s: EMIT MISMATCH" % name

# ---- the pre-registered paired bootstrap ------------------------------------
print("\npaired bootstrap vs `uniform`, 10,000 resamples of the SAME held-out rows")
print("positive = the reweighting is BETTER; the bar is a 95%% interval strictly above 0\n")
brng = np.random.default_rng(SEED + 1)
nv = len(va)
idx = brng.integers(0, nv, size=(10000, nv))
u = res["uniform"]
print("%-11s %26s %26s  %s" % ("weighting", "M1 delta (95% CI)", "M2 delta (95% CI)", "verdict"))
verdicts = {}
for name in WEIGHTS:
    if name == "uniform":
        continue
    r = res[name]
    out = []
    for metric, wt in (("M1", None), ("M2", w_m2[va])):
        du = u["e"] if wt is None else u["e"] * wt / wt.mean()
        dr = r["e"] if wt is None else r["e"] * wt / wt.mean()
        diff = du - dr
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append((diff.mean(), lo, hi))
    (m1, l1, h1), (m2, l2, h2) = out
    passes = l1 > 0 and not (h2 < 0)
    verdicts[name] = passes
    print("%-11s %+.6f [%+.6f,%+.6f] %+.6f [%+.6f,%+.6f]  %s"
          % (name, m1, l1, h1, m2, l2, h2, "PASSES BAR" if passes else "no"))

win = [k for k, v in verdicts.items() if v]
print("\nreweightings clearing the pre-registered bar: %s" % (", ".join(win) if win else "NONE"))
if win:
    best = min(win, key=lambda k: res[k]["e"].mean())
    json.dump({"weighting": best, "heldout": float(res[best]["e"].mean()),
               "heldout_emit": float(res[best]["e_emit"].mean()),
               "heldout_decoded": float(res[best]["e_dec"].mean()),
               "tables": res[best]["out"]},
              open(os.path.join(REPO, "tools/tune/candidates/fit_phase.json"), "w"))
    print("wrote candidates/fit_phase.json for %s -- a CANDIDATE, not an Elo claim." % best)
else:
    print("No third candidate. C1 and C2 stand as the screening pair.")

# ---- where the loss actually lives, by phase band ---------------------------
# A reweighting can only move loss BETWEEN phase bands. If the flat table is
# already at its capacity in every band, there is nothing to move, and that is
# a different diagnosis from "the training set is skewed".
print("\nheld-out loss by phase band (why reweighting has nothing to move):")
try:
    F = json.load(open(os.path.join(REPO, "tools/tune/candidates/fits.json")))
    B = F["qseam"]["tables"]
    w_on, w_off = unemit(B["qon"]), unemit(B["qoff"])
    qv = np.array(["Q" in fens[i].split()[0] and "q" in fens[i].split()[0] for i in va])
    p_t = 1.0 / (1.0 + np.exp(-np.where(qv, X[va] @ w_on, X[va] @ w_off) / K))
    e_taper = (p_t - t_all[va]) ** 2
except (IOError, KeyError):
    e_taper = None
bands = [(0, 6, "0-5  deep eg"), (6, 12, "6-11 endgame"),
         (12, 18, "12-17 middle"), (18, 25, "18-24 opening")]
hdr = "%-14s %6s %10s %10s %10s %10s" % ("band", "rows", "classic", "uniform", "mgtilt", "flatphase")
print(hdr + ("%12s" % "taper(768p)" if e_taper is not None else ""))
for lo, hi, lab in bands:
    m = (ph[va] >= lo) & (ph[va] < hi)
    if not m.any():
        continue
    line = "%-14s %6d %10.6f %10.6f %10.6f %10.6f" % (
        lab, m.sum(), base_e[m].mean(), res["uniform"]["e"][m].mean(),
        res["mgtilt"]["e"][m].mean(), res["flatphase"]["e"][m].mean())
    if e_taper is not None:
        line += "%12.6f" % e_taper[m].mean()
    print(line)
