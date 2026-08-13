"""Close the loop: fit -> emit -> encode -> DECODE -> held-out loss -> bytes.

The exact candidates priced in `price_candidates.py` answer "what does this fit
cost", and for the tapered one the answer was +670 bytes -- against a ledger
projection of ~134, because that projection assumed a MIRRORED STEP-8 second
table and this stored an exact one. Quantisation is not a detail here; it is
the whole question of whether a second table set is affordable.

Quantisation changes the evaluation, so it cannot be priced in bytes alone. The
loss below is computed from the tables the artifact will actually hold: the
emitted region source is EXECUTED, `pst` is read back out, the padding stripped
and the piece value removed. No reimplementation of the codec's rounding, so no
drift between what is measured and what ships.

usage: price_quantised.py FITS.json
"""
import json
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TOOLS, "eval4k"))
import codec  # noqa: E402
import measure  # noqa: E402
import splice  # noqa: E402

FITS = json.load(open(sys.argv[1]))
ROOT = measure.ROOT
BASE = open(os.path.join(ROOT, splice.ENTRY)).read()
PIECES = "PNBRQK"
K = 350.0
ROOT_OLD = '        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END\n'

# ---- the held-out split, reproduced exactly as the fit made it --------------
d = np.load(os.path.join(ROOT, "tools/tune/data/set20260813.npz"), allow_pickle=False)
X, y, fens = d["X"].astype(np.float64), d["y"].astype(np.float64), [str(f) for f in d["fens"]]
n = len(y)
queens = np.array(["Q" in f.split()[0] and "q" in f.split()[0] for f in fens])
perm = np.random.default_rng(20260813).permutation(n)
va = perm[int(0.8 * n):]
t_all = 1.0 / (1.0 + np.exp(-y / K))


def decoded_w(region_src, name="pst"):
    """Execute the emitted region and read the tables back as a weight vector.

    This is the artifact's own arithmetic, not a model of it."""
    ns = {}
    exec(region_src, ns)
    w = np.zeros(384)
    for pi, p in enumerate(PIECES):
        padded = ns[name][p]
        rows = [padded[20 + r * 10 + 1: 20 + r * 10 + 9] for r in range(8)]
        flat = np.array([v for row in rows for v in row], dtype=np.float64)
        # emit() writes rank 8 first; features index chess squares (A1 = 0)
        w[pi * 64:(pi + 1) * 64] = flat.reshape(8, 8)[::-1].reshape(64)
    return w


def loss_of(w_qon, w_qoff=None):
    if w_qoff is None:
        s = X[va] @ w_qon
    else:
        s = np.where(queens[va], X[va] @ w_qon, X[va] @ w_qoff)
    p = 1.0 / (1.0 + np.exp(-s / K))
    return float(np.mean((p - t_all[va]) ** 2))


def emit_for(tables, step=1, half=False, name="pst"):
    raw = {p: tables[p] for p in PIECES}
    vals = {p: tables["_value_" + p] for p in PIECES}
    src = codec.emit(vals, raw, step, half)
    src = src.replace(src.split("\n")[0] + "\n",
                      "piece = {%s}\n" % ", ".join('"%s": %d' % (p, vals[p]) for p in PIECES), 1)
    if name != "pst":
        src = src.replace("pst = {}", name + " = {}").replace("pst[_k]", name + "[_k]")
        src = src.replace('K_MID, K_END = pst["K"]', 'K_MID, K_END = %s["K"]' % name)
    return src


_cval, _craw = splice.classic_tables()
_classic = {p: [int(v) for v in _craw[p]] for p in PIECES}
_classic.update({"_value_" + p: int(_cval[p]) for p in PIECES})
base_va = loss_of(decoded_w(emit_for(_classic)))
print("classic, as the artifact decodes it:  held-out %.6f\n" % base_va)

_, BASESIZE = measure.pack(BASE, "base")
A, B = FITS["flat"]["tables"], FITS["qseam"]["tables"]

print("%-34s %6s %7s %9s %10s" % ("candidate", "bytes", "vs 3378", "spare", "held-out"))
rows = []


def row(label, src, w_on, w_off=None):
    packed, size = measure.pack(src, label.replace(" ", "_")[:24])
    mv = measure.standalone(packed)
    L = loss_of(w_on, w_off)
    print("%-34s %6d %+7d %9d %10.6f  %+6.2f%%  %s"
          % (label, size, size - BASESIZE, 4096 - size, L, 100 * (L - base_va) / base_va, mv))
    rows.append((label, size, L))
    return size, L


# ---- A, flat, at three quantisations ----------------------------------------
for step, half, tag in ((1, False, "exact"), (2, False, "step2"), (8, False, "step8")):
    reg = emit_for(A, step, half)
    row("A flat refit, %s" % tag, splice.splice(BASE, reg), decoded_w(reg))

# ---- B, queens seam: exact mg + progressively cheaper second set ------------
regB1 = emit_for(B["qon"])
for step, half, tag in ((1, False, "exact"), (4, False, "step4"), (8, False, "step8"),
                        (8, True, "mirrored step8")):
    reg2 = emit_for(B["qoff"], step, half, "pst2").replace(
        'K_MID, K_END = pst2["K"], tuple(piece["K"] + 70\n'
        "   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))\n", "")
    v2 = "piece2 = {%s}\n" % ", ".join('"%s": %d' % (p, B["qoff"]["_value_" + p]) for p in PIECES)
    reg2 = reg2.replace(reg2.split("\n")[0] + "\n", v2, 1).replace("piece[_k]", "piece2[_k]")
    reg = regB1 + "\n" + reg2
    src = splice.splice(BASE, reg).replace(
        ROOT_OLD,
        '        pst.update(pst2 if "Q" not in pos.board or "q" not in pos.board else pst1)\n'
        + ROOT_OLD)
    src = src.replace("pst = {}\n", "pst = {}\npst1 = {}\n", 1)
    src = src.replace(' pst[_k] = tuple([0] * 20', ' pst[_k] = pst1[_k] = tuple([0] * 20', 1)
    row("B queens-seam, 2nd set %s" % tag, src,
        decoded_w(regB1), decoded_w(reg2, "pst2"))

print("\nNo Elo is claimed. These are candidates and prices; games decide.")
json.dump([(a, b, c) for a, b, c in rows], open(
    os.path.join(os.path.dirname(sys.argv[1]), "prices.json"), "w"), indent=1)
