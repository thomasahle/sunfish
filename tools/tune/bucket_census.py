"""Positions per PARAMETER, per bucket -- the gate a budget-filling fit has to
pass before any of it is worth fitting.

Bytes are not the binding constraint on a bucketed eval; data is. Every bucket
is fitted from ITS OWN positions only, so the count that matters is not
19,491/N_params globally but

    (train positions falling in bucket b) / (free parameters in bucket b)

evaluated at the WORST bucket, because that is the one that will memorise.

The reference point is measured, not assumed: the queens-seam taper was
dropped at **11 positions per parameter**, where its train/held-out gap was the
widest of three fits and the packed artifact played a2a3 to depth 5. So 11 is
known-bad and anything near it is a prediction of the same failure.

Phase and king wings come from `fens`. `X` is a difference feature and cancels
mirrored pairs -- it can give neither.

usage: bucket_census.py [DATA.npz]
"""
import json
import os
import pathlib
import sys

import numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[2])
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "tools/tune/data/set20260813.npz")
SEED = 20260813
TAPER_FAILED_AT = 11.0

d = np.load(DATA, allow_pickle=False)
fens = [str(f) for f in d["fens"]]
n = len(fens)
meta = json.loads(str(d["meta"]))
ntr = int(0.8 * n)
tr = np.random.default_rng(SEED).permutation(n)[:ntr]

# Facts read off the board, exactly as the ROOT selector will read them.
queens = np.zeros(n, dtype=bool)
wk_king = np.zeros(n, dtype=bool)   # white king on the king side (file e..h)
bk_king = np.zeros(n, dtype=bool)
for i, fen in enumerate(fens):
    board = fen.split()[0]
    queens[i] = "Q" in board and "q" in board
    for row, rank in enumerate(board.split("/")):
        f = 0
        for c in rank:
            if c.isdigit():
                f += int(c)
                continue
            if c == "K":
                wk_king[i] = f > 3
            elif c == "k":
                bk_king[i] = f > 3
            f += 1

print("data: %s | %d positions (%d train) | %s depth %d"
      % (os.path.basename(DATA), n, ntr, meta["engine"], meta["depth"]))
print("queens on both sides: %.1f%% | white king on king side: %.1f%% | black: %.1f%%\n"
      % (100 * queens.mean(), 100 * wk_king.mean(), 100 * bk_king.mean()))

# Phase quantiles: as BALANCED AS AN INTEGER PHASE ALLOWS, which is much better
# than king wings (80.4% of our positions have the white king on the king side,
# so a 4-wing product has a 981-position corner and an 8-set one has 48) but is
# NOT equal-count -- see quantile_idx for why that distinction cost a factor of
# two in the reported figure for 8 buckets.
PHASE_W = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
ph = np.array([min(24, sum(PHASE_W[c.upper()] for c in f.split()[0] if c.isalpha()))
               for f in fens], dtype=np.float64)

def quantile_idx(k):
    """Phase quantiles AS THE ENGINE CAN IMPLEMENT THEM: a map from the integer
    phase 0..24 to a bucket. Whole phase values cannot be subdivided.

    An earlier version of this function ranked positions and cut the ranks into
    k equal parts, which reported 12.0 positions/parameter for 8 buckets. That
    partition is not implementable: phase is a coarse integer with a very lumpy
    histogram (2,300 of 15,592 training positions sit at phase 4 alone), so a
    rank cut splits a single phase value across two buckets and the root, which
    sees only the phase, cannot reproduce it. The honest number for 8 buckets is
    6.5, not 12.0 -- the instrument was measuring a partition that cannot ship.
    """
    # counts from TRAIN rows only -- the boundaries are a fitted choice
    cnt = np.bincount(ph[tr].astype(int), minlength=25)
    target = cnt.sum() / float(k)
    lut, b, acc = [], 0, 0
    for v in range(25):
        if acc + cnt[v] / 2.0 > target * (b + 1) and b < k - 1:
            b += 1
        lut.append(b)
        acc += cnt[v]
    LUTS[k] = "".join(str(x) for x in lut)
    return np.array([lut[int(p)] for p in ph], dtype=int)


LUTS = {}

PARTITIONS = [
    ("1 flat", np.zeros(n, dtype=int), 1),
    ("2 seam (queens-off)", (~queens).astype(int), 2),
    ("2 wings (same/opp)", (wk_king != bk_king).astype(int), 2),
    ("4 wings (wk x bk)", wk_king.astype(int) * 2 + bk_king.astype(int), 4),
    ("4 seam x wings2", (~queens).astype(int) * 2 + (wk_king != bk_king).astype(int), 4),
    ("8 seam x wings4", (~queens).astype(int) * 4 + wk_king.astype(int) * 2
     + bk_king.astype(int), 8),
    ("2 phase halves", quantile_idx(2), 2),
    ("4 phase quartiles", quantile_idx(4), 4),
    ("6 phase sextiles", quantile_idx(6), 6),
    ("7 phase septiles", quantile_idx(7), 7),
    ("8 phase octiles", quantile_idx(8), 8),
]

# 5 tables per set (K is held exact and unbucketed -- the landed kend fix)
for pname, params_per_set in (("step 8/4/2, MIRRORED", 160), ("unmirrored", 320)):
    print("=== %s: %d free parameters per set ===" % (pname, params_per_set))
    print("%-22s %6s %10s %12s %11s  %s"
          % ("partition", "sets", "params", "worst bucket", "pos/param", "verdict"))
    for label, idx, nsets in PARTITIONS:
        counts = np.array([(idx[tr] == b).sum() for b in range(nsets)])
        worst = counts.min()
        ppp = worst / float(params_per_set)
        verdict = ("OK" if ppp >= 4 * TAPER_FAILED_AT else
                   "THIN" if ppp >= TAPER_FAILED_AT else "BELOW THE TAPER'S FAILURE POINT")
        print("%-22s %6d %10d %12d %11.1f  %s"
              % (label, nsets, nsets * params_per_set, worst, ppp, verdict))
        if nsets > 1:
            print("%-22s        occupancy: %s" % ("", " ".join("%d" % c for c in counts)))
    print()

print("phase->bucket maps the root selector would index (from TRAIN counts):")
for k in sorted(LUTS):
    print("  %d buckets: %s" % (k, LUTS[k]))
print()
print("Reference: the queens-seam taper was DROPPED at %.0f positions/parameter." % TAPER_FAILED_AT)
print("A distilled teacher removes this constraint entirely -- it can label as many")
print("positions as we care to sample, so pos/param stops being a function of the")
print("Stockfish labelling budget. That is the point of using a net as a GENERATOR.")
