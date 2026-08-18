"""Cluster-robust Elo, clustered on the OPENING LINE.

WHY. `pair_elo.py` treats games (or colour-pairs) as the independent unit. That
is only true if each game comes from a different opening. Measured on this
lane's own round-robins, it does not: `-rounds 150 -games 2 -repeat` with three
engines drew ~49 distinct 16-ply book lines per pairing and played each of them
SIX times. Six games from one opening between two near-identical deterministic
engines are not six independent observations, so the naive interval is too
narrow by whatever the within-opening correlation is.

This does NOT assume a variance-inflation factor. It computes the
cluster-robust standard error of the mean score directly, with the OPENING as
the cluster, which recovers the naive interval when replays are independent and
widens it exactly as much as the observed correlation warrants.

    se^2 = sum_g ( sum_{i in g} (x_i - xbar) )^2 / N^2

and the Elo interval comes from the logistic delta method,
dElo/dp = 400 / (ln 10 * p * (1-p)).

usage: cluster_elo.py FILE.pgn [ARM_A ARM_B]
"""
import collections, math, re, sys

path = sys.argv[1]
want = tuple(sorted(sys.argv[2:4])) if len(sys.argv) > 3 else None
txt = open(path).read()
games = [g for g in re.split(r"\n(?=\[Event )", txt) if "[Result" in g]

recs = []
for g in games:
    w = re.search(r'^\[White "([^"]*)"\]', g, re.M).group(1)
    b = re.search(r'^\[Black "([^"]*)"\]', g, re.M).group(1)
    r = re.search(r'^\[Result "([^"]*)"\]', g, re.M).group(1)
    body = re.split(r"\]\n\n", g, maxsplit=1)
    mv = " ".join(body[1].split()) if len(body) > 1 else ""
    toks = [t for t in mv.split() if not t.endswith(".")
            and t not in ("1-0", "0-1", "1/2-1/2", "*")]
    line = " ".join(toks[:16])           # the book line is exactly 16 plies
    recs.append((w, b, r, line))

def elo(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -400 * math.log10(1 / p - 1)

pairs = sorted({tuple(sorted((r[0], r[1]))) for r in recs})
print("%s   %d games" % (path.split("/")[-1], len(recs)))
for pair in pairs:
    if want and pair != want: continue
    A, B = pair
    sub = [r for r in recs if tuple(sorted((r[0], r[1]))) == pair]
    xs, gs = [], []
    for w, b, res, line in sub:
        s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[res]
        xs.append(s if w == A else 1.0 - s)      # score for A
        gs.append(line)
    n = len(xs); xbar = sum(xs) / n
    by = collections.defaultdict(list)
    for x, g in zip(xs, gs): by[g].append(x)
    # naive (games independent)
    var_naive = sum((x - xbar) ** 2 for x in xs) / (n - 1)
    se_naive = math.sqrt(var_naive / n)
    # cluster-robust on the opening
    ssq = sum(sum(x - xbar for x in v) ** 2 for v in by.values())
    se_clu = math.sqrt(ssq) / n
    d = 400 / (math.log(10) * xbar * (1 - xbar))
    e = elo(xbar)
    G = len(by)
    # small-G correction, the usual G/(G-1) finite-cluster factor
    se_clu *= math.sqrt(G / (G - 1.0))
    print("  %-8s vs %-8s  n=%d  openings=%d  reuse=%.1fx  score=%.4f  Elo=%+.2f"
          % (A, B, n, G, n / G, xbar, e))
    print("      naive   +/- %6.2f   [%+7.2f, %+7.2f]  (games independent)"
          % (1.96 * se_naive * d, e - 1.96 * se_naive * d, e + 1.96 * se_naive * d))
    print("      CLUSTER +/- %6.2f   [%+7.2f, %+7.2f]  (opening as the unit)  inflation %.2fx"
          % (1.96 * se_clu * d, e - 1.96 * se_clu * d, e + 1.96 * se_clu * d,
             se_clu / se_naive))
