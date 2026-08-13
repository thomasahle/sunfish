"""Per-pairing Elo with 95% intervals from a multi-engine PGN.

A round-robin answers several questions at once, but fastchess only prints one
ranking table for the whole tournament. The questions this lane asks are
pairwise ("does the cap beat the entry", "is entry_nolmr below classic"), so
the PGN has to be split per pairing and each half scored on its own.

The interval is the PENTANOMIAL one fastchess reports, not a binomial on
individual games: games arrive in colour-swapped pairs from one opening, the
two halves of a pair are strongly correlated, and treating them as independent
understates the error. Validated against a fastchess-reported result before
first use -- see tools/screens/README-validation note in the ledger.

usage: pair_elo.py GAMES.pgn [-- name_a name_b ...]   (filter to these engines)
"""
import collections
import math
import re
import sys

Z = 1.959963984540054  # two-sided 95%


def elo(x):
    if x <= 0: return -800.0
    if x >= 1: return 800.0
    return -400.0 * math.log10(1.0 / x - 1.0)


def read(path):
    """Yield (round, white, black, white_score). One tag block per game."""
    cur = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = re.match(r'\[(\w+) "(.*)"\]', line.strip())
            if not m:
                continue
            k, v = m.groups()
            if k == "Round" and "Result" in cur:
                cur = {}
            cur[k] = v
            if k == "Result":
                s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(v)
                if s is not None and "White" in cur and "Black" in cur:
                    yield cur.get("Round", "?"), cur["White"], cur["Black"], s


def pentanomial(pairs):
    """pairs: list of A's score over a colour-swapped pair, each in [0,2]."""
    n = len(pairs)
    if n == 0:
        return None
    mu = sum(pairs) / (2.0 * n)                       # per-GAME score rate
    var = sum((p / 2.0 - mu) ** 2 for p in pairs) / n  # population, as fastchess
    se = math.sqrt(var / n)
    lo, hi = mu - Z * se, mu + Z * se
    return mu, elo(mu), (elo(hi) - elo(lo)) / 2.0, n


def main():
    path = sys.argv[1]
    keep = set(sys.argv[3:]) if "--" in sys.argv else None

    # Group by (pairing, round); a round holds the two colour-swapped games.
    slots = collections.defaultdict(dict)
    counts = collections.Counter()
    for rnd, w, b, s in read(path):
        if keep and (w not in keep or b not in keep):
            continue
        key = tuple(sorted((w, b)))
        counts[key] += 1
        slots[key].setdefault(rnd, []).append((w, s))

    print("%-22s %-22s %6s %6s %8s %9s" % ("engine A", "engine B", "pairs", "games", "score%", "Elo(A)"))
    for key in sorted(slots):
        a, b = key
        full = [v for v in slots[key].values() if len(v) == 2]
        pairs = [sum(s if w == a else 1.0 - s for w, s in v) for v in full]
        r = pentanomial(pairs)
        if r is None:
            continue
        mu, e, err, n = r
        odd = counts[key] - 2 * n
        print("%-22s %-22s %6d %6d %8.2f %9s" % (
            a, b, n, counts[key], 100 * mu, "%+.2f +/- %.2f" % (e, err))
            + ("   (%d unpaired game(s) dropped)" % odd if odd else ""))


if __name__ == "__main__":
    main()
