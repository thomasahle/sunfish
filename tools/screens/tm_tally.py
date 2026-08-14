"""Stage-1 read-out: outcomes, TERMINATION CLASS PER ARM, and the drain profile.

pair_elo.py already gives the pentanomial Elo; this adds the two readings that
were pre-registered as the mechanism evidence and that no Elo number carries:

  * time forfeits per arm. In this screen a forfeit in OUR pgn is DATA -- the
    pre-fix arm losing on time is the defect showing -- so they are counted per
    arm and per termination class rather than treated as an incident.
  * the drain profile. fastchess writes `tl=0.000s` for these engines (the
    packed builds emit no info lines for it to track), so the clock is
    RECONSTRUCTED from the per-move times it does record: an arm's clock at the
    end of a game is 60 s minus the sum of its own move times. Book moves cost
    nothing and are excluded by having no time comment.

usage: tally.py GAMES.pgn

Shipped to the bench-box arena as tally.py for the stage-1 run
(~/sunfish-bench/tmfix60-20260814/); this is the archived canonical copy.
"""
import collections
import re
import sys

TIME = re.compile(r"\{[^}]*?([0-9]+\.[0-9]+)s[^}]*\}")
TAG = re.compile(r'\[(\w+) "(.*)"\]')
CLOCK = 60.0


def games(path):
    tags, body, out = {}, [], []
    for line in open(path, errors="replace"):
        m = TAG.match(line.strip())
        if m:
            if body:
                out.append((tags, " ".join(body)))
                tags, body = {}, []
            tags[m.group(1)] = m.group(2)
        elif line.strip():
            body.append(line.strip())
    if body:
        out.append((tags, " ".join(body)))
    return out


def split_times(movetext):
    """White's and Black's move times, in order. Every played move carries
    exactly one time comment; book moves carry `{book}` and no time."""
    w, b = [], []
    side = 0
    for tok in re.finditer(r"\{[^}]*\}", movetext):
        c = tok.group(0)
        if "book" in c:
            side ^= 1
            continue
        m = TIME.match(c)
        (w if side == 0 else b).append(float(m.group(1)) if m else 0.0)
        side ^= 1
    return w, b


def median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


gs = games(sys.argv[1])
score = collections.Counter()          # tmfix W/L/D
term = collections.Counter()           # (loser, termination)
tclass = collections.Counter()         # termination -> n
left = collections.defaultdict(list)   # arm -> clock remaining at game end
cross = collections.defaultdict(list)  # arm -> (move it first went under 2.4 s, moves)
plies = []
illegal = []
ARMS = ("tmfix", "oldtm")

for tags, body in gs:
    r, W, B = tags.get("Result"), tags.get("White"), tags.get("Black")
    if r not in ("1-0", "0-1", "1/2-1/2"):
        continue
    t = tags.get("Termination", "normal")
    tclass[t] += 1
    if "illegal" in t.lower() or "illegal move" in body.lower():
        illegal.append((tags.get("Round"), W, B, t))
    if r == "1-0":
        score["W" if W == "tmfix" else "L"] += 1
        loser = B
    elif r == "0-1":
        score["W" if B == "tmfix" else "L"] += 1
        loser = W
    else:
        score["D"] += 1
        loser = None
    if t != "normal" and loser:
        term[(loser, t)] += 1
    wt, bt = split_times(body)
    left[W].append(CLOCK - sum(wt))
    left[B].append(CLOCK - sum(bt))
    for arm, ts in ((W, wt), (B, bt)):
        c, first = CLOCK, 0
        for k, dt in enumerate(ts, 1):
            c -= dt
            if c < 2.4:
                first = k
                break
        cross[arm].append((first, len(ts)))
    plies.append(int(tags.get("PlyCount", 0)))

n = sum(score.values())
print("games              %d  (tmfix %dW %dL %dD, score %.2f%%)"
      % (n, score["W"], score["L"], score["D"],
         100 * (score["W"] + 0.5 * score["D"]) / max(n, 1)))
print("median plies       %.0f" % median(plies))
print()
print("TERMINATION CLASSES")
for t, c in tclass.most_common():
    print("  %-22s %4d" % (t, c))
print()
print("DECISIVE LOSSES BY CLASS AND ARM  (the mechanism number)")
print("  %-10s %10s %10s" % ("arm", "time forfeit", "other non-normal"))
for a in ARMS:
    tf = sum(c for (arm, t), c in term.items() if arm == a and "time" in t.lower())
    ot = sum(c for (arm, t), c in term.items() if arm == a and "time" not in t.lower())
    print("  %-10s %10d %10d   (%.1f%% of its %d games lost on time)"
          % (a, tf, ot, 100.0 * tf / max(n, 1), n))
print()
print("DRAIN PROFILE  (clock remaining at game end, seconds, reconstructed)")
print("  %-10s %8s %8s %8s %8s" % ("arm", "median", "mean", "min", "<2s games"))
for a in ARMS:
    xs = left.get(a, [])
    if not xs:
        continue
    print("  %-10s %8.1f %8.1f %8.1f %8d" % (
        a, median(xs), sum(xs) / len(xs), min(xs), sum(1 for x in xs if x < 2.0)))
print()
print("MOVE AT WHICH THE CLOCK FIRST FALLS BELOW 2.4 s  (the negative-cap")
print("threshold, where the budget collapses to the 0.05 s floor). DESCRIPTIVE,")
print("added at analysis time off the same clocks as the reading above; the")
print("PRE-REGISTERED number is the median clock at game end.")
print("  %-10s %8s %8s %8s %14s" % ("arm", "median", "games", "never", "moves after"))
for a in ARMS:
    hit, never, after = [], 0, []
    for m, tot in cross.get(a, []):
        if m:
            hit.append(m)
            after.append(tot - m)
        else:
            never += 1
    print("  %-10s %8s %8d %8d %14s" % (
        a, ("%.0f" % median(hit)) if hit else "-", len(hit) + never, never,
        ("%.0f" % median(after)) if after else "-"))
print("  'moves after' = moves that arm then played with the budget collapsed to")
print("  the 0.05 s floor -- blind play, which is how the drain cashes out when")
print("  no flag falls.")
if illegal:
    print()
    print("*** ZERO-TOLERANCE FAIL: illegal move in %d game(s) ***" % len(illegal))
    for g in illegal[:8]:
        print("   round %s  %s vs %s  %s" % g)
