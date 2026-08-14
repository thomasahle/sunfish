"""TM screen read-out: outcomes, TERMINATION CLASS PER ARM, and the drain profile.

pair_elo.py already gives the pentanomial Elo; this adds the readings that were
pre-registered as the mechanism evidence and that no Elo number carries:

  * time forfeits per arm. In these screens a forfeit in OUR pgn is DATA -- an
    arm losing on time is the defect showing -- so they are counted per arm and
    per termination class rather than treated as an incident.
  * BLIND MOVES per arm: moves played at or under 0.06 s. Stage 1 established
    that the drain does not cash out as a flag, it cashes out as blind play --
    zero forfeits on either arm, and the pre-fix arm nonetheless played a median
    16 moves at the 0.05 s floor and got mated on the board. 0.06 s is the floor
    (0.05 s) plus a process-noise margin; move times are recorded to 0.001 s.
  * the drain profile. fastchess writes `tl=0.000s` for these engines (the
    packed builds emit no info lines for it to track), so the clock is
    RECONSTRUCTED from the per-move times it does record: an arm's clock at the
    end of a game is its starting clock, minus the sum of its own move times,
    plus one increment per move it played. Book moves cost nothing and are
    excluded by having no time comment.

usage: tally.py GAMES.pgn [ARM1 ARM2 [CLOCK_S [INC_S]]]

Defaults reproduce the stage-1 invocation exactly (tmfix vs oldtm at 60+0), so
the archived stage-1 numbers stay reproducible from this same file. The screens
of the SMOOTH budget pass their own arms and TC:

    tally.py match.pgn smooth step 60 0.1
    tally.py match.pgn smooth step 30 1

Shipped to each bench-box arena as tally.py; this is the archived canonical
copy, and there is exactly one -- a per-arena fork is the stale-copy failure
this project has paid for repeatedly.
"""
import collections
import re
import sys

TIME = re.compile(r"\{[^}]*?([0-9]+\.[0-9]+)s[^}]*\}")
TAG = re.compile(r'\[(\w+) "(.*)"\]')
if len(sys.argv) == 3:
    raise SystemExit("give BOTH arm names or neither: tally.py PGN [ARM1 ARM2 [CLOCK [INC]]]")
ARMS = tuple(sys.argv[2:4]) if len(sys.argv) > 3 else ("tmfix", "oldtm")
CLOCK = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
INC = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
# Where the OLD cap (wtime/2 - 1000 ms) turns negative and the budget collapses
# to the floor. A property of the superseded policy, kept as the descriptive
# reading it was in stage 1; the smooth cap has no such crossing by
# construction, which is the thing the blind-move count measures directly.
NEGCAP = 2.4


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
score = collections.Counter()          # ARMS[0] W/L/D
term = collections.Counter()           # (loser, termination)
tclass = collections.Counter()         # termination -> n
left = collections.defaultdict(list)   # arm -> clock remaining at game end
cross = collections.defaultdict(list)  # arm -> (move it first went under NEGCAP, moves)
blind = collections.defaultdict(list)  # arm -> blind moves in that game
starved = collections.defaultdict(list)  # arm -> starved moves (descriptive, see below)
tail = collections.defaultdict(list)   # arm -> move times over its last 20 moves
moves = collections.Counter()          # arm -> total moves played
plies = []
illegal = []

for tags, body in gs:
    r, W, B = tags.get("Result"), tags.get("White"), tags.get("Black")
    if r not in ("1-0", "0-1", "1/2-1/2"):
        continue
    t = tags.get("Termination", "normal")
    tclass[t] += 1
    if "illegal" in t.lower() or "illegal move" in body.lower():
        illegal.append((tags.get("Round"), W, B, t))
    if r == "1-0":
        score["W" if W == ARMS[0] else "L"] += 1
        loser = B
    elif r == "0-1":
        score["W" if B == ARMS[0] else "L"] += 1
        loser = W
    else:
        score["D"] += 1
        loser = None
    if t != "normal" and loser:
        term[(loser, t)] += 1
    wt, bt = split_times(body)
    for arm, ts in ((W, wt), (B, bt)):
        left[arm].append(CLOCK - sum(ts) + INC * len(ts))
        blind[arm].append(sum(1 for dt in ts if dt <= 0.06))
        tail[arm].extend(ts[-20:])
        starved[arm].append(sum(1 for dt in ts if dt <= max(0.06, 1.5 * INC)))
        moves[arm] += len(ts)
        c, first = CLOCK, 0
        for k, dt in enumerate(ts, 1):
            c -= dt - INC
            if c < NEGCAP:
                first = k
                break
        cross[arm].append((first, len(ts)))
    plies.append(int(tags.get("PlyCount", 0)))

n = sum(score.values())
print("arms               %s (engine1) vs %s     TC %g+%g" % (ARMS[0], ARMS[1], CLOCK, INC))
print("games              %d  (%s %dW %dL %dD, score %.2f%%)"
      % (n, ARMS[0], score["W"], score["L"], score["D"],
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
print("BLIND MOVES  (moves played at or under 0.06 s -- the 0.05 s floor plus")
print("process noise). THIS is how the drain cashes out: stage 1 saw zero")
print("forfeits on either arm and the pre-fix arm still played a median 16 moves")
print("here, mated on the board.")
print("  %-10s %10s %10s %10s %12s" % ("arm", "total", "per game", "median/game", "games with 0"))
for a in ARMS:
    xs = blind.get(a, [])
    if not xs:
        continue
    print("  %-10s %10d %10.2f %10.0f %12d   (%.1f%% of its %d moves)"
          % (a, sum(xs), sum(xs) / len(xs), median(xs), sum(1 for x in xs if x == 0),
             100.0 * sum(xs) / max(moves[a], 1), moves[a]))
print()
print("STARVATION PROFILE. **DESCRIPTIVE, added at analysis time** -- the")
print("PRE-REGISTERED mechanism number is the 0.06 s count above, and it stays")
print("as it was written. It is reported here because at an INCREMENT TC it can")
print("read 0 for a starved arm and thereby miss the effect: a capped budget")
print("does not settle at the 0.05 s floor, it settles wherever spend == income.")
print("An arm whose cap has collapsed plays at ~1 increment per move forever, so")
print("the starved band here is <= %.2f s (= max(0.06, 1.5 x %g s of increment);"
      % (max(0.06, 1.5 * INC), INC))
print("at a sudden-death TC that is 0.06 s and this reading reduces to the one")
print("above, which is how it was checked against the stage-1 run).")
print("  %-10s %12s %13s %18s" % ("arm", "starved mv", "% of its mv", "median last-20 mv"))
for a in ARMS:
    xs = starved.get(a, [])
    if not xs:
        continue
    print("  %-10s %12d %13.1f%% %18.3f s"
          % (a, sum(xs), 100.0 * sum(xs) / max(moves[a], 1), median(tail.get(a, []))))
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
print("MOVE AT WHICH THE CLOCK FIRST FALLS BELOW %.1f s  (the OLD cap's" % NEGCAP)
print("negative-crossing, where its budget collapses to the 0.05 s floor).")
print("DESCRIPTIVE, off the same clocks as the reading above; the PRE-REGISTERED")
print("numbers are the median clock at game end and the blind-move count.")
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
