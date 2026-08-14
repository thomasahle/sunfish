#!/usr/bin/env python3
"""STAGE 0 of the TM funnel: clock trajectories, with no games at all.

The cheapest ranking pass there is.  A time manager's most expensive failures
are ARITHMETIC -- a cap that goes negative, a budget that parks the clock at a
fixed point, a pool that stops spending -- and arithmetic does not need an
engine to see.  This module walks a clock forward through a game of a given
length under a given manager and reports the spend profile and the events:

  FLOOR   the budget collapsed to TM_FLOOR (blind play; the engine moves but
          does not search)
  PARK    the clock reached a FIXED POINT: spend + overhead == increment, so
          it stops falling and the engine plays out the game on the floor
  FLAG    the clock ran out

It generalises the drain walks in tests/test_tm_pool.py (which fix a spend
per move and count flags) in two ways: the spend is derived from the budget
by a discrete-iteration model rather than assumed, and the fixed points are
solved for rather than observed.

THE SPEND MODEL, and it is the only modelled thing here.  Budgets are not
spends: search stops at a yield, not at a wall, so what a manager actually
spends is the first LADDER STOP past its target.  The ladder is geometric --
iteration d costs t1*b^(d-1), b the effective branching factor at fixed nps
-- and the two stop rules read it at different granularities:

  yield_frac      probe granularity.  Both incumbents break at ANY yield past
                  frac*hard, so they land just past frac*hard and the wall
                  usually binds: realized ~ hard.
  mtd_converged   ITERATION granularity.  The pool may only stop where an
                  iteration ENDED, so it overshoots its soft limit by up to a
                  whole next iteration -- the 1.3-2.3x the pool's own
                  pre-registration recorded before its first game.

Everything else -- which move gets played, whether it was any good -- is
stage 1's job (vmatch.py).  Stage 0 ranks CLOCK BEHAVIOUR only.
"""
import argparse
import json

import tmlib

# Iteration ladder.  t1 is the depth-1 iteration in seconds at a middlegame
# node rate (2,000 nodes / ~23,000 nps from npsmodel.json) and b is the
# effective branching factor of classic sunfish's MTD-bi ladder.  Both are
# stage-0 parameters, swept by --ladder-t1/--branching so that what depends
# on them can be shown rather than argued.  The three calibration signatures
# do not: they are budget arithmetic and come out identical across the whole
# sweep (tests/test_tm_surrogate.py pins their values).
LADDER_T1 = 2000 / 23000
LADDER_B = 2.5
# Probes per iteration: the MTD bisection typically needs a handful.  Only
# used to give the yield_frac rule its granularity.
PROBES_PER_ITER = 4


def ladder_stop(target, wall, rule, t1=None, b=None):
    """Realized spend, SECONDS: the first ladder stop past `target`, walled.

    The ladder parameters are read from the MODULE at call time, not bound as
    default arguments: `t1=LADDER_T1` in the signature would freeze them at
    import and make every later sweep a silent no-op -- which is exactly what
    it did until this was caught, turning a robustness check into a check of
    nothing.
    """
    t1 = LADDER_T1 if t1 is None else t1
    b = LADDER_B if b is None else b
    if wall <= t1 / PROBES_PER_ITER:
        return wall                     # not even one probe fits: pure wall
    elapsed, d = 0.0, 1
    while elapsed <= wall and d < 64:
        it = t1 * b ** (d - 1)
        if rule == "mtd_converged":
            if elapsed >= target and elapsed > 0:
                return min(elapsed, wall)
            elapsed += it
        else:
            for _ in range(PROBES_PER_ITER):
                elapsed += it / PROBES_PER_ITER
                if elapsed > target:
                    return min(elapsed, wall)
        d += 1
    return wall


def walk(manager, base, inc, plies, overhead, knobs=None, floor=tmlib.TM_FLOOR):
    """Walk OUR clock through `plies` of our own moves.  Returns a trace."""
    knobs = knobs or {}
    fn = tmlib.MANAGERS[manager]
    clock, rows = base, []
    for ply in range(plies):
        b = fn(clock, inc, None, ply=ply, **knobs)
        target = b.soft if b.rule == "mtd_converged" else b.frac * b.hard
        spend = ladder_stop(target, b.hard, b.rule)
        spend = max(spend, 0.0)
        before = clock
        clock -= spend + overhead
        flagged = clock < 0
        if not flagged:
            clock += inc
        rows.append({"ply": ply, "clock_before": before, "soft": b.soft,
                     "hard": b.hard, "spend": spend, "clock_after": clock,
                     "floor": b.hard <= floor + 1e-12, "flag": flagged})
        if flagged:
            break
    return rows


def fixed_point(manager, inc, overhead, knobs=None, lo=1e-6, hi=1e5):
    """Solve for the clock the manager PARKS at, or None.

    A park is a clock T with drift(T) = increment - spend(T) - overhead == 0
    and drift > 0 below it (so it is an attractor from both sides).  Bisected
    rather than simulated, because a fixed point is an equation.
    """
    knobs = knobs or {}
    fn = tmlib.MANAGERS[manager]

    def drift(t):
        b = fn(t, inc, None, ply=40, **knobs)
        target = b.soft if b.rule == "mtd_converged" else b.frac * b.hard
        return inc - ladder_stop(target, b.hard, b.rule) - overhead

    if drift(hi) >= 0:
        return None                     # never spends enough to fall: no park
    if drift(lo) <= 0:
        return None                     # drains from everywhere: no park
    for _ in range(200):
        mid = (lo + hi) / 2
        if drift(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def knee(manager, inc, knobs=None, what="floor", lo=1e-6, hi=1e5):
    """The CLOCK at which a manager's shape changes, solved not observed.

    "floor"  the largest clock whose budget has already collapsed to
             TM_FLOOR -- below this the engine plays blind.
    "cap"    the largest clock at which the SAFETY term, rather than the
             per-move share, is what limits the budget (tmlib.cap_binds).
             For oldtm at winc == 0 that is 2.4 s exactly, the negative-cap
             threshold of the stage-1 forensics; for the pool it is
             (M+2)*O, the 8.4 s minimum end clock arm (a) measured.
    """
    knobs = knobs or {}
    fn = tmlib.MANAGERS[manager]

    def collapsed(t):
        if what == "floor":
            return fn(t, inc, None, ply=40, **knobs).hard <= tmlib.TM_FLOOR + 1e-12
        return tmlib.cap_binds(manager, t, inc, **knobs)

    if not collapsed(lo) or collapsed(hi):
        return None                      # no crossing in range: not a knee
    for _ in range(200):
        mid = (lo + hi) / 2
        if collapsed(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def summarize(rows, inc):
    """The readings the real matches pre-registered, computed analytically."""
    n = len(rows)
    floors = [r for r in rows if r["floor"]]
    below24 = [r["ply"] for r in rows if r["clock_before"] < 2.4]
    return {
        "plies": n,
        "flag_at": next((r["ply"] for r in rows if r["flag"]), None),
        "end_clock": rows[-1]["clock_after"] if rows else None,
        "min_clock": min(r["clock_after"] for r in rows) if rows else None,
        "median_spend": sorted(r["spend"] for r in rows)[n // 2] if n else None,
        "max_spend": max(r["spend"] for r in rows) if n else None,
        "floor_moves": len(floors),
        "first_floor_ply": floors[0]["ply"] if floors else None,
        "first_below_2v4": below24[0] if below24 else None,
        "moves_after_2v4": (n - below24[0]) if below24 else 0,
        "starved_moves": sum(1 for r in rows
                             if r["spend"] <= max(0.15, 1.5 * inc)),
    }


def parse_tc(tc):
    base, _, inc = tc.partition("+")
    return float(base), float(inc or 0)


def main():
    global LADDER_T1, LADDER_B
    ap = argparse.ArgumentParser()
    ap.add_argument("--tc", default="60+0")
    ap.add_argument("--plies", type=int, default=70,
                    help="OUR moves (half the ply count of a game)")
    ap.add_argument("--overhead", type=float, default=0.05,
                    help="SECONDS the environment charges per move on top of "
                         "the search: ~0.05 for a local fastchess arena, 0.2 "
                         "for the lichess deployment.  NOT the manager's own "
                         "O knob, which is --pool-overhead.")
    ap.add_argument("--pool-overhead", type=float, default=0.2,
                    help="the O the POOL FORMULA prices (shipped: 0.2)")
    ap.add_argument("--pool-scale", type=float, default=1.0)
    ap.add_argument("--pool-moves", type=int, default=40)
    ap.add_argument("--phase-m", action="store_true")
    ap.add_argument("--managers", default="oldtm,steptm,smooth,pool")
    ap.add_argument("--ladder-t1", type=float, default=2000 / 23000,
                    help="depth-1 iteration, SECONDS (spend-model parameter)")
    ap.add_argument("--branching", type=float, default=2.5,
                    help="effective branching of the iteration ladder")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    LADDER_T1, LADDER_B = args.ladder_t1, args.branching
    base, inc = parse_tc(args.tc)
    out = {}
    print("STAGE 0  TC %s  our plies %d  environment overhead %.3fs"
          % (args.tc, args.plies, args.overhead))
    print("%-9s %8s %8s %8s %7s %7s %7s %8s %8s %8s %8s"
          % ("manager", "median", "max", "endclk", "floors", "1stflr", "starv",
             "flag@", "park", "capknee", "flrknee"))
    for name in args.managers.split(","):
        knobs = {}
        if name == "pool":
            knobs = {"overhead": args.pool_overhead, "scale": args.pool_scale,
                     "moves": args.pool_moves, "phase_m": args.phase_m}
        rows = walk(name, base, inc, args.plies, args.overhead, knobs)
        s = summarize(rows, inc)
        s["park"] = fixed_point(name, inc, args.overhead, knobs)
        s["cap_knee"] = knee(name, inc, knobs, "cap")
        s["floor_knee"] = knee(name, inc, knobs, "floor")
        out[name] = {"summary": s, "rows": rows}
        fmt = lambda v: "-" if v is None else "%.2f" % v   # noqa: E731
        print("%-9s %8.3f %8.3f %8.2f %7d %7s %7d %8s %8s %8s %8s"
              % (name, s["median_spend"], s["max_spend"], s["end_clock"],
                 s["floor_moves"], s["first_floor_ply"], s["starved_moves"],
                 s["flag_at"], fmt(s["park"]), fmt(s["cap_knee"]),
                 fmt(s["floor_knee"])))
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
