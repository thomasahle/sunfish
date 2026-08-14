"""The classic builtin clock: one pool, spent down, and parked off the floor.

sunfish.py's `go` handler budgets time on a single inline line. Two
properties decide whether that line is safe, and no other gate can see
either -- the node ladder never starts a clock, and a match reports only
the result.

  * how HIGH the policy parks.  Iterate the self-clock recurrence
    T <- T - think(T) - O + I over our own moves (O is per-move overhead
    we cannot avoid, I the increment).

    A park is NOT caused by a cap, and an earlier draft of this file said
    it was.  At any increment TC the clock MUST come to rest where
    `spend + overhead == income`, whatever the budget's shape, so every
    manager parks -- these candidates included.  What the shape decides is
    not WHETHER the clock settles but HOW MUCH CLOCK IS STILL IN HAND when
    it does, and that is the whole of the safety argument.  The surrogate
    owns those altitudes and reads them off realized spend (60+0.1:
    one-max 6.17s, the step form 2.11s, min40-4 0.22s; 60+1: reserves of
    10.4s, 4.1s and 6.4s at a common ~1.06s spend).  This file asserts the
    MECHANISM that orders them and leaves the numbers to that instrument.

    The incumbent `min(t/12 + .9i, t/2 - 1s)` parks LOW and blind.  Once
    the cap binds the recurrence is T <- T/2 + 1 + I, fixed point
    T* = 2 + 2I seconds -- measured at 2.0s (60+0) and a 2.1s median
    (60+0.1).  Under a 2s clock that cap is already NEGATIVE, so the arm
    defining the park cannot be spent at all: the budget collapses to the
    0.05s floor and ~200ms/move of lag drains the remainder.  That is
    lichess EAThUL0P, lost on time at move 73 with no single move
    overrunning.  The defect is the ALTITUDE, not the existence.

  * how much RESERVE is banked when the floor is reached.  A pool policy
    is worth exactly the clock it still holds at its floor, in moves.

Two candidate one-liners are walked here against the shipped one:

    one-max   max((wtime - 8000) / 40 + winc, 50)
    min40-4   min(wtime / 40 + 0.9 * winc, wtime / 4)

Both drop the `t/2 - 1s` cap, which does not create the park but does set
it at the floor on a negative budget, and they reach a positive budget by
different routes -- one-max banks a named 8s overhead reserve, min40-4
clips to a quarter clock so the reserve is four increments and no
time-dimensioned constant appears at all.  This file is the shared
scaffolding: the candidates are literals, so every property below is
checked on both no matter which one is shipped.
"""
import re
import os

import pytest

ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sunfish.py")
SRC = open(ENGINE).read()

# The shipped budget statement, lifted from the engine rather than copied:
# if that line is reshaped this file fails loudly instead of quietly
# testing a stale duplicate.
SHIPPED = re.search(r"^ +(think = (?:min|max)\(.*\))$", SRC, re.M)

# The three policies as source, all in the MILLISECOND domain that the `go`
# handler actually works in -- `wtime`/`winc` arrive as integer ms and the
# next line in main() divides by 1000.  Mixing the domains is the trap that
# produced a 590-second move, so it is crossed in exactly one place below.
OLD = "think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000)"
ONE_MAX = "think = max((wtime - 8000) / 40 + winc, 50)"
MIN40_4 = "think = min(wtime / 40 + 0.9 * winc, wtime / 4)"

CANDIDATES = {"one-max": ONE_MAX, "min40-4": MIN40_4}

# The engine floors the budget once more where it arms the deadline
# (`max(think, .05)`), which is what guards a tiny explicit `movetime`.
FLOOR = 0.05
# Per-move overhead measured in production: process wakeup, I/O, lichess lag.
OVERHEAD = 0.2


def _think_ms(stmt, wtime_ms, winc_ms):
    ns = {"wtime": wtime_ms, "winc": winc_ms, "min": min, "max": max}
    exec(stmt, ns)
    return ns["think"]


def budget(stmt, wtime_s, winc_s=0.0):
    """Seconds of thinking for a clock given in SECONDS -- the one unit crossing.

    Takes and returns seconds because every regime and recurrence below is
    naturally stated in seconds; converts to and from the engine's
    millisecond domain here, once, so no test has to remember which side of
    the /1000 it is on.
    """
    return _think_ms(stmt, wtime_s * 1000.0, winc_s * 1000.0) / 1000.0


def armed(stmt, wtime_s, winc_s=0.0):
    """What the searcher deadline actually gets: the budget under main()'s floor."""
    return max(budget(stmt, wtime_s, winc_s), FLOOR)


# --------------------------------------------------------------------------
# the shipped line is one of the candidates
# --------------------------------------------------------------------------

def test_budget_statement_present():
    assert SHIPPED, "the inline budget statement is missing or reshaped"


def test_shipped_line_is_exactly_one_candidate():
    """No third form drifts in unmeasured."""
    assert SHIPPED.group(1) in CANDIDATES.values(), (
        "shipped budget %r is neither candidate; add it to CANDIDATES and "
        "give it a regime table before shipping" % SHIPPED.group(1))


def test_the_old_policy_is_the_one_being_replaced():
    """Pin the baseline as a literal so the contrast below cannot go stale."""
    assert SHIPPED.group(1) != OLD, "still on the parking policy"
    assert budget(OLD, 60) == pytest.approx(5.0)          # 60+0 -> t/12
    assert budget(OLD, 60, 1) == pytest.approx(5.9)       # 60+1 -> t/12 + .9i


# --------------------------------------------------------------------------
# regime tables
# --------------------------------------------------------------------------

@pytest.mark.parametrize("wtime,winc,want", [
    (300, 0, 7.3),      # (300 - 8) / 40
    (60, 0, 1.3),       # (60 - 8) / 40
    (30, 1, 1.55),      # (30 - 8) / 40 + 1
    (14, 0, 0.15),      # (14 - 8) / 40
    (8, 0, 0.05),       # the reserve is exactly spent; floor takes over
    (4, 0, 0.05),       # under the reserve: floor
    (0.5, 0, 0.05),     # nearly flagged: floor
])
def test_one_max_regimes(wtime, winc, want):
    assert budget(ONE_MAX, wtime, winc) == pytest.approx(want)


@pytest.mark.parametrize("wtime,winc,want", [
    (300, 0, 7.5),      # 300 / 40
    (60, 0, 1.5),       # 60 / 40
    (30, 1, 1.65),      # 30 / 40 + 0.9
    (14, 0, 0.35),      # 14 / 40
    (8, 0, 0.2),        # 8 / 40 -- still pacing, not floored
    (2, 1, 0.5),        # clock under four increments: the quarter-clock clip
    (0.5, 0, 0.0125),   # 0.5 / 40, under main()'s floor
])
def test_min40_4_regimes(wtime, winc, want):
    assert budget(MIN40_4, wtime, winc) == pytest.approx(want)


def test_sudden_death_is_the_same_pacing_law_for_both():
    """At I = 0 both are a plain 40-move split, one shifted by its reserve."""
    for t in (30, 60, 120, 300, 600):
        assert budget(MIN40_4, t) == pytest.approx(t / 40)
        assert budget(ONE_MAX, t) == pytest.approx((t - 8) / 40)


# --------------------------------------------------------------------------
# NO PARK: the recurrence has no fixed point above the floor
# --------------------------------------------------------------------------

def drift(stmt, wtime_s, winc_s, overhead=OVERHEAD):
    """One step of T <- T - think - O + I, as a signed change in the clock.

    Uses the ARMED budget, i.e. what the engine really spends including
    main()'s floor -- this is the recurrence a game actually walks.
    """
    return winc_s - overhead - armed(stmt, wtime_s, winc_s)


def drift_raw(stmt, wtime_s, winc_s, overhead=OVERHEAD):
    """The same step on the unfloored formula -- the analytic recurrence.

    The floor is what turns the old policy's park into a drain, so the two
    have to be kept apart: the fixed point below is a property of the
    FORMULA, and the floor is then what happens to a game that reaches it.
    """
    return winc_s - overhead - budget(stmt, wtime_s, winc_s)


def walk(stmt, base_s, inc_s, moves, overhead=OVERHEAD):
    """Simulate our own clock over `moves` of our moves; -1 == flagged."""
    clock = base_s
    for mv in range(moves):
        clock += drift(stmt, clock, inc_s, overhead)
        if clock <= 0:
            return -1, mv
    return clock, moves


def park_clock(stmt, winc_s, overhead):
    """The clock a game comes to rest at: the highest T with drift >= 0.

    `drift` is nonincreasing in T (the budget only grows with the clock), so
    the resting point is found by bisection.  Returns 0.0 when the clock
    never stops falling, i.e. when there is genuinely no park.
    """
    if drift(stmt, 0.0, winc_s, overhead) < 0:
        return 0.0
    lo, hi = 0.0, 600.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if drift(stmt, mid, winc_s, overhead) >= 0:
            lo = mid
        else:
            hi = mid
    return lo


@pytest.mark.parametrize("name,stmt", sorted(CANDIDATES.items()) + [("incumbent", OLD)])
@pytest.mark.parametrize("inc,over,parks", [
    (0.0, 0.05, False),    # sudden death: no income, so nothing to rest on
    (0.0, 0.20, False),
    (0.1, 0.20, False),    # lag exceeds the increment: still a pure drain
    (0.1, 0.05, True),     # the surrogate's charge: income wins, so a park
    (1.0, 0.05, True),
])
def test_a_park_exists_exactly_when_income_exceeds_overhead(name, stmt, inc, over, parks):
    """The correction, asserted: a park is not caused by a cap.

    The clock rests where `spend + overhead == income`.  Since spend is at
    least the floor, a resting point exists iff `income - overhead >= floor`
    -- a statement about the TIME CONTROL and the lag, not about the budget's
    shape.  Every policy here obeys it, the candidates included.
    """
    assert (park_clock(stmt, inc, over) > 0) is parks
    assert (inc - over >= FLOOR) is parks


def test_park_altitude_is_what_the_shape_actually_decides():
    """At 60+0.1 the three policies rest at three very different clocks.

    This reproduces the surrogate's ordering from the budget alone (it reads
    6.17 / 2.11 / 0.22 s off realized spend, which is higher than the budget
    model because the driver stops at 0.8*think; the numbers are that
    instrument's, the ordering is arithmetic and belongs here).

    Note where min40-4 lands: it parks LOWEST of the three, below even the
    incumbent.  That is its known cost -- it wastes almost no clock and has
    the thinnest flag margin -- and it is why the real-clock confirmation
    for it is a flag hammer, not another Elo match.
    """
    park = {n: park_clock(s, 0.1, 0.05) for n, s in
            [("one-max", ONE_MAX), ("incumbent", OLD), ("min40-4", MIN40_4)]}
    assert park["one-max"] == pytest.approx(6.0, abs=0.01)
    assert park["incumbent"] == pytest.approx(2.1, abs=0.01)
    assert park["min40-4"] == pytest.approx(0.2, abs=0.01)
    assert park["one-max"] > park["incumbent"] > park["min40-4"]


def test_only_the_incumbent_parks_blind_with_a_negative_budget():
    """Parking low is survivable; parking on a NEGATIVE budget is the defect.

    All three rest at the floor at 60+0.1.  The difference is that the
    incumbent gets there because its cap went negative -- it is not choosing
    a small budget, it has no budget at all -- while both candidates reach
    the floor with a positive, monotone budget behind them.
    """
    assert budget(OLD, 2.0, 0.1) < 0.1                      # cap already biting
    assert budget(OLD, 1.9, 0.0) < 0                        # and then negative
    for stmt in CANDIDATES.values():
        assert budget(stmt, 1.9, 0.0) > 0
        assert budget(stmt, 0.5, 0.0) > 0


@pytest.mark.parametrize("name,stmt", sorted(CANDIDATES.items()))
def test_no_park_at_sudden_death(name, stmt):
    """The one place "no park" survives the correction: winc == 0.

    Income is zero, so `spend + overhead == income` has no solution with a
    nonnegative budget and the clock falls monotonically.  This is the venue
    the real-clock arm tests, and it is the venue the candidates were chosen
    for.
    """
    grid = [t / 10.0 for t in range(1, 6001)]
    assert max(drift(stmt, t, 0.0) for t in grid) < 0
    assert park_clock(stmt, 0.0, OVERHEAD) == 0.0


def test_the_old_policy_does_park_at_two_plus_two_inc():
    """The contrast: the cap manufactures a stable fixed point at T* = 2 + 2I.

    Solved on the unfloored formula with no overhead, which is the pure form
    of the recurrence the cap induces: T <- T/2 + 1 + I.  At T* the capped
    arm returns exactly I, so income and spend cancel.  This is the
    equilibrium the candidates exist to remove, and the one measured at 2.0s
    (60+0) and a 2.1s median (60+0.1).
    """
    for inc in (0.0, 0.1, 1.0, 3.0):
        star = 2 + 2 * inc
        assert budget(OLD, star, inc) == pytest.approx(inc)
        assert drift_raw(OLD, star, inc, overhead=0.0) == pytest.approx(0.0, abs=1e-9)
        # and it ATTRACTS: above it the clock falls, below it the clock rises
        assert drift_raw(OLD, star + 0.5, inc, overhead=0.0) < 0
        assert drift_raw(OLD, star - 0.5, inc, overhead=0.0) > 0
    # neither candidate has any fixed point in that neighbourhood
    for stmt in CANDIDATES.values():
        assert drift_raw(stmt, 2.0, 0.0, overhead=0.0) < 0


def test_at_sudden_death_the_old_park_is_a_drain_not_a_park():
    """Why the 2s park kills: under it the budget is FLOORED, not held.

    `t/2 - 1` is already negative below 2s, so the arm that defines the park
    cannot actually be spent -- main()'s 0.05s floor takes over, income at
    I = 0 is nothing, and the remaining 2.1s leaves ~8 moves at 0.05 + 0.2.
    That is lichess EAThUL0P, and it is the reason the ALTITUDE of the park
    is the thing that matters rather than its existence.
    """
    assert budget(OLD, 1.9) < 0                    # the cap has gone negative
    assert armed(OLD, 1.9) == FLOOR                # so the engine plays blind
    left, reached = walk(OLD, 2.1, 0.0, 40)
    assert left == -1 and reached <= 9


# --------------------------------------------------------------------------
# the banked reserve, in moves of floor play
# --------------------------------------------------------------------------

def floor_clock(stmt, winc_s=0.0):
    """The clock at which the budget first reaches main()'s floor."""
    t = 600.0
    while t > 0.001 and budget(stmt, t, winc_s) > FLOOR:
        t -= 0.001
    return t


def test_one_max_banks_the_named_eight_second_reserve():
    """The (M+2)*O accounting is the point: reach the floor still holding 10s.

    The reserve constant is 8s and the floor is worth a further 40 * 0.05s,
    so the budget lands on the floor with 8 + 2 = 10s of clock untouched --
    40 further moves at 0.05 + 0.2 each.
    """
    assert floor_clock(ONE_MAX) == pytest.approx(8.0 + 40 * FLOOR, abs=0.01)
    assert (8.0 + 40 * FLOOR) / (FLOOR + OVERHEAD) >= 40


def test_min40_4_banks_no_named_reserve_but_never_collapses():
    """Recorded, not hidden: min40-4 reaches the floor at 2s, like the old form.

    Its safety is a different property -- the budget is t/40 all the way
    down and stays POSITIVE, so there is no negative-cap collapse and the
    approach to the floor is slow.  That is the trade against one-max's
    named reserve, and it is exactly what the surrogate has to price.
    """
    assert floor_clock(MIN40_4) == pytest.approx(40 * FLOOR, abs=0.01)
    for t in [x / 100.0 for x in range(1, 60001)]:
        assert budget(MIN40_4, t) > 0


def test_the_old_policy_reaches_its_floor_with_almost_nothing_banked():
    """Same measurement on the shipped policy: 2.1s, i.e. 8 moves. That is the loss."""
    assert floor_clock(OLD) == pytest.approx(2.1, abs=0.01)
    assert floor_clock(OLD) / (FLOOR + OVERHEAD) < 9


@pytest.mark.parametrize("moves", [80, 100, 120])
def test_candidates_survive_a_long_sudden_death_game(moves):
    """3+0, the control that actually lost lichess EAThUL0P."""
    for name, stmt in CANDIDATES.items():
        left, reached = walk(stmt, 180, 0, moves)
        assert left > 0, "%s flagged at move %d of %d in 3+0" % (name, reached, moves)


def test_the_old_policy_reproduces_the_lost_game():
    """The baseline really does flag -- otherwise the walk proves nothing."""
    left, reached = walk(OLD, 180, 0, 120)
    assert left == -1 and reached < 120


# --------------------------------------------------------------------------
# monotonicity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,stmt", sorted(CANDIDATES.items()))
@pytest.mark.parametrize("inc", [0.0, 0.1, 1.0, 3.0])
def test_monotone_nondecreasing_in_wtime(name, stmt, inc):
    """More clock never buys less thinking."""
    prev = -1.0
    for t in [x / 20.0 for x in range(1, 12001)]:    # 0.05s .. 600s
        cur = budget(stmt, t, inc)
        assert cur >= prev - 1e-12, "%s dips at wtime=%.2f inc=%.1f" % (name, t, inc)
        prev = cur


@pytest.mark.parametrize("name,stmt", sorted(CANDIDATES.items()))
@pytest.mark.parametrize("wtime", [1.0, 8.0, 30.0, 60.0, 300.0])
def test_monotone_nondecreasing_in_winc(name, stmt, wtime):
    """More increment never buys less thinking."""
    prev = -1.0
    for i in [x / 100.0 for x in range(0, 1001)]:    # 0s .. 10s
        cur = budget(stmt, wtime, i)
        assert cur >= prev - 1e-12, "%s dips at wtime=%.1f inc=%.2f" % (name, wtime, i)
        prev = cur


# --------------------------------------------------------------------------
# why no cap is needed (one-max) / what the cap is (min40-4)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("wtime", [8.5, 9, 10, 20, 60, 120, 300, 600])
@pytest.mark.parametrize("inc", [0.0, 0.1, 1.0, 2.0])
def test_one_max_stays_under_half_the_clock_without_a_cap(wtime, inc):
    """`(t - 8)/40` never approaches `t/2`, so the removed cap is not missed.

    Exact condition: (t - 8)/40 + i < t/2  <=>  i < (19t + 8)/40, which at
    the reserve clock t = 8.4s already allows i up to 4.19s -- past every
    increment in the tested field.
    """
    assert budget(ONE_MAX, wtime, inc) < wtime / 2
    assert inc < (19 * wtime + 8) / 40


def test_one_max_half_clock_bound_has_a_stated_edge():
    """It is a bound with a limit, not a universal law -- record where it ends."""
    # a huge increment on a nearly dead clock does exceed half the clock,
    # which is the regime the quarter-clock clip in min40-4 exists to hold.
    assert budget(ONE_MAX, 1.0, 5.0) > 1.0 / 2
    assert budget(MIN40_4, 1.0, 5.0) == pytest.approx(0.25)


@pytest.mark.parametrize("wtime", [0.5, 1, 2, 8, 60, 300])
@pytest.mark.parametrize("inc", [0.0, 0.1, 1.0, 5.0])
def test_min40_4_never_spends_over_a_quarter_clock(wtime, inc):
    assert budget(MIN40_4, wtime, inc) <= wtime / 4 + 1e-12


@pytest.mark.parametrize("inc", [0.1, 0.5, 1.0, 3.0])
def test_min40_4_reserve_is_four_increments(inc):
    """The clip engages exactly when the clock falls under 4 * increment."""
    assert budget(MIN40_4, 4 * inc, inc) == pytest.approx(inc)          # both arms agree
    assert budget(MIN40_4, 4 * inc - 0.1, inc) == pytest.approx((4 * inc - 0.1) / 4)
    below = budget(MIN40_4, 4 * inc + 0.1, inc)
    assert below == pytest.approx((4 * inc + 0.1) / 40 + 0.9 * inc)


def test_min40_4_never_binds_its_cap_at_sudden_death():
    """At I = 0 the policy is exactly t/40 -- the cap is inert, hence no park.

    Scoped deliberately to winc == 0, which is the only regime where "no
    park" survives the correction -- and it is the regime the real-clock arm
    tests.  Here min40-4 needs no reserve floor at all: the capped arm simply
    never binds, so the policy is a pure geometric drain.
    """
    for t in [x / 10.0 for x in range(1, 6001)]:
        assert budget(MIN40_4, t) == pytest.approx(t / 40)
        assert t / 40 < t / 4


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.001, 0.5, 2, 1000])
def test_min40_4_is_unit_independent(scale):
    """Homogeneous of degree 1: no time-dimensioned constant appears in it.

    Scaling both inputs scales the budget exactly, so the formula reads the
    same in seconds or milliseconds and the ms/s trap is unrepresentable.
    """
    for t, i in [(60, 0), (60, 1), (30, 0.1), (2, 1), (300, 5)]:
        assert (_think_ms(MIN40_4, t * scale, i * scale)
                == pytest.approx(scale * _think_ms(MIN40_4, t, i)))


@pytest.mark.parametrize("scale", [0.5, 2])
def test_one_max_is_not_unit_independent(scale):
    """The cost of the named reserve: 8000 and 50 are millisecond constants.

    Recorded, not tolerated -- this is the trade min40-4 buys out, and the
    reason one-max's line must never be copied into a seconds-domain loop.
    """
    t, i = 60, 1
    assert (_think_ms(ONE_MAX, t * scale, i * scale)
            != pytest.approx(scale * _think_ms(ONE_MAX, t, i)))


def test_the_shipped_line_is_in_the_millisecond_domain():
    """main() divides by 1000 on the next line; assert that is still true."""
    assert re.search(r"think = times\.get\(\"movetime\", think\) / 1000", SRC)
    assert re.search(r"searcher\.deadline = start \+ max\(think, \.05\)", SRC)
