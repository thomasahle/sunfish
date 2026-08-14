"""The classic builtin clock: one pool, spent down, never parked.

sunfish.py's `go` handler budgets time on a single inline line. Two
properties decide whether that line is safe, and no other gate can see
either -- the node ladder never starts a clock, and a match reports only
the result.

  * where the policy PARKS.  Iterate the self-clock recurrence
    T <- T - think(T) - O + I over our own moves (O is per-move overhead
    we cannot avoid, I the increment).  A policy with a fixed point above
    its floor stops spending down: it reaches that clock and plays the
    rest of the game at whatever budget the point allows.  The shipped
    `min(t/12 + .9i, t/2 - 1s)` has one.  Once the cap binds the
    recurrence is T <- T/2 + 1 + I, fixed point T* = 2 + 2I seconds --
    measured at 2.0s (60+0) and a 2.1s median (60+0.1).  At 3+0 that park
    is 2s, the budget under it has already collapsed to the 0.05s floor
    because `t/2 - 1s` went NEGATIVE, and ~200ms/move of lag drains the
    remainder: lichess EAThUL0P, lost on time at move 73 with no single
    move overrunning.

  * how much RESERVE is banked when the floor is reached.  A pool policy
    is worth exactly the clock it still holds at its floor, in moves.

Two candidate one-liners are walked here against the shipped one:

    one-max   max((wtime - 8000) / 40 + winc, 50)
    min40-4   min(wtime / 40 + 0.9 * winc, wtime / 4)

Both drop the `t/2 - 1s` cap that manufactures the park, and reach the
same place by different routes -- one-max banks a named 8s overhead
reserve, min40-4 clips to a quarter clock so the reserve is four
increments and no time-dimensioned constant appears at all.  This file
is the shared scaffolding: the candidates are literals, so every
property below is checked on both no matter which one is shipped.
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


@pytest.mark.parametrize("name,stmt", sorted(CANDIDATES.items()))
def test_no_park_at_the_reference_increment(name, stmt):
    """I = 0.1s: the clock strictly falls at every clock, so no fixed point.

    A fixed point is a T with drift(T) == 0.  Overhead alone (0.2s) already
    exceeds the 0.1s income, and both candidates spend a nonnegative budget
    on top, so the drift is bounded strictly below zero everywhere -- there
    is nothing to converge to and the pool is genuinely spent down.
    """
    grid = [t / 10.0 for t in range(1, 6001)]        # 0.1s .. 600s
    worst = max(drift(stmt, t, 0.1) for t in grid)
    assert worst < 0, "%s parks: drift reaches %.4f" % (name, worst)
    assert worst <= 0.1 - OVERHEAD - FLOOR + 1e-12


@pytest.mark.parametrize("name,stmt", sorted(CANDIDATES.items()))
def test_no_park_at_sudden_death(name, stmt):
    """I = 0: income is zero, so any nonnegative budget drains the clock."""
    grid = [t / 10.0 for t in range(1, 6001)]
    assert max(drift(stmt, t, 0.0) for t in grid) < 0


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
    That is lichess EAThUL0P, and it is the reason a park at a low clock is
    worse than no park at all.
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

    This is min40-4's no-park argument and it differs from one-max's: there
    is no reserve floor doing the work, the capped arm simply never binds.
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
