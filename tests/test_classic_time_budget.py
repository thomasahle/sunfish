"""The classic builtin clock: one whole-game pool, read at two limits.

`sunfish.py`'s `go` handler is the packed classic artifact's entire time
manager -- `pack.sh` deletes the `minifier-hide` block and the
`sunfish_ui.uci` import with it, so a checkout reaches the driver and only the
artifact runs that loop.  No other gate can see it: the node ladder never
starts a clock and a match reports only the result.

WHAT THIS FILE ASSERTS, and what it deliberately does not.  The pool's own
arithmetic -- monotonicity, continuity, the movestogo branch, the walks, the
five-fold headroom, the floor -- is characterised in `test_tm_pool.py` against
`uci.pool_budget`, and duplicating it here would be a second way to say the
same thing.  What only this file can say is that the ARTIFACT's inlined
millisecond copy is that same function, and that the loop AROUND it reads both
limits the way the measured arm did:

  1. the two shipped statements are lifted from the engine, so reshaping them
     fails here loudly instead of quietly testing a stale duplicate;
  2. they equal `uci.pool_budget` on a clock/increment grid -- one arithmetic
     at three sites (driver in seconds, 4k entry and this loop in
     milliseconds), which is the only thing that makes the duplication safe;
  3. the wall is armed as the deadline and the soft limit is read ONLY where
     the MTD bracket has closed.  A soft limit read at any yield stops at the
     soft limit, and the wall -- five times it -- is then unreachable, so the
     two-limit design would be worth nothing.  That is not a style point: the
     surrogate priced the budget alone at +40.7 [-41.7, +128.0] against the
     shipped min40_4 at 30+1 and the full pool at -223.3 [-345.5, -136.6] the
     other way, i.e. the whole effect is in the pair.

THE HISTORY THIS FILE REPLACES.  It used to walk two one-line candidates,
`min40-4` and `one-max`, chosen when the classic clock dropped its
`wtime/2 - 1s` cap (#196).  `min40-4` shipped and is kept below as the CONTROL
literal -- the arm this pool was measured against -- because a control that has
been deleted cannot be re-run.  `one-max` and the old parking policy are kept
for the same reason: they are the two failure shapes the pool has to not have.
"""
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sunfish_ui import uci                              # noqa: E402

SRC = (ROOT / "sunfish.py").read_text()

# ---- the shipped statements, lifted rather than copied ---------------------
SOFT_LINE = re.search(r"^ +(soft = min\(.*\))$", SRC, re.M)
THINK_LINE = re.search(r"^ +(think = max\(times\.get\(.*\))$", SRC, re.M)
CLIP_LINE = re.search(r"^ +(soft = min\(max\(soft / 1000, \.05\), think\))$", SRC, re.M)

# ---- the arms this one replaced, as literals, all MILLISECONDS ------------
# The `go` handler works in ms (`wtime`/`winc` arrive as integer ms) and
# crosses to seconds exactly once.  Mixing the domains produced a 590-second
# move once; every literal here is therefore labelled with its unit.
OLD = "think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000)"    # pre-#196
MIN40_4 = "think = min(wtime / 40 + 0.9 * winc, wtime / 4)"       # #196, the CONTROL
ONE_MAX = "think = max((wtime - 8000) / 40 + winc, 50)"           # #196's runner-up

FLOOR = uci.TM_FLOOR             # 0.05 s, and the engine's `.05`, same number
O = uci.MOVE_OVERHEAD            # 0.2 s, measured; the pool's O and the lag
OVERHEAD = 0.2                   # what a real deployment charges per move

CLOCKS = (0.001, 0.05, 0.2, 0.4, 1, 2, 5, 8.4, 10, 30, 60, 180, 300, 1800)
INCS = (0, 0.001, 0.05, 0.1, 0.5, 1, 2, 3, 5)


def shipped(wtime_ms, winc_ms, movetime_ms=None):
    """(soft, think) in SECONDS from the engine's own three statements.

    Runs the lifted text, so it cannot drift from what the artifact plays.
    """
    times = {} if movetime_ms is None else {"movetime": movetime_ms}
    ns = {"wtime": wtime_ms, "winc": winc_ms, "times": times,
          "min": min, "max": max}
    exec(SOFT_LINE.group(1), ns)      # noqa: S102 - the shipped expression
    exec(THINK_LINE.group(1), ns)     # noqa: S102
    exec(CLIP_LINE.group(1), ns)      # noqa: S102
    return ns["soft"], ns["think"]


def legacy(stmt, wtime_s, winc_s=0.0):
    """A retired one-liner's budget, SECONDS in and SECONDS out."""
    ns = {"wtime": wtime_s * 1000.0, "winc": winc_s * 1000.0,
          "min": min, "max": max}
    exec(stmt, ns)                    # noqa: S102
    return ns["think"] / 1000.0


# --------------------------------------------------------------------------
# (1) the statements are present and are the pool
# --------------------------------------------------------------------------

def test_the_three_budget_statements_are_present():
    for name, m in (("soft", SOFT_LINE), ("think", THINK_LINE),
                    ("soft clip", CLIP_LINE)):
        assert m, "the inline %s statement is missing or reshaped" % name


def test_the_shipped_budget_is_not_one_of_the_retired_one_liners():
    """A third form must not drift in unmeasured."""
    for stmt in (OLD, MIN40_4, ONE_MAX):
        assert stmt not in SRC, "a retired budget is back in the engine: %r" % stmt


def test_the_pool_constants_are_the_drivers_M_and_O():
    """39 and 42*200 are (M-1)*I and (M+2)*O, not free parameters.

    Read off the driver rather than restated, so retuning `POOL_MOVES` or
    `TM_OVERHEAD_MS` there without touching the artifact is a red test.
    """
    m, o_ms = uci.POOL_MOVES, 1000 * O
    assert "%d * winc" % (m - 1) in SOFT_LINE.group(1)
    assert "%d * %d" % (m + 2, o_ms) in SOFT_LINE.group(1)
    assert "%d) / 4" % (2 * o_ms) in SOFT_LINE.group(1)


# --------------------------------------------------------------------------
# (2) ONE ARITHMETIC AT THREE SITES -- the reason duplication is allowed
# --------------------------------------------------------------------------

def test_the_artifacts_millisecond_pool_is_the_drivers_pool():
    """t_ms(W, I) == 1000 * t_s(W/1000, I/1000) at every grid point, both limits.

    The driver works in seconds and this loop in milliseconds; the only thing
    between the two sources is a factor of 1000 in three constants.  The
    seconds/ms confusion has cost this project two incidents, which is why the
    crossing is asserted numerically and not argued.
    """
    worst = 0.0
    for w in CLOCKS:
        for i in INCS:
            got_s, got_h = shipped(w * 1000, i * 1000)
            want_s, want_h = uci.pool_budget(w, i)
            for got, want in ((got_s, want_s), (got_h, want_h)):
                worst = max(worst, abs(got - want) / max(abs(want), 1e-9))
    assert worst < 1e-12, f"artifact and driver disagree by {worst:.3e} relative"


def test_the_soft_limit_is_never_above_the_wall():
    for w in CLOCKS:
        for i in INCS:
            soft, think = shipped(w * 1000, i * 1000)
            assert soft <= think + 1e-12, (w, i, soft, think)


def test_the_wall_is_five_soft_limits_wherever_no_clamp_binds():
    """The headroom the bracket rule exists to make reachable.

    Where neither the quarter-clock nor the half-clock clamp binds, the wall is
    exactly 5x the soft limit.  min40_4's was 1/0.8 = 1.25x, which is why the
    same stop rule bought nothing there.
    """
    seen = 0
    for w in (30, 60, 180, 300, 1800):
        for i in (0, 0.1, 1):
            soft, think = shipped(w * 1000, i * 1000)
            if soft > FLOOR and think < (w - 2 * O) / 2 - 1e-9:
                assert think == pytest.approx(5 * soft, rel=1e-12)
                seen += 1
    assert seen >= 10, "grid never reached the unclamped regime"


# --------------------------------------------------------------------------
# (3) the wall cannot go negative -- the defect that lost a game
# --------------------------------------------------------------------------

@pytest.mark.parametrize("wtime", [0.001, 0.05, 0.5, 1, 1.9, 2, 2.4, 5, 60, 1800])
@pytest.mark.parametrize("winc", [0, 0.1, 1, 5])
def test_the_wall_is_always_positive_and_at_least_the_floor(wtime, winc):
    """`wtime/2 - 1s` goes negative under a 2 s clock; half of `A` cannot.

    A negative wall is an already-expired deadline, which is how
    lichess.org/EAThUL0P was lost: ~16 moves at no search at all.
    """
    soft, think = shipped(wtime * 1000, winc * 1000)
    assert think >= FLOOR - 1e-12
    assert soft >= min(FLOOR, think) - 1e-12
    assert legacy(OLD, 1.9) < 0, "the control no longer demonstrates the defect"


def test_a_movetime_overrides_both_limits_and_keeps_the_floor():
    """`go movetime` is the GUI's own number: it becomes the wall, and the soft
    limit is clipped to it so no break can fire after the wall has passed."""
    for mt in (1, 30, 300, 5000):
        soft, think = shipped(60000, 0, movetime_ms=mt)
        assert think == pytest.approx(max(mt / 1000, FLOOR))
        assert soft <= think + 1e-12
    # CI runs exactly this on the minified engine, so it is pinned here too.
    soft, think = shipped(60000, 0, movetime_ms=300)
    assert think == pytest.approx(0.3) and soft == pytest.approx(0.3)


# --------------------------------------------------------------------------
# (4) THE DISCLOSED HOLE, asserted so it cannot be forgotten
# --------------------------------------------------------------------------

def test_the_sudden_death_collapse_exists_and_is_bounded():
    """With no increment the pool is empty below (M+2)*O and soft hits the floor.

    Recorded rather than hidden: the driver measured -209.91 +/- 60.11 at a 1 s
    clock for exactly this, and the code comment says so.  What is asserted
    here is the SHAPE -- the collapse is to the floor and never below it, the
    engine always gets a positive budget, and it starts exactly at the knee --
    so that a future fix (scoping the pool to P > 0) changes this test on
    purpose instead of by accident.
    """
    knee = (uci.POOL_MOVES + 2) * O                     # 8.4 s at the shipped O
    for t in (0.05, 0.5, 1, 2, 5, 8.0):
        soft, think = shipped(t * 1000, 0)
        assert soft == pytest.approx(FLOOR) and think == pytest.approx(FLOOR)
        assert legacy(MIN40_4, t) > 0                   # the control still paces
    for t in (12, 20, 60):
        soft, _ = shipped(t * 1000, 0)
        assert soft > FLOOR, "the pool should be spending again above the knee"
    assert knee == pytest.approx(8.4)
    # ANY increment removes it: the pool is then income-fed, not overhead-bound.
    for i in (0.1, 1, 3):
        soft, _ = shipped(1000, i * 1000)
        assert soft > FLOOR or i * (uci.POOL_MOVES - 1) < knee


# --------------------------------------------------------------------------
# (5) the park, and the reserve it banks -- the safety argument
# --------------------------------------------------------------------------

def drift(spend, wtime_s, winc_s, overhead=OVERHEAD):
    """One step of T <- T - spend - O + I, as a signed change in the clock."""
    return winc_s - overhead - spend(wtime_s, winc_s)


def park_clock(spend, winc_s, overhead=OVERHEAD):
    """The clock a game rests at: the highest T with drift >= 0, by bisection.

    `drift` is nonincreasing in T because the budget only grows with the clock.
    0.0 means the clock never stops falling, i.e. there is genuinely no park.
    """
    if drift(spend, 0.0, winc_s, overhead) < 0:
        return 0.0
    lo, hi = 0.0, 600.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if drift(spend, mid, winc_s, overhead) >= 0:
            lo = mid
        else:
            hi = mid
    return lo


def pool_soft(wtime_s, winc_s):
    return shipped(wtime_s * 1000, winc_s * 1000)[0]


def test_a_park_exists_exactly_when_income_exceeds_overhead():
    """The 2026-08-15 correction, asserted: a park is not caused by a cap.

    The clock rests where `spend + overhead == income`.  Spend is at least the
    floor, so a resting point exists iff `income - overhead >= floor` -- a fact
    about the time control and the lag, not about the budget's shape.  The pool
    obeys it like every other manager.
    """
    for inc, over, parks in ((0.0, 0.05, False), (0.0, 0.20, False),
                             (0.1, 0.20, False), (0.1, 0.05, True),
                             (1.0, 0.05, True)):
        assert (park_clock(pool_soft, inc, over) > 0) is parks
        assert (inc - over >= FLOOR) is parks


def test_the_pool_banks_a_bigger_reserve_than_the_control_it_replaces():
    """min40_4's recorded cost was the thinnest flag margin in the field.

    On the SOFT limit -- what the loop stops at on a settled move -- the pool
    rests at a higher clock than min40_4 at every increment and both charged
    overheads, so the reserve it carries into an endgame is larger.  That is
    the half of the trade the budget can prove.

    WHAT THIS MODEL CANNOT SEE, stated rather than left to be discovered: on an
    unsettled move the pool runs PAST soft toward a wall five times higher, so
    its realized spend is larger than the number walked here and its realized
    park is lower.  The surrogate measures that directly and it comes out the
    other way -- the pool ends a 30+1 game on a 4.08 s median clock against
    min40_4's 17.85 s, while min40_4 flagged 3 of 120 modelled 60+0 games and
    the pool flagged none.  Both readings are real; this one is the floor of
    the pool's spend, not its expectation.
    """
    control = lambda t, i: max(legacy(MIN40_4, t, i), FLOOR)    # noqa: E731
    for over in (0.05, 0.2):
        for inc in (0.1, 0.5, 1.0, 3.0):
            assert park_clock(pool_soft, inc, over) >= park_clock(control, inc, over)
    # And the park is reached on a POSITIVE budget, which is the whole
    # difference from the policy that parked at T* = 2 + 2I on a negative cap.
    for inc in (0.1, 1.0, 3.0):
        rest = park_clock(pool_soft, inc, 0.05)
        assert pool_soft(rest, inc) > 0
        assert legacy(OLD, 2 + 2 * inc, inc) == pytest.approx(inc)


@pytest.mark.parametrize("moves", [80, 100, 120])
def test_the_pool_survives_a_long_sudden_death_game(moves):
    """3+0, the control that actually lost lichess EAThUL0P, walked on spend.

    The wall is what a game really pays when a search runs long, so the walk
    charges the SOFT limit (what the loop stops at) plus the lag -- and the
    clock must still be positive.  The old policy flags; the pool does not.
    """
    clock = 180.0
    for _ in range(moves):
        clock += drift(pool_soft, clock, 0.0)
        assert clock > 0
    old = 180.0
    flagged = False
    for _ in range(120):
        old += drift(lambda t, i: max(legacy(OLD, t, i), FLOOR), old, 0.0)
        if old <= 0:
            flagged = True
            break
    assert flagged, "the control no longer reproduces the lost game"


# --------------------------------------------------------------------------
# (6) the loop that reads the two limits
# --------------------------------------------------------------------------

def test_the_wall_and_soft_deadlines_are_armed_together():
    assert re.search(
        r"searcher\.deadline, searcher\.soft = start \+ think, start \+ soft$",
        SRC, re.M)


def test_the_soft_limit_is_read_only_where_the_bracket_has_closed():
    """The other half of the pool, and the half that carries the Elo.

    A check in the UCI loop would have to duplicate the MTD interval.  The
    search generator already owns the exact interval, so it reads the soft
    clock immediately after its inner loop closes and before the next depth.
    """
    search = SRC.split("def search(self, history):", 1)[1].split(
        "# UCI User interface", 1)[0]
    assert "while lower < upper - EVAL_ROUGHNESS:" in search
    assert re.search(r"^ {12}if time\.time\(\) > self\.soft: return$", search, re.M)
    packed = SRC.split('elif args[0] == "go":', 1)[1]
    assert "think * 0.8" not in packed
    assert "lo, up" not in packed


def test_elapsed_soft_time_finishes_the_current_bracket(monkeypatch):
    """Both reports closing depth 1 are yielded; depth 2 is never entered."""
    import sunfish                                      # noqa: E402

    searcher = sunfish.Searcher()
    searcher.soft = 0
    searcher.bound = lambda pos, gamma, depth, root=False: 0
    monkeypatch.setattr(sunfish.time, "time", lambda: 1)

    reports = list(searcher.search([sunfish.hist[0]]))
    assert [depth for depth, gamma, score, move in reports] == [1, 1]
    assert [score >= gamma for depth, gamma, score, move in reports] == [True, False]


def test_a_hard_stop_prefers_the_last_completed_depth():
    """Keep a depth-one candidate only if no complete depth exists yet."""
    packed = SRC.split('elif args[0] == "go":', 1)[1]
    assert re.search(r"^ {12}except Stop:\n {16}cand = best or cand$", packed, re.M)
    assert 'print("bestmove", cand or best or \'(none)\')' in packed


def test_default_soft_deadline_is_unbounded():
    import sunfish                                      # noqa: E402
    assert sunfish.Searcher().soft == 1 << 63


def test_the_bracket_width_is_the_engines_own_convergence_window():
    """EVAL_ROUGHNESS is the width the driver's MTD-bi loop stops at, so the
    loop reads convergence with the same constant the search converges on."""
    import sunfish                                      # noqa: E402
    assert sunfish.EVAL_ROUGHNESS == 15
    search = SRC.split("def search(self, history):", 1)[1].split(
        "# UCI User interface", 1)[0]
    assert "EVAL_ROUGHNESS" in search


# --------------------------------------------------------------------------
# (7) units
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [0.5, 2])
def test_the_pool_is_not_unit_independent_and_that_is_the_trade(scale):
    """min40_4 was homogeneous of degree 1; the pool cannot be.

    (M+2)*O and 2*O are absolute times -- 8400 ms and 400 ms here, 8.4 s and
    0.4 s in the driver -- because an overhead is a property of the deployment
    and not of the clock.  Recorded rather than tolerated: it is why the
    crossing above is asserted numerically at every grid point, and why the
    unit is named at every site in the engine's comment.
    """
    t, i = 60, 1
    a = shipped(t * 1000, i * 1000)[1]
    b = shipped(t * scale * 1000, i * scale * 1000)[1]
    assert b != pytest.approx(scale * a)
    # the control it replaced WAS unit-independent, which is the thing lost
    assert (legacy(MIN40_4, t * scale, i * scale)
            == pytest.approx(scale * legacy(MIN40_4, t, i)))
