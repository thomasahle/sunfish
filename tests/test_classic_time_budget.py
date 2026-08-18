"""The classic builtin clock: three per-move numbers, and a wall that is five.

`sunfish.py`'s `go` handler is the packed classic artifact's entire time
manager -- `pack.sh` deletes the `minifier-hide` block and the
`sunfish_ui.uci` import with it, so a checkout reaches the driver and only the
artifact runs that loop.  No other gate can see it: the node ladder never
starts a clock and a match reports only the result.

WHAT CHANGED, 2026-08-18, and why this file changed shape with it.  The loop
ran the POOL, whose defining property was that all three shipped sites (this
loop, `sunfish_ui/uci.py`, the 4k entry) evaluated ONE arithmetic -- so the
duplication was safe because it was checked numerically against
`uci.pool_budget` on a grid.  It now runs a per-move BUDGET that subtracts the
lag from each limit instead of reserving it in a pool, and the driver keeps
the pool.  **That equality is therefore gone, and this file no longer asserts
it.**  What replaces it is not weaker but different: the artifact's own three
statements are lifted and grid-checked against `tools/ctwin/tmlib.budget`, the
mirror the surrogate ranked, so the thing that plays and the thing that was
measured are provably the same function; and the divergence from the driver is
CHARACTERISED here rather than merely noted.

The other structural change: `go movetime` is gone from this loop.
`sunfish.py` is the simplified, clock-only UCI the TCEC-4k rules ask for;
`sunfish_ui/uci.py` remains the full interface and still honours `movetime`.
So the two limits no longer need a clip line coupling them -- `think >= soft`
falls out of the arithmetic, and that is asserted below rather than enforced.

WHAT THIS FILE ASSERTS:

  1. the three shipped statements are lifted from the engine, so reshaping
     them fails here loudly instead of quietly testing a stale duplicate;
  2. they ARE the surrogate's mirror, on a grid -- the gate that caught a real
     packed-arm mismatch on 2026-08-18, six values apart, 24 games into a
     match;
  3. `think >= soft` everywhere, with no statement coupling them;
  4. the floors, the clamps, and the fact that neither can go negative;
  5. the loop AROUND them reads both limits the way the measured arm did --
     the wall armed as the deadline, the soft limit read ONLY where the MTD
     bracket has closed.  A soft limit read at any yield stops at the soft
     limit and the wall, five times it, is then unreachable; the surrogate
     priced that pair at +223.3 against +40.7 and +64.4 for the halves.

THE HISTORY THIS FILE REPLACES.  `min40-4` and `one-max` (#196) and the pool
(#217) are kept below as CONTROL literals -- arms that have been deleted
cannot be re-run -- together with the failure shapes each of them had.
"""
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ctwin"))

from sunfish_ui import uci                              # noqa: E402
import tmlib                                            # noqa: E402

SRC = (ROOT / "sunfish.py").read_text()

# ---- the shipped statements, lifted rather than copied ---------------------
BUDGET_LINE = re.search(r"^ +(budget = wtime / 40 \+ winc - DELAY)$", SRC, re.M)
SOFT_LINE = re.search(r"^ +(soft = max\(min\(budget, .*)$", SRC, re.M)
THINK_LINE = re.search(r"^ +(think = max\(min\(5 \* budget, .*)$", SRC, re.M)
DELAY_LINE = re.search(r"^DELAY = (\d+)$", SRC, re.M)
DELAY = float(DELAY_LINE.group(1)) if DELAY_LINE else None

# ---- the arms this one replaced, as literals, all MILLISECONDS ------------
# The `go` handler works in ms (`wtime`/`winc` arrive as integer ms) and
# crosses to seconds exactly once.  Mixing the domains produced a 590-second
# move once; every literal here is therefore labelled with its unit.
OLD = "think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000)"    # pre-#196
MIN40_4 = "think = min(wtime / 40 + 0.9 * winc, wtime / 4)"       # #196
ONE_MAX = "think = max((wtime - 8000) / 40 + winc, 50)"           # #196's runner-up
POOL = ("soft = max(0, min((wtime + 39 * winc - 42 * 200) / 40, "
        "(r := wtime - 400) / 4))")                               # #217

SOFT_FLOOR = 0.100               # the shipped 100, SECONDS
THINK_FLOOR = 0.200              # the shipped 200, SECONDS
O = uci.MOVE_OVERHEAD            # 0.2 s, measured; the lag DELAY names
OVERHEAD = 0.2                   # what a real deployment charges per move

CLOCKS = (0.001, 0.05, 0.2, 0.4, 1, 2, 5, 8.4, 10, 30, 60, 180, 300, 1800)
INCS = (0, 0.001, 0.05, 0.1, 0.5, 1, 2, 3, 5)


def shipped(wtime_ms, winc_ms):
    """(soft, think) in SECONDS from the engine's own three statements.

    Runs the lifted text, so it cannot drift from what the artifact plays.
    There is no `movetime` parameter any more, and that is the point: this
    loop does not read one.
    """
    ns = {"wtime": wtime_ms, "winc": winc_ms, "DELAY": DELAY,
          "min": min, "max": max}
    exec(BUDGET_LINE.group(1), ns)    # noqa: S102 - the shipped expression
    exec(SOFT_LINE.group(1), ns)      # noqa: S102
    exec(THINK_LINE.group(1), ns)     # noqa: S102
    return ns["soft"], ns["think"]


def legacy(stmt, wtime_s, winc_s=0.0):
    """A retired one-liner's budget, SECONDS in and SECONDS out."""
    ns = {"wtime": wtime_s * 1000.0, "winc": winc_s * 1000.0,
          "min": min, "max": max}
    exec(stmt, ns)                    # noqa: S102
    return ns["think"] / 1000.0


# --------------------------------------------------------------------------
# (1) the statements are present, and they are the measured arithmetic
# --------------------------------------------------------------------------

def test_the_three_budget_statements_are_present():
    for name, m in (("budget", BUDGET_LINE), ("soft", SOFT_LINE),
                    ("think", THINK_LINE)):
        assert m, "the inline %s statement is missing or reshaped" % name
    assert DELAY_LINE, "DELAY is no longer a module constant"


def test_the_shipped_budget_is_not_one_of_the_retired_forms():
    """A fourth form must not drift in unmeasured."""
    for stmt in (OLD, MIN40_4, ONE_MAX, POOL):
        assert stmt not in SRC, "a retired budget is back in the engine: %r" % stmt


def test_the_artifact_is_the_surrogate_mirror():
    """THE GATE THAT EARNED ITS PLACE.  On 2026-08-18 a packed arm went to the
    box with a soft limit the mirror floored and the artifact did not: 6 of
    378 grid values apart, all below a 2 s clock, and it was found 24 games
    into a live match.  The arm that plays and the arm that was ranked are the
    same function here, or this is red.
    """
    worst = 0.0
    for w in tmlib.GRID_T:
        for i in tmlib.GRID_I:
            got_s, got_h = shipped(w * 1000, i * 1000)
            want = tmlib.budget(w, i, delay=DELAY)
            worst = max(worst, abs(got_s - want.soft), abs(got_h - want.hard))
    assert worst < 1e-12, f"artifact and mirror disagree by {worst:.3e} s"


def test_the_divisor_is_the_drivers_M_and_the_increment_is_whole():
    """What is still coupled to the driver, and what deliberately is not.

    The 40 is `uci.POOL_MOVES`, read off the driver rather than restated, so
    retuning the horizon there without touching the artifact is a red test.
    The increment coefficient is NOT: the pool spent `(M-1)/M` of it (39/40 =
    0.975, the horizon minus this move), and this form spends the whole thing,
    because the increment is what THIS move earns back rather than a share of
    what the game will earn.  Stated as the fact it is instead of forced into
    the old shape.
    """
    assert "wtime / %d" % uci.POOL_MOVES in BUDGET_LINE.group(1)
    assert "+ winc -" in BUDGET_LINE.group(1), "the increment is spent whole"
    assert "%d * winc" % (uci.POOL_MOVES - 1) not in SRC, "that was the pool's"
    # DELAY is the lag, and the driver's own measured figure is the default.
    assert DELAY in (100.0, 200.0), "an unranked DELAY shipped: %r" % DELAY
    if DELAY == 1000 * O:
        assert True                                     # the measured lichess lag
    assert "wtime / 4 - DELAY" in SOFT_LINE.group(1)
    assert "wtime / 2 - DELAY" in THINK_LINE.group(1)


# --------------------------------------------------------------------------
# (2) the two limits, and why nothing couples them
# --------------------------------------------------------------------------

def test_the_wall_is_never_below_the_soft_limit_and_nothing_enforces_it():
    """STRUCTURAL, not clipped.  `min` is monotone in both arguments, 5*b >= b
    wherever b >= 0, the second arguments are ordered at every clock, and where
    b < 0 both sit on floors that are ordered 200 >= 100.  So the clip line the
    pool needed (`soft = min(soft, think)`) is gone, and its absence is part of
    the claim.
    """
    for w in CLOCKS:
        for i in INCS:
            soft, think = shipped(w * 1000, i * 1000)
            assert think >= soft - 1e-12, (w, i, soft, think)
    packed = SRC.split('elif args[0] == "go":', 1)[1]
    assert "min(soft, think)" not in packed
    assert "min(max(soft" not in packed


def test_the_wall_is_five_budgets_wherever_no_clamp_binds():
    """The headroom the bracket rule exists to make reachable.

    Where neither clamp binds, the wall is exactly 5x the soft limit -- the
    pool's ratio, kept deliberately, because the 2x2 priced the pair at +223.3
    and that ratio is what the stop rule can spend.  min40_4's was 1.25x, which
    is why the same rule bought nothing there.
    """
    seen = 0
    for w in (30, 60, 180, 300, 1800):
        for i in (0, 0.1, 1):
            soft, think = shipped(w * 1000, i * 1000)
            if soft > SOFT_FLOOR and think < (w / 2 - DELAY / 1000) - 1e-9:
                assert think == pytest.approx(5 * soft, rel=1e-12)
                seen += 1
    assert seen >= 10, "grid never reached the unclamped regime"


def test_the_five_fold_is_taken_off_the_unclamped_budget_and_it_changes_nothing():
    """The one formal difference from the pool's wall, and it is EMPTY in the
    regime a game is played in.  Said out loud so it is never claimed as an
    advantage.

    The pool's wall was five times its CLAMPED soft; this one is five times
    the unclamped `budget`, so in principle a move whose share is clipped by
    the quarter clock keeps a wall worth reaching.  In practice the two clamps
    bind together: the quarter clamp needs I > 9T/40 and the half clamp fails
    to bind only when I <= 3T/40 + 4*DELAY/5, and those two overlap only below
    a ~1.1 s clock, where both floors have taken over anyway.

    So above a second, wherever the quarter clamp binds, the wall is the HALF
    clamp and the headroom is (T/2 - D)/(T/4 - D) -- about 2x, not 5x.
    """
    both = 0
    for w in (2, 3, 4, 6, 10):
        for i in (1, 2, 3, 5):
            if not (w / 4 < w / 40 + i):                # quarter clamp idle
                continue
            soft, think = shipped(w * 1000, i * 1000)
            if soft <= SOFT_FLOOR + 1e-12:
                continue
            assert think == pytest.approx(max(w / 2 - DELAY / 1000, THINK_FLOOR))
            assert think < 5 * soft, "the 5x survived a clamp; recheck the claim"
            both += 1
    assert both >= 8, "grid never reached the clamped regime"
    # Where NOTHING clamps, the 5x is exactly the pool's and is the ratio the
    # bracket rule was priced on.
    soft, think = shipped(60000, 0)
    assert think == pytest.approx(5 * soft)


# --------------------------------------------------------------------------
# (3) the floors, and a wall that cannot go negative
# --------------------------------------------------------------------------

@pytest.mark.parametrize("wtime", [0.001, 0.05, 0.5, 1, 1.9, 2, 2.4, 5, 60, 1800])
@pytest.mark.parametrize("winc", [0, 0.1, 1, 5])
def test_the_limits_are_always_positive_and_at_least_their_floors(wtime, winc):
    """`wtime/2 - 1s` goes negative under a 2 s clock; a floored max cannot.

    A negative wall is an already-expired deadline, which is how
    lichess.org/EAThUL0P was lost: ~16 moves at no search at all.
    """
    soft, think = shipped(wtime * 1000, winc * 1000)
    assert soft >= SOFT_FLOOR - 1e-12
    assert think >= THINK_FLOOR - 1e-12
    assert legacy(OLD, 1.9) < 0, "the control no longer demonstrates the defect"


def test_the_floors_are_asymmetric_and_that_is_the_flag_trade():
    """100 and 200, against the pool's flat 50/50.

    A flagging engine therefore spends 2x to 4x faster here than the pool
    does, which is not free and is not hidden: the real-clock 60+0 forfeit
    cell is what prices it.  The two floors have DIFFERENT knees, which is
    worth pinning because it is the shape of the endgame: the wall floors
    below 40*(DELAY + 40) of clock and the soft limit below 40*(DELAY + 100),
    so between the two the engine still paces its wall while its target sits
    on the floor.
    """
    assert THINK_FLOOR == pytest.approx(2 * SOFT_FLOOR)
    assert SOFT_FLOOR == pytest.approx(2 * tmlib.TM_FLOOR)
    soft_knee = 40 * (DELAY + 100) / 1000               # 12.0 s at DELAY=200
    think_knee = 40 * (DELAY + 40) / 1000               # 9.6 s at DELAY=200
    assert think_knee < soft_knee
    for t in (0.05, 0.5, 1, 2, 5, think_knee - 0.5):
        soft, think = shipped(t * 1000, 0)
        assert soft == pytest.approx(SOFT_FLOOR)
        assert think == pytest.approx(THINK_FLOOR)
    for t in (think_knee + 0.5, soft_knee - 0.5):       # the band between them
        soft, think = shipped(t * 1000, 0)
        assert soft == pytest.approx(SOFT_FLOOR)
        assert think > THINK_FLOOR
    for t in (soft_knee + 1, 60, 300):
        soft, _ = shipped(t * 1000, 0)
        assert soft > SOFT_FLOOR, "the budget should be pacing again above the knee"
    # An increment lifts the SHARE off the floor, but at a 1 s clock the
    # quarter clamp (250 - DELAY ms) is itself under the floor, so no
    # increment can: both terms have to clear 100 ms, not just the share.
    for t, i in ((1, 0.5), (1, 3), (30, 0.5), (60, 1)):
        soft, _ = shipped(t * 1000, i * 1000)
        share = t * 1000 / 40 + i * 1000 - DELAY
        assert (soft > SOFT_FLOOR) is (min(share, t * 1000 / 4 - DELAY) > 100)


def test_this_loop_does_not_read_movetime():
    """`sunfish.py` is the SIMPLIFIED, clock-only UCI (TCEC-4k); the full
    interface lives in `sunfish_ui/uci.py` and still honours `movetime`.
    Asserted on both sides so the split cannot rot: removing it here without
    keeping it there would break every GUI, and re-adding it here would put
    back the clip line this form does without.
    """
    packed = SRC.split('elif args[0] == "go":', 1)[1]
    assert "movetime" not in packed
    assert "movetime" in (ROOT / "sunfish_ui" / "uci.py").read_text()


# --------------------------------------------------------------------------
# (4) the driver keeps the pool -- so state the divergence, do not hide it
# --------------------------------------------------------------------------

def test_the_artifact_and_the_driver_are_no_longer_one_arithmetic():
    """The pool's safety argument for duplication was numerical equality with
    `uci.pool_budget`.  This form gives that up on purpose, so the honest
    replacement is to SHOW the divergence rather than assert an equality that
    would have to be weakened until it passed.

    Both are per-move budgets over the same M = 40 horizon charging the same
    lag, so they stay close where a game is actually played -- and they part
    company exactly where the pool's reserve empties.
    """
    for w, i in ((30, 1), (60, 1), (60, 0.1), (300, 3)):
        mine, _ = shipped(w * 1000, i * 1000)
        theirs, _ = uci.pool_budget(w, i)
        assert mine == pytest.approx(theirs, rel=0.35), (w, i, mine, theirs)
    # ...and below the pool's (M+2)*O knee they do not: the pool is on its
    # 50 ms floor with an empty pool, this form still paces a fortieth.
    mine, _ = shipped(5000, 0)
    theirs, _ = uci.pool_budget(5.0, 0)
    assert theirs == pytest.approx(tmlib.TM_FLOOR)
    assert mine == pytest.approx(SOFT_FLOOR)
    assert mine > theirs


def test_the_driver_and_the_4k_entry_still_run_the_pool():
    """Retiring the pool from THIS loop retires it from nowhere else, and the
    tmlib pin says so per site rather than going quiet."""
    assert hasattr(uci, "pool_budget")
    assert "budget_classic" in tmlib.PINNED
    assert "pool_classic" not in tmlib.PINNED


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


def budget_soft(wtime_s, winc_s):
    return shipped(wtime_s * 1000, winc_s * 1000)[0]


def test_a_park_exists_exactly_when_income_exceeds_overhead():
    """The 2026-08-15 correction, asserted: a park is not caused by a cap.

    The clock rests where `spend + overhead == income`.  Spend is at least the
    floor, so a resting point exists iff `income - overhead >= floor` -- a fact
    about the time control and the lag, not about the budget's shape.  This
    form obeys it like every other manager, and its floor being 100 ms rather
    than the pool's 50 ms MOVES THE LINE: a 0.1 s increment against a 0.05 s
    charge parked the pool and does not park this.  That is the asymmetric
    floor showing up as a policy difference, priced in the forfeit cell.
    """
    for inc, over, parks in ((0.0, 0.05, False), (0.0, 0.20, False),
                             (0.1, 0.20, False), (0.1, 0.05, False),
                             (1.0, 0.05, True), (1.0, 0.20, True)):
        assert (park_clock(budget_soft, inc, over) > 0) is parks
        assert (inc - over >= SOFT_FLOOR) is parks


def test_the_budget_banks_a_bigger_reserve_than_the_one_line_control():
    """min40_4's recorded cost was the thinnest flag margin in the field.

    On the SOFT limit -- what the loop stops at on a settled move -- this form
    rests at a higher clock than min40_4 at every increment and both charged
    overheads, so the reserve it carries into an endgame is larger.  That is
    the half of the trade the arithmetic can prove.

    WHAT THIS MODEL CANNOT SEE, stated rather than left to be discovered: on an
    unsettled move the loop runs PAST soft toward a wall five times higher, so
    its realized spend is larger than the number walked here and its realized
    park is lower.  The surrogate measures that directly and it comes out the
    other way -- the pool ends a 30+1 game on a 4.08 s median clock against
    min40_4's 17.85 s, while min40_4 flagged 3 of 120 modelled 60+0 games and
    the pool flagged none.  Both readings are real; this one is the floor of
    the pool's spend, not its expectation.
    """
    control = lambda t, i: max(legacy(MIN40_4, t, i), tmlib.TM_FLOOR)    # noqa: E731
    for over in (0.05, 0.2):
        for inc in (0.5, 1.0, 3.0):
            assert park_clock(budget_soft, inc, over) >= park_clock(control, inc, over)
    # THE EXCEPTION, which is the asymmetric floor again and is not smoothed
    # over: at a 0.1 s increment against a 0.05 s charge the net income is
    # 50 ms, under this form's 100 ms floor, so it has NO park at all where
    # min40_4 rests at 0.2 s.  A no-park arm drains to zero and flags; that
    # regime is 10+0-shaped and the surrogate cells cover it.
    assert park_clock(budget_soft, 0.1, 0.05) == 0.0
    assert park_clock(control, 0.1, 0.05) > 0
    # And the park is reached on a POSITIVE budget, which is the whole
    # difference from the policy that parked at T* = 2 + 2I on a negative cap.
    for inc in (1.0, 3.0):
        rest = park_clock(budget_soft, inc, 0.05)
        assert budget_soft(rest, inc) > 0
        assert legacy(OLD, 2 + 2 * inc, inc) == pytest.approx(inc)


@pytest.mark.parametrize("moves", [80, 100, 120])
def test_the_budget_survives_a_long_sudden_death_game(moves):
    """3+0, the control that actually lost lichess EAThUL0P, walked on spend.

    The wall is what a game really pays when a search runs long, so the walk
    charges the SOFT limit (what the loop stops at) plus the lag -- and the
    clock must still be positive.  The old policy flags; this one does not --
    on the SOFT limit.  The wall is five times higher and the surrogate flags
    this form at 60+0, which is exactly why a real-clock forfeit cell was run
    instead of this walk being called safety.
    """
    clock = 180.0
    for _ in range(moves):
        clock += drift(budget_soft, clock, 0.0)
        assert clock > 0
    old = 180.0
    flagged = False
    for _ in range(120):
        old += drift(lambda t, i: max(legacy(OLD, t, i), tmlib.TM_FLOOR), old, 0.0)
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
def test_the_budget_is_not_unit_independent_and_that_is_the_trade(scale):
    """min40_4 was homogeneous of degree 1; this form cannot be.

    DELAY and the two floors are absolute times, because a lag is a property
    of the deployment and not of the clock.  Recorded rather than tolerated:
    it is why the mirror is asserted numerically at every grid point, and why
    the unit is named in the engine's comment and in tmlib's docstring.
    """
    t, i = 60, 1
    a = shipped(t * 1000, i * 1000)[1]
    b = shipped(t * scale * 1000, i * scale * 1000)[1]
    assert b != pytest.approx(scale * a)
    # the control it replaced WAS unit-independent, which is the thing lost
    assert (legacy(MIN40_4, t * scale, i * scale)
            == pytest.approx(scale * legacy(MIN40_4, t, i)))
