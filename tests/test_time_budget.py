"""The time budget: one smooth function of (wtime, winc), clipped once.

    think = min(wtime * (1 + 20*winc) / (40 + 240*winc) + 0.9*winc,
                wtime * wtime / (2*wtime + 4))                     # seconds

Three superseded policies are kept below as LITERALS, because every claim
this file makes is a claim *relative to one of them*:

  OLD12  min(wtime/12 + 0.9*winc, wtime/2 - 1)                     (d3f7f12)
  STEP   min(wtime/(12 if winc else 40) + 0.9*winc, wtime/2 - 1)
  MS     the MILLISECONDS mirror, which the packed engine runs

OLD12 lost lichess.org/EAThUL0P on time at move 73 of a 3+0 game WITHOUT a
single move overrunning: /12 spent 12.8 s of a 180 s budget on ply 9, and
once the clock fell under 2 s the `wtime/2 - 1` cap went NEGATIVE, the budget
collapsed, and ~200 ms/move of unavoidable lag drained the rest. STEP fixed
that by pacing winc == 0 at /40, and the packed engine's twin of STEP measured
+235.5 +/- 65.4 head-to-head at 60+0 over its own pre-fix arm.

STEP's defect is that it is DISCONTINUOUS at winc == 0: one millisecond of
increment moves the divisor 40 -> 12 and puts 60+0.1 -- a sudden-death clock
in all but name -- back in the drain regime the /40 branch exists to close.
The shipped form ramps the divisor instead, and replaces the cap with one that
cannot go negative. What it gives up is EXACTNESS at increment TCs: it is
/12 + 0.9*inc asymptotically, within 10% for every winc >= 1 s and always on
the spend-less side, and that price is measured in games rather than assumed.

No existing gate can see any of this: the suite checks protocol and search
correctness, and a match would need a real 3+0 game to reproduce the loss. So
the budget curve is walked directly here, the same way
nnue_4k/tests/test_time_budget.py walks the packed engine's twin.

The formula is inline in run(), so it is extracted from the source rather than
duplicated -- if its shape changes, this test fails loudly instead of silently
testing a stale copy. uci.py works in SECONDS (wtime is divided by 1000 before
the formula), unlike the packed engine's milliseconds version; the two are
asserted equal under t_ms = 1000*t_s below, because that confusion has cost
this project two incidents.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "sunfish_ui" / "uci.py").read_text()

# THE budget statement. There is exactly one.
DEV = re.search(r"^\s+think = min\(wtime \* \(1 \+ 20 \* winc\)"
                r" / \(40 \+ 240 \* winc\) \+ 0\.9 \* winc,\n"
                r"\s+wtime \* wtime / \(2 \* wtime \+ 4\)\)\s*$", SRC, re.M)

# ---- the superseded policies, as literals -------------------------------
OLD12_LINE = "think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1)"
STEP_LINE = "think = min(wtime / (12 if winc else 40) + 0.9 * winc, wtime / 2 - 1)"
# The packed engine's MILLISECONDS mirror, kept here so this repo pins the
# sibling's text and vice versa (sunfish-packed nnue_4k/tests/test_time_budget
# .py holds the seconds mirror). A divergence between the two engines' time
# managers is then a red test on both sides, not a discovery made in a game.
MS_LINE = ("think = min(wtime * (1000 + 20 * winc) / (40000 + 240 * winc) + 0.9 * winc,"
           " wtime * wtime / (2 * wtime + 4000))")


def _eval(line, wtime, winc):
    ns = {"wtime": wtime, "winc": winc, "min": min}
    exec(line, ns)
    return ns["think"]


def budget(wtime_s, winc_s):
    """seconds of thinking time, straight from the shipped source"""
    return _eval(DEV.group(0).strip(), wtime_s, winc_s)


def old12(wtime_s, winc_s):
    return _eval(OLD12_LINE, wtime_s, winc_s)


def step(wtime_s, winc_s):
    return _eval(STEP_LINE, wtime_s, winc_s)


def ms_form(wtime_ms, winc_ms):
    """the packed engine's milliseconds mirror, evaluated in milliseconds"""
    return _eval(MS_LINE, wtime_ms, winc_ms)


def cap(wtime_s):
    """the safety clip alone"""
    return wtime_s * wtime_s / (2 * wtime_s + 4)


def base(wtime_s, winc_s):
    """the smooth term alone, before the clip"""
    return wtime_s * (1 + 20 * winc_s) / (40 + 240 * winc_s) + 0.9 * winc_s


CLOCKS = (0.001, 0.01, 0.1, 0.5, 1, 1.9, 2.667, 5, 30, 60, 180, 300, 1800)
INCS = (0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 3, 5)

# THE TWO BOUNDARIES AT winc == 0, both exact rationals. Everything the
# sudden-death argument rests on is a statement about which side of these a
# clock sits, so they are pinned rather than described.
#
#   B_CAP = 2/19 s: below it the NEW cap binds.
#           wtime/40 == wtime**2/(2*wtime + 4)  <=>  40*wtime == 2*wtime + 4
#   B_ID  = 40/19 s: at or above it the new policy and STEP are IDENTICAL,
#           because both reduce to wtime/40.
#           wtime/40 == wtime/2 - 1  <=>  wtime * 19/40 == 1
#
# Between them the base wtime/40 binds for the new policy while STEP is still
# clipped by wtime/2 - 1 -- which is nonpositive for wtime <= 2 s.
B_CAP = 2 / 19             # 0.10526 s
B_ID = 40 / 19             # 2.10526 s


def test_budget_statement_present():
    assert DEV, "smooth budget statement missing or reshaped in sunfish_ui/uci.py"
    assert SRC.count("think = min(wtime * ") == 1, "more than one budget statement"


# ---- the seconds/ms trap, closed numerically ----------------------------

def test_the_packed_mirror_is_this_formula_scaled():
    """t_ms(W, I) == 1000 * t_s(W/1000, I/1000) at every grid point.

    The two engines must budget the same time; the only thing separating
    their source lines is a factor of 1000 in three constants. That has been
    got wrong twice, so it is checked rather than reasoned about.
    """
    worst = 0.0
    for w in CLOCKS:
        for i in INCS:
            got, want = ms_form(w * 1000, i * 1000), 1000 * budget(w, i)
            worst = max(worst, abs(got - want) / max(abs(want), 1e-12))
    assert worst < 1e-12, "seconds and ms forms disagree by %.3e relative" % worst


# ---- (a) sudden death is EXACTLY /40 ------------------------------------

def test_sudden_death_is_exactly_wtime_over_40():
    """winc == 0 collapses the rational base to wtime/40, bit-for-bit."""
    for w in CLOCKS:
        assert base(w, 0) == w / 40, "winc == 0 is not wtime/40 at %s s" % w


def test_sudden_death_matches_the_step_policy_above_40_over_19():
    """At or above 40/19 s of clock the shipped budget == STEP at winc == 0.

    THE BOUNDARY IS 40/19 = 2.10526 s, NOT 2.667 s. 8/3 was carried over from
    an ABANDONED cap, max(wtime/2 - 1, wtime/8), whose branches cross there.
    This cap is wtime**2/(2*wtime + 4) and the crossing solves a different
    equation. The error was conservative in the useful direction: the true
    identity region is WIDER by 0.56 s, so every argument that leaned on it
    survives with more margin.

    Everything the sudden-death fix was argued from -- the EAThUL0P
    reconstruction, the packed twin's +235 at 60+0 -- lives above 40/19, so
    all of it carries to this form unchanged.
    """
    for w in CLOCKS + (2.106, 2.4, 2.667):
        if w < B_ID:
            continue
        assert budget(w, 0) == step(w, 0), (
            "sudden-death budget moved off the validated policy at %s s" % w)


def test_the_three_regimes_at_winc_zero_and_their_exact_boundaries():
    """The complete sudden-death picture, pinned at both crossings.

        wtime >= 40/19  : new == STEP, both = wtime/40      (IDENTICAL)
        2/19 .. 40/19   : new = wtime/40; STEP = wtime/2 - 1, which is
                          SMALLER, and nonpositive at wtime <= 2 s
        wtime <  2/19   : new = wtime**2/(2*wtime + 4)  (the cap binds)

    Only the first regime carries the validation, and it is the one every
    clock in that run actually occupied.
    """
    # UCI clocks arrive as integer milliseconds and are divided by 1000, so
    # that is the reachable domain. Over all of it above the boundary the two
    # policies are BIT-EQUAL: every integer ms from 2106 to 400000.
    assert all(budget(W / 1000, 0) == step(W / 1000, 0)
               for W in range(2_106, 400_001))
    # At real-valued clocks they agree to ~1e-16 rather than exactly, because
    # `wtime*1/40` and `wtime/40` can round apart and because `wtime/2 - 1`
    # CANCELS at this boundary. Neither is a policy difference, and neither is
    # reachable through UCI.
    assert abs(budget(B_ID, 0) - step(B_ID, 0)) < 1e-15
    # exactly at the cap boundary base and cap coincide; either side swaps
    assert abs(base(B_CAP, 0) - cap(B_CAP)) < 1e-17
    assert cap(B_CAP * 0.99) < base(B_CAP * 0.99, 0)
    assert cap(B_CAP * 1.01) > base(B_CAP * 1.01, 0)
    # the middle regime: our base binds, the old cap bites STEP
    for w in (0.2, 0.5, 1.0, 1.9, 2.0, 2.1):
        assert budget(w, 0) == w / 40, "base should bind at %s s" % w
        assert step(w, 0) < budget(w, 0)
    assert step(2.0, 0) <= 0, "the old cap must be nonpositive at a 2 s clock"


def test_the_2667_figure_is_not_a_boundary_of_this_policy():
    """A regression test against the specific error this file once carried.

    2.667 s belongs to max(wtime/2 - 1, wtime/8), designed and abandoned.
    Nothing happens at 2.667 s here: it is an interior point of the identity
    region, which already holds 0.56 s lower.
    """
    assert B_ID < 2.667
    assert budget(2.667, 0) == step(2.667, 0)
    assert budget(2.4, 0) == step(2.4, 0)
    assert 2.4 - B_ID > 0.25


# ---- (b) the clip ---------------------------------------------------------

def test_cap_is_strictly_positive_for_every_positive_clock():
    """The negative cap was the doorway into blind play, and it is gone.

    Under a 2 s clock `wtime/2 - 1` is negative; the budget then collapses to
    whatever floor is downstream and the engine plays on at no search --
    losing on the board rather than on the flag. wtime**2/(2*wtime + 4) has
    no such crossing.
    """
    for w in (0.001, 0.01, 0.1, 0.5, 1, 1.5, 1.999, 2, 2.666, 2.667, 10):
        assert cap(w) > 0, "cap non-positive at wtime=%s s" % w
        assert budget(w, 0) > 0, "budget non-positive at wtime=%s s" % w
    # and the old cap really did go negative there -- the test has teeth
    assert min(w / 2 - 1 for w in (0.001, 0.5, 1.999)) < 0


def test_cap_never_exceeds_half_the_clock():
    for w in CLOCKS:
        assert cap(w) <= w / 2 + 1e-15, "cap over wtime/2 at %s s" % w


def test_cap_tracks_the_old_cap_within_5pct_above_10s():
    """cap == (wtime/2 - 1) + 2/(wtime + 2), so the gap is exactly
    4/(wtime^2 - 4): 4.2% at a 10 s clock, 0.11% at 60 s, 0.004% at 300 s.
    Every regime the audit measured sits in the tail of that."""
    for w in (10, 20, 30, 60, 180, 300, 1800):
        old = w / 2 - 1
        assert abs(cap(w) - old) / old < 0.05, "cap moved more than 5% at %s s" % w
        assert abs(cap(w) - (old + 2 / (w + 2))) < 1e-12   # the closed form


# ---- (c) continuity -------------------------------------------------------

@pytest.mark.parametrize("w", (1, 30, 60, 300, 1800))
def test_continuous_in_winc(w):
    """The critique this change answers, as a test: no jump anywhere in winc.

    Walked at 0.1 ms resolution from 0 to 2 s of increment. The bound is
    scale-free (5% of the sudden-death budget at the same clock) so it means
    the same thing at a 1 s clock and a 30 min one.
    """
    lim = 0.05 * budget(w, 0)
    prev, worst = budget(w, 0.0), 0.0
    for k in range(1, 20_001):
        cur = budget(w, 2.0 * k / 20_000)
        worst = max(worst, abs(cur - prev))
        prev = cur
    assert worst < lim, "jump of %.4f s in winc at wtime=%s (limit %.4f)" % (worst, w, lim)


def test_the_step_form_fails_the_continuity_bound():
    """The bound above has teeth: STEP jumps 3.5 s at a 60 s clock."""
    prev, worst = step(60, 0.0), 0.0
    for k in range(1, 2_001):
        cur = step(60, 2.0 * k / 2_000)
        worst = max(worst, abs(cur - prev))
        prev = cur
    assert worst > 3.0, "STEP no longer jumps -- this file is testing the wrong thing"


# ---- (d) monotone in wtime ------------------------------------------------

@pytest.mark.parametrize("w", (0.1, 1, 30, 60, 300, 1800))
def test_monotone_nondecreasing_in_winc(w):
    """More increment may never buy less thinking -- and winc is the
    dimension the defect was in, so it is the one that most needed a test.

    Analytically the base is STRICTLY increasing in winc:

        dB/dI = 560*wtime/(40 + 240*winc)**2 + 0.9  >  0

    and the cap does not depend on winc, so the clipped allocation is
    nondecreasing. Walked here anyway, at 0.1 ms resolution.
    """
    prev = -1.0
    for k in range(0, 20_001):
        i = k * 0.0001
        cur = budget(w, i)
        assert cur >= prev - 1e-12, "budget fell at wtime=%s winc=%.4f" % (w, i)
        prev = cur


def test_the_base_derivative_in_winc_is_the_analytic_one():
    """dB/dI = 560*wtime/(40 + 240*winc)**2 + 0.9, checked numerically.

    Positive for every wtime >= 0, which is what makes the monotonicity above
    a theorem rather than a grid observation, and what makes the allocation
    continuous in the dimension the step form broke.
    """
    h = 1e-7
    for w in (0, 60, 1800):
        for i in (0, 0.1, 1, 5):
            analytic = 560 * w / (40 + 240 * i) ** 2 + 0.9
            numeric = (base(w, i + h) - base(w, i)) / h
            assert analytic > 0
            assert abs(analytic - numeric) < 1e-3 * max(1.0, abs(analytic)), (
                "dB/dI mismatch at wtime=%s winc=%s: %.6f vs %.6f"
                % (w, i, analytic, numeric))


@pytest.mark.parametrize("i", (0, 0.001, 0.05, 0.1, 0.5, 1, 3))
def test_monotone_nondecreasing_in_wtime(i):
    """More clock may never buy less thinking."""
    prev = -1.0
    for k in range(0, 20_001):
        cur = budget(k * 0.1, i)
        assert cur >= prev - 1e-12, "budget fell at wtime=%.1f winc=%s" % (k * 0.1, i)
        prev = cur


# ---- (e) the increment TCs the audit measured -----------------------------

@pytest.mark.parametrize("w,i", ((30, 1), (60, 1), (300, 3)))
def test_increment_tcs_within_10pct_of_the_audited_policy(w, i):
    """The 11-game production audit measured /12 + 0.9*inc from 60+1 to
    300+5. The shipped form approaches that instead of reproducing it, so
    what carries is a BOUND, not an identity: -7.4% at 30+1, -8.5% at 60+1,
    -3.3% at 300+3. Whether that price is free is a games question, answered
    by a pre-registered 30+1 non-inferiority match, not asserted here."""
    got, want = budget(w, i), step(w, i)
    assert abs(got - want) / want < 0.10, (
        "%s+%s moved %.2f%% off the audited policy" % (w, i, 100 * (got - want) / want))


def test_the_10pct_bound_holds_for_every_winc_at_or_above_1s():
    """Not just the three named TCs, and the deviation is one-sided.

    The base ratio is (12 + 240i)/(40 + 240i): 0.900 at i = 1 s, rising to 1,
    and the identical +0.9*inc term on both sides only pulls the full ratio
    up. So over the whole family the shipped form SPENDS LESS than the
    audited policy, never more, and never by as much as 10% -- 10% is the
    asymptote (i = 1 s, clock -> infinity), approached and not reached; the
    worst point on this grid is -9.9% at a 30-minute clock.

    Restricted to where neither policy is clipped: in the clipped regime the
    two caps differ on purpose, which the cap tests above cover."""
    for w in CLOCKS:
        for i in (1, 2, 3, 5, 30):
            if step(w, i) >= w / 2 - 1 or budget(w, i) >= cap(w):
                continue                      # clipped on one side or the other
            rel = (budget(w, i) - step(w, i)) / step(w, i)
            assert -0.10 < rel < 0, "%s+%s deviates %+.3f%%" % (w, i, 100 * rel)


def test_the_transition_band_is_where_the_shipped_form_differs():
    """Below 1 s of increment the two policies genuinely part company, and
    that is the point: STEP pays /12 for a single millisecond of increment."""
    assert step(60, 0.1) / budget(60, 0.1) > 1.7
    assert budget(60, 0.1) < 3.0 < step(60, 0.1)


# ---- (f) the walks --------------------------------------------------------

def walk(base_s, inc_s, moves, fn=None, overhead=0.2):
    """simulate our own clock over `moves` of our moves; -1 == flagged.

    overhead is per-move lag (network + process turnaround) the budget cannot
    see; lichess games show ~200 ms.
    """
    fn = fn or budget
    clock, floored = base_s, 0
    for mv in range(moves):
        think = max(fn(clock, inc_s), 0.05)
        floored += think <= 0.05 + 1e-12
        clock -= think + overhead
        clock += inc_s
        if clock <= 0:
            return -1, mv, floored
    return clock, moves, floored


def test_old_policy_reproduces_the_lost_game():
    """/12 flags a 73-move 3+0 game -- the walk has teeth."""
    left, reached, _ = walk(180, 0, 73, fn=old12)
    assert left == -1, "the old /12 policy no longer flags: walk model is stale"
    assert reached > 30, "flagged implausibly early at move %d" % reached


@pytest.mark.parametrize("moves", [80, 100, 120])
def test_sudden_death_survives_long_games(moves):
    """3+0, the control that actually lost a game."""
    left, reached, _ = walk(180, 0, moves)
    assert left > 0, "flagged at move %d of %d in 3+0" % (reached, moves)


def test_the_lost_game_would_now_be_survived():
    """73 moves at 3+0 -- the exact game, with time to spare."""
    left, _, _ = walk(180, 0, 73)
    assert left > 5, "only %.1fs left after the lost game's length" % left


def test_tiny_increment_no_longer_drains():
    """60+0.1 is the regime the step got wrong, and the walk shows why.

    A 0.1 s increment against ~0.2 s of lag is a sudden-death clock with
    extra steps, but STEP reads `winc != 0` and pays /12. Neither policy
    survives 100 moves under this pessimistic 200 ms overhead -- the point is
    the margin: the shipped form reaches move ~58 having played 3 moves at
    the floor, STEP reaches ~44 having played 14 blind."""
    _, smooth_reach, smooth_floor = walk(60, 0.1, 100)
    _, step_reach, step_floor = walk(60, 0.1, 100, fn=step)
    assert smooth_reach > step_reach + 8, (
        "no drain margin at 60+0.1: smooth %d moves vs step %d" % (smooth_reach, step_reach))
    assert smooth_floor * 2 < step_floor, (
        "no blind-play margin at 60+0.1: smooth %d floored moves vs step %d"
        % (smooth_floor, step_floor))


def test_the_old_cap_has_a_parking_fixed_point_at_2_plus_2_inc():
    """WHY the losing arm's clock plateaus instead of falling to zero.

    Once `wtime/2 - 1` is the binding term, the arm spends exactly that and
    banks one increment, so its clock obeys

        T_next = T - (T/2 - 1) + inc  =  T/2 + 1 + inc

    a contraction (slope 1/2) with the attracting fixed point

        T* = 2 + 2*inc   seconds

    One expression, two confirmations: at inc = 0 it gives 2.0 s and the
    packed twin's 60+0 run measured the pre-fix arm asymptoting at exactly
    2.0 s; at inc = 0.1 it gives 2.2 s and the 60+0.1 run measured the step
    arm at a 2.1 s median with a 2.0 s minimum over 438 games. It is also why
    that arm never flags: at the fixed point its spend equals its income.

    The shipped cap has no such fixed point -- it never goes nonpositive and
    never stops paying out.
    """
    for inc in (0.0, 0.1, 1.0):
        T = 60.0
        for _ in range(200):
            T = T / 2 + 1 + inc
        assert abs(T - (2 + 2 * inc)) < 1e-9, "fixed point moved at inc=%.1f" % inc


def test_tournament_control_unaffected():
    """1800+3 over a long game keeps a healthy reserve."""
    left, _, _ = walk(1_800, 3, 120)
    assert left > 5, "1800+3 left only %.1fs" % left


def test_early_move_is_no_longer_front_loaded():
    """The lost game spent 12.8s on ply 9; the fix must be far below that."""
    assert budget(180, 0) < 5.0, "first move still front-loaded"
