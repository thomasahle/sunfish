"""The time budget: one smooth function of (wtime, winc), clipped once.

    think = min(wtime * (1000 + 20*winc) / (40000 + 240*winc) + 0.9*winc,
                wtime * wtime / (2*wtime + 4000))            # milliseconds

Three superseded policies are kept below as LITERALS, because every claim
this file makes is a claim *relative to one of them*:

  OLD12  min(wtime/12 + 0.9*winc, wtime/2 - 1000)               (pre-e73da7d)
  STEP   min(wtime/(12 if winc else 40) + 0.9*winc, wtime/2 - 1000)
  SEC    the normative SECONDS form, which sunfish_ui/uci.py runs

OLD12 lost lichess.org/EAThUL0P on time at move 73 of a 3+0 game WITHOUT a
single move overrunning: /12 spent 12.8 s of a 180 s budget on ply 9, and
once the clock fell under 2 s the `wtime/2 - 1000` cap went NEGATIVE, the
budget collapsed to the 0.05 s floor, and ~200 ms/move of unavoidable lag
drained the rest. STEP fixed that by pacing winc == 0 at /40, and measured
+235.5 +/- 65.4 head-to-head at 60+0 (MEASUREMENTS.md stage 1).

STEP's defect is that it is DISCONTINUOUS at winc == 0: one millisecond of
increment moves the divisor 40 -> 12 and puts 60+0.1 -- a sudden-death clock
in all but name -- back in the drain regime the /40 branch exists to close.
The shipped form ramps the divisor instead, and replaces the cap with one
that cannot go negative. What it gives up is EXACTNESS at increment TCs: it
is /12 + 0.9*inc asymptotically, within 10% for every winc >= 1 s, and that
price is measured in games rather than assumed.

The formula is inline in main(), so it is extracted from the source rather
than duplicated -- if its shape changes, this test fails loudly instead of
silently testing a stale copy. It is in MILLISECONDS (main() divides by 1000
on the next line); the seconds/ms confusion has cost this project two
incidents, so the two domains are asserted equal on a grid below.
"""
import os
import re

import pytest

ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "sunfish_nnue.py")
SRC = open(ENGINE).read()

# THE budget statement. There is exactly one, and it is NOT inside
# minifier-hide: source and artifact run the same formula. The old layout kept
# a plain /12 line for the artifact and hid the sudden-death branch behind
# minifier-hide -- and the 300+0 ladder then played the known-bad /12 branch on
# 97.2% of 4,158 matched moves while the fix sat dead in source
# (LOSS_TAXONOMY.md P0).
DEV = re.search(r"^\s+think = min\(wtime \* \(1000 \+ 20 \* winc\)"
                r" / \(40000 \+ 240 \* winc\) \+ 0\.9 \* winc,\n"
                r"\s+wtime \* wtime / \(2 \* wtime \+ 4000\)\)\s*$", SRC, re.M)

# ---- the superseded policies, as literals -------------------------------
OLD12_LINE = "think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000)"
STEP_LINE = "think = min(wtime / (12 if winc else 40) + 0.9 * winc, wtime / 2 - 1000)"
# The NORMATIVE seconds form, mirrored here so this repo pins the sibling's
# text and vice versa (sunfish tests/test_time_budget.py holds the ms mirror).
# A divergence between the two engines' time managers is then a red test on
# both sides, not a discovery made in a game.
SEC_LINE = ("think = min(wtime * (1 + 20 * winc) / (40 + 240 * winc) + 0.9 * winc,"
            " wtime * wtime / (2 * wtime + 4))")


def _eval(line, wtime, winc):
    ns = {"wtime": wtime, "winc": winc, "min": min}
    exec(line, ns)
    return ns["think"]


def budget(wtime_ms, winc_ms):
    """seconds of thinking time, i.e. the engine's `think` after its /1000"""
    return _eval(DEV.group(0).strip(), wtime_ms, winc_ms) / 1000.0


def old12(wtime_ms, winc_ms):
    return _eval(OLD12_LINE, wtime_ms, winc_ms) / 1000.0


def step(wtime_ms, winc_ms):
    return _eval(STEP_LINE, wtime_ms, winc_ms) / 1000.0


def sec_form(wtime_s, winc_s):
    """the normative SECONDS form, evaluated in seconds"""
    return _eval(SEC_LINE, wtime_s, winc_s)


def cap(wtime_ms):
    """the safety clip alone, in seconds"""
    return wtime_ms * wtime_ms / (2 * wtime_ms + 4000) / 1000.0


def base(wtime_ms, winc_ms):
    """the smooth term alone, before the clip, in seconds"""
    return (wtime_ms * (1000 + 20 * winc_ms) / (40000 + 240 * winc_ms)
            + 0.9 * winc_ms) / 1000.0


CLOCKS = (1, 10, 100, 500, 1_000, 1_900, 2_667, 5_000, 30_000, 60_000,
          180_000, 300_000, 1_800_000)
INCS = (0, 1, 10, 50, 100, 250, 500, 1_000, 3_000, 5_000)

# THE TWO BOUNDARIES AT winc == 0, both exact rationals. Everything the
# sudden-death argument rests on is a statement about which side of these a
# clock sits, so they are pinned rather than described.
#
#   B_CAP = 2000/19 ms: below it the NEW cap binds. wtime/40 == wtime^2 /
#           (2*wtime + 4000)  <=>  40*wtime == 2*wtime + 4000.
#   B_ID  = 40000/19 ms: at or above it the new policy and STEP are
#           IDENTICAL, because both reduce to wtime/40. wtime/40 ==
#           wtime/2 - 1000  <=>  wtime * 19/40 == 1000.
#
# Between them the base wtime/40 binds for the new policy while STEP is still
# clipped by wtime/2 - 1000 -- which is nonpositive for wtime <= 2000 ms.
B_CAP = 2000 / 19          # 105.263 ms
B_ID = 40000 / 19          # 2105.263 ms


# ---- the line actually ships --------------------------------------------

def test_budget_statement_ships_in_the_artifact():
    """One budget statement, outside minifier-hide, so the artifact gets it."""
    assert DEV, "budget statement missing or reshaped"
    assert SRC.count("think = min(wtime") == 1, (
        "more than one budget statement -- the hide-block split is back")
    stripped = re.sub(r"# minifier-hide start.*?# minifier-hide end", "", SRC, flags=re.S)
    assert "40000 + 240 * winc" in stripped, (
        "budget statement is inside minifier-hide: artifact would drop the fix")


# ---- the ms/seconds trap, closed numerically ----------------------------

def test_ms_form_is_the_seconds_form_scaled():
    """t_ms(W, I) == 1000 * t_s(W/1000, I/1000) at every grid point.

    The two engines must budget the same time; the only thing separating
    their source lines is a factor of 1000 in three constants. That has been
    got wrong twice, so it is checked rather than reasoned about.
    """
    worst = 0.0
    for W in CLOCKS:
        for I in INCS:
            got, want = budget(W, I), sec_form(W / 1000.0, I / 1000.0)
            worst = max(worst, abs(got - want) / max(abs(want), 1e-12))
    assert worst < 1e-12, "ms and seconds forms disagree by %.3e relative" % worst


# ---- (a) sudden death is EXACTLY /40 ------------------------------------

def test_sudden_death_is_exactly_wtime_over_40():
    """winc == 0 collapses the rational base to wtime/40, bit-for-bit.

    This is the whole reason stage 1's +235.5 +/- 65.4 at 60+0 carries: at
    winc == 0 the shipped budget IS the arm that won that match, provided the
    clip agrees too (next test).
    """
    for W in CLOCKS:
        assert base(W, 0) == W / 40 / 1000.0, "winc == 0 is not wtime/40 at %d ms" % W


def test_sudden_death_matches_the_validated_step_arm_above_40000_over_19():
    """At or above 40000/19 ms of clock, the shipped budget == STEP at
    winc == 0, exactly. The whole 60+0 validation lives above that.

    THE BOUNDARY IS 40000/19 = 2105.26 ms, NOT 2667 ms. 2667 (= 8000/3) was
    carried over from an ABANDONED cap, max(wtime/2 - 1000, wtime/8), whose
    two branches cross there. This cap is wtime**2/(2*wtime + 4000) and the
    relevant crossing is a different equation -- see the boundary tests
    below. The mistake was conservative in the useful direction: the true
    identity region is WIDER by 562 ms, so every argument that leaned on it
    survives with more margin, not less.
    """
    for W in CLOCKS + (2_106, 2_400, 2_667):
        if W < B_ID:
            continue
        assert budget(W, 0) == step(W, 0), (
            "sudden-death budget moved off the validated arm at %s ms" % W)


def test_the_three_regimes_at_winc_zero_and_their_exact_boundaries():
    """The complete sudden-death picture, pinned at both crossings.

        wtime >= 40000/19   : new == STEP, both = wtime/40   (IDENTICAL)
        2000/19 .. 40000/19 : new = wtime/40; STEP = wtime/2 - 1000, which is
                              SMALLER, and nonpositive at wtime <= 2000 ms
        wtime <  2000/19    : new = wtime^2/(2*wtime + 4000) (the cap binds)

    Only the first regime carries the +235.5 +/- 65.4 validation, and it is
    the one every clock in that run actually occupied.
    """
    # UCI clocks are integer milliseconds (`int(next(tokens))`), and over
    # that whole domain above the boundary the two policies are BIT-EQUAL:
    # every integer from 2106 to 400000 ms, no exceptions.
    assert all(budget(W, 0) == step(W, 0) for W in range(2_106, 400_001))
    # At real-valued clocks they agree to ~1e-16 s rather than exactly. Two
    # separate float effects, neither of them a policy difference:
    #   * `wtime*1000/40000` (ours) and `wtime/40` (STEP's) are different
    #     expressions of the same number and can round apart in the last bit;
    #   * `wtime/2 - 1000` CANCELS at this boundary -- 1052.63 - 1000 keeps
    #     only ~13 significant digits -- so STEP's own cap is the less
    #     accurate of the two there.
    # Both are unreachable through UCI, which parses integer milliseconds.
    assert abs(budget(B_ID, 0) - step(B_ID, 0)) < 1e-15
    assert abs(budget(B_ID * 1.000001, 0) - step(B_ID * 1.000001, 0)) < 1e-15
    # Below the boundary the difference is real and large, not an ulp: the
    # new policy is the strictly more generous one.
    assert budget(B_ID * 0.999999, 0) > step(B_ID * 0.999999, 0), (
        "below 40000/19 the new policy must be the more generous one")
    assert budget(2_000, 0) - step(2_000, 0) == 0.05
    # exactly at the cap boundary: base and cap coincide, either side swaps
    assert abs(base(B_CAP, 0) - cap(B_CAP)) < 1e-15
    assert cap(B_CAP * 0.99) < base(B_CAP * 0.99, 0)      # cap binds below
    assert cap(B_CAP * 1.01) > base(B_CAP * 1.01, 0)      # base binds above
    # the middle regime: base binds for us, the old cap bites STEP
    for W in (200, 500, 1_000, 1_900, 2_000, 2_100):
        assert budget(W, 0) == W / 40 / 1000.0, "base should bind at %d ms" % W
        assert step(W, 0) < budget(W, 0)
    assert step(2_000, 0) <= 0, "the old cap must be nonpositive at a 2 s clock"


def test_the_2667ms_figure_is_not_a_boundary_of_this_policy():
    """A regression test against the specific error this file once carried.

    2667 ms belongs to max(wtime/2 - 1000, wtime/8), a cap that was designed
    and abandoned. Nothing happens at 2667 ms in the shipped policy: it is an
    interior point of the identity region, and the identity already holds
    562 ms below it.
    """
    assert B_ID < 2_667
    assert budget(2_667, 0) == step(2_667, 0)          # identity, unremarkably
    assert budget(2_400, 0) == step(2_400, 0)          # and already 267 ms lower
    # stage 1's measured minimum clock was 2.4 s, so the run never left the
    # identity region -- with 295 ms of margin, not the 0 the old figure implied
    assert 2_400 - B_ID > 250


# ---- (b) the clip ---------------------------------------------------------

def test_cap_is_strictly_positive_for_every_positive_clock():
    """The negative cap was the doorway into the blind floor. It is gone.

    Stage 1 watched the pre-fix arm cross `wtime/2 - 1000 < 0` at median move
    42 and then play a median 16 moves at the 0.05 s floor -- mated on the
    board, never flagging. wtime^2/(2*wtime + 4000) has no such crossing.
    """
    for W in (1, 2, 5, 10, 50, 100, 500, 999, 1_000, 1_999, 2_000, 2_666, 2_667, 10_000):
        assert cap(W) > 0, "cap non-positive at wtime=%d ms" % W
        assert budget(W, 0) > 0, "budget non-positive at wtime=%d ms" % W
    # and the old cap really did go negative there -- the test has teeth
    assert min(W / 2 - 1000 for W in (1, 500, 1_999)) < 0


def test_cap_never_exceeds_half_the_clock():
    for W in CLOCKS:
        assert cap(W) <= W / 2 / 1000.0 + 1e-12, "cap over wtime/2 at %d ms" % W


def test_cap_tracks_the_old_cap_within_5pct_above_10s():
    """cap == (wtime/2 - 1000) + 2e6/(wtime + 2000), so the gap is 4/(t^2-4)
    in seconds -- 4.2% at a 10 s clock and 0.004% at 300 s. Every measured
    regime sits in the tail of that, so none of them moved."""
    for W in (10_000, 20_000, 30_000, 60_000, 180_000, 300_000, 1_800_000):
        old = W / 2 - 1000
        assert abs(cap(W) * 1000 - old) / old < 0.05, "cap moved >5% at %d ms" % W
        # the closed form, asserted rather than trusted
        assert abs(cap(W) * 1000 - (old + 2e6 / (W + 2000))) < 1e-6


# ---- (c) continuity -------------------------------------------------------

@pytest.mark.parametrize("W", (1_000, 30_000, 60_000, 300_000, 1_800_000))
def test_continuous_in_winc(W):
    """Thomas's objection, as a test: no jump anywhere in winc.

    Walked at 0.1 ms resolution from 0 to 2 s of increment. The bound is
    scale-free (5% of the sudden-death budget at the same clock) so it means
    the same thing at a 1 s clock and a 30 min one.
    """
    lim = 0.05 * budget(W, 0)
    prev, worst = budget(W, 0.0), 0.0
    for k in range(1, 20_001):
        cur = budget(W, 2_000 * k / 20_000)
        worst = max(worst, abs(cur - prev))
        prev = cur
    assert worst < lim, "jump of %.4f s in winc at wtime=%d ms (limit %.4f)" % (worst, W, lim)


def test_the_step_form_fails_the_continuity_bound():
    """The bound above has teeth: STEP jumps 3.5 s at a 60 s clock."""
    prev, worst = step(60_000, 0.0), 0.0
    for k in range(1, 2_001):
        cur = step(60_000, 2_000 * k / 2_000)
        worst = max(worst, abs(cur - prev))
        prev = cur
    assert worst > 3.0, "STEP no longer jumps -- this file is testing the wrong thing"


# ---- (d) monotone in wtime ------------------------------------------------

@pytest.mark.parametrize("W", (100, 1_000, 30_000, 60_000, 300_000, 1_800_000))
def test_monotone_nondecreasing_in_winc(W):
    """More increment may never buy less thinking -- and winc is the
    dimension the defect was in, so it is the one that most needed a test.

    Analytically the base is STRICTLY increasing in winc:

        dB/dI = 560*wtime/(40000 + 240*I)^2 + 0.9  >  0

    and the cap does not depend on winc at all, so the clipped allocation is
    nondecreasing. Walked here anyway, at 0.1 ms resolution.
    """
    prev = -1.0
    for k in range(0, 20_001):
        I = k * 0.1
        cur = budget(W, I)
        assert cur >= prev - 1e-12, "budget fell at wtime=%d winc=%.1f" % (W, I)
        prev = cur


def test_the_base_derivative_in_winc_is_the_analytic_one():
    """dB/dI = 560*wtime/(40000 + 240*winc)^2 + 0.9, checked numerically.

    Positive everywhere for wtime >= 0, which is what makes the monotonicity
    above a theorem rather than a grid observation.
    """
    h = 1e-4
    for W in (0, 60_000, 1_800_000):
        for I in (0, 100, 1_000, 5_000):
            # 560_000, not 560: that constant is the SECONDS-domain one
            # (dB/dI = 560*T/(40 + 240*I)^2 + 0.9). Rescaling T -> W/1000 and
            # I -> J/1000 turns it into 560_000. Both evaluate to the same
            # dimensionless 21.9 at a 60 s clock with no increment, which is
            # the arithmetic that catches the slip.
            analytic = 560_000 * W / (40_000 + 240 * I) ** 2 + 0.9
            numeric = (base(W, I + h) - base(W, I)) / h * 1000.0
            assert analytic > 0
            assert abs(analytic - numeric) < 1e-3 * max(1.0, abs(analytic)), (
                "dB/dI mismatch at wtime=%d winc=%d: %.6f vs %.6f"
                % (W, I, analytic, numeric))


@pytest.mark.parametrize("I", (0, 1, 50, 100, 500, 1_000, 3_000))
def test_monotone_nondecreasing_in_wtime(I):
    """More clock may never buy less thinking. Both terms of the min are
    increasing in wtime, so the min is -- checked anyway, cheaply."""
    prev = -1.0
    for k in range(0, 20_001):
        cur = budget(k * 100, I)
        assert cur >= prev - 1e-12, "budget fell at wtime=%d ms winc=%d" % (k * 100, I)
        prev = cur


# ---- (e) the increment TCs the audit measured -----------------------------

@pytest.mark.parametrize("W,I", ((30_000, 1_000), (60_000, 1_000), (300_000, 3_000)))
def test_increment_tcs_within_10pct_of_the_audited_policy(W, I):
    """The 11-game production audit measured /12 + 0.9*inc from 60+1 to
    300+5. The shipped form approaches it instead of reproducing it, so what
    carries is a BOUND, not an identity: -7.4% at 30+1, -8.5% at 60+1,
    -3.3% at 300+3. Whether that price is free is a games question, and it is
    pre-registered as a 30+1 non-inferiority match, not asserted here."""
    got, want = budget(W, I), step(W, I)
    assert abs(got - want) / want < 0.10, (
        "%d+%d moved %.2f%% off the audited policy" % (W, I, 100 * (got - want) / want))


def test_the_10pct_bound_holds_for_every_winc_at_or_above_1s():
    """Not just the three named TCs, and the deviation is one-sided.

    The base ratio is (12 + 240i)/(40 + 240i): 0.900 at i = 1 s, rising to 1,
    and the identical +0.9*inc term on both sides only pulls the full ratio
    up. So over the whole family the shipped form SPENDS LESS than the
    audited policy, never more, and never by as much as 10% -- 10% is the
    asymptote (i = 1 s, clock -> infinity), approached and not reached; the
    worst point on this grid is -9.95% at a 30-minute clock.

    Restricted to where neither policy is clipped: in the clipped regime the
    two caps differ on purpose, which the cap tests above cover."""
    for W in CLOCKS:
        for I in (1_000, 2_000, 3_000, 5_000, 30_000):
            if step(W, I) >= (W / 2 - 1000) / 1000.0 or budget(W, I) >= cap(W):
                continue                      # clipped on one side or the other
            rel = (budget(W, I) - step(W, I)) / step(W, I)
            assert -0.10 < rel < 0, "%d+%d deviates %+.3f%%" % (W, I, 100 * rel)


def test_the_transition_band_is_where_the_shipped_form_differs():
    """Below 1 s of increment the two policies genuinely part company, and
    that is the point: STEP pays /12 for a single millisecond of increment."""
    assert step(60_000, 100) / budget(60_000, 100) > 1.7
    assert budget(60_000, 100) < 3.0 < step(60_000, 100)


# ---- (f) the walks --------------------------------------------------------

def walk(base_ms, inc_ms, moves, fn=budget, overhead=0.2):
    """simulate our own clock over `moves` of our moves; -1 == flagged.

    overhead is per-move lag (network + process turnaround) the budget cannot
    see; the lichess games show ~200 ms.
    """
    clock, floored = base_ms, 0
    for mv in range(moves):
        think = max(fn(clock, inc_ms), 0.05)
        floored += think <= 0.05 + 1e-12
        clock -= (think + overhead) * 1000
        clock += inc_ms
        if clock <= 0:
            return -1, mv, floored
    return clock, moves, floored


def test_the_old_policy_reproduces_the_lost_game():
    """OLD12 flags a 73-move 3+0 game -- the walk model has teeth."""
    left, reached, _ = walk(180_000, 0, 73, fn=old12)
    assert left == -1, "the old /12 policy no longer flags: walk model is stale"
    assert reached > 30, "flagged implausibly early at move %d" % reached


@pytest.mark.parametrize("moves", [80, 100, 120])
def test_sudden_death_survives_long_games(moves):
    """3+0, the control that actually lost a game."""
    left, reached, _ = walk(180_000, 0, moves)
    assert left > 0, "flagged at move %d of %d in 3+0" % (reached, moves)


def test_the_lost_game_would_now_be_survived():
    """73 moves at 3+0 -- the exact game, with time to spare."""
    left, _, _ = walk(180_000, 0, 73)
    assert left > 5_000, "only %.1fs left after the lost game's length" % (left / 1000)


def test_tiny_increment_no_longer_drains():
    """60+0.1 is the regime the step got wrong, and the walk shows why.

    A 0.1 s increment against ~0.2 s of lag is a sudden-death clock with
    extra steps, but STEP reads `winc != 0` and pays /12. Neither policy
    survives 100 moves under this pessimistic 200 ms overhead -- the point is
    the margin: the shipped form reaches move ~58 having played 3 moves at
    the floor, STEP reaches ~44 having played 14 blind."""
    smooth_left, smooth_reach, smooth_floor = walk(60_000, 100, 100)
    step_left, step_reach, step_floor = walk(60_000, 100, 100, fn=step)
    assert smooth_reach > step_reach + 8, (
        "no drain margin at 60+0.1: smooth %d moves vs step %d" % (smooth_reach, step_reach))
    assert smooth_floor * 2 < step_floor, (
        "no blind-play margin at 60+0.1: smooth %d floored moves vs step %d"
        % (smooth_floor, step_floor))


def test_the_old_cap_has_a_parking_fixed_point_at_2_plus_2_inc():
    """WHY the losing arm's clock plateaus instead of falling to zero.

    Once `wtime/2 - 1000` is the binding term, the arm spends exactly that
    and banks one increment, so its clock obeys

        T_{n+1} = T_n - (T_n/2 - 1) + I  =  T_n/2 + 1 + I

    a contraction (slope 1/2) with the attracting fixed point

        T* = 2 + 2*I   seconds

    That single expression predicts BOTH runs. At I = 0 it gives 2.0 s, and
    stage 1 measured the pre-fix arm asymptoting at exactly 2.0 s. At
    I = 0.1 it gives 2.2 s, and match 1 measured the step arm at a 2.1 s
    median with a 2.0 s minimum across all 438 games -- the fixed point less
    per-move overhead. It is also why the arm never flags: at the fixed point
    its spend equals its income exactly.

    The shipped cap has no such fixed point in the starving regime, because
    it never goes nonpositive and never stops paying out.
    """
    for I in (0.0, 0.1, 1.0):
        T = 60.0
        for _ in range(200):
            T = T / 2 + 1 + I
        assert abs(T - (2 + 2 * I)) < 1e-9, "fixed point moved at inc=%.1f" % I
    assert abs((2 + 2 * 0.0) - 2.0) < 1e-12      # stage 1 observed 2.0 s
    assert abs((2 + 2 * 0.1) - 2.2) < 1e-12      # match 1 observed 2.1 s median


def test_tournament_control_unaffected():
    """1800+3 over a long game keeps a healthy reserve."""
    left, _, _ = walk(1_800_000, 3_000, 120)
    assert left > 5_000, "1800+3 left only %.1fs" % (left / 1000)


def test_early_move_is_no_longer_front_loaded():
    """The lost game spent 12.8s on ply 9; the fix must be far below that."""
    assert budget(180_000, 0) < 5.0, "first move still front-loaded"
