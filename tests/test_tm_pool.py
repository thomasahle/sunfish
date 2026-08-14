"""The POOL time manager: a whole-game budget, split into soft and hard.

    P = max(0, T + (M-1)*I - (M+2)*O)          the pool the game still has
    A = max(0, T - 2*O)                        what THIS move can reach
    t_soft = min(s * P/M, A/4)                 stop STARTING iterations
    t_hard = min(5 * t_soft, A/2)              the in-search deadline

All quantities in SECONDS here, because sunfish_ui/uci.py works in seconds
(the UCI millisecond fields are divided by 1000 as they are parsed). The
packed engine runs the same arithmetic in MILLISECONDS, and its text is
pinned below as a literal and asserted equal under t_ms = 1000*t_s -- the
seconds/ms confusion has cost this project two incidents, so it is checked
numerically rather than reasoned about.

The incumbent single-curve manager is kept here as a literal too, for the
same reason tests/test_time_budget.py keeps its predecessors: every claim
made below is a claim *relative to* the arm the pool has to beat.

What no other gate can see: the suite checks protocol and search
correctness, and the two failures this manager exists to prevent -- the
EAThUL0P drain at 3+0 and the 60+0 blind-floor tail -- only show up in a
game or in a clock walk. So the clock is walked here.
"""
import pathlib
import re
import sys
import threading

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunfish                                          # noqa: E402
from sunfish_ui import uci                              # noqa: E402

# NOTE: uci.sunfish is the module global run() injects, and it is bound PER
# TEST below rather than here. tests/test_regressions.py loads its own copy of
# the engine and binds it at import time; a module-level binding here would
# win the collection race and hand that file's fixed-depth searches positions
# built from a different module's tables -- which is exactly what it did
# (the KPK conversion regression went red, only in a full-suite run).

SRC = (ROOT / "sunfish_ui" / "uci.py").read_text()
O = uci.MOVE_OVERHEAD            # the measured per-move overhead, SECONDS
FLOOR = uci.TM_FLOOR             # smallest positive budget, SECONDS

# ---- the arms, as literals ------------------------------------------------
# The INCUMBENT curve, i.e. the control arm of every match below. Pinned as a
# literal so this file fails loudly if the pool knob ever changes it, and so
# that a change to the incumbent (PR #188 replaces this exact line with the
# smooth two-line form) is a red test rather than a stale comparison.
INCUMBENT_LINE = "think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1)"
# PR #188's smooth curve, which the PACKED entry already ships and which the
# packed pool arm is measured against. Kept here because the classic driver
# will inherit it and the pool must beat whichever of the two is live.
SMOOTH_LINE = ("think = min(wtime * (1 + 20 * winc) / (40 + 240 * winc) + 0.9 * winc,"
               " wtime * wtime / (2 * wtime + 4))")
# THE PACKED MIRROR, MILLISECONDS: the replacement text of the `pooltm` mod in
# the packed repo's tools/build/make_variants.py. Same arithmetic, 1000x the
# constants. If the two ever drift, the grid assertion below goes red on this
# side; the packed side pins the seconds form the same way.
POOL_MS_LINES = (
    "M = 40\n"
    "P = max(0, wtime + (M - 1) * winc - (M + 2) * 200)\n"
    "A = max(0, wtime - 400)\n"
    "soft = min(P / M, A / 4)\n"
    "think = min(5 * soft, A / 2)\n"
)

CLOCKS = (0.001, 0.05, 0.2, 0.4, 1, 2, 5, 8.4, 10, 30, 60, 180, 300, 1800)
INCS = (0, 0.001, 0.05, 0.1, 0.5, 1, 2, 3, 5)


def _eval(text, **ns):
    ns["min"], ns["max"] = min, max
    exec(text, ns)
    return ns


def incumbent(wtime_s, winc_s):
    return _eval(INCUMBENT_LINE, wtime=wtime_s, winc=winc_s)["think"]


def smooth(wtime_s, winc_s):
    return _eval(SMOOTH_LINE, wtime=wtime_s, winc=winc_s)["think"]


def pool_ms(wtime_ms, winc_ms):
    """(soft, hard) in MILLISECONDS, straight from the packed mod's text."""
    ns = _eval(POOL_MS_LINES, wtime=wtime_ms, winc=winc_ms)
    return ns["soft"], ns["think"]


def pool(wtime_s, winc_s, **kw):
    """(soft, hard) in SECONDS, from the shipped driver."""
    return uci.pool_budget(wtime_s, winc_s, **kw)


# ---- (0) the arms are what we think they are ------------------------------

def test_the_control_arm_is_untouched():
    """The pool knob is additive: the incumbent expression still ships.

    If PR #188 lands first this literal becomes the smooth two-liner, and
    THIS LINE is where that is noticed -- a red test, not a silent drift.
    """
    assert INCUMBENT_LINE in SRC, "the incumbent budget line moved or changed"
    assert SRC.count("think = min(wtime /") == 2, "budget statements added or lost"


def test_the_manager_knob_defaults_to_the_incumbent():
    assert uci.TM_MANAGER in ("smooth", "pool")
    assert 'os.environ.get("TM_MANAGER", "smooth")' in SRC, "the default arm moved"


def test_an_unknown_manager_fails_loudly():
    """No silent fallback to a manager the operator did not ask for: a typo in
    a match script must not quietly play the control arm on both sides."""
    import subprocess
    env = {**__import__("os").environ, "TM_MANAGER": "poool"}
    p = subprocess.run([sys.executable, "-c", "import sunfish_ui.uci"],
                       cwd=str(ROOT), env=env, capture_output=True, text=True)
    assert p.returncode != 0, "an unknown TM_MANAGER was accepted"
    assert "poool" in p.stderr


# ---- (1) the formula, walked ----------------------------------------------

@pytest.mark.parametrize("w,i,soft,hard", [
    # T      I     t_soft    t_hard        (SECONDS, from the closed form)
    (60,     0,    1.290,     6.450),      # 60+0   P = 60 - 8.4
    (60,     1,    2.265,    11.325),      # 60+1   P = 60 + 39 - 8.4
    (30,     1,    1.515,     7.575),      # 30+1
    (180,    0,    4.290,    21.450),      # 3+0, the lost game's TC
    (1800,   3,   47.715,   238.575),      # TCEC 1800+3
])
def test_named_time_controls(w, i, soft, hard):
    s, h = pool(w, i)
    assert s == pytest.approx(soft, abs=5e-4), f"{w}+{i} soft"
    assert h == pytest.approx(hard, abs=5e-4), f"{w}+{i} hard"


def test_the_pool_prices_increment_as_income_and_overhead_as_tax():
    """P = T + (M-1)*I - (M+2)*O, term by term, at a TC where nothing clamps."""
    M, T, I = uci.POOL_MOVES, 60.0, 1.0
    expect = (T + (M - 1) * I - (M + 2) * O) / M
    assert pool(T, I)[0] == pytest.approx(expect, abs=1e-12)
    # income: an extra second of increment is worth (M-1)/M of a second/move
    assert pool(T, I + 1)[0] - pool(T, I)[0] == pytest.approx((M - 1) / M, abs=1e-12)
    # tax: an extra 10ms of overhead costs (M+2)/M of it per move
    a = uci.pool_budget(T, I, overhead=O)[0]
    b = uci.pool_budget(T, I, overhead=O + 0.01)[0]
    assert a - b == pytest.approx(0.01 * (M + 2) / M, abs=1e-12)


def test_soft_never_exceeds_hard_anywhere():
    """The invariant the whole design rests on, over the whole grid."""
    for w in CLOCKS:
        for i in INCS:
            for mtg in (None, 1, 5, 40, 200):
                for ply in (0, 30, 100):
                    for phase in (False, True):
                        s, h = pool(w, i, movestogo=mtg, ply=ply, phase_m=phase)
                        assert 0 < s <= h, f"soft {s} hard {h} at {w}+{i} mtg={mtg}"


def test_hard_is_five_times_soft_until_the_availability_clamp_binds():
    """The closed form, recomputed here rather than read off the code: the
    wall is 5x the UNFLOORED share, clipped by A/2 and floored last."""
    for w in CLOCKS:
        for i in INCS:
            s, h = pool(w, i)
            avail = max(0.0, w - 2 * O)
            raw = min(max(0.0, w + 39 * i - 42 * O) / 40, avail / 4)
            assert h == pytest.approx(max(min(5 * raw, avail / 2), FLOOR), rel=1e-12)
            assert s == pytest.approx(min(max(raw, FLOOR), h), rel=1e-12)
            if raw > FLOOR and 5 * raw < avail / 2:
                assert h == pytest.approx(5 * s, rel=1e-12)


def test_the_wall_can_never_go_negative_and_never_takes_half_the_clock():
    """The EAThUL0P failure, as an assertion. `wtime/2 - 1` is NEGATIVE under a
    2s clock -- and a negative wall is not "spend less", it is the collapse to
    a blind floor that lost the game. A/2 has no such crossing."""
    for w in (0.001, 0.01, 0.1, 0.5, 1, 1.999, 2, 2.667, 10, 60):
        s, h = pool(w, 0)
        assert h > 0 and s > 0, f"non-positive budget at {w}s"
        assert h <= max(w / 2, FLOOR) + 1e-12, f"wall over half the clock at {w}s"
    # teeth: the old cap really did go negative in that range
    assert min(w / 2 - 1 for w in (0.001, 0.5, 1.999)) < 0


def test_the_exhausted_pool_falls_to_a_minimal_positive_think():
    """P = 0 is reached at T = (M+2)*O = 8.4s of sudden death, and it is not a
    bug: 8.4s does not contain 40 more moves' worth of ANYTHING at 200ms of
    overhead a move, so the right answer is to move instantly. What must not
    happen is a zero or negative budget -- the floor keeps it positive."""
    assert pool(8.4, 0)[0] == pytest.approx(FLOOR)
    for w in (0.0, 0.001, 0.05, 0.5, 2, 8.4):
        s, h = pool(w, 0)
        assert s == h == FLOOR, f"pool-exhausted budget at {w}s is ({s}, {h})"
    # The crossover is where the share itself clears the floor: P/M > 0.05s,
    # i.e. T > (M+2)*O + M*FLOOR = 10.4s at 60+0. Named, not approximated.
    assert pool(10.4 - 0.01, 0)[0] == pytest.approx(FLOOR)
    assert pool(10.4 + 0.01, 0)[0] > FLOOR


def test_monotone_in_the_clock_and_in_the_increment():
    """More clock, or more increment, may never buy LESS thinking."""
    for i in (0, 0.1, 1, 3):
        prev = -1.0
        for k in range(0, 4001):
            cur = pool(k * 0.5, i)[0]
            assert cur >= prev - 1e-12, f"soft fell at wtime={k * 0.5} winc={i}"
            prev = cur
    for w in (10, 60, 300):
        prev = -1.0
        for k in range(0, 501):
            cur = pool(w, k * 0.01)[0]
            assert cur >= prev - 1e-12, f"soft fell at winc={k * 0.01}"
            prev = cur


def test_continuous_in_the_increment():
    """No step anywhere -- the defect that made PR #188 necessary, asserted
    for the pool as well. Bound is scale-free: 2% of the sudden-death soft
    limit at the same clock."""
    for w in (1, 30, 60, 300, 1800):
        lim, prev, worst = 0.02 * pool(w, 0)[0] + 1e-9, pool(w, 0.0)[0], 0.0
        for k in range(1, 20_001):
            cur = pool(w, 2.0 * k / 20_000)[0]
            worst = max(worst, abs(cur - prev))
            prev = cur
        assert worst < lim, f"jump of {worst:.5f}s in winc at {w}s (limit {lim:.5f})"


# ---- (2) movestogo --------------------------------------------------------

def test_movestogo_is_clamped_into_1_50():
    """GUIs send 0 (meaning "no control"), 1, and absurdities. A 200-move
    horizon would pace us at a fortieth of nothing; a 0 must not divide."""
    assert pool(60, 0, movestogo=0) == pool(60, 0)          # falsy: no horizon
    assert pool(60, 0, movestogo=200) == pool(60, 0, movestogo=50)
    assert pool(60, 0, movestogo=-3) == pool(60, 0, movestogo=1)


def test_movestogo_one_spends_the_clock_down():
    """UNTESTED IN GAMES (no staged-TC match has run): with one move to the
    control, holding back A/4 would be leaving the game unplayed, so the
    known-horizon variant spends up to 85% of what is reachable."""
    s, h = pool(60, 0, movestogo=1)
    assert 0.8 * 60 < s <= h <= 0.85 * (60 - 2 * O) + 1e-12
    # ...and it is still a horizon, not a licence: 10 moves to go is a tenth-ish
    s10, h10 = pool(60, 0, movestogo=10)
    assert 4.0 < s10 < 6.0
    # THE FOOT-GUN, pinned rather than left to be discovered: 5x a horizon
    # share is a far bigger bite than 5x a fortieth, and the 0.85*A clamp does
    # not bind here. Measured through the real driver, this wall lets one move
    # take 18.9 s of a 60 s clock. Nothing we measure with sends movestogo, and
    # nothing on the ladder tests it.
    assert h10 == pytest.approx(5 * s10) and h10 > 24


# ---- (3) the phase-M arm --------------------------------------------------

def test_phase_m_rises_through_the_middlegame_and_then_holds():
    """M = max(20, 46 - ply/2) on the driver's ply counter: slightly tighter
    than M=40 at move 1, looser from the middlegame on, flat after ply 52."""
    early, mid, late = (pool(60, 1, ply=p, phase_m=True)[0] for p in (0, 30, 60))
    assert early < pool(60, 1)[0] < mid < late
    assert pool(60, 1, ply=52, phase_m=True) == pool(60, 1, ply=200, phase_m=True)
    for ply in range(0, 200):
        s, h = pool(60, 1, ply=ply, phase_m=True)
        assert 0 < s <= h


# ---- (4) the seconds/ms trap, closed numerically --------------------------

def test_the_packed_mirror_is_this_manager_scaled():
    """t_ms(W, I) == 1000 * t_s(W/1000, I/1000) at every grid point, for BOTH
    limits. The two engines must budget the same time; the only thing between
    their sources is a factor of 1000 in three constants."""
    worst = 0.0
    for w in CLOCKS:
        for i in INCS:
            got_s, got_h = pool_ms(w * 1000, i * 1000)
            # the packed mod leaves the floor to the driver's max(think, .05),
            # so compare the unfloored formula against the same grid
            want_s = max(0.0, min((w + 39 * i - 42 * O) / 40, max(0.0, w - 2 * O) / 4))
            want_h = min(5 * want_s, max(0.0, w - 2 * O) / 2)
            for got, want in ((got_s / 1000, want_s), (got_h / 1000, want_h)):
                worst = max(worst, abs(got - want) / max(abs(want), 1e-9))
    assert worst < 1e-12, f"seconds and ms forms disagree by {worst:.3e} relative"


def test_the_packed_mirror_matches_the_shipped_function_where_no_clamp_binds():
    for w in (30, 60, 180, 300, 1800):
        for i in (0, 0.1, 1, 3):
            s_ms, h_ms = pool_ms(w * 1000, i * 1000)
            s, h = pool(w, i)
            assert s_ms / 1000 > FLOOR, "grid point is in the floored regime"
            assert s_ms / 1000 == pytest.approx(s, rel=1e-12)
            assert h_ms / 1000 == pytest.approx(h, rel=1e-12)


# ---- (5) the walks --------------------------------------------------------

def walk(base_s, inc_s, moves, spend, overhead=0.2):
    """Simulate OUR clock over `moves` of our own moves.

    `overhead` is per-move lag the budget cannot see -- network plus process
    turnaround; lichess games show ~200ms, which is also the O the manager
    budgets for. Returns a dict; flag=None means we never ran out.
    """
    clock, floored, first_floor, lowest = base_s, 0, None, base_s
    for mv in range(1, moves + 1):
        lowest = min(lowest, clock)
        think = max(spend(clock, inc_s), FLOOR)
        if think <= FLOOR + 1e-12:
            floored += 1
            first_floor = first_floor or mv
        clock -= think + overhead
        clock += inc_s
        if clock <= 0:
            return dict(flag=mv, left=-1.0, floored=floored, first_floor=first_floor,
                        lowest=lowest)
    return dict(flag=None, left=clock, floored=floored, first_floor=first_floor,
                lowest=lowest)


def pool_soft(w, i):
    return pool(w, i)[0]


def realized(factor):
    """The spend a soft limit ACTUALLY produces. Iterations are discrete: the
    search stops at the first one that ENDS past the soft limit, so the spend
    lands somewhere in [soft, soft x growth], never exactly on soft. Measured
    on the packed artifact through tm_smoke, cold table, from the start
    position: 2.26s against a 1.29s limit at 60+0 (1.75x), 1.74s against 1.39s
    at 60+0.1 (1.25x), 3.10s against 2.27s at 60+1 (1.37x).

    Nothing in the design hides this -- s (SOFT_SCALE) is the knob that
    recalibrates it -- but a walk that assumes the ideal 1.0x is a walk that
    flatters the manager, so the tail is walked at the measured factor too.
    """
    def spend(w, i):
        s, h = pool(w, i)
        return min(factor * s, h)
    return spend


def test_the_lost_game_replayed_at_3_plus_0():
    """lichess.org/EAThUL0P: 73 of our moves at 3+0, 200ms/move of lag.

    The incumbent /12 flags -- the walk has teeth. The pool finishes the same
    game with a healthy clock, never plays a floored move, and never lets the
    clock into the sub-2*O regime where the manager has nothing left to give.
    """
    lost = walk(180, 0, 73, incumbent)
    assert lost["flag"], "the /12 policy no longer flags: this walk is stale"

    got = walk(180, 0, 73, pool_soft)
    assert got["flag"] is None, "the pool flagged the lost game"
    assert got["left"] > 20, f"only {got['left']:.1f}s left after 73 moves"
    assert got["floored"] == 0, f"{got['floored']} blind moves in the lost game"
    assert got["lowest"] > 2 * O, "the clock entered the sub-2*O regime while searching"
    # and it is not merely surviving: it ends with more clock than the smooth
    # curve does, having thought longer on the moves that mattered
    assert got["left"] > walk(180, 0, 73, smooth)["left"]


@pytest.mark.parametrize("factor,flags_by", [(1.0, None), (1.5, 95), (1.75, 89)])
def test_the_60_plus_0_drain_walk(factor, flags_by):
    """100 of our moves at 60+0 under a pessimistic 200ms of lag -- longer than
    almost any real game, which is the point: this is the tail where the
    incumbents die. The incumbent /12 flags at move 39 and the smooth curve at
    84, both after a stretch of blind play.

    THE HONEST 60+0 PICTURE, and the reason that arm of the ladder is a
    non-regression check rather than a win claim: at the measured 1.75x
    realized spend the pool flags at move 89 -- five moves LATER than the
    smooth curve's own 84, and 50 later than /12, but this is not the rout the
    ideal-1.0x walk (never flags, 3.0s left) would suggest. Sudden death is
    where the pool and the incumbent curve are closest by construction (1.29s
    vs 1.50s of budget), so it is where the design has least to give.
    """
    old = walk(60, 0, 100, incumbent)
    sm = walk(60, 0, 100, smooth)
    assert old["flag"] == 39 and sm["flag"] == 84, "the incumbent walks are stale"

    got = walk(60, 0, 100, realized(factor))
    if flags_by is None:
        assert got["flag"] is None, f"the ideal pool arm flagged at move {got['flag']}"
        assert got["lowest"] > 2 * O, "the clock entered the sub-2*O regime while searching"
    else:
        assert got["flag"] == flags_by, f"flagged at {got['flag']}, expected {flags_by}"
        assert got["flag"] > sm["flag"], "the pool drains faster than the smooth curve"


@pytest.mark.parametrize("factor", (1.0, 1.5, 1.75))
def test_the_increment_tail_survives_at_every_realized_spend(factor):
    """60+1 and 3+0, walked at the same measured overshoot factors. This is
    where the pool has room: the increment is income the divisor never priced,
    so even a 1.75x realized spend finishes 100 moves of 60+1 and the lost
    game's 73 moves of 3+0 without a floored move."""
    got = walk(60, 1, 100, realized(factor))
    assert got["flag"] is None and got["floored"] == 0, got
    got = walk(180, 0, 73, realized(factor))
    assert got["flag"] is None and got["floored"] == 0, got
    assert got["lowest"] > 2 * O


def test_no_flooring_at_a_realistic_game_length_and_local_overhead():
    """60 of our moves, 20ms of turnaround (a local match, not lichess): the
    regime the bench-box matches actually play. Nothing floors, and the pool
    ends with more clock than either incumbent."""
    got = walk(60, 0, 60, pool_soft, overhead=0.02)
    assert got["floored"] == 0 and got["flag"] is None
    assert got["left"] > walk(60, 0, 60, smooth, overhead=0.02)["left"]
    assert got["left"] > walk(60, 0, 60, incumbent, overhead=0.02)["left"]


def test_tiny_increment_is_paced_like_the_sudden_death_it_is():
    """60+0.1 -- 0.1s/move of income against ~0.9s/move of spend. Both
    incumbents drain it; the pool does not."""
    assert walk(60, 0.1, 100, incumbent)["flag"]
    assert walk(60, 0.1, 100, smooth)["flag"]
    got = walk(60, 0.1, 100, pool_soft)
    assert got["flag"] is None and got["lowest"] > 2 * O


def test_the_increment_tc_tradeoff_is_what_the_matches_are_buying():
    """THE HONEST NUMBERS, pinned so nobody has to take the PR's word for it.

    BUDGETS (this test): at 60+1 the pool's soft limit is 2.4x under the smooth
    curve's whole budget, at 30+1 2.1x, at 60+0 only 1.16x.

    REALIZED SPEND is not the same thing and is measured, not derived --
    tm_smoke on the two packed artifacts, cold table, start position:

        TC        smooth   pool    ratio
        60+0      1.50 s   2.26 s  1.5x MORE
        60+0.1    2.46 s   1.74 s  1.4x less
        60+1      5.40 s   3.10 s  1.7x less

    because the pool stops at the first iteration that ENDS past its soft
    limit while the incumbent stops at the first probe past 0.8 of its single
    budget. So the tradeoff this design is buying is real but smaller than the
    budgets suggest, and at sudden death it runs the other way. Which of those
    is worth Elo is a games question (ladder arms a and b), not argued here.
    """
    assert smooth(60, 1) / pool_soft(60, 1) == pytest.approx(2.38, abs=0.05)
    assert smooth(30, 1) / pool_soft(30, 1) == pytest.approx(2.08, abs=0.05)
    # ...while sudden death barely moves, which is why 60+0 is the
    # NON-REGRESSION arm of the ladder and not the interesting one
    assert smooth(60, 0) / pool_soft(60, 0) == pytest.approx(1.16, abs=0.05)


# ---- (6) the dynamic target (v1.1, TM_DYNAMIC=1) --------------------------

def test_dynamic_target_is_the_static_one_when_nothing_is_moving():
    """A settled search: stable for two iterations, same move, no drop."""
    assert uci.dynamic_target(2.0, stable_iters=0) == pytest.approx(2.0 * 1.15)
    assert uci.dynamic_target(2.0, stable_iters=2) == pytest.approx(2.0 * 0.99)


def test_dynamic_target_clamps_at_both_ends():
    lo = uci.dynamic_target(2.0, stable_iters=99)
    assert lo == pytest.approx(2.0 * 0.65), "stability factor is not floored"
    hi = uci.dynamic_target(2.0, stable_iters=0, changed=True, score_drop=10_000)
    assert hi == pytest.approx(2.0 * 1.15 * 1.35 * 1.75), "extensions are not capped"
    for st in range(0, 40):
        for ch in (False, True):
            for dr in (0, 50, 200, 5000):
                f = uci.dynamic_target(1.0, st, ch, dr)
                assert 0.65 <= f <= 1.15 * 1.35 * 1.75 + 1e-12


def test_a_score_drop_extends_and_a_gain_does_not():
    """1 + drop/200, so 100cp is +50% and the 1.75 cap binds from 150cp on."""
    assert uci.dynamic_target(1.0, 1, False, 100) == pytest.approx(1.07 * 1.5)
    assert uci.dynamic_target(1.0, 1, False, 150) == pytest.approx(1.07 * 1.75)
    assert uci.dynamic_target(1.0, 1, False, 400) == pytest.approx(1.07 * 1.75), "cap"
    assert uci.dynamic_target(1.0, 1, False, 0) == pytest.approx(1.07)
    assert uci.dynamic_target(1.0, 1, False, -400) == pytest.approx(1.07), "a gain extended"


def test_mate_scores_bypass_the_drop_term():
    """A mate score is not an evaluation; differencing it against centipawns
    produces a meaningless 30,000cp "drop" that would extend every move of
    every won endgame to the wall."""
    assert uci.dynamic_target(1.0, 1, False, 29_000, mate=True) == pytest.approx(1.07)
    assert uci.dynamic_target(1.0, 1, False, 29_000, mate=False) == pytest.approx(1.07 * 1.75)


# ---- (7) the driver, end to end -------------------------------------------

class TapeSearcher:
    """Searcher stand-in replaying (depth, gamma, score, move) with a FAKE
    clock: each yield advances it by `step` seconds, and `used` counts what the
    driver actually consumed. Deterministic -- a timing test that sleeps is a
    timing test that flakes."""

    def __init__(self, tape, clock, step=0.04):
        self.tape, self.clock, self.step = tape, clock, step
        self.nodes, self.tp_move, self.tp_score, self.used = 0, {}, {}, 0
        self.deadline = float("inf")

    def search(self, hist):
        for entry in self.tape:
            self.clock[0] += self.step
            self.used += 1
            yield entry


class Clock:
    def __init__(self, cell):
        self.cell = cell

    def __call__(self):
        return self.cell[0]


def mv(s):
    return sunfish.Move(sunfish.parse(s[:2]), sunfish.parse(s[2:4]), "")


# FOUR ITERATIONS, two probes each, exactly as MTD-bi runs them: a fail-high
# that endorses the move, then a fail-low that closes the bracket to inside
# EVAL_ROUGHNESS (15). Convergence therefore lands on the EVEN yields, at
# t = 0.08, 0.16, 0.24, 0.32 with 0.04s steps -- and the depth TRANSITIONS
# land one probe later, at 0.12, 0.20, 0.28, which is the gap this design is
# about.
TAPE = [
    (1, 0, 50, mv("e2e4")), (1, 60, 55, mv("e2e4")),      # depth 1 -> e2e4
    (2, 50, 70, mv("d2d4")), (2, 80, 75, mv("d2d4")),     # depth 2 -> d2d4
    (3, 70, 90, mv("g1f3")), (3, 100, 95, mv("g1f3")),    # depth 3 -> g1f3
    (4, 90, 110, mv("b1c3")), (4, 120, 115, mv("b1c3")),  # depth 4 -> b1c3
]


def run_tape(monkeypatch, capsys, tape, step=0.04, **kw):
    cell = [0.0]
    monkeypatch.setattr(uci, "sunfish", sunfish, raising=False)   # what run() injects
    monkeypatch.setattr(uci.time, "perf_counter", Clock(cell))
    hist = [sunfish.Position(sunfish.initial, 0, (True, True), (True, True), 0, 0)]
    searcher = TapeSearcher(tape, cell, step)
    uci.go_loop(searcher, hist, threading.Event(), max_depth=100, **kw)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("bestmove")]
    assert len(lines) == 1
    return lines[0].split()[1], searcher.used


def test_the_soft_limit_stops_at_an_iteration_end_not_a_probe_later(monkeypatch, capsys):
    """Depth 2 converges at t=0.16 against a 0.12s soft limit, so the search
    stops THERE -- four probes -- and plays depth 2's answer.

    Stopping at the depth transition instead would consume a fifth probe, a
    full first probe of depth 3. That is not a detail: measured on the packed
    twin, the transition rule spends 2.64s against a 1.29s soft limit at 60+0
    and 6.82s against 2.27s at 60+1."""
    got, used = run_tape(monkeypatch, capsys, TAPE, max_movetime=10.0, soft_movetime=0.12)
    assert (got, used) == ("d2d4", 4), f"played {got} after {used} probes"


def test_the_soft_limit_never_stops_inside_an_iteration(monkeypatch, capsys):
    """A 0.10s limit is already passed at t=0.12, the FIRST probe of depth 2 --
    and an iteration that has started is allowed to finish. The search stops at
    depth 2's convergence, one probe later, with depth 2's answer."""
    got, used = run_tape(monkeypatch, capsys, TAPE, max_movetime=10.0, soft_movetime=0.10)
    assert (got, used) == ("d2d4", 4), f"played {got} after {used} probes"


def test_the_incumbent_rule_is_untouched_without_a_soft_limit(monkeypatch, capsys):
    """Same tape, no soft limit: the incumbent breaks at ANY yield past 2/3 of
    the single budget, including one inside an iteration -- and then plays the
    last COMPLETED depth's move, which is depth 1's. Exactly as on master."""
    got, used = run_tape(monkeypatch, capsys, TAPE, max_movetime=0.15)
    assert (got, used) == ("e2e4", 3), f"played {got} after {used} probes"


def test_the_hard_limit_is_still_the_deadline_not_the_loop(monkeypatch, capsys):
    """Nothing in the loop stops at max_movetime under the pool manager: the
    wall is armed as searcher.deadline by run() and enforced inside bound().
    With a soft limit past the tape's end the whole tape is consumed."""
    got, used = run_tape(monkeypatch, capsys, TAPE, max_movetime=0.05, soft_movetime=10.0)
    assert (got, used) == ("b1c3", 8), f"played {got} after {used} probes"


def test_the_dynamic_target_searches_longer_on_an_unstable_root(monkeypatch, capsys):
    """Same tape, same 0.12s soft limit, TM_DYNAMIC on. The root changes its
    mind at every iteration, so the target is 1.15 x 1.35 = 1.55 soft limits
    (0.186s) and depth 3 -- which converges at 0.16 -- is searched too."""
    static = run_tape(monkeypatch, capsys, TAPE, max_movetime=10.0, soft_movetime=0.12)
    monkeypatch.setattr(uci, "TM_DYNAMIC", True)
    dynamic = run_tape(monkeypatch, capsys, TAPE, max_movetime=10.0, soft_movetime=0.12)
    assert static == ("d2d4", 4) and dynamic == ("g1f3", 6), f"{static} vs {dynamic}"


def test_a_terminal_root_still_answers_under_the_pool(monkeypatch, capsys):
    """bound()'s contract: a root fail-high without a move is a verified
    terminal. The soft limit must not change what happens there."""
    got, _ = run_tape(monkeypatch, capsys, [(1, 0, 0, None)], max_movetime=10.0,
                      soft_movetime=0.01)
    assert got == "(none)"


def test_the_pool_manager_requires_the_engines_roughness(monkeypatch):
    """The mirror needs the engine's own bisection bound, so an engine without
    it must be named at startup rather than discovered inside a search -- and
    only under the pool manager, since nothing else reads it."""
    assert uci.POOL_ENGINE_API == ("EVAL_ROUGHNESS",)
    assert all(hasattr(sunfish, a) for a in uci.POOL_ENGINE_API)

    class Blind:                       # every required attribute but that one
        __name__ = "blind_engine"
    for attr in uci.ENGINE_API + ("pst",):
        setattr(Blind, attr, 1)
    uci.check_engine_module(Blind)                      # smooth: accepted
    monkeypatch.setattr(uci, "TM_MANAGER", "pool")
    with pytest.raises(TypeError, match="EVAL_ROUGHNESS"):
        uci.check_engine_module(Blind)
