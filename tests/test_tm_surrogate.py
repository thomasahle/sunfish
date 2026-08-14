"""The TM surrogate's instrument checks (tools/ctwin/: tmlib, tmsim, vmatch).

These run without the C twin, without pypy and without a clock: everything
here is arithmetic, which is the point of the surrogate.  The twin-driven
half (vmatch) is exercised only for its replay semantics, on synthetic probe
traces, so the suite stays a unit test rather than a match.

The three CALIBRATION SIGNATURES are asserted here as well, because they are
the reason to believe the surrogate at all and a regression in any of them
invalidates every ranking it produces:

  * oldtm's negative-cap threshold is 2.4 s
  * the step budget PARKS the clock at a fixed point (2.1 s at 60+0.1 with a
    50 ms per-move charge -- the equilibrium the 438-game match measured)
  * the pool's floor knee is (M+2)*O -- 8.4 s at the shipped O = 200 ms,
    the "minimum end-clock is 8.4 s exactly" of the arm-(a) telemetry
"""
import os
import sys

import pytest

CTWIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools", "ctwin")
sys.path.insert(0, CTWIN)

import tmlib                                       # noqa: E402
import tmsim                                       # noqa: E402


# ------------------------------------------------------- the mirrors are real
def test_every_mirror_matches_its_pinned_source_literal():
    """2,000+ grid values, and a drift tripwire on the source text itself."""
    assert tmlib.verify(verbose=False) > 2000


def test_the_managers_disagree_where_the_matches_said_they_do():
    # winc == 0: the step IS wtime/40 and oldtm is wtime/12 -- a 3.3x gap,
    # which is the whole of stage 1.
    assert tmlib.steptm(60, 0).hard == pytest.approx(1.5)
    assert tmlib.oldtm(60, 0).hard == pytest.approx(5.0)
    # winc == 0: smooth is bit-for-bit the step (the fix carries untouched).
    for t in (5, 10, 30, 60, 300):
        assert tmlib.smooth(t, 0).hard == pytest.approx(tmlib.steptm(t, 0).hard)
    # a 0.1s increment: the step jumps to /12, smooth slides to /21.3.
    assert tmlib.steptm(60, 0.1).hard == pytest.approx(tmlib.oldtm(60, 0.1).hard)
    assert tmlib.smooth(60, 0.1).hard < tmlib.steptm(60, 0.1).hard


def test_the_divisor_slides_forty_to_twelve():
    """The smooth base's effective divisor, from the PR's own table."""
    for winc, want in ((0, 40), (0.05, 26.0), (0.1, 21.3), (0.2, 17.6),
                       (0.5, 14.5), (1.0, 13.3), (3.0, 12.5)):
        base = (40000 + 240 * 1000 * winc) / (1000 + 20 * 1000 * winc)
        assert base == pytest.approx(want, abs=0.05)


# --------------------------------------------------- calibration signature 1
def test_oldtms_negative_cap_threshold_is_2v4_seconds():
    """Below 2.4 s the cap, not the base, is what binds -- and the old cap
    heads for zero.  The stage-1 forensics put the crossing at a median move
    42 of a 60+0 game; this is the clock it happens at."""
    assert tmsim.knee("oldtm", 0.0, what="cap") == pytest.approx(2.4, abs=1e-3)
    assert tmlib.oldtm(2.4, 0).hard == pytest.approx(0.2)
    assert tmlib.oldtm(2.1, 0).hard == pytest.approx(tmlib.TM_FLOOR)


def test_the_old_cap_goes_negative_and_the_new_one_cannot():
    for t in (0.1, 0.5, 1.0, 1.9):
        assert 1000 * t / 2 - 1000 < 0                # the raw old cap
        assert tmlib.smooth(t, 0).hard > 0            # wtime^2/(2wtime+4000)
        assert tmlib.pool(t, 0).hard > 0


# --------------------------------------------------- calibration signature 2
def test_the_step_budget_parks_the_clock_at_a_fixed_point():
    """THE 2.1 s PARKING EQUILIBRIUM.  It is arithmetic: the cap wtime/2-1000
    buys exactly the increment at wtime = 2*(inc + 1) seconds, less whatever
    the environment charges per move.  438 games measured 2.1 s median and
    2.0 s minimum; the solver puts it at 2.1 s with a 50 ms charge and at the
    textbook 2.2 s with none."""
    assert tmsim.fixed_point("steptm", 0.1, 0.05) == pytest.approx(2.1, abs=0.02)
    assert tmsim.fixed_point("steptm", 0.1, 0.0) == pytest.approx(2.2, abs=0.02)
    # It is an ATTRACTOR: the clock rises to it from below and falls to it
    # from above, which is why every one of those games ended there.
    for start in (2.05, 2.6, 6.0, 20.0):
        rows = tmsim.walk("steptm", start, 0.1, 120, 0.05)
        assert rows[-1]["clock_after"] == pytest.approx(2.1, abs=0.05), start


def test_the_smooth_budget_has_no_such_park_and_keeps_spending():
    """The park is the step's pathology, not a property of increments: the
    smooth cap is positive everywhere, so its fixed point sits an order of
    magnitude lower and the arm is still searching where the step is not."""
    step = tmsim.fixed_point("steptm", 0.1, 0.05)
    smooth = tmsim.fixed_point("smooth", 0.1, 0.05)
    assert smooth < step / 3
    late_step = [r["spend"] for r in tmsim.walk("steptm", 60, 0.1, 63, 0.05)][-20:]
    late_smooth = [r["spend"] for r in tmsim.walk("smooth", 60, 0.1, 63, 0.05)][-20:]
    assert sorted(late_smooth)[10] > 3 * sorted(late_step)[10]


# --------------------------------------------------- calibration signature 3
@pytest.mark.parametrize("overhead", [0.1, 0.2, 0.3])
def test_the_pool_stops_spending_at_exactly_m_plus_two_times_o(overhead):
    """The arm-(a) telemetry's "minimum end-clock is 8.4 s = (M+2)*O exactly".
    It is the clock at which the pool P hits zero, so it is (M+2)*O for any O
    and the 8.4 s is not a coincidence of the shipped 200 ms."""
    knobs = {"overhead": overhead}
    assert tmlib.pool(42 * overhead, 0, **knobs).soft == pytest.approx(tmlib.TM_FLOOR)
    assert tmlib.pool(42 * overhead + 10.0, 0, **knobs).soft > tmlib.TM_FLOOR
    knee = tmsim.knee("pool", 0.0, knobs, what="cap")
    assert knee == pytest.approx(42 * overhead, rel=0.02)


def test_the_pool_prices_increment_as_income_and_overhead_as_tax():
    plain = tmlib.pool(60, 0)
    with_inc = tmlib.pool(60, 1)
    assert with_inc.soft > plain.soft                 # (M-1) moves earn it
    assert tmlib.pool(60, 0, overhead=0.3).soft < plain.soft   # (M+2) pay it


def test_phase_m_raises_the_middlegame_share_then_holds():
    shares = [tmlib.pool(60, 0, ply=p, phase_m=True).soft for p in range(0, 120, 8)]
    assert shares == sorted(shares)                   # rises...
    assert shares[-1] == pytest.approx(shares[-2])    # ...then holds at M=20


# ------------------------------------------ the min40_4 candidate's promises
def test_min40_4s_expression_commutes_exactly_with_unit_scaling():
    """The property that makes this candidate different in kind: every term
    is degree-1 homogeneous, so there is no seconds version and milliseconds
    version to get wrong.  Checked on the EXPRESSION, since the TM_FLOOR the
    driver applies afterwards is an absolute constant by design."""
    def raw(t, i):
        return min(t / 40 + 0.9 * i, t / 4)

    for t in (0.05, 1, 5, 20, 60, 300, 1800):
        for i in (0, 0.05, 0.1, 1, 3, 5):
            for k in (1e-6, 1e-3, 7, 1000, 1e6):
                assert raw(k * t, k * i) == pytest.approx(k * raw(t, i), rel=1e-12)
    # And the manager itself is homogeneous wherever the floor does not bind.
    for t in (5, 20, 60, 300):
        for i in (0, 0.1, 1, 3):
            for k in (2, 7, 1000):
                assert (tmlib.min40_4(k * t, k * i).hard
                        == pytest.approx(k * tmlib.min40_4(t, i).hard, rel=1e-12))


def test_no_other_manager_has_that_property():
    """The contrast is the point.  Every shipped budget carries an absolute
    constant, so scaling the clock does not scale the budget -- though only
    where that constant is what binds, which is exactly the regime each of
    these managers gets its pathology from."""
    probes = [(t, i, k) for t in (1, 2, 4, 60, 300) for i in (0, 0.1, 1, 3)
              for k in (2, 10)]
    for name in ("legacy12", "oldtm", "steptm", "smooth", "pool"):
        fn = tmlib.MANAGERS[name]
        broken = [(t, i, k) for t, i, k in probes
                  if abs(fn(k * t, k * i).hard - k * fn(t, i).hard) > 1e-9]
        assert broken, "%s looks unit-independent; it should not be" % name


def test_min40_4_is_the_smooth_budget_at_sudden_death():
    """wtime/4 never binds at winc == 0, so the arm IS wtime/40 -- which is
    the smooth budget's own sudden-death branch.  Stage 1's +235.5 carries,
    and 60+0 is not a regime this candidate needs ranked."""
    for t in (2.0, 3, 5, 10, 30, 60, 120, 300, 1800):
        assert tmlib.min40_4(t, 0).hard == pytest.approx(tmlib.smooth(t, 0).hard)
        assert tmlib.min40_4(t, 0).hard == pytest.approx(max(t / 40, tmlib.TM_FLOOR))


def test_min40_4_has_no_high_park_but_does_have_a_low_reserve_fixed_point():
    """Recorded honestly, because the claim it is registered under is "no
    park possible" and that is only true of the STEP's kind of park.

    At T = 4*I both branches equal I exactly, so the reserve is a real
    landmark -- but the per-move balance there is 0.1*I - O, so it holds
    only for I >= 10*O.  Below that the clock drifts down to the genuine
    fixed point at T = 4*(I - O), which is LOW: the arm spends its clock
    instead of sitting on it.  That is a different trade, not the absence
    of one, and the flag margin is where it gets paid for.
    """
    assert tmlib.min40_4(4 * 1.0, 1.0).hard == pytest.approx(1.0)
    assert tmlib.min40_4(4 * 0.1, 0.1).hard == pytest.approx(0.1)
    # No park at all with zero overhead and I >= 10*O: T = 4I is neutral.
    step_park = tmsim.fixed_point("steptm", 0.1, 0.05)
    m_park = tmsim.fixed_point("min40_4", 0.1, 0.05)
    assert m_park is None or m_park < step_park / 5
    # I >= 10*O: the reserve self-maintains, so the clock settles high.
    rows = tmsim.walk("min40_4", 60, 1.0, 150, 0.05)
    assert rows[-1]["clock_after"] > 3.0
    assert all(not r["flag"] for r in rows)


# ------------------------------------------------------------ the spend model
def test_a_budget_is_not_a_spend_and_the_pool_overshoots_its_soft_limit():
    """Iterations are discrete, so the pool stops at the first one that ENDS
    past its limit.  Its own pre-registration recorded 1.3-2.3x before a game
    was played; the ladder model has to produce the same class of number, or
    it is not modelling the thing that made the pool's realized spend a
    surprise."""
    b = tmlib.pool(60, 0)
    spend = tmsim.ladder_stop(b.soft, b.hard, b.rule)
    assert 1.0 < spend / b.soft < 3.0


def test_the_incumbents_land_near_their_wall_not_near_their_target():
    b = tmlib.smooth(60, 0)
    spend = tmsim.ladder_stop(b.frac * b.hard, b.hard, b.rule)
    assert b.frac * b.hard < spend <= b.hard


def test_the_signatures_do_not_depend_on_the_spend_model():
    """The reason the three calibration signatures are worth anything: they
    are BUDGET arithmetic, so no modelled quantity can move them.  The
    realized spend, which is modelled, does move -- and this test asserts
    both halves, because a sweep that changes nothing at all is a sweep that
    is not connected (`ladder_stop` froze its parameters as default arguments
    once, and the robustness check it was silently no-op-ing was reported as
    a pass before that was caught)."""
    keep = (tmsim.LADDER_T1, tmsim.LADDER_B)
    spends = []
    try:
        for t1, b in ((2000 / 23000, 2.5), (1000 / 23000, 2.5),
                      (4000 / 23000, 2.5), (2000 / 23000, 2.0),
                      (2000 / 23000, 3.5)):
            tmsim.LADDER_T1, tmsim.LADDER_B = t1, b
            assert tmsim.knee("oldtm", 0.0, what="cap") == pytest.approx(2.4, abs=1e-3)
            assert tmsim.fixed_point("steptm", 0.1, 0.05) == pytest.approx(2.1, abs=0.02)
            assert tmsim.knee("pool", 0.0, {"overhead": 0.2},
                              what="cap") == pytest.approx(8.4, abs=1e-3)
            rows = tmsim.walk("steptm", 60, 0, 54, 0.05)
            spends.append(tmsim.summarize(rows, 0)["median_spend"])
    finally:
        tmsim.LADDER_T1, tmsim.LADDER_B = keep
    assert len(set(round(s, 4) for s in spends)) > 1, "the ladder sweep is inert"


# ----------------------------------------------------------- replay semantics
def _trace(pairs):
    """(depth, kind, nodes) -> the twin's probe tuples, at a fixed score."""
    return [(d, 0 if k != "lower" else 40, k, "e2e4" if k == "lower" else None, n)
            for d, k, n in pairs]


def test_the_wall_aborts_inside_the_probe_that_would_cross_it():
    """The real deadline fires inside bound(), so the crossing probe never
    finishes: the spend is exactly `hard`, not the probe's end time.  The
    twin's own node cap cannot express this -- it is a yield-boundary rule --
    which is why the surrogate replays rather than trusting the twin's stop."""
    import vmatch
    probes = _trace([(1, "lower", 1000), (2, "lower", 2000), (2, "upper", 9000)])
    b = tmlib.Budget(0.05, 0.25, "yield_frac", 0.8)
    mv, spend, info = vmatch.replay(probes, "a1a1", b, nps=10000)
    assert spend == pytest.approx(0.25) and info["stop"] == "deadline"
    assert mv == "e2e4"                               # the committed move


def test_a_budget_too_small_for_one_probe_plays_the_structural_floor():
    import vmatch
    probes = _trace([(1, "lower", 5000)])
    b = tmlib.Budget(0.05, 0.05, "yield_frac", 0.8)
    mv, spend, info = vmatch.replay(probes, "h2h3", b, nps=10000)
    assert info["fallback"] and info["blind"] and mv == "h2h3"
    assert spend == pytest.approx(0.05)


def test_the_soft_limit_stops_at_a_converged_iteration_not_a_probe_later():
    """The pool's load-bearing detail.  Reading the soft limit at "a new depth
    appeared" arrives one FULL PROBE of the next depth late -- measured 2.64 s
    against a 1.29 s limit.  The bracket mirror stops at the convergence."""
    import vmatch
    # depth 2 converges at 12,000 nodes (1.2 s); the next probe would take us
    # to 2.5 s.  A converged stop spends 1.2 s, a depth-transition stop 2.5 s.
    probes = [(1, 40, "lower", "e2e4", 2000), (1, 10, "upper", None, 4000),
              (2, 40, "lower", "d2d4", 9000), (2, 30, "upper", None, 12000),
              (3, 40, "lower", "d2d4", 25000)]
    b = tmlib.Budget(1.0, 5.0, "mtd_converged", None)
    mv, spend, info = vmatch.replay(probes, "a1a1", b, nps=10000)
    assert info["stop"] == "soft" and spend == pytest.approx(1.2)
    assert mv == "d2d4"


def test_the_soft_limit_never_stops_inside_an_iteration():
    import vmatch
    probes = [(1, 40, "lower", "e2e4", 2000), (1, 10, "upper", None, 4000),
              (2, 400, "lower", "d2d4", 30000), (2, 30, "upper", None, 40000)]
    b = tmlib.Budget(0.5, 100.0, "mtd_converged", None)
    mv, spend, info = vmatch.replay(probes, "a1a1", b, nps=10000)
    # Depth 1 converged at 0.4s, under the 0.5s target, so depth 2 was
    # STARTED and must be allowed to finish -- 4.0s, far past the target.
    assert spend == pytest.approx(4.0) and mv == "d2d4"


def test_a_terminal_root_is_reported_as_complete_not_truncated():
    import vmatch
    probes = [(1, 40, "lower", "e2e4", 1000), (2, -29000, "exact", None, 3000)]
    b = tmlib.Budget(1.0, 5.0, "yield_frac", 0.8)
    mv, spend, info = vmatch.replay(probes, "a1a1", b, nps=10000)
    assert info["stop"] == "terminal" and mv == "e2e4"


# ------------------------------------------------------------------ nps model
def test_the_nps_model_is_monotone_and_refuses_to_extrapolate():
    import json

    import npsprofile
    model = json.load(open(os.path.join(CTWIN, "npsmodel.json")))
    lo, hi = model["clamp_pieces"]
    vals = [npsprofile.nps_for(model, p) for p in range(hi, lo - 1, -1)]
    assert vals == sorted(vals)                       # nps rises into endgames
    assert npsprofile.nps_for(model, 2) == npsprofile.nps_for(model, lo)
    assert npsprofile.nps_for(model, 64) == npsprofile.nps_for(model, hi)
    assert all(v > 0 for v in vals)
    # The residual is the honest width of the node budget, and it is large:
    # piece count is not the only thing that sets a node rate.
    assert 0.10 < model["residual_rms_frac"] < 0.30


def test_the_dynamic_target_extends_only_where_uci_py_says_it_does():
    assert tmlib.dynamic_target(1.0, stable_iters=0) == pytest.approx(1.15)
    assert tmlib.dynamic_target(1.0, stable_iters=99) == pytest.approx(0.65)
    assert tmlib.dynamic_target(1.0, 0, changed=True) == pytest.approx(1.15 * 1.35)
    assert tmlib.dynamic_target(1.0, 0, score_drop=100) == pytest.approx(1.15 * 1.5)
    assert tmlib.dynamic_target(1.0, 0, score_drop=1e6) == pytest.approx(1.15 * 1.75)
    # A mate score is not an evaluation; differencing it is meaningless.
    assert (tmlib.dynamic_target(1.0, 0, score_drop=1e6, mate=True)
            == pytest.approx(1.15))


def test_the_surrogate_never_reads_a_wall_clock():
    """The whole point.  A surrogate that timed anything would have imported
    the variance it exists to remove."""
    for name in ("tmlib.py", "tmsim.py", "vmatch.py"):
        body = open(os.path.join(CTWIN, name)).read()
        code = "\n".join(ln for ln in body.splitlines()
                         if not ln.lstrip().startswith("#"))
        assert "time.time()" not in code and "perf_counter" not in code, name
        assert "import time" not in code, name
