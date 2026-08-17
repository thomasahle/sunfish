#!/usr/bin/env python3
"""Time-management formula plugins: mirrors of every shipped TM expression.

The virtual-clock surrogate needs the budget arithmetic as a PURE FUNCTION so
it can be evaluated without an engine, a clock or a game.  Every manager here
is a transcription of a line that a real engine actually ran, and every
transcription is checked against that line rather than trusted:

  * `verify()` re-evaluates each PINNED SOURCE LITERAL -- the exact text of
    the shipped expression, quoted below with its repo/branch/file/line -- on
    a grid of clocks and increments and asserts the mirror agrees to 1e-12.
  * When the source file is present in this checkout, `verify()` also greps
    for the pinned text and FAILS if it has drifted.  When it is absent (the
    packed arms live on another branch), it says so, per position, instead of
    quietly skipping.

UNITS.  The interface is SECONDS in, SECONDS out.  The packed artifact runs
its arithmetic in MILLISECONDS and divides once at the end; that is mirrored
literally (multiply by 1000, run the ms expression, divide) rather than
algebraically simplified, so the mirror is bit-identical to the artifact and
not merely equal in exact arithmetic.  The seconds/ms confusion has cost this
project two incidents.

WHAT A BUDGET IS.  Two numbers and a stop rule, because a budget that does
not say when it is read is not a budget:

  soft   when to stop STARTING work
  hard   the wall, armed as the in-search deadline
  rule   how `soft` is read by the driver's iteration loop:
         "yield_frac"     break at ANY yield past frac*hard  (both incumbents;
                          frac = 0.8 packed, 2/3 classic)
         "mtd_converged"  break at a yield where the MTD bracket has CLOSED
                          and elapsed > soft  (the pool manager)
"""
import os
import sys
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

Budget = namedtuple("Budget", "soft hard rule frac")

# Smallest positive budget, SECONDS.  The packed entry's max(think, .05) and
# uci.py's TM_FLOOR are the same number for the same reason: a degenerate
# clock must still produce a legal move rather than a zero-length search.
TM_FLOOR = 0.05
# The engine's own convergence window; the pool's soft rule reads it.
# sunfish.py:141 and nnue_4k/pst_entry.py:102 both say 15.
EVAL_ROUGHNESS = 15


# ---------------------------------------------------------------------------
# PINNED SOURCE LITERALS.  Each is the exact expression the named engine ran,
# in the units it ran it in.  verify() evals these; the mirrors below must
# agree with them everywhere on the grid.
# ---------------------------------------------------------------------------
# Managers that no shipped engine carries yet.  A missing pin is news, not a
# failure, for exactly these -- and the moment one lands, the same grep starts
# reporting where, with no edit here.
CANDIDATES = {"onemax"}

PINNED = {
    # THE CLASSIC BUILTIN CLOCK, MILLISECONDS -- sunfish.py's `go` handler, the
    # packed classic artifact's whole time manager (a checkout reaches the
    # driver instead).  It became the pool on 2026-08-17; before that it ran
    # min40_4 (#196), whose pin is retired just below.
    #
    # Only the SOFT line is pinned, and on purpose: it carries every constant
    # (39*winc, 42*200, /40, the 400 and the /4), so drift in any of them shows
    # up here.  The wall and the soft clip that follow it are asserted in
    # tests/test_classic_time_budget.py, which lifts all three statements and
    # grid-checks the pair against uci.pool_budget -- the artifact and the
    # driver being ONE arithmetic is the thing that makes the duplication safe,
    # and it is checked numerically rather than by pinning text twice.
    "pool_classic": (
        "sunfish.py",
        "soft = min(max(0, wtime + 39 * winc - 42 * 200) / 40, max(0, wtime - 400) / 4)"),
    # sunfish_ui/uci.py:467 (master f95f49c) -- classic's incumbent, SECONDS.
    # Also uci.py:712 on branch tm-pool-manager, unchanged there.
    "legacy12": (
        "sunfish_ui/uci.py", "min(wtime / 12 + 0.9 * winc, wtime / 2 - 1)"),
    # sunfish_ui/uci.py:452 -- the movestogo branch of the same manager.
    "legacy12_mtg": (
        "sunfish_ui/uci.py", "min(wtime / movestogo + winc, wtime / 2 - 1)"),
    # CANDIDATE, the classic lane's sibling to min40_4 competing for the same
    # line: branch classic/tm-one-max-pool, sunfish.py:601 at 3a48984.
    # MILLISECONDS, and unlike min40_4 that matters -- the 8000 and the 50
    # are absolute, so this literal is grid-checked in the ms domain only.
    "onemax": ("sunfish.py", "max((wtime - 8000) / 40 + winc, 50)"),
}
# ---------------------------------------------------------------------------
# RE-PINNED 2026-08-15 after the 4k entry's pooltm landing (nnue-4k 5f16bae;
# PR #201's handoff flagged exactly this breakage). oldtm/steptm/pool_ms used
# to anchor in tools/build/make_variants.py, smooth in nnue_4k/pst_entry.py.
# That commit retired oldtm/steptm/pooltm from make_variants.py (tombstoned,
# their shared anchor gone) and replaced the entry's smooth budget line with
# the pool. All four pins broke. None of the four belongs back in PINNED
# above, for two different reasons:
#
#   oldtm, steptm  DROPPED, not re-anchored -- RETIRED. The smooth line they
#                  shared as an anchor no longer exists anywhere, on any
#                  branch, so there is no live text left to pin. Their
#                  mirrors stay (stage 1's -235.5 +/- 65.4, and every test
#                  that calls them, still need them), just without a
#                  source-drift check: a retired formula cannot drift, it can
#                  only be misremembered, and the explicit numeric asserts in
#                  test_tm_surrogate.py are what guard against that instead.
#   smooth         DROPPED too, same reason: superseded by the pool, and
#                  nnue_4k/pst_entry.py no longer carries this expression on
#                  ANY branch. (It never carried it on master to begin with --
#                  master's nnue_4k predates the smooth budget entirely,
#                  which is the OTHER reason this pin drifted; see PR #201's
#                  comment.)
#   pool_ms        NOT retired -- LANDED, and still the live truth. Its text
#                  is unchanged: the same M/P/A/soft/think block that lived
#                  in make_variants.py's `pooltm` mod now lives, byte-for-byte,
#                  in tools/build/make_pst_entry.py's `_pooltm` list (checked
#                  by hand against nnue-4k 5f16bae; the two are identical).
#                  It is not re-anchored to that path here because the path
#                  does not resolve on THIS checkout: master's own
#                  tools/build/make_pst_entry.py is a different, older
#                  generator (builds pst_entry.py from nnue_4k's retired
#                  accumulator source, not from make_variants.py's mod
#                  system, and has no `_pooltm`), so a live-file check there
#                  would find a real file with the WRONG content and report a
#                  drift that is really a wrong-checkout lookup -- exactly
#                  what _pinned_present's own docstring warns about, and
#                  exactly what job 2's root fix exists to stop happening via
#                  a sibling. The grid-assert below still runs in full
#                  against this text (_POOL_MS_TEXT), so the MIRROR is proven
#                  correct against the landed formula; only the automatic
#                  "is the formula still there" tripwire is unavailable from
#                  a master-only checkout, until nnue_4k here is next synced
#                  past 5f16bae.
# ---------------------------------------------------------------------------
_POOL_MS_TEXT = (
    "M = 40\n"
    "P = max(0, wtime + (M - 1) * winc - (M + 2) * 200)\n"
    "A = max(0, wtime - 400)\n"
    "soft = min(P / M, A / 4)\n"
    "think = min(5 * soft, A / 2)\n")


# --------------------------------------------------------------- managers ---
def legacy12(wtime, winc, movestogo=None, ply=0, **kw):
    """Classic sunfish's incumbent.  sunfish_ui/uci.py:452/467, SECONDS.

    The 2/3 rule is uci.py:399 -- `elapsed > max_movetime * 2 / 3` at any
    yield with depth > 1 -- so this manager's soft limit is not a separate
    number, it is a fraction of the single one.  NOTE: classic never received
    the sudden-death fix; this is the /12 drain that stage 1 measured at
    -235.5 +/- 65.4 on the packed artifact.
    """
    if movestogo:
        think = min(wtime / movestogo + winc, wtime / 2 - 1)
    else:
        think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1)
    return Budget(2 * think / 3, think, "yield_frac", 2 / 3)


def oldtm(wtime, winc, movestogo=None, ply=0, **kw):
    """Packed pre-fix, MILLISECONDS internally.  make_variants.py "oldtm"."""
    wtime, winc = 1000 * wtime, 1000 * winc
    think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000) / 1000
    think = max(think, TM_FLOOR)
    return Budget(0.8 * think, think, "yield_frac", 0.8)


def steptm(wtime, winc, movestogo=None, ply=0, **kw):
    """The stage-1 winner (`tmfix`), MILLISECONDS.  make_variants.py "steptm"."""
    wtime, winc = 1000 * wtime, 1000 * winc
    think = min(wtime / (12 if winc else 40) + 0.9 * winc, wtime / 2 - 1000) / 1000
    think = max(think, TM_FLOOR)
    return Budget(0.8 * think, think, "yield_frac", 0.8)


def smooth(wtime, winc, movestogo=None, ply=0, **kw):
    """The shipped smooth budget, MILLISECONDS.  pst_entry.py:766-767."""
    wtime, winc = 1000 * wtime, 1000 * winc
    think = min(wtime * (1000 + 20 * winc) / (40000 + 240 * winc) + 0.9 * winc,
                wtime * wtime / (2 * wtime + 4000)) / 1000
    think = max(think, TM_FLOOR)
    return Budget(0.8 * think, think, "yield_frac", 0.8)


def min40_4(wtime, winc, movestogo=None, ply=0, **kw):
    """CANDIDATE (Thomas, 2026-08-14).  Not yet shipped anywhere.

        think = min(wtime/40 + 0.9*winc, wtime/4)

    UNIT-INDEPENDENT, alone among the managers here: every term is degree-1
    homogeneous, so the expression commutes exactly with unit scaling and
    there is no seconds/milliseconds version of it to get wrong.  Every other
    budget in this file carries an absolute constant (-1, -1000, +4000, the
    pool's O) and therefore only means what it means in its own unit -- the
    confusion that has cost this project two incidents cannot arise here.
    Asserted, not asserted-in-prose: tests/test_tm_surrogate.py.

    SUDDEN DEATH is exactly wtime/40 (wtime/4 never binds), which is the
    smooth budget's own winc == 0 branch above a 2 s clock -- so stage 1's
    +235.5 +/- 65.4 carries to it untouched and 60+0 is not a question it
    needs asked.  The open regimes are the increment ones.

    THE RESERVE.  The cap is a fraction of the clock rather than a clock
    minus a constant, so it cannot go negative and it cannot produce the
    step's high blind park.  What it does produce is a reserve equilibrium
    near T = 4*I: at T = 4*I both terms equal I exactly.  With a real
    per-move overhead O the balance at that clock is 0.1*I - O per move, so
    the reserve SELF-MAINTAINS only for I >= 10*O and otherwise drifts down
    -- to the genuine fixed point at T = 4*(I - O), where the cap branch
    buys exactly the increment back.  That is a LOW park, not the step's
    high one: the engine keeps its clock rather than sitting on 2.1 s of it
    unspent, at the price of a thinner flag margin.  Which of those two is
    worth more is a match question, and it is the one this arm is for.
    """
    think = min(wtime / 40 + 0.9 * winc, wtime / 4)
    think = max(think, TM_FLOOR)
    return Budget(0.8 * think, think, "yield_frac", 0.8)


def onemax(wtime, winc, movestogo=None, ply=0, **kw):
    """CANDIDATE (classic lane, branch classic/tm-one-max-pool at 3a48984,
    sunfish.py:601).  MILLISECONDS -- the 8000 and the 50 are ms.

        think = max((wtime - 8000) / 40 + winc, 50)

    The pool manager's arithmetic collapsed to one line: P/M at M = 40 and
    O = 200 ms is wtime/40 + 0.975*winc - 210 ms, and rounding the increment
    coefficient to 1 and the overhead reserve to a flat 8 s gives this.

    THE min BECAME A max, which is the whole idea.  There is no cap to park
    on, so the budget FLOORS instead of collapsing: it reaches the 50 ms
    floor at wtime = 10 - 40*winc seconds, i.e. still holding 10 s at sudden
    death -- 40 further moves at 0.05 + 0.2 each -- where the step form
    reached its floor holding 2.1 s, which is eight.

    NOT unit-independent (the 8000 and 50 are absolute), unlike min40_4.
    That is a real difference between the two classic candidates and the
    grid check below only passes in the millisecond domain.
    """
    wtime, winc = 1000 * wtime, 1000 * winc
    think = max((wtime - 8000) / 40 + winc, 50) / 1000
    return Budget(0.8 * think, think, "yield_frac", 0.8)


def pool(wtime, winc, movestogo=None, ply=0, overhead=0.2, moves=40,
         scale=1.0, phase_m=False, **kw):
    """The pool manager.  Transcribed from sunfish_ui/uci.py's `pool_budget`
    on branch tm-pool-manager (commit 7e8e1ff, lines 101-140), SECONDS.

        P = max(0, T + (M-1)*I - (M+2)*O)     the pool this game still has
        A = max(0, T - 2*O)                   what THIS move can reach
        soft = min(s*P/M, A/4)                stop STARTING iterations
        hard = min(5*soft, A/2)               the wall

    movestogo, phase-M and the scale knob are the branch's own knobs, kept
    here because the knob matrix sweeps them.
    """
    if movestogo:
        m = min(50, max(1, movestogo))
    elif phase_m:
        m = max(20, 46 - ply / 2)
    else:
        m = moves
    p = max(0.0, wtime + (m - 1) * winc - (m + 2) * overhead)
    avail = max(0.0, wtime - 2 * overhead)
    share = scale * p / m
    if movestogo:
        soft, wall = min(0.85 * share, 0.85 * avail), 0.85 * avail
    else:
        soft, wall = min(share, avail / 4), avail / 2
    hard = max(min(5 * soft, wall), TM_FLOOR)
    return Budget(min(max(soft, TM_FLOOR), hard), hard, "mtd_converged", None)


def poolyield(wtime, winc, movestogo=None, ply=0, **kw):
    """CANDIDATE (classic lane, 2026-08-17): THE POOL'S NUMBERS, READ BY
    CLASSIC'S OWN STOP RULE.

    Identical (soft, hard) to `pool` -- it DELEGATES rather than restating the
    arithmetic, so the two cannot drift -- but the soft limit is read at EVERY
    yield, `if (best or cand) and elapsed > soft: break`, instead of only at a
    yield where the MTD bracket has closed.

    Why the arm exists.  Classic's builtin loop already has the two-limit
    shape the pool needs: a `searcher.deadline` wall and a break tested at
    every yield.  So the pool's BUDGET ports into `sunfish.py` in three lines
    while the bracket rule is a second, separable mechanism costing three
    more.  On a 152-line engine that difference is the whole elegance
    argument, so it gets measured instead of assumed: `pool` minus
    `poolyield` is the price of the bracket rule, and `poolyield` minus
    `min40_4` is what the cheap port actually buys.

    `frac` is soft/hard rather than a constant because replay reads
    `frac * hard` for a yield_frac arm, and the target here is `soft` itself
    -- the pool's wall is 5x its soft limit, not 1.25x, so no fixed fraction
    of the wall would name the same number.
    """
    b = pool(wtime, winc, movestogo, ply=ply, **kw)
    return Budget(b.soft, b.hard, "yield_frac", b.soft / b.hard)


def min40_4c(wtime, winc, movestogo=None, ply=0, **kw):
    """CANDIDATE (classic lane, 2026-08-17): min40_4's NUMBERS under the pool's
    stop rule -- the other diagonal of the 2x2, and the arm that says whether
    the pool's win is its arithmetic, its stop rule, or the INTERACTION.

    The mechanism to attribute: the bracket rule's whole effect is to let an
    unsettled search run PAST the soft limit, out to the wall.  How much that
    is worth is bounded by hard/soft -- 5x for the pool, and only 1/0.8 = 1.25x
    here, because min40_4 derives its soft limit as a fraction of its own wall.
    So this arm is predicted small, and the prediction is what makes it worth
    one cell: a null here plus a null on `poolyield` says the pool's Elo lives
    in neither half alone.
    """
    b = min40_4(wtime, winc, movestogo, ply=ply, **kw)
    return Budget(b.soft, b.hard, "mtd_converged", None)


def dynamic_target(soft, stable_iters=0, changed=False, score_drop=0.0, mate=False):
    """uci.py `dynamic_target` (tm-pool-manager 7e8e1ff, lines 143-165).

    Scales the SOFT limit by how settled the search looks: stability decays
    1.15 -> 0.65, a best-move change multiplies 1.35, a score drop adds
    drop/200 capped at 1.75, and mate scores bypass the drop term entirely.
    """
    factor = min(1.15, max(0.65, 1.15 - 0.08 * stable_iters))
    if changed:
        factor *= 1.35
    if not mate:
        factor *= min(1.75, max(1.0, 1 + score_drop / 200))
    return soft * factor


def opening_ramp(think, ply, rng):
    """uci.py:474-475 -- `if len(hist) < 8: think = min(think, len(hist) + random())`.

    CLASSIC ONLY.  The packed artifact has no ramp (grepped: `random` does
    not occur in nnue_4k/pst_entry.py), so a surrogate reproducing a PACKED
    measurement must not apply it and one reproducing CLASSIC must.  The rng
    is passed in so a virtual game is reproducible from its seed.
    """
    if ply < 8:
        return min(think, ply + rng.random())
    return think


def cap_binds(manager, wtime, winc, ply=40, **knobs):
    """Is the SAFETY term, rather than the per-move share, what limits this
    budget?  The clock where the answer flips is a manager's knee, and every
    TM pathology this project has measured lives at one:

      oldtm / steptm   the share crosses the cap at wtime = 2*(1 + 0.9*inc)/
                       (1 - 2/D) seconds -- 2.4 s at winc == 0 and D == 12,
                       the "negative-cap threshold" of the stage-1 forensics
      smooth           the cap wtime^2/(2*wtime+4000) is above the share for
                       every clock a game reaches, so the knee sits where the
                       old one used to collapse and nothing collapses there
      pool             the POOL itself hits zero: wtime == (M+2)*O - (M-1)*I,
                       i.e. 8.4 s at the shipped O = 200 ms and winc == 0 --
                       the arm-(a) telemetry's exact minimum end clock

    A predicate, not a perturbation: each manager's two competing terms are
    named and compared, so the answer does not depend on a probe step.
    """
    t, i = 1000 * wtime, 1000 * winc          # the packed arms' own units
    if manager == "oldtm":
        return t / 2 - 1000 < t / 12 + 0.9 * i
    if manager == "steptm":
        return t / 2 - 1000 < t / (12 if i else 40) + 0.9 * i
    if manager == "smooth":
        return t * t / (2 * t + 4000) < t * (1000 + 20 * i) / (40000 + 240 * i) + 0.9 * i
    if manager == "legacy12":
        return wtime / 2 - 1 < wtime / 12 + 0.9 * winc
    if manager in ("min40_4", "min40_4c"):
        return wtime / 4 < wtime / 40 + 0.9 * winc      # binds below T = 4*I
    if manager == "onemax":
        # No cap exists; the only shape change is the FLOOR, which the max()
        # applies at wtime == 10 - 40*winc seconds.
        return (t - 8000) / 40 + i < 50
    if manager in ("pool", "poolyield"):
        # Same budget, so the same knee: poolyield changes only how the soft
        # limit is READ, and a knee is a property of the two competing terms.
        overhead = knobs.get("overhead", 0.2)
        if knobs.get("phase_m"):
            m = max(20, 46 - ply / 2)
        else:
            m = knobs.get("moves", 40)
        return wtime + (m - 1) * winc - (m + 2) * overhead <= 0
    raise KeyError(manager)


MANAGERS = {"legacy12": legacy12, "oldtm": oldtm, "steptm": steptm,
            "smooth": smooth, "pool": pool, "min40_4": min40_4,
            "onemax": onemax, "poolyield": poolyield,
            "min40_4c": min40_4c}
# Which DRIVER each manager was measured in.  It selects exactly one thing:
# classic's opening ramp (`if len(hist) < 8: think = min(think, len(hist) +
# random())`, uci.py:474), which the packed artifact does not have.
#
# `pool` is listed as packed ON PURPOSE.  Its arithmetic is uci.py's
# `pool_budget`, but the arm that measured +119.9 +/- 36.4 was the PACKED
# `pooltm` mod, and the packed driver has no ramp.  Reproducing that number
# with a ramp attached would be reproducing a different manager.
#
# CORRECTED 2026-08-17.  An earlier note here said the driver ramps the pool,
# so a classic pool would differ from the measured one for eight plies.  That
# is no longer true and the fix is deliberate: uci.py scopes the ramp to the
# incumbent manager ("THE RAMP BELONGS TO THE INCUMBENT MANAGER ONLY, and that
# is a measurement rule before it is a design one"), and the classic builtin
# clock -- which is where the pool landed on 2026-08-17 -- has never had a
# ramp at all.  So all three shipped pools ARE the measured manager.
FAMILY = {"legacy12": "classic", "oldtm": "packed", "steptm": "packed",
          "smooth": "packed", "pool": "packed", "min40_4": "packed",
          # Same classic-builtin driver as min40_4: max(think, .05) deadline,
          # `(best or cand) and elapsed > think * 0.8` at every yield, no ramp.
          "onemax": "packed", "poolyield": "packed",
          "min40_4c": "packed"}


# ----------------------------------------------------------------- verify ---
GRID_T = [0.05, 0.2, 0.5, 1.0, 1.9, 2.0, 2.1, 2.2, 2.4, 2.667, 3.0, 5.0, 8.4,
          10.0, 20.0, 30.0, 60.0, 120.0, 180.0, 300.0, 1800.0]
GRID_I = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]


def _pinned_present(relpath, literal, roots):
    """Where is the pinned text still checked out?

    `roots` is a list for callers that pass one explicitly, but verify()'s
    OWN default is this checkout alone (job 2, 2026-08-15) -- a second,
    sibling root used to be the default and made the verdict depend on that
    other checkout's mutable state, which is not what a drift tripwire is
    for. EVERY given root is still searched in order, not just the first:
    `nnue_4k/pst_entry.py` can exist in more than one checkout with different
    content, and stopping at the first hit would report a drift that is
    really a wrong-copy lookup.
    """
    # Line by line and IN ORDER, on collapsed whitespace.  A multi-line
    # pinned block is a generator MOD in make_variants.py, i.e. it lives in
    # the file as quoted Python string literals with escaped newlines -- so
    # the block never appears contiguously, but each of its lines does, in
    # order.  Single-line pins are the degenerate case of the same rule.
    lines = [" ".join(ln.split()) for ln in literal.strip().splitlines()]
    seen = []
    for root in roots:
        path = os.path.join(root, relpath)
        if not os.path.exists(path):
            continue
        seen.append(path)
        flat = " ".join(open(path, encoding="utf-8", errors="replace").read().split())
        at, ok = 0, True
        for ln in lines:
            at = flat.find(ln, at)
            if at < 0:
                ok = False
                break
            at += len(ln)
        if ok:
            return path, True
    return (seen[0] if seen else None), (False if seen else None)


def verify(roots=(), verbose=True):
    """Grid-check every mirror against its pinned literal.  Returns coverage."""
    # Pinned to THIS checkout only (job 2, 2026-08-15): a verifier whose
    # verdict depends on a SIBLING checkout's mutable state is broken --
    # observed directly (PR #201's handoff): the same commit read green, then
    # 4 drifted an hour later, purely because ~/repos/sunfish-packed moved
    # underneath it. ROOT is already derived upward from __file__, so pinning
    # to it is the whole fix; no env-var override is added; see it here.
    roots = list(roots) or [ROOT]
    checked = drift = 0
    report = []

    def grid(name, mirror, literal, scale_in, post):
        """post() re-applies whatever the engine does AFTER the pinned line
        (the /1000 and the max(think, .05) floor), so the comparison is
        against the shipped expression itself and not against a reworded
        version of it."""
        nonlocal checked
        for t in GRID_T:
            for i in GRID_I:
                env = {"min": min, "max": max, "wtime": scale_in * t,
                       "winc": scale_in * i, "movestogo": None}
                want = post(eval(literal, env))  # noqa: S307
                got = mirror(t, i).hard
                assert abs(got - want) < 1e-12, (
                    "%s mirror != pinned literal at T=%s I=%s: %r vs %r"
                    % (name, t, i, got, want))
                checked += 1

    # Classic arms the deadline at exactly `think`, with no floor: a negative
    # budget is an already-expired deadline, and that IS the drain pathology.
    grid("legacy12", legacy12, PINNED["legacy12"][1], 1, lambda v: v)
    # oldtm/steptm/smooth used to grid-assert here against PINNED[name][1],
    # the packed arms' divide-then-floor-at-.05 shape. Retired 2026-08-15 (see
    # the RE-PINNED note above): no PINNED entry survives for any of the
    # three, so there is no live text left to grid them against here -- their
    # mirrors are still fully exercised, by the explicit numeric asserts in
    # test_tm_surrogate.py (e.g. test_the_managers_disagree_where_the_matches_
    # said_they_do), which is a stronger check on frozen, no-longer-shipping
    # formulas than re-deriving a pin from nothing would be.
    #
    # min40_4 kept its own pin until 2026-08-17, grid-checked in BOTH units
    # because it is the one unit-independent expression here.  RETIRED there,
    # not re-anchored: the classic builtin clock it lived in now runs the pool
    # (the `pool_classic` pin above), so no live text remains on any branch.
    # The mirror stays -- it is the CONTROL arm in every cell this lane has
    # measured, and a control that has been deleted cannot be re-run -- and
    # the unit-independence claim is asserted numerically instead, in
    # test_tm_surrogate.py's test_min40_4s_expression_commutes_exactly_with_
    # unit_scaling.
    #
    # The landed classic pool, MILLISECONDS: the same _POOL_MS_TEXT block the
    # 4k entry runs, so it is gridded against the pool mirror below rather than
    # a second time here.  What IS checked here is that the shipped soft line
    # reproduces that block's `soft` exactly, in the ms domain it is written in.
    for t in GRID_T:
        for i in GRID_I:
            env = {"min": min, "max": max, "wtime": 1000 * t, "winc": 1000 * i}
            exec(PINNED["pool_classic"][1], env)   # noqa: S102 - shipped text
            ref = {"min": min, "max": max, "wtime": 1000 * t, "winc": 1000 * i}
            exec(_POOL_MS_TEXT, ref)               # noqa: S102 - landed formula
            assert abs(env["soft"] - ref["soft"]) < 1e-9, (
                "classic's shipped soft line != the landed pool at T=%s I=%s: "
                "%r vs %r" % (t, i, env["soft"], ref["soft"]))
            checked += 1
    # one-max carries its own floor inside the pinned expression (the 50), so
    # nothing is re-applied after it.
    grid("onemax", onemax, PINNED["onemax"][1], 1000, lambda v: v / 1000)

    # movestogo branch of the classic manager.
    for t in GRID_T:
        for i in GRID_I:
            for mtg in (1, 2, 5, 10, 40, 200):
                want = eval(PINNED["legacy12_mtg"][1],  # noqa: S307
                            {"min": min, "max": max, "wtime": t, "winc": i,
                             "movestogo": mtg})
                assert abs(legacy12(t, i, mtg).hard - want) < 1e-12
                checked += 1

    # The pool, against the packed millisecond text (no movestogo there, and
    # the packed mod has no TM_FLOOR on soft, so re-apply the clamps here).
    # Text is _POOL_MS_TEXT (see the RE-PINNED note above), not a PINNED
    # entry: the landed formula, verified by hand identical to nnue-4k
    # 5f16bae's tools/build/make_pst_entry.py `_pooltm`, but not re-derivable
    # from a live file in THIS checkout.
    for t in GRID_T:
        for i in GRID_I:
            env = {"min": min, "max": max, "wtime": 1000 * t, "winc": 1000 * i}
            exec(_POOL_MS_TEXT, env)               # noqa: S102 - the landed pool formula
            s_ms, h_ms = env["soft"], env["think"]
            b = pool(t, i)
            want_soft = min(max(s_ms / 1000, TM_FLOOR), b.hard)
            want_hard = max(h_ms / 1000, TM_FLOOR)
            assert abs(b.hard - want_hard) < 1e-9, (t, i, b.hard, want_hard)
            assert abs(b.soft - want_soft) < 1e-9, (t, i, b.soft, want_soft)
            checked += 2

    # Live source, when this checkout has it.
    for name, (relpath, literal) in sorted(PINNED.items()):
        path, present = _pinned_present(relpath, literal, roots)
        if name in CANDIDATES and not present:
            report.append("  %-13s CANDIDATE: not landed in %s here (branch "
                          "classic/tm-one-max-pool has it)" % (name, relpath))
            continue
        if path is None:
            report.append("  %-13s UNCHECKED against source: %s not in %s"
                          % (name, relpath, " or ".join(roots)))
        elif present:
            report.append("  %-13s pinned text FOUND in %s" % (name, path))
        else:
            report.append("  %-13s *** DRIFTED: pinned text absent from %s"
                          % (name, path))
            drift += 1

    # The four re-pinned names (see the RE-PINNED note above): not in PINNED,
    # so the loop above never touches them, but "it says so, per position,
    # instead of quietly skipping" applies to these as much as to any other.
    report.append("  oldtm         RETIRED at 5f16bae (shared anchor gone); "
                  "mirror kept for test_tm_surrogate.py, not source-checked")
    report.append("  steptm        RETIRED at 5f16bae (same anchor as oldtm); "
                  "mirror kept, not source-checked")
    report.append("  smooth        RETIRED at 5f16bae (superseded by the pool); "
                  "mirror kept, not source-checked")
    report.append("  min40_4       RETIRED 2026-08-17 (classic's builtin clock "
                  "became the pool; see the pool_classic pin); mirror kept as "
                  "the CONTROL arm, not source-checked")
    report.append("  pool_ms       LANDED at 5f16bae into make_pst_entry.py's "
                  "_pooltm (make_variants.py no longer has it); that file does "
                  "not resolve on this checkout, so grid-assert only (below), "
                  "no source check")

    # THE LIVE pool_budget, which is the pool mirror's real proof.  It lives
    # on branch tm-pool-manager, so on master there is nothing to check
    # against and the report says so.  Point TM_UCI at any checkout of that
    # uci.py to run the check without merging it:
    #
    #   git show origin/tm-pool-manager:sunfish_ui/uci.py > /tmp/uci_pool.py
    #   TM_UCI=/tmp/uci_pool.py python3 tmlib.py
    #
    # Measured that way against 7e8e1ff: 45,360 pool_budget values and 160
    # dynamic_target values identical to 1e-12.  When the branch lands, the
    # plain import below picks it up and the env var stops being needed.
    live = "  pool_budget   UNCHECKED: no pool_budget here; set TM_UCI to check"
    sys.path.insert(0, ROOT)
    try:
        alt = os.environ.get("TM_UCI")
        if alt:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_tm_uci", alt)
            uci = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(uci)
        else:
            import sunfish_ui.uci as uci
        if hasattr(uci, "pool_budget"):
            n = 0
            for t in GRID_T:
                for i in GRID_I:
                    for ply in (0, 10, 40, 52, 120):
                        for pm in (False, True):
                            for sc in (0.8, 1.0, 1.2):
                                for mtg in (None, 1, 10, 40):
                                    want = uci.pool_budget(t, i, mtg, ply=ply,
                                                           phase_m=pm, scale=sc)
                                    got = pool(t, i, mtg, ply=ply, phase_m=pm,
                                               scale=sc)
                                    assert abs(got.soft - want[0]) < 1e-12
                                    assert abs(got.hard - want[1]) < 1e-12
                                    n += 2
            for si in range(10):
                for ch in (False, True):
                    for dr in (0.0, 50.0, 200.0, 1000.0):
                        for mate in (False, True):
                            assert abs(dynamic_target(1.0, si, ch, dr, mate)
                                       - uci.dynamic_target(1.0, si, ch, dr, mate)) < 1e-15
                            n += 1
            checked += n
            live = "  pool_budget   %d values identical to %s" % (n, alt or "uci.py")
    except Exception as exc:                       # noqa: BLE001 - reported, not hidden
        live = "  pool_budget   UNCHECKED: loading uci.py raised %r" % (exc,)

    if verbose:
        print("tmlib.verify: %d grid values checked against pinned literals"
              % checked)
        for line in report + [live]:
            print(line)
    if drift:
        raise SystemExit("tmlib: %d pinned literal(s) drifted - re-pin before "
                         "any surrogate number is used" % drift)
    return checked


if __name__ == "__main__":
    verify()
