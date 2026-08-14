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
CANDIDATES = {"min40_4"}

PINNED = {
    # sunfish_ui/uci.py:467 (master f95f49c) -- classic's incumbent, SECONDS.
    # Also uci.py:712 on branch tm-pool-manager, unchanged there.
    "legacy12": (
        "sunfish_ui/uci.py", "min(wtime / 12 + 0.9 * winc, wtime / 2 - 1)"),
    # sunfish_ui/uci.py:452 -- the movestogo branch of the same manager.
    "legacy12_mtg": (
        "sunfish_ui/uci.py", "min(wtime / movestogo + winc, wtime / 2 - 1)"),
    # tools/build/make_variants.py "oldtm" replacement text (sunfish-packed,
    # branch nnue-4k, commit adf1313).  MILLISECONDS.  This is the line the
    # ladder actually played and lost -235.5 +/- 65.4 with.
    "oldtm": (
        "tools/build/make_variants.py",
        "min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000)"),
    # make_variants.py "steptm" replacement text, same commit.  MILLISECONDS.
    # Byte-identical to the stage-1 tmfix winner (packed sha fe22791b409b1fba).
    "steptm": (
        "tools/build/make_variants.py",
        "min(wtime / (12 if winc else 40) + 0.9 * winc, wtime / 2 - 1000)"),
    # nnue_4k/pst_entry.py:766-767 (sunfish-packed, nnue-4k) -- the shipped
    # smooth budget, packed sha 14b69a606b743a37.  MILLISECONDS.
    "smooth": (
        "nnue_4k/pst_entry.py",
        "min(wtime * (1000 + 20 * winc) / (40000 + 240 * winc) + 0.9 * winc,"
        " wtime * wtime / (2 * wtime + 4000))"),
    # CANDIDATE (in CANDIDATES above), so its absence from the tree is
    # reported as "not landed yet" rather than as drift.  It IS implemented on
    # branch classic/tm-min40-4: sunfish.py:601 at a7d9a6c, in the classic
    # built-in clock, and the text there is byte-identical to this literal.
    #
    # THAT BRANCH RUNS IT IN MILLISECONDS and this mirror runs it in SECONDS,
    # and both are right -- which is the candidate's whole claim, now
    # demonstrated by two independent implementations rather than argued.
    # verify() grid-checks this literal in BOTH units for exactly that reason.
    # Its driver matches the packed family too: max(think, .05) as the
    # deadline, `(best or cand) and elapsed > think * 0.8` at every yield,
    # and no opening ramp -- which is what FAMILY says about it below.
    "min40_4": ("sunfish.py", "min(wtime / 40 + 0.9 * winc, wtime / 4)"),
    # make_variants.py "pooltm" replacement text, commit 629cba2.
    # MILLISECONDS; packed sha cddf392e21449054.  This one is a STATEMENT
    # block, not an expression, and it is checked by exec-ing the literal
    # itself -- so the text that is grepped for and the text that is
    # evaluated are the same string, with no reformulation in between.
    "pool_ms": (
        "tools/build/make_variants.py",
        "M = 40\n"
        "P = max(0, wtime + (M - 1) * winc - (M + 2) * 200)\n"
        "A = max(0, wtime - 400)\n"
        "soft = min(P / M, A / 4)\n"
        "think = min(5 * soft, A / 2)\n"),
}


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
    if manager == "min40_4":
        return wtime / 4 < wtime / 40 + 0.9 * winc      # binds below T = 4*I
    if manager == "pool":
        overhead = knobs.get("overhead", 0.2)
        if knobs.get("phase_m"):
            m = max(20, 46 - ply / 2)
        else:
            m = knobs.get("moves", 40)
        return wtime + (m - 1) * winc - (m + 2) * overhead <= 0
    raise KeyError(manager)


MANAGERS = {"legacy12": legacy12, "oldtm": oldtm, "steptm": steptm,
            "smooth": smooth, "pool": pool, "min40_4": min40_4}
# Which DRIVER each manager was measured in.  It selects exactly one thing:
# classic's opening ramp (`if len(hist) < 8: think = min(think, len(hist) +
# random())`, uci.py:474), which the packed artifact does not have.
#
# `pool` is listed as packed ON PURPOSE.  Its arithmetic is uci.py's
# `pool_budget`, but the arm that measured +119.9 +/- 36.4 was the PACKED
# `pooltm` mod, and the packed driver has no ramp.  Reproducing that number
# with a ramp attached would be reproducing a different manager.  Worth
# knowing when the pool lands in classic: uci.py:719-722 DOES ramp it (and
# clamps the soft limit to the ramped value), so classic's pool and the
# measured pool differ for the first eight plies of every game.
FAMILY = {"legacy12": "classic", "oldtm": "packed", "steptm": "packed",
          "smooth": "packed", "pool": "packed", "min40_4": "packed"}


# ----------------------------------------------------------------- verify ---
GRID_T = [0.05, 0.2, 0.5, 1.0, 1.9, 2.0, 2.1, 2.2, 2.4, 2.667, 3.0, 5.0, 8.4,
          10.0, 20.0, 30.0, 60.0, 120.0, 180.0, 300.0, 1800.0]
GRID_I = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]


def _pinned_present(relpath, literal, roots):
    """Where is the pinned text still checked out?

    EVERY root is searched, not the first one that happens to hold a file of
    that name: `nnue_4k/pst_entry.py` exists in both checkouts and only the
    packed one carries the shipped budget, so stopping at the first hit
    reports a drift that is really a wrong-copy lookup.
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
    roots = list(roots) or [ROOT, os.path.expanduser("~/repos/sunfish-packed")]
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
    # The packed arms divide by 1000 and then floor at .05 (pst_entry.py's
    # `searcher.deadline = start + max(think, .05)`).
    for name in ("oldtm", "steptm", "smooth"):
        grid(name, MANAGERS[name], PINNED[name][1], 1000,
             lambda v: max(v / 1000, TM_FLOOR))
    # The candidate is unit-independent, so its literal is checked in BOTH
    # units -- if the mirror ever acquired an absolute constant this is where
    # it would show.
    grid("min40_4", min40_4, PINNED["min40_4"][1], 1, lambda v: max(v, TM_FLOOR))
    grid("min40_4/ms", min40_4, PINNED["min40_4"][1], 1000,
         lambda v: max(v / 1000, TM_FLOOR))

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
    for t in GRID_T:
        for i in GRID_I:
            env = {"min": min, "max": max, "wtime": 1000 * t, "winc": 1000 * i}
            exec(PINNED["pool_ms"][1], env)        # noqa: S102 - the pinned text
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
                          "classic/tm-min40-4 has it)" % (name, relpath))
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
