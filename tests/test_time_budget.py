"""The clock must survive a long sudden-death game.

sunfish-nnue-engine lost lichess.org/EAThUL0P on time at move 73 of a
3+0 game WITHOUT a single move overrunning: `wtime/12` spent 12.8s of a
180s budget on ply 9, and once the clock fell under 2s the `wtime/2 - 1`
cap went negative, the budget collapsed, and ~200ms/move of unavoidable
lag drained the rest.

No existing gate can see this: the suite checks protocol and search
correctness, and a match would need a real 3+0 game to reproduce it. So
the budget curve is walked directly here, the same way
nnue_4k/tests/test_time_budget.py walks the packed engine's twin fix.

The formula under test is inline in run(), so it is extracted from the
source rather than duplicated -- if its shape changes, this test fails
loudly instead of silently testing a stale copy. uci.py works in
SECONDS (wtime is divided by 1000 before the formula), unlike the
packed engine's milliseconds version.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "sunfish_ui" / "uci.py").read_text()

# THE sudden-death-aware budget line. There is exactly one.
DEV = re.search(r"^\s+think = min\(wtime / \(12 if winc else 40\) \+ 0\.9 \* winc,"
                r" wtime / 2 - 1\)\s*$", SRC, re.M)
# the OLD unconditional /12 policy (d3f7f12), kept as a literal reference: it
# reproduces the defect below, and proves the winc > 0 path is unchanged
OLD_LINE = "think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1)"


def test_budget_line_present():
    assert DEV, "sudden-death budget line missing or reshaped in sunfish_ui/uci.py"


def _eval(line, wtime_s, winc_s):
    ns = {"wtime": wtime_s, "winc": winc_s, "min": min}
    exec(line, ns)
    return ns["think"]


def budget(wtime_s, winc_s):
    """seconds of thinking time, straight from the shipped source line"""
    return _eval(DEV.group(0).strip(), wtime_s, winc_s)


def old_budget(wtime_s, winc_s):
    """the pre-fix policy, from the literal above"""
    return _eval(OLD_LINE, wtime_s, winc_s)


def walk(base_s, inc_s, moves, fn=budget, overhead=0.2):
    """simulate our own clock over `moves` of our moves; -1 == flagged.

    overhead is per-move lag (network + process turnaround) the budget
    cannot see; lichess games show ~200ms.
    """
    clock = base_s
    for mv in range(moves):
        spent = max(fn(clock, inc_s), 0) + overhead
        clock -= spent
        clock += inc_s
        if clock <= 0:
            return -1, mv
    return clock, moves


def test_old_policy_reproduces_the_lost_game():
    """/12 flags a 73-move 3+0 game -- the walk has teeth."""
    left, reached = walk(180, 0, 73, fn=old_budget)
    assert left == -1, "the old /12 policy no longer flags: walk model is stale"
    assert reached > 30, "flagged implausibly early at move %d" % reached


@pytest.mark.parametrize("moves", [80, 100, 120])
def test_sudden_death_survives_long_games(moves):
    """3+0, the control that actually lost a game."""
    left, reached = walk(180, 0, moves)
    assert left > 0, "flagged at move %d of %d in 3+0" % (reached, moves)


def test_the_lost_game_would_now_be_survived():
    """73 moves at 3+0 -- the exact game, with time to spare."""
    left, _ = walk(180, 0, 73)
    assert left > 5, "only %.1fs left after the lost game's length" % left


def test_increment_behaviour_unchanged():
    """winc > 0 must budget exactly what the audited /12 policy did: that
    path measured well in the 11-game production audit, so the fix may
    only change winc == 0."""
    for wtime in (1, 30, 180, 1_800):
        for winc in (0.001, 0.1, 3):
            assert budget(wtime, winc) == old_budget(wtime, winc), (
                "winc > 0 budget moved at wtime=%s winc=%s" % (wtime, winc))


def test_tournament_control_unaffected():
    """1800+3 over a long game keeps a healthy reserve."""
    left, _ = walk(1_800, 3, 120)
    assert left > 5, "1800+3 left only %.1fs" % left


def test_early_move_is_no_longer_front_loaded():
    """The lost game spent 12.8s on ply 9; the fix must be far below that."""
    assert budget(180, 0) < 5.0, "first move still front-loaded"
