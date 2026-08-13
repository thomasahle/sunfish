"""The clock must survive a long sudden-death game.

sunfish-nnue-engine lost lichess.org/EAThUL0P on time at move 73 of a
3+0 game WITHOUT a single move overrunning: `wtime/12` spent 12.8s of a
180s budget on ply 9, and once the clock fell under 2s the
`wtime/2 - 1000` cap went negative, the budget collapsed to the 0.05s
floor, and ~200ms/move of unavoidable lag drained the rest.

No existing gate can see this. The ladder checks nodes, bytes and
correctness; a match would need a real 3+0 game to reproduce it. So the
budget curve is walked directly here.

The formula under test is inline in main(), so it is extracted from the
source rather than duplicated -- if its shape changes, this test fails
loudly instead of silently testing a stale copy.
"""
import os
import re

import pytest

ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "sunfish_nnue.py")
SRC = open(ENGINE).read()

# THE budget line. There is exactly one, and it is NOT inside minifier-hide:
# source and artifact run the same formula. The old layout kept a plain /12
# line for the artifact and hid the sudden-death branch behind minifier-hide
# -- and the 300+0 ladder then played the known-bad /12 branch on 97.2% of
# 4,158 matched moves while the fix sat dead in source (LOSS_TAXONOMY.md P0).
DEV = re.search(r"^\s+think = min\(wtime / \(12 if winc else 40\) \+ 0\.9 \* winc,"
                r" wtime / 2 - 1000\)\s*$", SRC, re.M)
# the OLD artifact-only /12 policy, kept here as a literal reference so the
# winc > 0 path is provably unchanged by the unification
OLD_ART_LINE = "think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000)"


def test_budget_line_ships_in_the_artifact():
    """One budget line, outside minifier-hide, so the artifact gets the fix."""
    assert DEV, "budget line missing or reshaped"
    assert SRC.count("think = min(wtime") == 1, (
        "more than one budget line -- the hide-block split is back")
    # the line must not sit inside a minifier-hide region, or the artifact
    # silently loses the sudden-death branch again
    stripped = re.sub(r"# minifier-hide start.*?# minifier-hide end", "", SRC, flags=re.S)
    assert re.search(r"12 if winc else 40", stripped), (
        "budget line is inside minifier-hide: artifact would drop the fix")


def budget(wtime_ms, winc_ms):
    """seconds of thinking time, i.e. the engine's `think` after its /1000.

    The extracted expression yields MILLISECONDS (wtime is ms); main()
    divides by 1000 on the next line. Getting this wrong is exactly the
    ms/seconds confusion that produced a 590-second move earlier today,
    so the conversion lives in one place and is named.
    """
    ns = {"wtime": wtime_ms, "winc": winc_ms, "min": min}
    exec(DEV.group(0).strip(), ns)
    return ns["think"] / 1000.0


def old_artifact_budget(wtime_ms, winc_ms):
    """the pre-fix artifact policy (/12 unconditionally), from the literal above"""
    ns = {"wtime": wtime_ms, "winc": winc_ms, "min": min}
    exec(OLD_ART_LINE, ns)
    return ns["think"] / 1000.0


def walk(base_ms, inc_ms, moves, overhead=0.2):
    """simulate our own clock over `moves` of our moves; -1 == flagged"""
    clock = base_ms
    for mv in range(moves):
        think = max(budget(clock, inc_ms), 0.05)
        spent = think + overhead                      # lag we cannot avoid
        clock -= spent * 1000
        clock += inc_ms
        if clock <= 0:
            return -1, mv
    return clock, moves


@pytest.mark.parametrize("moves", [80, 100, 120])
def test_sudden_death_survives_long_games(moves):
    """3+0, the control that actually lost a game."""
    left, reached = walk(180_000, 0, moves)
    assert left > 0, "flagged at move %d of %d in 3+0" % (reached, moves)


def test_the_lost_game_would_now_be_survived():
    """73 moves at 3+0 -- the exact game, with time to spare."""
    left, _ = walk(180_000, 0, 73)
    assert left > 5_000, "only %.1fs left after the lost game's length" % (left / 1000)


def test_increment_behaviour_unchanged_by_unification():
    """winc > 0 must budget exactly what the old /12 policy did: that path
    measured fine at 30+1 and on lichess, so the unification may only change
    winc == 0."""
    for wtime in (1_000, 30_000, 180_000, 1_800_000):
        for winc in (1, 100, 3_000):
            assert budget(wtime, winc) == old_artifact_budget(wtime, winc), (
                "winc > 0 budget moved at wtime=%d winc=%d" % (wtime, winc))


def test_tournament_control_unaffected():
    """1800+3 over a long game keeps a healthy reserve."""
    left, _ = walk(1_800_000, 3_000, 120)
    assert left > 5_000, "1800+3 left only %.1fs" % (left / 1000)


def test_early_move_is_no_longer_front_loaded():
    """The lost game spent 12.8s on ply 9; the fix must be far below that."""
    assert budget(180_000, 0) < 5.0, "first move still front-loaded"
