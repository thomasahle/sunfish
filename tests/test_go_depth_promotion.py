"""Driver-rule regression tests for go_loop at "go depth N".

The commit-on-completed-depth rule plays only moves endorsed by a
finished depth. But the max_depth break fires on the FIRST yield of
depth N+1 - a probe that ran to completion at the sanest window of the
whole search (its gamma sits inside depth N's converged bracket, before
the per-depth bracket reset introduces absurd windows). Discarding a
fail-high on that yield threw away paid-for information and dropped the
WAC depth-3 battery from 94/300 to 52/300 (master CI, 2026-08-10, the
#158 merge).

The mid-dive protection must survive the fix: on a TIMED stop, a
fail-high from a later probe of an unfinished depth (the Qxc6/eviction
class: absurd gamma after the bracket reset) must not override the last
completed depth's answer.
"""
import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunfish
from sunfish_tools import uci

# go_loop/render_move/pv reach the engine through the module global that
# run() normally injects; bind it the same way run() does.
uci.sunfish = sunfish


class ScriptedSearcher:
    """Searcher stand-in replaying a fixed (depth, gamma, score, move) tape."""

    def __init__(self, tape):
        self.tape = tape
        self.nodes = 0
        self.tp_move = {}
        self.tp_score = {}
        self.deadline = float("inf")

    def search(self, hist):
        yield from self.tape


def mv(s):
    return sunfish.Move(sunfish.parse(s[:2]), sunfish.parse(s[2:4]), "")


def bestmove_of(capsys, tape, max_movetime=10**6, max_depth=3):
    hist = [sunfish.Position(sunfish.initial, 0, (True, True), (True, True), 0, 0)]
    uci.go_loop(ScriptedSearcher(tape), hist, threading.Event(),
                max_movetime, max_depth)
    lines = [l for l in capsys.readouterr().out.splitlines()
             if l.startswith("bestmove")]
    assert len(lines) == 1
    return lines[0].split()[1]


def test_depth_break_plays_first_probe_fail_high(capsys):
    """A fail-high on the breaking yield (first probe of depth N+1, sane
    window, ran to completion) must be played - it is what the pv-walk
    driver played at "go depth N" for years."""
    tape = [
        (1, 0, 5, mv("e2e4")),    # depth 1: fail-high on e2e4
        (2, 3, 1, mv("e2e4")),    # depth 2: fail-low, converges
        (3, 2, 4, mv("e2e4")),    # depth 3: fail-high on e2e4, converges
        (4, 3, 10, mv("d2d4")),   # depth 4 FIRST probe: fail-high on d2d4
    ]
    assert bestmove_of(capsys, tape, max_depth=3) == "d2d4"


def test_depth_break_keeps_committed_answer_on_fail_low(capsys):
    """If the depth-N+1 first probe fails low it endorses no move: the
    last completed depth's answer stands."""
    tape = [
        (1, 0, 5, mv("e2e4")),
        (3, 2, 4, mv("e2e4")),    # depth 3 fail-high, converges
        (4, 3, 1, mv("g1h3")),    # depth 4 first probe: fail-LOW (stale move)
    ]
    assert bestmove_of(capsys, tape, max_depth=3) == "e2e4"


def test_timed_stop_never_plays_mid_dive_artifact(capsys):
    """The Qxc6/eviction class stays closed: on a timed stop, a fail-high
    from a later probe of the unfinished depth must not be played over the
    committed answer."""
    tape = [
        (1, 0, 5, mv("e2e4")),        # depth 1: fail-high, completes
        (2, 3, 8, mv("e2e4")),        # depth 2 first probe: fail-high
        (2, -30000, 9, mv("g1h3")),   # depth 2 mid-dive: absurd-window artifact
    ]
    # max_movetime=0 forces the timed break at the first depth>1 yield
    # processed after it; the artifact's cand must lose to the committed
    # depth-1 answer.
    assert bestmove_of(capsys, tape, max_movetime=0, max_depth=100) == "e2e4"


def test_wac004_go_depth_3_finds_qxh7(capsys):
    """End-to-end on the real engine and the real go_loop: the CI
    battery's WAC.004. Depth 3 converges on h6f4; the depth-4 first probe
    fails high on Qxh7+. 'go depth 3' must answer h6h7 (this exact case
    regressed the master CI floor)."""
    fen = "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1"
    hist = [uci.from_fen(*fen.split())]
    uci.go_loop(sunfish.Searcher(), hist, threading.Event(),
                max_movetime=10**6, max_depth=3)
    lines = [l for l in capsys.readouterr().out.splitlines()
             if l.startswith("bestmove")]
    assert len(lines) == 1
    assert lines[0].split()[1] == "h6h7"
