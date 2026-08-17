"""The reported PV must stop at the fifty-move rule.

pv() walks the transposition table with no notion of the fifty-move clock, so
in the shuffling endgames of a real 300-game run it reported lines continuing
past a legally-drawn position: 1,232 "PV continues after fifty-move rule"
warnings from the tournament manager, each with a full position dump, 4 MB of
log. The engine deliberately ignores the rule (README) and Position is too
size-constrained to carry the clock, so the UI tracks it alongside hist and
hands it to the loops, which hand it to pv().

Reporting only, and these tests pin that: raising the clock truncates the
reported line and changes nothing else about it.

The walk is driven off a completed fixed-depth search rather than go_loop,
because go_loop's probe sequence is wall-clock sensitive - two runs of the
same search differ in nodes and scores. The walk itself, given the finished
table, is deterministic. One end-to-end test covers go_loop with an assertion
that holds whatever the probes did.
"""
import contextlib
import io
import pathlib
import re
import sys
import threading

import chess

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunfish
from sunfish_ui import uci

# go_loop/render_move/pv reach the engine through the module global that run()
# normally injects; bind it the same way run() does.
uci.sunfish = sunfish

INFO_PV = re.compile(r"^info .*\bpv ((?:[a-h][1-8][a-h][1-8][qrbn]?(?: |$))+)", re.M)

# Quiet positions sitting one ply from the fifty-move draw, the shape that
# produced the warnings. At clock 99 any quiet continuation is already past the
# rule, so a walk that ignores the clock overshoots on its very first move.
FENS = [
    "2kr3r/pp3ppp/8/8/8/8/PP3PPP/2KR3R w - - 99 126",
    "r2q1rk1/pp1bbppp/2n1pn2/3p4/3P4/2NBPN2/PP1B1PPP/R2Q1RK1 w - - 99 128",
    "5k2/5p2/5P2/8/8/5P2/5K2/8 w - - 99 133",
]


def walk(fen, hclock, depth=4):
    """The PV pv() reports from the table a finished depth-N search leaves."""
    # search() picks the king table from the root and leaves it in the module
    # global (sunfish.py:537), but from_fen scores the position BEFORE that
    # happens - so a position built after somebody else's search is scored with
    # their table, shifting every score by a constant and changing the table the
    # walk reads. Pick it here, the way search() will, to stay order-independent.
    board = fen.split()[0]
    sunfish.pst["K"] = sunfish.K_MID if "Q" in board and "q" in board else sunfish.K_END
    hist = [uci.from_fen(*fen.split())]
    searcher = sunfish.Searcher()
    for reported_depth, _, _, _ in searcher.search(hist):
        if reported_depth > depth: break
    return uci.pv(searcher, hist[-1], include_scores=False, hclock=hclock)


def overshoot(fen, moves):
    """Plies the line plays from an already-drawn position - what the
    tournament manager flags. Replayed on python-chess, which owns the rule."""
    board, past = chess.Board(fen), 0
    for move in moves:
        if board.halfmove_clock >= 100: past += 1
        board.push(chess.Move.from_uci(move))
    return past


def test_pv_stops_at_the_fifty_move_rule():
    for fen in FENS:
        assert overshoot(fen, walk(fen, hclock=int(fen.split()[4]))) == 0, fen


def test_the_positions_have_teeth():
    """Told the clock is fresh, the same walk does overshoot - so the test above
    closes a real gap rather than merely reporting short PVs."""
    assert sum(overshoot(fen, walk(fen, hclock=0)) for fen in FENS) > 0


def test_the_clock_only_ever_truncates():
    """Reporting-only, in the strongest form available to a test: the stopped
    line is a prefix of the unstopped one. Nothing is re-ordered or replaced,
    and in particular the first move - the bestmove fallback - is untouched."""
    for fen in FENS:
        full, stopped = walk(fen, hclock=0), walk(fen, hclock=99)
        assert stopped, fen
        assert full[: len(stopped)] == stopped, fen


def test_a_drawn_root_still_reports_a_move():
    """bestmove falls back to my_pv[0] when no depth committed, so the stop must
    never empty the PV: it fires only once a move is already in it."""
    for fen in FENS:
        assert walk(fen, hclock=300), fen


def test_a_stopped_pv_carries_no_ponder_hint():
    """go_loop prints "ponder my_pv[1]" only when the walk has a second move. A
    move that completes the fifty-move count has no reply worth pondering, so
    the hint is correctly omitted - the one output change beyond the PV text.

    Asserted over the set rather than per position: a capturing first move resets
    the clock and legitimately keeps walking, so which of these stops at the rule
    depends on what the search likes, and that is allowed to change."""
    assert any(len(walk(fen, hclock=99)) == 1 for fen in FENS)


def test_go_loop_reports_stopped_pvs_end_to_end():
    """The whole path run() uses, on the real engine. Timing decides how many
    info lines appear; every one of them must respect the rule."""
    for fen in FENS:
        hist = [uci.from_fen(*fen.split())]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            uci.go_loop(sunfish.Searcher(), hist, threading.Event(),
                        max_movetime=10**6, max_depth=4, hclock=int(fen.split()[4]))
        out = buf.getvalue()
        reported = [m.group(1).split() for m in INFO_PV.finditer(out)]
        assert reported, fen
        assert all(overshoot(fen, moves) == 0 for moves in reported), fen
        assert re.search(r"^bestmove [a-h][1-8][a-h][1-8]", out, re.M), fen


def test_quiet_move_advances_the_clock_only_when_it_should():
    """Captures and pawn pushes reset it; castling does not. En passant is a
    pawn push whose destination looks empty, which is why the test is here."""
    pos = uci.from_fen(*"4k3/8/8/3pP3/8/8/8/R3K2R w KQ d6 0 1".split())
    by_uci = {uci.render_move(m, True): m for m in pos.gen_moves()}

    assert not uci.quiet_move(pos, by_uci["e5d6"])  # en passant capture
    assert not uci.quiet_move(pos, by_uci["e5e6"])  # quiet pawn push
    assert uci.quiet_move(pos, by_uci["e1g1"])      # castling
    assert uci.quiet_move(pos, by_uci["a1a5"])      # rook to an empty square
