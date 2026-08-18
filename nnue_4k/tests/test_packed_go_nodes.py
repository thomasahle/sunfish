"""`go nodes N` in the ARTIFACT's own UCI loop.

The 4k artifact is what we want to benchmark, and until 2026-08-18 it could not
play a fixed-node game: `nodes` was parsed only inside `minifier-hide`, so the
packed build ignored it and searched to the clock instead.  A harness that pins
`tc=6000+0` (the standard way to make a node cap the binding limit) therefore
handed the artifact a ~150 s per-move budget, and a "fixed-node" match silently
became a movetime match -- observed in the book lane's gauntlet as eight engine
processes at 98% CPU with no game finishing, and the reason that lane had to
measure byte-verified checkouts instead of the artifact itself.

These tests drive the artifact's loop the way the packer leaves it: with the
minifier-hide blocks stripped, exactly as CI does for sunfish.py.  Packing is not
involved -- lzma round-tripping is `tools/build/pack.sh`'s job, not this file's.
"""
import pathlib
import re
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "nnue_4k" / "pst_entry.py"
HIDE = re.compile(r"^\s*# minifier-hide start$.*?^\s*# minifier-hide end$\n", re.M | re.S)
# A clock so large the engine's own budget is minutes: only a node cap can end
# the search early, which is exactly the harness configuration that exposed the
# defect (`-each nodes=N tc=6000+0`).
BIG_CLOCK = "go wtime 6000000 btime 6000000"


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    """pst_entry.py with the minifier-hide blocks removed -- what the packer ships."""
    out = tmp_path_factory.mktemp("packedish") / "artifact.py"
    out.write_text(HIDE.sub("", SRC.read_text()))
    return out


def run(artifact, commands, timeout):
    p = subprocess.Popen([sys.executable, str(artifact)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    t0 = time.time()
    try:
        p.stdin.write(commands)
        p.stdin.flush()
        while time.time() - t0 < timeout:
            line = p.stdout.readline()
            if not line: break
            if line.startswith("bestmove"):
                return line.split()[1], time.time() - t0
    except (BrokenPipeError, OSError):
        pass
    finally:
        p.kill()
    return None, time.time() - t0


def test_node_cap_binds_before_the_clock(artifact):
    """With a minutes-long budget, `nodes` must be what stops the search."""
    move, elapsed = run(artifact, "uci\nisready\nposition startpos\n%s nodes 3000\n" % BIG_CLOCK,
                        timeout=90)
    assert move is not None, "no bestmove within 90 s: the node cap did not bind"
    assert re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move), move
    # The poll runs every 2048 nodes, so the cap is honoured to within one poll,
    # never exactly. What matters is that it ended the search at all: the same
    # command without `nodes` would run for ~150 s.
    assert elapsed < 60, "took %.1fs -- the cap did not bind" % elapsed


def test_no_nodes_means_the_clock_still_rules(artifact):
    """The production path: no `nodes`, so the budget comes from the clock as before."""
    move, elapsed = run(artifact, "uci\nisready\nposition startpos moves e2e4\ngo wtime 4000 btime 4000\n",
                        timeout=90)
    assert move is not None and re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move), move
    # wtime 4000 -> a budget of order 0.1 s; a couple of seconds of slack for a
    # cold interpreter on a loaded CI box, but nowhere near a node-capped search.
    assert elapsed < 30, "took %.1fs on a 4 s clock" % elapsed


def test_a_zero_node_request_still_produces_a_legal_move(artifact):
    """`nodes 0` must not mean "no cap" by accident and must never answer (none).

    The structural bestmove floor exists because "(none)" is scored as an ILLEGAL
    MOVE by tournament managers (19/400 games at 1+0, 2026-08-14). A cap that
    fires before the first root fail-high has to fall through to it.
    """
    move, _ = run(artifact, "uci\nisready\nposition startpos\n%s nodes 1\n" % BIG_CLOCK, timeout=90)
    assert move is not None and move != "(none)", move
    assert re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move), move
