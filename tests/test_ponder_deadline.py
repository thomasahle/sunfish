"""The ponder search must be bounded, but only as a backstop.

sunfish_tools/uci.py used to leave ``searcher.deadline`` unset for "go ponder" and
"go infinite", so such a search ended only on "stop"/"ponderhit". If that
command is ever lost the search pins a CPU forever -- on a 0.25 vCPU
shared-core VM that starves everything else on the box, including the event
loop that has to read the opponent's next move.

It is now capped at ``UNBOUNDED_MAX_SECONDS``. The risk in that change is
making ponder terminate *early*, which would be a UCI protocol violation
(a spontaneous "bestmove" the GUI never asked for) and would throw away the
ponder work. These tests pin both halves: the cap exists and is finite, and
a ponder search still runs until it is told to stop.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunfish_tools.uci as uci  # noqa: E402


def test_cap_is_finite_and_generous():
    """Finite (the whole point) but far longer than any real ponder turn."""
    cap = uci.UNBOUNDED_MAX_SECONDS
    assert isinstance(cap, (int, float))
    assert cap == cap, "cap must not be NaN"
    assert 60 <= cap <= 3600, f"cap {cap}s is outside the sensible range"


def test_cap_applies_to_the_unbounded_think_time():
    """'go ponder'/'go infinite' set think = 10**6; the cap must win."""
    think = 10**6
    assert min(think, uci.UNBOUNDED_MAX_SECONDS) == uci.UNBOUNDED_MAX_SECONDS
    # ...and a normal search must be completely unaffected by it.
    for normal in (0.05, 1.0, 15.0):
        assert min(normal, uci.UNBOUNDED_MAX_SECONDS) == normal


class Engine:
    """Drive sunfish.py over UCI as a subprocess, collecting stdout."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, "sunfish.py"],
            cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.lines: list[str] = []
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        for line in self.proc.stdout:
            self.lines.append(line.strip())

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait_for(self, token, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if any(ln.startswith(token) for ln in list(self.lines)):
                return True
            time.sleep(0.02)
        return False

    def saw(self, token):
        return any(ln.startswith(token) for ln in list(self.lines))

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.proc.kill()


@pytest.fixture
def engine():
    e = Engine()
    yield e
    e.close()


def test_ponder_runs_until_told_to_stop(engine):
    """A ponder search must not end on its own, and must end on 'stop'."""
    engine.send("uci")
    assert engine.wait_for("uciok", 30), f"no uciok; got {engine.lines}"
    engine.send("isready")
    assert engine.wait_for("readyok", 30)

    engine.send("position startpos moves e2e4")
    engine.send("go ponder")

    # It should still be pondering a few seconds later: no bestmove yet.
    time.sleep(3.0)
    assert not engine.saw("bestmove"), (
        "ponder search terminated on its own -- the cap is firing far too "
        f"early. Output: {engine.lines}")

    # And it must stop promptly when asked.
    engine.send("stop")
    assert engine.wait_for("bestmove", 15), (
        f"no bestmove within 15s of 'stop'. Output: {engine.lines}")
