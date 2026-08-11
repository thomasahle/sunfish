"""WinBoard/XBoard compatibility through the PolyGlot adapter.

A 2023 user report (WinBoard 4.6.2 + polyglot + a 5-ply book) described
sunfish crashing on Python 3.7 and losing on time under pypy3.  The
crash era is gone (sunfish requires >= 3.8) and the time losses match
the since-fixed between-iteration deadline bugs - but nothing guarded
the adapter path itself.  This drives the engine through a real
PolyGlot process speaking XBoard protocol exactly as WinBoard does:
feature negotiation, `level`/`time`/`otim` clocks, raw-coordinate
moves.  Skipped when polyglot is not installed (CI installs it).
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("polyglot") is None, reason="polyglot adapter not installed")


def test_winboard_session_plays_moves(tmp_path):
    ini = tmp_path / "polyglot.ini"
    ini.write_text(
        "[Polyglot]\n"
        "EngineName=Sunfish\n"
        f"EngineCommand={sys.executable} {ROOT / 'sunfish.py'}\n"
        f"EngineDirectory={ROOT}\n"
        "[Engine]\n")
    proc = subprocess.Popen(
        ["polyglot", str(ini)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1, cwd=tmp_path)

    def send(*lines):
        for l in lines:
            proc.stdin.write(l + "\n")
        proc.stdin.flush()

    moves = []
    deadline = time.time() + 60

    def read_until_move():
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith("move "):
                moves.append(line.split()[1])
                return
        raise AssertionError(
            f"no move from the adapter (got {len(moves)} so far)")

    try:
        send("xboard", "protover 2", "new",
             "level 40 5 0", "time 30000", "otim 30000", "e2e4")
        read_until_move()
        send("time 28000", "otim 28000", "g1f3")
        read_until_move()
    finally:
        try:
            send("quit")
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    assert len(moves) == 2, f"expected 2 engine moves, got {moves}"
    for m in moves:
        assert len(m) in (4, 5) and m[0] in "abcdefgh" and m[2] in "abcdefgh", \
            f"malformed move {m!r}"
