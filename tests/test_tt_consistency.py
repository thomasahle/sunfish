"""Transposition-table point-spec consistency.

The doctrine (formal/README.md): every stored bound describes one value
function determined by the TT key, so no entry may ever hold
lower > upper.  These tests probe the shapes that historically violated
it: terminal (mate/stalemate) QS nodes whose stand-pat had already been
served to another probe before the exact correction fired.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.uci as uci  # noqa: E402


def load_sunfish():
    import importlib.util

    src = (ROOT / "sunfish.py").read_text()
    stripped = src[: src.index("def main():")]
    probe = ROOT / "tests" / "_sunfish_probe.py"
    probe.write_text(stripped)
    spec = importlib.util.spec_from_file_location("sunfish_probe", probe)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sunfish_probe"] = mod
    spec.loader.exec_module(mod)
    uci.sunfish = mod
    return mod


def crossed_entries(searcher):
    return [(k, e) for k, e in searcher.tp_score.items() if e.lower > e.upper]


def test_mated_qs_node_two_probe_consistency():
    """Bare king mated by defended pieces: every pseudo-move is a
    valuable-but-illegal capture, so the QS exhaustion gate passes while
    stand-pat carries a normal score.  A stand-pat fail-high followed by
    an exhaustive fail-low at the same key must not cross."""
    sf = load_sunfish()
    pos = uci.from_fen("6Rk/6QP/8/8/8/8/8/K7", "b", "-", "-", "0", "1")
    s = sf.Searcher()
    s.history = set()
    s.bound(pos, pos.score, 0)      # stand-pat fails high
    s.bound(pos, pos.score + 1, 0)  # exhaustion reaches the correction
    assert crossed_entries(s) == []


@pytest.mark.parametrize(
    "fen",
    [
        "4k1QK/5q1P/8/8/8/8/8/8",  # Q+P(h7) fortress; stalemate tricks at QS
        "5qQK/4k2P/8/8/8/8/8/8",
        # Natural corner mate found by formal/scripts/standpat_terminal_search.py:
        # K_MID first-row deltas price cornered-king quiet moves above QS=40,
        # so ordinary corner mates pass the exhaustion gate at depth 0.
        "8/8/8/7p/7P/1K5P/p7/k5R1",
    ],
)
def test_driver_leaves_no_crossed_entries(fen):
    """Full driver runs on positions whose search trees contain terminal
    QS nodes with positive stand-pat.  These two previously 'passed' the
    stalemate suite only via transient exact-0 reports produced by
    crossed bounds (settled scores were far from 0)."""
    sf = load_sunfish()
    pos = uci.from_fen(fen, "b", "-", "-", "0", "1")
    s = sf.Searcher()
    for depth, gamma, score, move in s.search([pos]):
        if depth > 5:
            break
    assert crossed_entries(s) == []
