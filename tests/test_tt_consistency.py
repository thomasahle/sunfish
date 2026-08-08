"""Transposition-table point-spec consistency.

The doctrine (formal/README.md): every stored bound describes one value
function determined by the TT key, so no entry may ever hold
lower > upper.  These tests pin the witnesses discovered in the
verify-on-suspicion arc.  The doctrine's sharpest lesson:

    A fail-soft score is evidence about value, not evidence about
    legality.  Mobility is certified only by a stored fail-high move or
    by the dedicated legality probe.

Witness 1 (fail-low arm / stand-pat masking): a bare king mated by
defended pieces - every pseudo-move is a valuable-but-illegal capture,
so stand-pat carries a normal score while the exact terminal value is
mate.  Historically produced Entry(lower=-1711, upper=-47923).

Witness 2 (fail-high arm / positive null at stalemate): the +175
"ahead-stalemate" - a king-defended pinned knight, a promotion-blocked
guard pawn, and frozen central pawn towers give the stalemated side a
positive static score, so the null pass fails high (+11) at a genuine
stalemate while the correction later stores the exact 0.  This refuted
NullAtStalemateNonpositive as a real-chess assumption.

Both witnesses run in BOTH probe orders: lower-then-upper and
upper-then-lower, to catch asymmetric update mistakes.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.uci as uci  # noqa: E402

MATED_QS_FEN = ("6Rk/6QP/8/8/8/8/8/K7", "b", "-", "-", "0", "1")
AHEAD_STALEMATE_FEN = ("8/8/8/5k1p/3b1P1P/1p1P1P1P/pN1P1P2/K7", "w", "-", "-", "0", "1")


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


def crossed(searcher):
    return [(k, e) for k, e in searcher.tp_score.items() if e.lower > e.upper]


@pytest.mark.parametrize("order", ["lower_first", "upper_first"])
def test_faillow_verifier_mated_qs_node(order):
    """Fail-low arm: a stand-pat fail-high followed (or preceded) by an
    exhaustive probe at the same key must not cross."""
    sf = load_sunfish()
    pos = uci.from_fen(*MATED_QS_FEN)
    s = sf.Searcher()
    s.history = set()
    gammas = (pos.score, pos.score + 1)
    if order == "upper_first":
        gammas = gammas[::-1]
    for g in gammas:
        s.bound(pos, g, 0)
    assert crossed(s) == []


@pytest.mark.parametrize("order", ["lower_first", "upper_first"])
def test_failhigh_null_verifier_ahead_stalemate(order):
    """Fail-high arm: a positive uncertified null cutoff at a genuine
    stalemate must be verified, not stored as a lower bound above the
    exact draw. The +175 construction crosses on any design without the
    null-side verifier (master pre-fix: lower=11, upper=0)."""
    sf = load_sunfish()
    pos = uci.from_fen(*AHEAD_STALEMATE_FEN)
    s = sf.Searcher()
    s.history = set()
    gammas = (5, 300)
    if order == "upper_first":
        gammas = gammas[::-1]
    results = [s.bound(pos, g, 4) for g in gammas]
    assert crossed(s) == []
    # The pass's +11 must never be served as a fail-high lower bound: any
    # probe that fails high here claims the stalemate is worth > 0.
    # (Fail-low uppers may exceed 0 - a loose upper bound on a 0-valued
    # position is sound, e.g. via the open sentinel-masking channel.)
    for g, r in zip(gammas, results):
        assert not (r >= g and r > 0), (g, r)


def test_ahead_stalemate_full_driver_consistent():
    """Full iterative-deepening driver over the fail-high witness leaves
    an ordered table. (Not asserted over arbitrary positions: the
    orthogonal, still-open sentinel-masking channel - a king-capturable
    child soundly cutting off on a partial bound - can cross entries
    elsewhere; see the ledger of open obligations in formal/README.md.)"""
    sf = load_sunfish()
    pos = uci.from_fen(*AHEAD_STALEMATE_FEN)
    s = sf.Searcher()
    for depth, gamma, score, move in s.search([pos]):
        if depth > 5:
            break
    assert crossed(s) == []


@pytest.mark.parametrize(
    "fen",
    [
        "4k1QK/5q1P/8/8/8/8/8/8",  # Q+P(h7) fortress; QS stalemate tricks
        "5qQK/4k2P/8/8/8/8/8/8",
    ],
)
def test_fortress_driver_consistent(fen):
    """Positions whose old suite 'successes' were transient exact-0 lines
    produced by crossed bounds (settled scores were far from 0)."""
    sf = load_sunfish()
    pos = uci.from_fen(fen, "b", "-", "-", "0", "1")
    s = sf.Searcher()
    for depth, gamma, score, move in s.search([pos]):
        if depth > 5:
            break
    assert crossed(s) == []


def test_legality_oracle_vs_python_chess():
    """Differential test of the legality oracle: for pseudo-legal moves,
    probe == MATE_UPPER iff the move is illegal (leaves the king
    capturable). Deterministic corpus from random playouts."""
    chess = pytest.importorskip("chess")
    import random

    from tools.uci import render_move

    sf = load_sunfish()
    random.seed(7)
    checked = 0
    for game in range(30):
        b = chess.Board()
        for _ in range(random.randint(4, 70)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(random.choice(moves))
        if b.is_game_over():
            continue
        fen = b.fen().split()
        pos = uci.from_fen(fen[0], fen[1], fen[2], fen[3], "0", "1")
        s = sf.Searcher()
        s.history = set()
        for mv in pos.gen_moves():
            u = render_move(mv, white_pov=b.turn == chess.WHITE)
            try:
                cm = chess.Move.from_uci(u)
            except ValueError:
                continue
            if b.piece_at(cm.from_square) is None or not b.is_pseudo_legal(cm):
                continue
            oracle_illegal = (
                s.bound(pos.move(mv), sf.MATE_UPPER, 0, root=True) == sf.MATE_UPPER
            )
            assert oracle_illegal == (not b.is_legal(cm)), (b.fen(), u)
            checked += 1
    assert checked > 300
