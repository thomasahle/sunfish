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

Also here, because they are the premises the point spec rests on: the
depth-0 half of the stratified king-capture contract, and the
correctness of the legality oracle the fail-low correction calls -
tested directly as the board predicate Position.king_capture(), both
differentially against python-chess and on a deterministic corpus of
the special rules a random playout corpus cannot reach.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunfish_tools.uci as uci  # noqa: E402

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
def test_faillow_mated_qs_node(order):
    """Fail-low arm: a stand-pat fail-high followed (or preceded) by an
    exhaustive probe at the same key must not cross.  (The repair is no
    longer a "verifier" bolted onto the null: at depth 0 QS simply
    evaluates the fold and claims no exact terminal value, and the
    fail-low correction at depth >= 1 re-derives legality with the
    board predicate.  The witness is kept because it is the position
    that first exposed stand-pat masking.)"""
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
    # position is sound, and QS, which only has to fail high at a
    # capturable node, can hand one up.)
    for g, r in zip(gammas, results):
        assert not (r >= g and r > 0), (g, r)


def test_ahead_stalemate_full_driver_consistent():
    """Full iterative-deepening driver over the fail-high witness leaves
    an ordered table. (The sentinel-masking channel that used to make
    this unassertable over arbitrary positions is closed on this branch:
    every virtual fail-high is validated before it may cut. The wider
    statement is now carried by test_terminal_bench.py, which asserts an
    ordered table over all 148 bench positions.)"""
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
    """Differential test of the legality oracle against python-chess.

    The PRIMARY assertion is the board predicate the shipped correction
    actually calls - child.king_capture() is not None iff the move was
    illegal.  The search probe is kept as a SECONDARY assertion: it is
    the old oracle, no longer on the shipped path, but it must still
    agree.  Corpus: deterministic random playouts."""
    chess = pytest.importorskip("chess")
    import random

    from sunfish_tools.uci import render_move

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
            child = pos.move(mv)
            assert (child.king_capture() is not None) == (not b.is_legal(cm)), \
                (b.fen(), u, "direct")
            assert (s.bound(child, sf.MATE_UPPER, 0, root=True) == sf.MATE_UPPER) \
                == (not b.is_legal(cm)), (b.fen(), u, "search")
            checked += 1
    assert checked > 300


# Deterministic special-rule corpus: (fen, uci, legal).  Random playouts
# cover these poorly or not at all - the three illegal castlings are not
# even PSEUDO-legal to python-chess, so the differential test above
# skips them and can never exercise the kp rule.  Ground truth is
# hardcoded (and cross-checked against python-chess when it is
# installed) so these cases run even without it.
SPECIAL_RULES = [
    ("5r2/8/8/k7/8/8/8/4K2R w K -", "e1g1", False, "castle-through-check-kp"),
    ("6r1/8/8/k7/8/8/8/4K2R w K -", "e1g1", False, "castle-into-check"),
    ("4r3/8/8/k7/8/8/8/4K2R w K -", "e1g1", False, "castle-out-of-check"),
    ("8/8/8/k7/8/8/8/4K2R w K -", "e1g1", True, "castle-clean"),
    # The rook, unlike the king, may cross an attacked square: b1 is hit
    # but b1 is outside the kp window, so this must stay legal.
    ("1r6/8/8/7k/8/8/8/R3K3 w Q -", "e1c1", True, "castle-rook-crosses-attack"),
    ("2r5/8/8/7k/8/8/8/R3K3 w Q -", "e1c1", False, "castle-queenside-into-check"),
    ("3r4/8/8/7k/8/8/8/R3K3 w Q -", "e1c1", False, "castle-queenside-through-check"),
    ("8/8/8/K2pP2r/8/8/8/7k w - d6", "e5d6", False, "ep-exposes-rook"),
    ("6bk/8/8/3pP3/8/1K6/8/8 w - d6", "e5d6", False, "ep-exposes-bishop"),
    ("7k/8/8/K2pP3/8/8/8/8 w - d6", "e5d6", True, "ep-clean"),
    ("4r2k/8/8/8/8/8/4R3/4K3 w - -", "e2a2", False, "pinned-rook-off-file"),
    ("4r2k/8/8/8/8/8/4R3/4K3 w - -", "e2e5", True, "pinned-rook-along-file"),
    ("7k/8/8/8/8/4b3/5B2/6K1 w - -", "f2e1", False, "pinned-bishop-off-diagonal"),
    ("7k/8/8/8/8/4b3/5B2/6K1 w - -", "f2e3", True, "pinned-bishop-takes-pinner"),
    ("8/8/8/8/8/4k3/8/4K3 w - -", "e1e2", False, "king-adjacent-to-king"),
    ("8/8/8/8/8/4k3/8/4K3 w - -", "e1d1", True, "king-steps-away"),
    ("r6k/1P6/8/8/8/8/8/K7 w - -", "b7a8q", True, "promotion-capture-legal"),
    ("nr4k1/1P6/8/8/8/8/8/1K6 w - -", "b7a8q", False, "promotion-capture-unpins"),
    ("6k1/8/8/8/8/8/1p6/RN5K b - -", "b2a1q", True, "promotion-capture-black"),
]


def special_case(fen, u):
    """Parse a SPECIAL_RULES row into (parent, move, child).

    Requires load_sunfish() to have run (it binds uci.sunfish)."""
    parts = fen.split()
    pos = uci.from_fen(parts[0], parts[1], parts[2], parts[3], "0", "1")
    mv = uci.parse_move(u, white_pov=(parts[1] == "w"))
    assert mv in set(pos.gen_moves()), (fen, u, "not pseudo-legal in sunfish")
    return pos, mv, pos.move(mv)


@pytest.mark.parametrize("fen,u,legal,label", SPECIAL_RULES)
def test_king_capture_special_rules(fen, u, legal, label):
    """Position.king_capture() on the deterministic special-rule corpus:
    castling through / into / out of check (the kp rule), en passant
    that uncovers a rook or a bishop, pins, king-next-to-king, and
    promotion captures."""
    load_sunfish()  # binds uci.sunfish, which special_case() parses through
    _, _, child = special_case(fen, u)
    assert (child.king_capture() is None) == legal, label


@pytest.mark.parametrize("fen,u,legal,label", SPECIAL_RULES)
def test_special_rules_ground_truth(fen, u, legal, label):
    """The hardcoded ground truth above is what python-chess says."""
    chess = pytest.importorskip("chess")
    b = chess.Board(fen + " 0 1")
    assert b.is_valid(), (label, b.status())
    assert b.is_legal(chess.Move.from_uci(u)) == legal, label


@pytest.mark.parametrize("order", ["lower_first", "upper_first"])
def test_qs_stratified_contract(order):
    """The depth-0 half of the stratified king-capture contract, on
    every capturable child of the special-rule corpus.

    At depth 0 bound() only promises to FAIL HIGH, and the two halves
    of that promise are what the depth-1 argument rests on:

      - stand-pat MAY fail high, so a probe at gamma = pos.score is
        allowed to return the static score and nothing stronger;
      - stand-pat CANNOT cut one point above it, so at
        gamma = pos.score + 1 the capture is proved and the node
        reports the sentinel exactly.

    Asserted in both probe orders, cold, warm by repetition, and warm
    behind a positive-depth search that has filled tp_move at the key."""
    sf = load_sunfish()
    capturable = 0
    for fen, u, legal, label in SPECIAL_RULES:
        if legal:
            continue
        _, _, pos = special_case(fen, u)
        assert pos.king_capture() is not None, label
        capturable += 1
        s = sf.Searcher()
        s.history = set()
        for phase in ("cold", "warm", "deep"):
            if phase == "deep":
                s.bound(pos, 1, 3)
            probes = ["standpat", "capture"]
            if order == "upper_first":
                probes = probes[::-1]
            for kind in probes:
                if kind == "standpat":
                    assert s.bound(pos, pos.score, 0) >= pos.score, (label, phase)
                else:
                    assert s.bound(pos, pos.score + 1, 0) == sf.MATE_UPPER, \
                        (label, phase)
    assert capturable >= 9
