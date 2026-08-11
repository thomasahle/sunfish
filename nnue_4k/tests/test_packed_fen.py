"""FEN glue for the packed engine (lichess deployment path).

The bot serves FEN positions through tools/uci.py's from_fen, which for
accumulator-carrying engines delegates to sunfish.from_board -- the same
from-scratch construction verify.py trusts.  These tests prove the glue:

1. round-trip: replaying a real game move by move, rendering each position
   to FEN and re-parsing it reproduces the position EXACTLY -- board,
   score, ps, castling rights, en passant, accumulator, perspective flag,
   king-bucket index and piece count.  (kp is excluded: the king-passant
   square is a transient of the immediately preceding castling move and
   has no FEN representation; it only ever matters for the very next reply,
   which a position served over FEN does not have.)
2. en passant: a FEN carrying an ep square yields a position whose ep
   field and legal-move set include the capture.
3. black-to-move orientation: from_fen returns the mover-oriented
   position; its accumulator equals the incremental one.
"""
import importlib.util
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sunfish_ui"))


def load():
    os.environ["SF_NET"] = str(ROOT / "nnue_4k" / "packed" / "net128.sfnn")
    spec = importlib.util.spec_from_file_location(
        "sunfish", ROOT / "nnue_4k" / "sunfish_packed.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sunfish_packed_fen"] = mod
    spec.loader.exec_module(mod)
    import uci
    uci.sunfish = mod          # what uci.run() would do
    return mod, uci


GAME = (
    "g1f3 g8f6 d2d4 e7e6 b1c3 f8b4 c1f4 e8g8 e2e3 f6d5 d1d2 c7c6 f1d3 h7h6 "
    "e1g1 d5f4 e3f4 d7d5 a2a3 b4d6 a1d1 b8d7 f1e1 d7b6 f3e5 d8f6 h2h3 c8d7 "
    "e5d7 b6d7 c3e2 c6c5 c2c3 c5c4 d3b1 a8e8 d2c2 g7g6 c2a4 d7b8 a4a7 e8e7"
).split()


def to_fen(mod, uci, pos):
    """Render a mover-oriented packed Position as a FEN string."""
    white = uci.get_color(pos) == uci.WHITE
    p = pos if white else pos.rotate()
    rows = []
    for r in range(8):
        row, run = "", 0
        for c in p.board[21 + 10 * r:29 + 10 * r]:
            if c == ".":
                run += 1
            else:
                row += (str(run) if run else "") + c
                run = 0
        rows.append(row + (str(run) if run else ""))
    castle = ("K" if p.wc[1] else "") + ("Q" if p.wc[0] else "") + \
             ("k" if p.bc[0] else "") + ("q" if p.bc[1] else "")
    ep = mod.render(p.ep) if p.ep else "-"
    return "%s %s %s %s 0 1" % ("/".join(rows), "w" if white else "b",
                                castle or "-", ep)


def test_fen_round_trip_full_game():
    mod, uci = load()
    pos = mod.hist[0]
    checked = 0
    for ply, mv in enumerate([None] + GAME):
        if mv is not None:
            i, j, prom = mod.parse(mv[:2]), mod.parse(mv[2:4]), mv[4:].upper()
            if ply % 2 == 0:
                i, j = 119 - i, 119 - j
            pos = pos.move(mod.Move(i, j, prom))
        fen = to_fen(mod, uci, pos)
        got = uci.from_fen(*fen.split())
        assert got.board == pos.board, f"board mismatch at ply {ply}: {fen}"
        for field in ("score", "ps", "wc", "bc", "ep", "acc", "pf", "kb"):
            assert getattr(got, field) == getattr(pos, field), (
                f"{field} mismatch at ply {ply}: {fen}\n"
                f"  replay: {getattr(pos, field)!r}\n"
                f"  fen:    {getattr(got, field)!r}")
        # the accumulator must also equal a fresh from-scratch build
        assert got.acc == mod.from_board(got.board, pf=got.pf).acc
        checked += 1
    assert checked == len(GAME) + 1


def test_fen_en_passant_capture_available():
    mod, uci = load()
    # after 1. e4 c5 2. e5 d5 the ep capture exd6 must exist
    pos = uci.from_fen(
        "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR", "w", "KQkq",
        "d6", "0", "3")
    assert pos.ep == mod.parse("d6")
    ep_moves = [m for m in pos.gen_moves()
                if m.j == pos.ep and pos.board[m.i] == "P"]
    assert ep_moves, "en passant capture missing after FEN load"
    after = pos.move(ep_moves[0])
    # the captured pawn is gone: the incremental accumulator matches a
    # from-scratch build
    assert after.acc == mod.from_board(after.board, pf=after.pf).acc


def test_fen_black_to_move_matches_incremental():
    mod, uci = load()
    pos = mod.hist[0]
    for ply, mv in enumerate(GAME[:9]):        # odd count: black to move
        i, j, prom = mod.parse(mv[:2]), mod.parse(mv[2:4]), mv[4:].upper()
        if ply % 2 == 1:
            i, j = 119 - i, 119 - j
        pos = pos.move(mod.Move(i, j, prom))
    fen = to_fen(mod, uci, pos)
    assert fen.split()[1] == "b"
    got = uci.from_fen(*fen.split())
    assert got.pf == pos.pf and got.acc == pos.acc and got.score == pos.score
