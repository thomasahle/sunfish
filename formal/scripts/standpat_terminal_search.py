#!/usr/bin/env python3
"""The sweep that REFUTED StandPatAtTerminal (formal/Sunfish/Stalemate.lean).

Historical role: the Lean side once carried StandPatAtTerminal as a
hypothesis -- at a depth-0 correction-terminal position the stand-pat
pseudo-option (`yield None, pos.score`) must not exceed the terminal
value.  This sweep found 100+ real-board violations (ordinary corner
mates), refuting it; the verify-on-suspicion landing then removed the
consumer instead of defending the hypothesis (`if depth and ...`
excludes depth 0 from the correction, sunfish.py line 463 on
`d2-verify-pending`), and the Lean ledger keeps the definition only as
a countermodel.  The script is preserved as the refutation's
reproducer and as a template for sweeping the still-open channel
(sentinel masking -- see KingCapturableReportsExact in the ledger).

A violation on a real board is the crisp crossing shape of
`cexT_crossing` played by the stand-pat instead of the null yield:

* the side to move is stalemated or checkmated (python-chess is the
  ground truth here),
* the static `pos.score` EXCEEDS the terminal value (0 for stalemate,
  `-MATE_LOWER` for mate -- for mates the score condition is vacuous, so
  the gate is the whole test), and
* EVERY pseudo-legal `pos.gen_moves()` has `pos.value(m) >= 40`
  (`val_lower` at depth 0), so the PRE-d2 correction gate
  `all(pos.value(m) >= val_lower ...)` passed.

At such a position, on the pre-d2 engine, a low-gamma depth-0 probe cut
off on the stand-pat and stored `lower = pos.score > terminal`, while a
high-gamma probe ran the loop dry (`best_real` stayed at the sentinel),
corrected, and stored `upper = terminal`: a transposition-table
crossing.  On the verified engine the depth-0 correction is gone and
hits are expected to leave ordered entries (see
tests/test_regressions.py::TestStandPatTerminalDepth0).

Sources searched:

1. uniform-random playouts from the initial position (the terminal
   position of every game that ends in mate or stalemate is tested);
2. random sparse positions (kings plus 1..7 random pieces, pawn-biased
   for blockage, both colors to move), keeping the valid ones that are
   mate or stalemate.

Any hit is printed prominently with its FEN (and should become an xfail
regression test in tests/).  No hit prints the attempted counts and the
nearest misses (fewest below-threshold pseudo-legal moves).

Run from the repo root:

    uv run python formal/scripts/standpat_terminal_search.py \
        --games 4000 --sparse 500000 --seed 0
"""

import argparse
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

import sunfish  # noqa: E402
from tools import uci  # noqa: E402

uci.sunfish = sunfish  # tools/uci binds its engine module in run(); do it here

VAL_LOWER_D0 = 40  # val_lower = QS - 0 * QS_A at depth 0 (sunfish.py line 359)


def sunfish_pos(board):
    """python-chess Board -> sunfish Position, side to move's perspective."""
    return uci.from_fen(*board.fen().split())


def classify(board):
    if board.is_checkmate():
        return "mate"
    if board.is_stalemate():
        return "stalemate"
    return None


def check_terminal(board, stats, near):
    """Return a hit tuple if `board` is the counterexample shape."""
    kind = classify(board)
    if kind is None:
        return None
    stats[kind] += 1
    pos = sunfish_pos(board)
    terminal = 0 if kind == "stalemate" else -sunfish.MATE_LOWER
    if pos.score <= terminal:
        return None
    stats[kind + "+score"] += 1
    vals = [pos.value(m) for m in pos.gen_moves()]
    below = [v for v in vals if v < VAL_LOWER_D0]
    if not below:
        return (board.fen(), kind, pos.score, terminal, sorted(vals))
    near.append((len(below), len(vals), min(below), kind, board.fen()))
    return None


def search_playouts(games, rng, stats, near, hits, max_plies=400):
    for _ in range(games):
        board = chess.Board()
        while not board.is_game_over() and board.ply() < max_plies:
            board.push(rng.choice(list(board.legal_moves)))
        stats["playout_positions"] += 1
        hit = check_terminal(board, stats, near)
        if hit:
            hits.append(hit)


PIECE_POOL = "PPPPNBRQ"  # pawn-biased: blockage is what makes terminals


def random_sparse(rng):
    board = chess.Board(None)
    squares = rng.sample(chess.SQUARES, 2 + rng.randint(1, 7))
    board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
    for sq in squares[2:]:
        sym = rng.choice(PIECE_POOL)
        color = rng.random() < 0.5
        if sym == "P" and not (8 <= sq < 56):
            continue
        board.set_piece_at(sq, chess.Piece(chess.Piece.from_symbol(sym).piece_type, color))
    board.turn = rng.random() < 0.5
    board.clear_stack()
    return board if board.is_valid() else None


def search_sparse(count, rng, stats, near, hits):
    for _ in range(count):
        board = random_sparse(rng)
        if board is None:
            continue
        stats["sparse_positions"] += 1
        hit = check_terminal(board, stats, near)
        if hit:
            hits.append(hit)


def corner_packs(stats, near, hits, max_pieces=4):
    """Systematic family: the shape a hit must take is a king smothered by
    pieces whose every pseudo-legal move is blocked or expensive, so
    enumerate ALL packings of the 3x3-plus-fringe box around each corner
    with up to `max_pieces` pieces from a small palette, both colors to
    move.  (A lone escape square or a cheap quiet move anywhere kills the
    gate, which is exactly why this family is where counterexamples would
    live.)"""
    import itertools

    palette_syms = ["P", "N", "B", "Q", "p", "n"]
    for corner, white_to_move, ek_sq in (
        (chess.A1, True, chess.E6), (chess.H1, True, chess.D6),
        (chess.A8, False, chess.E3), (chess.H8, False, chess.D3),
    ):
        cf, cr = chess.square_file(corner), chess.square_rank(corner)
        df = 1 if cf == 0 else -1
        dr = 1 if cr == 0 else -1
        box = [chess.square(cf + df * i, cr + dr * j)
               for i in range(3) for j in range(3) if (i, j) != (0, 0)]
        own = chess.WHITE if white_to_move else chess.BLACK
        for k in range(1, max_pieces + 1):
            for squares in itertools.combinations(box, k):
                for syms in itertools.product(palette_syms, repeat=k):
                    board = chess.Board(None)
                    board.set_piece_at(corner, chess.Piece(chess.KING, own))
                    board.set_piece_at(ek_sq, chess.Piece(chess.KING, not own))
                    for sq, sym in zip(squares, syms):
                        pt = chess.Piece.from_symbol(sym).piece_type
                        color = own if sym.isupper() else (not own)
                        board.set_piece_at(sq, chess.Piece(pt, color))
                    board.turn = own
                    if not board.is_valid():
                        continue
                    stats["corner_positions"] += 1
                    hit = check_terminal(board, stats, near)
                    if hit:
                        hits.append(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--sparse", type=int, default=500000)
    ap.add_argument("--corner-pieces", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = {k: 0 for k in ("playout_positions", "sparse_positions", "corner_positions",
                            "mate", "stalemate", "mate+score", "stalemate+score")}
    near, hits = [], []

    search_playouts(args.games, rng, stats, near, hits)
    search_sparse(args.sparse, rng, stats, near, hits)
    corner_packs(stats, near, hits, max_pieces=args.corner_pieces)

    print(f"playout games:          {stats['playout_positions']}")
    print(f"valid sparse positions: {stats['sparse_positions']}")
    print(f"valid corner packings:  {stats['corner_positions']}")
    print(f"terminal positions:     {stats['mate']} mates, "
          f"{stats['stalemate']} stalemates")
    print(f"score above terminal:   {stats['mate+score']} mates, "
          f"{stats['stalemate+score']} stalemates")

    if hits:
        print("\n*** COUNTEREXAMPLE(S) FOUND -- StandPatAtTerminal is FALSE on real boards ***")
        for fen, kind, score, terminal, vals in hits:
            print(f"  {kind}: FEN {fen}")
            print(f"    pos.score = {score} > terminal = {terminal}; "
                  f"pseudo-legal move values (all >= {VAL_LOWER_D0}): {vals}")
        sys.exit(1)

    near.sort()
    print("\nNo counterexample. Nearest misses "
          "(#moves below val_lower / #pseudo-legal, min value):")
    for below, total, minval, kind, fen in near[:5]:
        print(f"  {below}/{total} below, min {minval}, {kind}: {fen}")


if __name__ == "__main__":
    main()
