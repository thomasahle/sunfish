#!/usr/bin/env python3
"""
Test that all files in tools/test_files are well-formed and parse correctly.
"""
import pytest
import chess
import chess.pgn
from pathlib import Path



@pytest.mark.parametrize('fen_file',
    [f for f in Path('tools/test_files').glob('*.fen')]
)
def test_fen_files_parse(fen_file):
    # Each line in a .fen file should produce a valid Board
    for ln, line in enumerate(fen_file.read_text().splitlines(), start=1):
        fen = line.strip()
        if not fen or fen.startswith('#'):
            continue
        try:
            # Strip any annotations after semicolon to get pure FEN
            pure_fen = fen.split(';', 1)[0]
            board = chess.Board(pure_fen)
        except Exception as e:
            pytest.fail(f"Invalid FEN in {fen_file} at line {ln}: {e}")
        assert isinstance(board, chess.Board)

@pytest.mark.parametrize('epd_file',
    [f for f in Path('tools/test_files').glob('*.epd')]
)
def test_epd_files_parse(epd_file):
    # Each line in a .epd file should parse via Board.from_epd
    for ln, line in enumerate(epd_file.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            # Extract EPD fields: board, turn, castling, ep
            # Then append dummy halfmove and fullmove counters
            fields = line.split(';', 1)[0].split()
            fen_six = ' '.join(fields[:4] + ['0', '1'])
            board = chess.Board(fen_six)
        except Exception as e:
            pytest.fail(f"Invalid EPD in {epd_file} at line {ln}: {e}")
        assert isinstance(board, chess.Board)

@pytest.mark.parametrize('pgn_file',
    [f for f in Path('tools/test_files').glob('*.pgn')]
)
def test_pgn_files_parse(pgn_file):
    # Each .pgn file should contain at least one valid game
    text = pgn_file.read_text(encoding='utf-8', errors='ignore')
    # Use a stream to read games
    stream = pgn_file.open(encoding='utf-8', errors='ignore')
    games = 0
    try:
        for game in chess.pgn.read_game(stream):
            if game is None:
                break
            games += 1
            # Play through moves to ensure legality
            board = game.board()
            for move in game.mainline_moves():
                assert move in board.legal_moves, f"Illegal move in {pgn_file}: {move}"
                board.push(move)
    finally:
        stream.close()
    assert games > 0, f"No games found in {pgn_file}"