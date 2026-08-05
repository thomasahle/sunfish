#!/usr/bin/env python3
import pytest
import chess
import chess.engine
from pathlib import Path
from tools.tester import uci_perft

@pytest.mark.asyncio
async def test_perft(engine):
    """Test move generation correctness with UCI perft command."""
    test_files_dir = Path(__file__).parent.parent / "tools" / "test_files"
    perft_file = test_files_dir / "perft.epd"
    lines = perft_file.read_text().splitlines()

    positions_tested = 0
    # Only test a few positions to keep the suite fast
    for idx, line in enumerate(lines[:3]):
        board, opts = chess.Board.from_epd(line)
        # Set engine to this position
        engine._position(board)
        # For the starting position, test depths 1 and 2; others just depth 1
        max_depth = 2 if idx == 0 else 1
        for depth in range(1, max_depth + 1):
            key = f"D{depth}"
            if key not in opts:
                continue
            positions_tested += 1
            moves = await uci_perft(engine, depth)
            count = sum(cnt for _, cnt in moves)
            expected = int(opts[key])
            assert count == expected, (
                f"Perft mismatch for position {idx+1} at depth {depth}: "
                f"expected {expected}, got {count}"
            )
    assert positions_tested > 0, "No positions were tested"