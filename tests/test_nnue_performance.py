#!/usr/bin/env python3
import pytest
import chess
import chess.engine
import asyncio
import sys
from tools.tester import uci_perft
from pathlib import Path

@pytest.mark.asyncio
async def test_nnue_perft():
    """Test that the NNUE engine can calculate perft correctly"""
    root_dir = Path(__file__).parent.parent
    model_dir = root_dir / "nnue" / "models"
    models = list(model_dir.glob("*.pickle"))
    
    if not models:
        pytest.skip("No NNUE models found in nnue/models directory")
    
    # Test with the first model we find
    model_path = models[0]
    
    # Initialize engine with a longer timeout for initialization
    transport, engine = await asyncio.wait_for(
        chess.engine.popen_uci([sys.executable, str(root_dir / "sunfish_nnue.py"), str(model_path)]),
        timeout=5.0
    )
    
    try:
        # Test perft (depth 1 from initial position should give 20 moves)
        # Set the engine to the standard starting position
        board = chess.Board()
        engine._position(board)

        # Run perft depth 1 via UCI perft command and count nodes
        moves = await asyncio.wait_for(
            uci_perft(engine, 1),
            timeout=5.0
        )
        count = sum(cnt for _, cnt in moves)
        assert count == 20, f"Expected 20 perft nodes at depth 1, got {count}"

    finally:
        await engine.quit()