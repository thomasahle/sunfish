#!/usr/bin/env python3
import pytest
import chess
import chess.engine
import asyncio
import time
import sys
from pathlib import Path

@pytest.mark.asyncio
async def test_nnue_engine_initialization():
    """Test that the NNUE engine initializes correctly with different models"""
    root_dir = Path(__file__).parent.parent
    model_dir = root_dir / "nnue" / "models"
    models = list(model_dir.glob("*.pickle"))
    
    if not models:
        pytest.skip("No NNUE models found in nnue/models directory")
    else:
        print(f"Found models: {models}")
    
    # Test with the first model we find
    model_path = models[0]
    
    # Initialize engine
    transport, engine = await asyncio.wait_for(
        chess.engine.popen_uci([sys.executable, str(root_dir / "sunfish_nnue.py"), str(model_path)]),
        timeout=5.0
    )
    
    try:
        # Check that the engine responds to UCI commands
        await asyncio.wait_for(engine.ping(), timeout=2.0)
        
        # Success - we just needed to verify the engine initializes properly
        assert True
    finally:
        await engine.quit()

@pytest.mark.asyncio
async def test_nnue_engine_moves():
    """Test that the NNUE engine can generate legal moves"""
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
        # Check that the engine responds to UCI commands
        await asyncio.wait_for(engine.ping(), timeout=2.0)
        
        # Test standard starting position - this should be enough to verify moves work
        board = chess.Board()  # Starting position
        result = await asyncio.wait_for(
            engine.play(board, chess.engine.Limit(time=0.1)),
            timeout=3.0
        )
        assert result.move is not None, "Engine should return a move in starting position"
        assert result.move in board.legal_moves, "Engine should return a legal move"
    finally:
        await engine.quit()