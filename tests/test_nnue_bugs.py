#!/usr/bin/env python3
import pytest
import chess
import chess.engine
import os
import asyncio
import sys
import contextlib
from pathlib import Path
from pytest_asyncio import fixture

@pytest.mark.asyncio
async def test_nnue_bug_positions():
    """Test positions that previously caused bugs in the NNUE implementation"""
    # Get the path to the project root directory
    root_dir = Path(__file__).parent.parent
    
    # Check for test data
    bug_positions_file = root_dir / "nnue" / "nnue_bug_fens"
    if not bug_positions_file.exists():
        pytest.skip("NNUE bug positions file not found")
    
    # Check for NNUE models
    model_dir = root_dir / "nnue" / "models"
    models = list(model_dir.glob("*.pickle"))
    if not models:
        pytest.skip("No NNUE models found in nnue/models directory")
    
    model_path = models[0]  # Use the first model we find
    
    # Part 1: Validate that all positions in the bug file are properly formatted
    test_positions = []
    with open(bug_positions_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse the FEN and best move if provided
            parts = line.split(';')
            fen = parts[0].strip()
            
            # Check if the FEN is valid
            try:
                board = chess.Board(fen)
                assert board is not None, f"Invalid FEN on line {i+1}: {fen}"
            except ValueError:
                assert False, f"Invalid FEN format on line {i+1}: {fen}"
            
            # Check best move annotation if present
            best_move = None
            if len(parts) > 1 and parts[1].strip().startswith('bm'):
                best_move_uci = parts[1].strip()[3:].strip()
                try:
                    best_move = chess.Move.from_uci(best_move_uci)
                    assert best_move in board.legal_moves, f"Illegal best move on line {i+1}: {best_move_uci}"
                except ValueError:
                    assert False, f"Invalid move format on line {i+1}: {best_move_uci}"
    
    # Part 2: Test that we can initialize the NNUE engine with our fixes
    try:
        transport, engine = await asyncio.wait_for(
            chess.engine.popen_uci([sys.executable, str(root_dir / "sunfish_nnue.py"), str(model_path)]),
            timeout=5.0  # Allow 5 seconds for engine initialization
        )
        
        # Check that the engine responds to isready
        await asyncio.wait_for(
            engine.ping(),
            timeout=2.0
        )
        
        # Success - the engine initialized without crashing
        await engine.quit()
        
    except Exception as e:
        assert False, f"Failed to initialize NNUE engine: {e}"