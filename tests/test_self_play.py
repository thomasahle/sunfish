#!/usr/bin/env python3
import os
import pytest
import pytest_asyncio
import chess
import chess.engine
import asyncio
import time
# Using engine fixture from conftest.py

@pytest.mark.asyncio
async def test_self_play(engine):
    """Test if the engine can complete a game without crashing"""
    # Just test a few moves to keep the test fast
    max_moves = 5  
    
    wtime = btime = 1.0  # seconds
    inc = 0.1  # seconds
    board = chess.Board()
    move_count = 0
    
    # Play a few moves
    while not board.is_game_over() and move_count < max_moves:
        limit = chess.engine.Limit(
            white_clock=wtime, 
            black_clock=btime, 
            white_inc=inc, 
            black_inc=inc
        )
        
        start = time.time()
        result = await engine.play(board, limit)
        elapsed = time.time() - start
        
        # Update clock
        if board.turn == chess.WHITE:
            wtime -= elapsed - inc
        else:
            btime -= elapsed - inc
            
        # Ensure we have a valid move
        assert result.move is not None, "Engine returned None move"
        
        # Make the move
        board.push(result.move)
        move_count += 1
    
    # If we get here without exceptions, the test passed
    assert move_count > 0, "No moves were played"