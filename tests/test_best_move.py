#!/usr/bin/env python3
import os
import pytest
import chess
import chess.engine
import re
import asyncio
from pathlib import Path

@pytest.mark.asyncio
async def test_win_at_chess_puzzles(engine):
    """Test if the engine can find best moves in standard test positions"""
    test_files_dir = Path(__file__).parent.parent / "tools" / "test_files"
    with open(test_files_dir / "win_at_chess_test.epd", 'r') as f:
        lines = f.readlines()
    
    points = 0
    total = 0
    movetime = 100  # ms
    
    # Limit to just a few puzzles to keep tests fast
    for line in lines[:5]:  
        board, opts = chess.Board.from_epd(line)
        
        # Handle PV moves if present
        if "pv" in opts:
            for move in opts["pv"]:
                board.push(move)
        
        # Parse c0 comments which contain best/avoid move info
        if "c0" in opts:
            for key, val in re.findall(r"(\w+) (\w+)", opts["c0"]):
                opts[key] = [chess.Move.from_uci(val)]
        
        if "am" not in opts and "bm" not in opts:
            continue
            
        # Set time limit
        limit = chess.engine.Limit(time=movetime/1000)
        
        # Get engine's move
        result = await engine.play(board, limit, info=chess.engine.INFO_SCORE)
        
        # Check if it matches best move or avoids bad move
        if "bm" in opts:
            total += 1
            if result.move in opts["bm"]:
                points += 1
                
        if "am" in opts:
            total += 1
            if result.move not in opts["am"]:
                points += 1
    
    # For tests, we might accept a lower threshold than 100%
    # as these are harder puzzles that might need more time
    if total > 0:
        assert points >= total * 0.5, f"Only found {points}/{total} best moves"