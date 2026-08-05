#!/usr/bin/env python3
import os
import pytest
import pytest_asyncio
import chess
import chess.engine
import asyncio
import time
from pathlib import Path
# Using engine fixture from conftest.py

# Helper functions
async def engine_finds_mate(engine, fen_file, movetime=10000, limit=None):
    """Test if the engine can find a mate in the given position"""
    test_files_dir = Path(__file__).parent.parent / "tools" / "test_files"
    with open(test_files_dir / fen_file, 'r') as f:
        lines = f.readlines()
        
    if limit:
        lines = lines[:limit]
        
    success = 0
    total = len(lines)
    
    for line in lines:
        # Parse the FEN string (the first part of the line)
        board = chess.Board(line.strip())
        limit = chess.engine.Limit(time=movetime/1000)
        
        with await engine.analysis(board, limit) as analysis:
            mate_found = False
            async for info in analysis:
                if not "score" in info:
                    continue
                score = info["score"]
                if score.is_mate() or score.relative.cp > 10000:
                    if "pv" in info and info["pv"]:
                        b = board.copy()
                        for move in info["pv"]:
                            b.push(move)
                        if b.is_game_over():
                            mate_found = True
                            success += 1
                            break
            
    return success, total

@pytest.mark.asyncio
async def test_mate_in_one(engine):
    """Test if the engine can find mate in 1"""
    success, total = await engine_finds_mate(engine, "mate1.fen", limit=3)  # Limit to 3 positions
    if total > 0:
        assert success == total, f"Only found {success}/{total} mates in 1"

@pytest.mark.asyncio
async def test_mate_in_two(engine):
    """Test if the engine can find mate in 2"""
    success, total = await engine_finds_mate(engine, "mate2.fen", limit=3)  # Limit to 3 positions
    if total > 0:
        assert success == total, f"Only found {success}/{total} mates in 2"

@pytest.mark.asyncio
async def test_mate_in_three(engine):
    """Test if the engine can find mate in 3"""
    success, total = await engine_finds_mate(engine, "mate3.fen", limit=2)  # Limit to 2 positions
    if total > 0:
        assert success == total, f"Only found {success}/{total} mates in 3"