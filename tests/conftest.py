#!/usr/bin/env python3
import os
import pytest
import pytest_asyncio
import chess
import chess.engine
import asyncio
import sys
import contextlib
from typing import AsyncGenerator
from pathlib import Path

# Add the parent directory to the path so we can import sunfish
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set chess engine event loop policy globally
asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())

@contextlib.asynccontextmanager
async def get_engine() -> AsyncGenerator[chess.engine.UciProtocol, None]:
    """Context manager for creating and cleaning up chess engines"""
    # Determine path to the sunfish engine script relative to this file
    engine_path = Path(__file__).parent.parent / "sunfish.py"
    # Launch the engine using the current Python interpreter
    transport, engine = await chess.engine.popen_uci([sys.executable, str(engine_path)])
    try:
        yield engine
    finally:
        await engine.quit()

@pytest_asyncio.fixture
async def engine():
    """Create the chess engine for testing using pytest-asyncio"""
    async with get_engine() as engine:
        yield engine