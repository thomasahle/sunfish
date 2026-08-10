"""Regression tests for the killer-table eviction race and the in-search
deadline, ported from classic (lichess cy7H1Mhx, 26. Qxc6??).

Mechanism: tp_move persists across searches and is FIFO-capped. When a
single deep probe stores more than TABLE_SIZE distinct killers, the
current search root's entry ages out MID-SEARCH. MTD-bi's next fail-low
dive then probes at an absurdly low gamma, the sorted loop leads with the
highest capture, it trivially fails high, is stored as the root move -
and a timeout answers it.

Fixes under test: (1) eviction never removes Searcher.root, so the root
killer survives arbitrary churn; (2) an armed deadline actually stops the
search from inside (the builtin go loop arms it; before, the apparatus
was dead code and a deep iteration could overrun the soft break
arbitrarily).
"""
import importlib.util
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_engine(table_size):
    os.environ["SF_NET"] = str(ROOT / "packed" / "net128.pickle")
    spec = importlib.util.spec_from_file_location(
        "sunfish_packed_race", ROOT / "sunfish_packed.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sunfish_packed_race"] = mod
    spec.loader.exec_module(mod)
    mod.TABLE_SIZE = table_size
    return mod


# The production game up to the position before 26. Qxc6??
GAME = (
    "g1f3 g8f6 d2d4 e7e6 b1c3 f8b4 c1f4 e8g8 e2e3 f6d5 d1d2 c7c6 f1d3 h7h6 "
    "e1g1 d5f4 e3f4 d7d5 a2a3 b4d6 a1d1 b8d7 f1e1 d7b6 f3e5 d8f6 h2h3 c8d7 "
    "e5d7 b6d7 c3e2 c6c5 c2c3 c5c4 d3b1 a8e8 d2c2 g7g6 c2a4 d7b8 a4a7 e8e7 "
    "g2g3 b8c6 a7b6 g6g5 f4g5 h6g5 b1c2 d6b8"
).split()


def build_history(mod):
    hist = [mod.hist[0]]
    for ply, mv in enumerate(GAME):
        i, j, prom = mod.parse(mv[:2]), mod.parse(mv[2:4]), mv[4:].upper()
        if ply % 2 == 1:
            i, j = 119 - i, 119 - j
        hist.append(hist[-1].move(mod.Move(i, j, prom)))
    return hist


def test_root_killer_survives_table_churn():
    """Once stored, the root's tp_move entry must be present at every
    subsequent driver yield - even when the table churns many times its
    capacity within single probes (TABLE_SIZE is tiny here to force the
    production regime cheaply)."""
    mod = load_engine(table_size=300)
    hist = build_history(mod)
    root = hist[-1]
    searcher = mod.Searcher()
    seen_root_entry = False
    evictions = 0
    for depth, gamma, score, move in searcher.search(hist):
        if seen_root_entry:
            assert root in searcher.tp_move, (
                f"root killer evicted mid-search at depth {depth} "
                f"(table len {len(searcher.tp_move)}) - the Qxc6 race is open"
            )
        elif root in searcher.tp_move:
            seen_root_entry = True
        if len(searcher.tp_move) >= mod.TABLE_SIZE:
            evictions += 1
        if depth == 6:
            break
    assert seen_root_entry, "test never observed a root store - not exercising the race"
    assert evictions > 0, "table never reached capacity - not exercising the race"


def test_no_capture_giveaway_at_dive_windows():
    """End-to-end: at the production position, with heavy churn, the move
    stored for the root must never become the unprotected b6c6 (Qxc6)
    giveaway."""
    mod = load_engine(table_size=300)
    hist = build_history(mod)
    root = hist[-1]
    searcher = mod.Searcher()
    for depth, gamma, score, move in searcher.search(hist):
        killer = searcher.tp_move.get(root)
        if killer is not None:
            uci = mod.render(killer.i) + mod.render(killer.j)
            assert uci != "b6c6", (
                f"Qxc6 stored as root move at depth {depth}, gamma {gamma} - "
                "a timeout here plays the queen giveaway"
            )
        if depth == 6:
            break


def test_armed_deadline_stops_search():
    """With a deadline in the past the search must raise Stop within a
    bounded number of driver yields (the %2048 node check), not run on
    to arbitrary depth - the builtin go loop relies on this to avoid
    the deep-iteration time-forfeit class."""
    mod = load_engine(table_size=10**6)
    hist = build_history(mod)
    searcher = mod.Searcher()
    searcher.deadline = time.time() - 1
    gen = searcher.search(hist)
    raised = False
    try:
        for _ in range(200):
            next(gen)
    except mod.Stop:
        raised = True
    assert raised, "armed past deadline did not stop the search in 200 yields"
