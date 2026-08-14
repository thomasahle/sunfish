"""Regression tests for history-independent QSearch TT persistence."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sunfish as sf  # noqa: E402


def start(searcher, history):
    """Run enough of search() to execute its table initialization."""
    search = searcher.search(history)
    next(search)
    search.close()


def test_same_king_phase_keeps_only_qsearch_entries():
    pos = sf.Position(sf.initial, 0, (True, True), (True, True), 0, 0)
    qkey, deepkey = (pos, 0), (pos, 2)
    qentry, deepentry = sf.Entry(-12, 34), sf.Entry(-56, 78)
    searcher = sf.Searcher()
    searcher.king = sf.K_MID
    searcher.tp_score.update({qkey: qentry, deepkey: deepentry})
    searcher.tp_deep.add(deepkey)

    old_king = sf.pst["K"]
    try:
        sf.pst["K"] = sf.K_MID
        start(searcher, [pos])
    finally:
        sf.pst["K"] = old_king

    assert searcher.tp_score[qkey] == qentry
    assert deepkey not in searcher.tp_score
    assert not searcher.tp_deep


def test_king_phase_change_discards_qsearch_entries():
    pos = sf.Position(sf.initial, 0, (True, True), (True, True), 0, 0)
    qkey = pos, 0
    searcher = sf.Searcher()
    searcher.king = sf.K_END
    searcher.tp_score[qkey] = sf.Entry(-12, 34)

    old_king = sf.pst["K"]
    try:
        sf.pst["K"] = sf.K_END
        start(searcher, [pos])
    finally:
        sf.pst["K"] = old_king

    assert searcher.king == sf.K_MID
    assert qkey not in searcher.tp_score


def test_eviction_also_bounds_the_deep_key_tracker():
    pos = sf.Position(sf.initial, 0, (True, True), (True, True), 0, 0)
    searcher = sf.Searcher()
    searcher.root, searcher.history = pos, set()

    old_size = sf.TABLE_SIZE
    try:
        sf.TABLE_SIZE = 8
        searcher.bound(pos, 0, 4)
    finally:
        sf.TABLE_SIZE = old_size

    assert len(searcher.tp_score) <= 8
    assert searcher.tp_deep <= searcher.tp_score.keys()
