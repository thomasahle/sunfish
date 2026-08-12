"""The MTD driver must survive an UNSTABLE search.

The packed engine deliberately gives up one-value-per-key: reductions key
off move index, history is mutable global state, and gamma-dependent
cutoffs let two probes of the same position disagree. MTD-bi has no real
window to absorb that -- it only probes null windows and bisects on the
answers -- so a contradiction can cross the bracket (lower > upper) and
either spin forever or return nonsense.

Instability is fine. Nontermination and garbage are not. These tests
assert the driver's guards hold even when bound() is actively lying, and
they run with a WARM table, which is when contradictions actually appear.
"""
import importlib.util
import os
import random
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(os.path.dirname(HERE), "sunfish_nnue.py")
NET = os.path.join(os.path.dirname(HERE), "net128kb8.sfnn")


def load():
    os.environ.setdefault("SF_NET", NET)
    spec = importlib.util.spec_from_file_location("sunfish_mtd", ENGINE)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sunfish_mtd"] = m
    spec.loader.exec_module(m)
    return m


sunfish = load()

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def board120(fen_board):
    import re
    b = re.sub(r"\d", lambda m: "." * int(m.group(0)), fen_board)
    b = list(21 * " " + "  ".join(b.split("/")) + 21 * " ")
    b[9::10] = ["\n"] * 12
    return "".join(b)


def position(fen):
    board, color, cast, ep = fen.split()[:4]
    wc = ("Q" in cast, "K" in cast)
    bc = ("k" in cast, "q" in cast)
    e = sunfish.parse(ep) if ep != "-" else 0
    pos = sunfish.from_board(board120(board), wc, bc, e, 0)
    return pos.rotate() if color == "b" else pos


def run(searcher, pos, max_depth=4, budget=25.0):
    """drive the generator like the real consumers do; return per-depth probes"""
    searcher.deadline = time.time() + budget
    probes, last = {}, None
    for depth, gamma, score, move in searcher.search([pos]):
        probes[depth] = probes.get(depth, 0) + 1
        last = (depth, score)
        if depth > max_depth:
            break
        assert time.time() < searcher.deadline + 5, "driver overran its deadline badly"
    return probes, last


@pytest.mark.parametrize("fen", FENS)
def test_terminates_with_warm_table(fen):
    """Second search on a warm table must still terminate per depth."""
    pos = position(fen)
    s = sunfish.Searcher()
    run(s, pos)
    probes, last = run(s, pos)          # warm: tp_move/tp_score populated
    assert last is not None
    for depth, n in probes.items():
        assert n <= sunfish.PROBE_CAP + 1, (
            "depth %d used %d probes, cap is %d" % (depth, n, sunfish.PROBE_CAP))


def test_survives_a_lying_bound():
    """Adversarial: make bound() contradict itself and demand termination.

    This is the failure mode the guards exist for -- a bound that returns
    ">= gamma" and "< gamma" for the same position at different gammas.
    Without monotone tightening the bracket oscillates and the loop never
    exits; without the probe cap it spins.
    """
    pos = position(FENS[1])
    s = sunfish.Searcher()
    rng = random.Random(7)
    real = s.bound

    def lying_bound(p, gamma, depth, root=False):
        v = real(p, gamma, depth, root=root)
        if root and rng.random() < 0.5:        # contradict half the probes
            return gamma + rng.choice([-40, 40])
        return v

    s.bound = lying_bound
    s.deadline = time.time() + 30
    seen = {}
    t0 = time.time()
    for depth, gamma, score, move in s.search([pos]):
        seen[depth] = seen.get(depth, 0) + 1
        assert seen[depth] <= sunfish.PROBE_CAP + 1, (
            "unstable search exceeded the probe cap at depth %d" % depth)
        if depth > 3 or time.time() - t0 > 25:
            break
    assert seen, "driver produced nothing under instability"


def test_guards_are_present():
    """The guards are load-bearing; a refactor must not quietly drop them."""
    src = open(ENGINE).read()
    assert "PROBE_CAP" in src, "probe cap missing"
    assert "lower = max(lower, score)" in src, "monotone lower tightening missing"
    assert "upper = min(upper, score)" in src, "monotone upper tightening missing"
    assert "lower > upper" in src, "bracket-crossing guard missing"
