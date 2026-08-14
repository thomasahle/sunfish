#!/usr/bin/env python3
"""Invariant suite for packed128_compose_proto.py (the golf gate).

Run after EVERY golf step. Checks, on the prototype's own decoded rows:
  1. mirror identity: ROWS[1][p][s] == ROWS[0][swapcase(p)][119-s]
  2. 40-ply random walk: incremental acc == from-scratch acc,
     ps identity, score identity
  3. exact antisymmetry: eval(rotate(pos)) recomputed == -eval(pos)
  4. the net actually fires (nn_cp != 0 somewhere on the walk)
A step that breaks any of these is REVERTED, not patched around.
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import packed128_compose_proto as e

for p in e._PIECES:
    for s in range(120):
        assert e.ROWS[1][p][s] == e.ROWS[0][p.swapcase()][119 - s], (p, s)

random.seed(20260814)
pos = e.from_board(e.initial)
fired = e.nn_cp(pos.acc, pos.pf) != 0
for step in range(40):
    moves = [m for m in pos.gen_moves() if not pos.move(m).king_capture()]
    if not moves:
        break
    pos = pos.move(random.choice(moves))
    fresh = e.from_board(pos.board, pos.wc, pos.bc, pos.ep, pos.kp, pos.pf)
    assert pos.acc == fresh.acc, (step, "acc")
    assert pos.ps == fresh.ps, (step, "ps")
    assert pos.score == fresh.score, (step, "score")
    r = pos.rotate()
    fr = e.from_board(r.board, r.wc, r.bc, r.ep, r.kp, r.pf)
    assert fr.score == -pos.score, (step, "antisymmetry")
    fired = fired or e.nn_cp(pos.acc, pos.pf) != 0
assert fired, "net never fired -- rows are dead"
print("proto_check: mirror + %d-ply walk (acc/ps/score) + antisymmetry + nn-fires PASS" % (step + 1))
