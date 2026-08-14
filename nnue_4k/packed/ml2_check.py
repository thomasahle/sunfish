#!/usr/bin/env python3
"""ml2 pricing-build verifier: independent payload decode + the
packed_layers int-bridge (the certified training twin, used verbatim) as
reference against the engine's nn_cp, over FEN probes and a 40-ply walk.

usage: ml2_check.py [SPLICED_ML2_ENTRY]
With no argument it derives the variant from ../replnet_proto.py via
make_ml2_proto.py, splices a real-shaped --u2 4 payload, and checks that.
"""
import importlib.util
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", ".."))          # repo root, for the fen file
sys.path.insert(0, "nnue_4k/train")
import packed_layers as PL                         # noqa: E402

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    built = subprocess.run([sys.executable, os.path.join(HERE, "make_ml2_proto.py")],
                           capture_output=True, text=True, check=True).stdout.strip()
    pay = subprocess.run([sys.executable, os.path.join(HERE, "make_proto_payload.py"),
                          "--u2", "4", "--zeros", "0.596"],
                         capture_output=True, text=True, check=True).stdout.strip()
    src = open(built).read()
    m = re.search(r'^for _c in "(.*)":$', src, re.M)
    path = built[:-3] + "_spliced.py"
    open(path, "w").write(src[:m.start(1)] + pay + src[m.end(1):])
spec = importlib.util.spec_from_file_location("ml2e", path)
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)

# --- independent decode (own code path, not the engine's)
src = open(path).read()
pay = re.search(r'^for _c in "(.*)":$', src, re.M).group(1)
w = 0
for c in pay:
    d = ord(c) - 35; w = w * 90 + d - (d > 4) - (d > 56)
w, shift = divmod(w, 90)
g = []
for _ in range(4): w, d = divmod(w, 90); g.append(d)
bd = []
for _ in range(4): w, d = divmod(w, 90); bd.append(d - 44)
u2 = []
for _ in range(4): w, d = divmod(w, 8100); u2.append(d - 4050)
trits = {}
for p in e._PIECES:
    for f in range(64):
        w, d = divmod(w, 90)
        trits[p, 21 + f // 8 * 10 + f % 8] = [d // 3 ** k % 3 - 1 for k in range(4)]
assert u2 == e.U2, (u2, e.U2)
assert shift == e.SHIFT

def ref(board, pf):
    us, them = list(bd), list(bd)
    for i, p in enumerate(board):
        if p.isalpha():
            for k in range(4):
                us[k] += g[k] * trits[p, i][k]
                them[k] += g[k] * trits[p.swapcase(), 119 - i][k]
    # pf selects the engine's block layout only -- the evaluated value is
    # pf-independent (the flip and the layout cancel), so the reference
    # ignores it (verify_export asserts the same r0 == r1 property).
    caps = [32 * x for x in g]
    yu = [min(max(v, 0), c) for v, c in zip(us, caps)]
    yt = [min(max(v, 0), c) for v, c in zip(them, caps)]
    v1 = sum(yu) - sum(yt)
    ha = PL.bigint_circular_conv(yu, yu, 4, 32)   # the certified twin, verbatim
    hb = PL.bigint_circular_conv(yt, yt, 4, 32)
    w2 = sum(u2[k] * (ha[k] - hb[k]) for k in range(4))
    v = int(v1 / (1 << shift)) + int(w2 / 1024)
    return max(-e.CLAMP, min(e.CLAMP, v))

def probe_boards(n):
    boards = []
    for line in open("nnue_4k/packed/shapecheck_fens.txt"):
        parts = line.split()
        if not parts: continue
        b = parts[0]
        b = re.sub(r"\d", lambda m: "." * int(m.group(0)), b)
        rows = b.split("/")
        board = "".join(" " * 21) + "  ".join(rows) + " " * 21
        board = list(board)
        board[9::10] = ["\n"] * 12
        board = "".join(board)
        if len(parts) > 1 and parts[1] == "b":
            board = board[::-1].swapcase()
        boards.append(board)
        if len(boards) >= n: break
    return boards

fired = 0
for i, board in enumerate(probe_boards(60)):
    pos0 = e.from_board(board, pf=0)
    got0 = e.nn_cp(pos0.acc, pos0.pf)
    assert got0 == ref(board, 0), ("fen", i, got0, ref(board, 0))
    pos1 = e.from_board(board, pf=1)
    got1 = e.nn_cp(pos1.acc, pos1.pf)
    assert got1 == got0, ("pf changed the eval", i, got0, got1)
    rr = e.nn_cp(e.from_board(board[::-1].swapcase(), pf=0).acc, 0)
    assert rr == -got0, ("antisymmetry", i, got0, rr)
    fired += got0 != 0

random.seed(20260814)
pos = e.from_board(e.initial)
assert pos.ps == 0 and pos.score == e.nn_cp(pos.acc, pos.pf)
for step in range(40):
    moves = [m for m in pos.gen_moves() if not pos.move(m).k()]
    if not moves: break
    pos = pos.move(random.choice(moves))
    fresh = e.from_board(pos.board, pos.wc, pos.bc, pos.ep, pos.kp, pos.pf)
    assert (pos.acc, pos.ps, pos.score) == (fresh.acc, fresh.ps, fresh.score), step
    assert e.nn_cp(pos.acc, pos.pf) == ref(pos.board, pos.pf), (step, "walk-ref")
    r = pos.rotate()
    fr = e.from_board(r.board, r.wc, r.bc, r.ep, r.kp, r.pf)
    assert fr.score == -pos.score, (step, "antisymmetry")
    fired += e.nn_cp(pos.acc, pos.pf) != 0
assert fired
print("ml2_check: independent decode == engine (u2/shift), engine == "
      "packed_layers int-bridge BIT-EXACT on 60 fens x 2 pf + rotations "
      "+ 40-ply walk (incremental/antisymmetry) PASS; L2 fired on %d probes" % fired)
