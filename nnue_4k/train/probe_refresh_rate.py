"""Refresh-rate census: what fraction of GENERATED moves would force a
bucket switch (and therefore an accumulator rebuild) under each partition.

Every generated move calls Position.move(), which is where the incremental
accumulator update lives, so this fraction multiplied by the rebuild cost IS
the runtime price of a per-node bucketed design.
"""
import sys
import numpy as np
sys.path.insert(0, ".")
sys.path.insert(0, "..")
import features
import sunfish as classic

PH = {"P": 0, "N": 1, "B": 1, "R": 2, "Q": 4, "K": 0}
PHCUT = 11          # the measured median phase (ph2 cut)

def band(s):        # own-frame: 1 if the king is past its second rank
    return int(s // 10 <= 7)

d = np.load("pool10m.npz", allow_pickle=False)
fens = d["fens"]
step = max(1, len(fens) // 4000)
tot = kingmv = bandcross = cap = phcross = 0
npos = 0
for s in fens[::step][:4000]:
    f = s.decode() if isinstance(s, bytes) else s
    parts = f.split()
    board = features.fen_to_board120(parts[0])
    if parts[1] == "b":
        board = board[::-1].swapcase()
    pos = classic.Position(board, 0, (True, True), (True, True), 0, 0)
    ph0 = sum(PH[c.upper()] for c in board if c.isalpha())
    wk = board.index("K")
    b0 = band(wk)
    npos += 1
    for m in pos.gen_moves():
        i, j = m.i, m.j
        p, q = board[i], board[j]
        tot += 1
        if p == "K":
            kingmv += 1
            if band(j) != b0:
                bandcross += 1
        if q != "." and q.islower():
            cap += 1
            if (ph0 >= PHCUT) != (ph0 - PH[q.upper()] >= PHCUT):
                phcross += 1

print("positions            %d" % npos)
print("generated moves      %d   (%.1f per position)" % (tot, tot / npos))
print("king moves           %.4f of generated" % (kingmv / tot))
print("  band-crossing      %.4f of generated   <- kb2 refresh rate" % (bandcross / tot))
print("captures             %.4f of generated" % (cap / tot))
print("  phase-cut-crossing %.4f of generated   <- ph2 refresh rate" % (phcross / tot))
