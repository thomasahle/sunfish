"""Emit a random packed net of a given width.

Weights are drawn to have realistic magnitudes, not to play chess: this is
the artefact used to measure nps and to verify the packed arithmetic before
any training has happened.

usage: make_random_net.py OUT.pickle [N] [--relu]
"""
import sys, random, pnet

out = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else 64
crelu = "--relu" not in sys.argv
random.seed(1234)

W = [{p: [0.0] * 120 for p in pnet.PIECES} for _ in range(N)]
for k in range(N):
    for p in pnet.PIECES:
        col = W[k][p]
        for s in pnet.SQUARES:
            col[s] = random.gauss(0, 0.06)
bias = [random.gauss(0.2, 0.1) for _ in range(N)]
v = [random.gauss(0, 12.0) for _ in range(N)]

shift, worst, sabs = pnet.pick_shift(W, bias, v)
d = pnet.build(W, bias, v, shift, clampcp=600)
pnet.save(out, d)
print("N=%d lanes=%d shift=%d (C=%d) worst |v|*excursion=%.1f sum|v|=%.0f max lane excursion=%d"
      % (N, 2 * N, shift, 1 << shift, worst, sabs, d["excursion"]))
print("wrote", out)
