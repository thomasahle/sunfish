"""The packed head against a float reference of the same net.

`build()` is the only place where the float net becomes integers, so this
is the test that says how much the quantisation costs.  It reports the
centipawn error distribution over random legal positions, for the exact
same weights evaluated both ways.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pnet

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
NPOS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
NSEG = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SEGS = tuple(i / NSEG for i in range(NSEG))
random.seed(99)

W = [{p: [0.0] * 120 for p in pnet.PIECES} for _ in range(N)]
for k in range(N):
    for p in pnet.PIECES:
        for s in pnet.SQUARES:
            W[k][p][s] = random.gauss(0, 0.06)
bias = [random.gauss(0.2, 0.1) for _ in range(N)]
v = [random.gauss(0, 12.0) for _ in range(N)]

shift, worst, sabs = pnet.pick_shift(W, bias, v, segs=SEGS)
d = pnet.build(W, bias, v, shift, clampcp=10 ** 6, segs=SEGS)
net = pnet.PackedNet(d)
C = 1 << shift
print("N=%d segs=%s shift=%d C=%d rigorous excursion bound=%d" % (N, SEGS, shift, C, d["excursion"]))


def random_board(rng):
    """A random legal-ish placement: two kings plus a random army."""
    sq = list(pnet.SQUARES)
    rng.shuffle(sq)
    b = [" "] * 120
    for r in range(2, 10):
        for f in range(1, 9):
            b[r * 10 + f] = "."
        b[r * 10 + 9] = "\n"
    for r in (0, 1, 10, 11):
        for c in range(10):
            b[r * 10 + c] = " "
    b[9::10] = ["\n"] * 12
    for i in range(0, 10):
        b[i] = b[110 + i] = " "
    pool = ["K", "k"] + rng.sample(list("QRRBBNNPPPPPPPPqrrbbnnpppppppp"),
                                   rng.randrange(0, 26))
    for p in pool:
        b[sq.pop()] = p
    return "".join(b)


errs = []
rng = random.Random(7)
for _ in range(NPOS):
    board = random_board(rng)
    for pf in (0, 1):
        acc = net.from_board(board, pf)
        assert net.check_lanes(acc), "lane overflow on a random position"
        got = net.raw(acc, pf) / C
        want = pnet.float_nn(W, bias, v, board, pf, segs=SEGS)
        errs.append(got - want)
        # antisymmetry of the packed head
        assert net.raw(acc, pf) == -net.raw(acc, pf ^ 1)

errs.sort()
absmax = max(abs(e) for e in errs)
mean = sum(errs) / len(errs)
rms = (sum(e * e for e in errs) / len(errs)) ** 0.5
print("packed vs float over %d evaluations: mean %+.3f cp, rms %.3f cp, max |err| %.3f cp"
      % (len(errs), mean, rms, absmax))
print("p50 %+.3f  p95 %+.3f  p99 %+.3f (cp)"
      % (errs[len(errs) // 2], errs[int(len(errs) * .95)], errs[int(len(errs) * .99)]))
