"""Price the READ-OUT against accumulator width, the way the ledger did at N=4/5/6.

The ACCUMULATOR PRICING entry measured three widths and found the read-out
`nn_cp` is ~92 % of the net's per-move cost and the only part that grows.  The
factored design buys UNITS, and units are lanes, so the width tax is the price
of the whole idea -- and three points at N=4/5/6 cannot be extrapolated to
N=32.  This measures the same two primitives out to the widths a factored
payload can actually afford.

Instrument: whatever interpreter runs it (report it), one built variant per
width from make_factor_proto.py, timed on a real accumulator and a real row
table.  Relative numbers within one run are the deliverable; absolute ns are
machine-specific and are NOT comparable with the box's.

    pypy3 price_width.py [--widths 4,5,6,16,32,48] [--reps 200000]
"""
import argparse
import importlib.util
import os
import platform
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_factor_proto as M                                    # noqa: E402


def load(r, N, lane_bits, zeros):
    with open(os.path.join(HERE, "..", "replnet_proto.py")) as f:
        src = f.read()
    out, _, _ = M.build(src, r, N, lane_bits, zeros)
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(out)
    spec = importlib.util.spec_from_file_location("w%d" % N, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    os.unlink(path)
    return m


def bench(m, reps, rng):
    """delta = one incremental accumulator update; readout = one nn_cp."""
    rows = m.ROWS[0]
    pcs = list(m._PIECES)
    sqs = [21 + f // 8 * 10 + f % 8 for f in range(64)]
    # a plausible accumulator: the empty-board base plus 32 men
    acc = m._B
    for _ in range(32):
        acc += rows[rng.choice(pcs)][rng.choice(sqs)]
    moves = [(rng.choice(pcs), rng.choice(sqs), rng.choice(sqs))
             for _ in range(1024)]

    def timeit(fn, n):
        best = None
        for _ in range(3):                       # 3 stable repeats, take min
            t = time.perf_counter()
            fn(n)
            dt = time.perf_counter() - t
            best = dt if best is None else min(best, dt)
        return best / n * 1e9

    def do_delta(n):
        a = acc
        for i in range(n):
            p, j, k = moves[i & 1023]
            a = a + rows[p][j] - rows[p][k]
        return a

    def do_read(n):
        s = 0
        f = m.nn_cp
        for i in range(n):
            s += f(acc, 0)
        return s

    def do_loop(n):                              # the empty loop, subtracted
        s = 0
        for i in range(n):
            s += i
        return s

    over = timeit(do_loop, reps)
    return timeit(do_delta, reps) - over, timeit(do_read, reps) - over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="4,5,6,8,16,24,32,48")
    ap.add_argument("--reps", type=int, default=300000)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--lane-bits", type=int, default=16)
    ap.add_argument("--zeros", type=float, default=0.43)
    a = ap.parse_args()
    rng = random.Random(20260817)
    print("%s %s on %s" % (platform.python_implementation(),
                           platform.python_version(), platform.machine()))
    print("%-5s %8s %9s %10s %9s" % ("N", "delta", "readout", "combined", "vs N=4"))
    base = None
    for N in [int(x) for x in a.widths.split(",")]:
        m = load(a.r, N, a.lane_bits, a.zeros)
        d, rd = bench(m, a.reps, rng)
        c = d + rd
        base = c if base is None else base
        print("%-5d %8.1f %9.1f %10.1f %8.1f%%" % (N, d, rd, c, 100 * (c / base - 1)))


if __name__ == "__main__":
    main()
