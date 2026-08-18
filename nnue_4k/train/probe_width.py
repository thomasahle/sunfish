"""nn_cp + accumulator-delta cost vs lane count N, on the target interpreter.

The engine's own arithmetic, parameterised by N.  Search-free: this times the
PRIMITIVES, which is the width-isolated form the ledger already prefers for
the N=6 estimate (the direct engine measurement is confounded by search shape).
"""
import random, sys, time

CLAMP = 600

def build(N):
    half = 16 * N
    U = ((1 << 2 * half) - 1) // 65535
    MH, MVAL, MLO = U << 15, U * 32767, U << 14
    g = [37] * N
    MGP = 0
    for k in range(N):
        MGP += g[k] * 32 << 16 * k
    MGP *= (1 | 1 << half)
    MGH = MGP | MH
    return half, MH, MVAL, MLO, MGP, MGH

def make_nn_cp(N):
    half, MH, MVAL, MLO, MGP, MGH = build(N)
    HALFP = 1 << half
    def nn_cp(acc, pf, SHIFT=8):
        m = ((acc & MLO) >> 14) * 32767
        y = ((acc & m) | MLO) - MLO
        m = (((MGH - y) & MH) >> 15) * 32767
        y = (y & m) | (MGP & (m ^ MVAL))
        v = y % HALFP % 65535 - (y >> half) % 65535
        if pf:
            v = -v
        return max(-CLAMP, min(CLAMP, int(v / (1 << SHIFT))))
    return nn_cp

def timeit(fn, reps):
    t0 = time.perf_counter()
    fn(reps)
    return (time.perf_counter() - t0) / reps * 1e9   # ns per rep

rng = random.Random(7)
print("%4s %7s %12s %12s %12s %12s" % ("N", "bits", "readout_ns", "delta_ns", "combined", "vs N=4"))
base = None
for N in (4, 5, 6, 8, 12, 16, 24, 32, 48, 64):
    half, MH, MVAL, MLO, MGP, MGH = build(N)
    nn_cp = make_nn_cp(N)
    # a plausible accumulator: offsets + a few dozen rows added in
    rows = [rng.getrandbits(2 * half) & ((1 << (2 * half)) - 1) for _ in range(64)]
    acc = MLO
    for i in range(32):
        acc += rows[i] >> 6
    reps = max(2000, int(200000 / N))
    def run_readout(n, acc=acc, nn_cp=nn_cp):
        s = 0
        for i in range(n):
            s += nn_cp(acc + i, i & 1)
        return s
    def run_delta(n, acc=acc, rows=rows):
        a = acc
        for i in range(n):
            a = a + rows[i & 63] - rows[(i + 7) & 63]
        return a
    for _ in range(2):            # warm the jit
        run_readout(reps); run_delta(reps)
    r = min(timeit(run_readout, reps) for _ in range(3))
    d = min(timeit(run_delta, reps) for _ in range(3))
    c = r + d
    if base is None:
        base = c
    print("%4d %7d %12.1f %12.1f %12.1f %12.2fx" % (N, 2 * half, r, d, c, c / base))
