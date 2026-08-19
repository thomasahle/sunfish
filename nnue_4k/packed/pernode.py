"""TREE-INDEPENDENT per-node profile: what an evaluator actually costs per node.

Why not a search: two engines whose EVALUATIONS differ search different trees,
so a whole-search nps ratio silently mixes "this eval is slower" with "this eval
steers into cheaper nodes".  Every arm here walks the IDENTICAL deterministic
sequence of 4,096 (position, move) pairs and is timed on `pos.move(mv)` and
nothing else, so the work is fixed by construction and the DELTAS are the
evaluator's true per-node price.

Absolutes are box- and interpreter-specific and must never be quoted against a
figure measured elsewhere.  Only the deltas within one run decide anything.

The reading that produced the 2026-08-19 ledger entry "THE EVALUATION TAX,
DECOMPOSED" (development laptop, pypy3 7.3.23, min-of-40 x3 replicates):

    pst_entry.py                          0.245 - 0.325 us/node
    + incremental packed accumulator      0.498 - 0.613
    + full N=32 read-out                  1.235 - 1.323
    sunfish_nnue.py, same net             2.860 - 2.997
    accumulator rebuilt every node        3.163 - 3.190

i.e. incrementality is already worth 1.89 us/node, and the CARRIER is worth
1.65 -- more than the evaluator it carries.

usage: pernode.py ENGINE.py [ENGINE.py ...]      (SF_NET set for net engines)
"""
import importlib.util, os, random, sys, time

NPAIRS = int(os.environ.get("NPAIRS", "4096"))
REPS = int(os.environ.get("REPS", "40"))
SEED = int(os.environ.get("SEED", "20260819"))


def load(path):
    name = "arm_" + os.path.basename(path).replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pairs(M):
    """The same walk for every arm: same seed, same sorted move list, same
    board -- so `move` is called on identical inputs in identical order."""
    random.seed(SEED)
    out, pos = [], M.hist[0]
    while len(out) < NPAIRS:
        ms = sorted(tuple(m) for m in pos.gen_moves())
        if not ms:
            random.seed(SEED + len(out))
            pos = M.hist[0]
            continue
        mv = ms[random.randrange(len(ms))]
        out.append((pos, mv))
        pos = pos.move(mv)
        if abs(pos.score) > 30000:
            pos = M.hist[0]
    return out


def main():
    arms = sys.argv[1:]
    if not arms:
        raise SystemExit(__doc__)
    print("per-node profile: %d fixed (pos, move) pairs, min of %d reps "
          "(ABSOLUTES are box-specific; the DELTAS decide)" % (NPAIRS, REPS))
    base = None
    for path in arms:
        M = load(path)
        P = pairs(M)
        for _ in range(3):                            # JIT warm-up
            for pos, mv in P:
                pos.move(mv)
        best = min(_time(P) for _ in range(REPS))
        us = best / len(P) * 1e6
        if base is None:
            base = us
        print("  %-26s %8.3f us/node   delta %+7.3f us  (%.2fx)"
              % (os.path.basename(path), us, us - base, us / base))


def _time(P):
    t0 = time.perf_counter()
    for pos, mv in P:
        pos.move(mv)
    return time.perf_counter() - t0


if __name__ == "__main__":
    main()
