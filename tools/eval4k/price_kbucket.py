"""Price king-BUCKETED table sets on the real generator. Bytes only, no Elo.

The ledger carries a ~134 B/bucket estimate. It was arithmetic on a codec
figure, never a build, so this measures it: N table sets decoded at startup and
selected once at the root, packed, measured off disk, run alone in an empty
directory.

WHAT A KING BUCKET CAN BE IN THIS ENGINE, AND WHAT IT CANNOT
------------------------------------------------------------
The entry's score is INCREMENTAL and both sides read one shared `pst` (black
through `119 - i`). A per-side own-king bucket would change the table when a
king moves, invalidating every carried score in the tree -- that is not a
pricing question, it is a different engine.

What IS available for free is the mechanism the `kend` fix already uses: a
POSITION-GLOBAL property, read once at the root, fixed for the whole search.
So the buckets here are king-WING buckets:

    4 buckets   white king wing x black king wing
    2 buckets   kings on the same wing / on opposite wings (a fold of the 4)

Both are a `pst.update` on an index computed from the root board -- the same
shape as the queens-off swap, so no new per-node cost and no second
accumulator. Whether they carry Elo is not asked here.

The bucket tables are FILLER (see price_taper.fillers), so these are SHAPE
prices. The taper cross-check measured filler at ~60-75 bytes CHEAPER than
real fitted data of the same shape, because fitted values are less round than
classic's; read these as a floor for a fitted version, not a ceiling.

usage: price_kbucket.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec  # noqa: E402
import measure  # noqa: E402
import price_taper as pt  # noqa: E402
import splice  # noqa: E402

# Bucket 0 is always `pst1`, the set the decoder already wrote into `pst`.
ROOT_KB2 = '''        pst.update(pst1 if (pos.board.index("K") % 10 > 5)
                   == (pos.board.index("k") % 10 > 5) else KB2)
        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END
'''

ROOT_KB4 = '''        pst.update((pst1, KB2, KB3, KB4)[(pos.board.index("K") % 10 > 5) * 2
                                          + (pos.board.index("k") % 10 > 5)])
        pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END
'''


def main():
    base = open(os.path.join(pt.ROOT, splice.ENTRY)).read()
    _, cur = measure.pack(base, "cur")
    piece, raw = splice.classic_tables()
    print("entry as landed: %d bytes, %d spare" % (cur, 4096 - cur))
    print("bucket data is FILLER -- SHAPE prices, a FLOOR for a fitted version. No Elo claimed.\n")

    rows = [(1, False, True), (8, False, True), (8, True, True)]
    print("one-set references (classic tables, root unchanged):")
    ref = {}
    for step, half, kexact in rows:
        src = splice.splice(base, codec.emit(piece, raw, step, half, exact="K" if kexact else ""))
        _, size = measure.pack(src, "ref")
        ref[(step, half, kexact)] = size
        print("  step %d%s%-12s %6d bytes" % (step, ", mirrored" if half else "",
                                              ", K exact" if kexact else "", size))
    print()

    print("%-10s %-24s %-10s %6s %7s %8s %9s %10s  %s"
          % ("buckets", "encoding", "filler", "bytes", "spare", "vs entry", "vs 1-set",
             "per bucket", "standalone"))
    last = None
    for nb, rootcode in ((2, ROOT_KB2), (4, ROOT_KB4)):
        for step, half, kexact in rows:
            enc = "step %d%s%s" % (step, ", mirrored" if half else "", ", K exact" if kexact else "")
            for fname in ("same", "perturbed", "shuffled"):
                # one filler set per EXTRA bucket, each with its own seed so the
                # buckets are genuinely distinct data
                egs = [("KB%d" % (i + 2), pt.fillers(raw, 20260813 + i)[fname])
                       for i in range(nb - 1)]
                src = pt.build(base, egs, rootcode, step, half, kexact, valline=False)
                packed, size = measure.pack(src, "kb%d" % nb)
                bm = measure.standalone(packed)
                extra = size - ref[(step, half, kexact)]
                print("%-10d %-24s %-10s %6d %7d %+8d %+9d %10.1f  %s"
                      % (nb, enc, fname, size, 4096 - size, size - cur, extra,
                         extra / float(nb - 1), bm))
                last = src
        print()
    print("decode of the largest build: %.2f ms (startup budget 60 s)"
          % measure.decode_time(last, "kb"))


if __name__ == "__main__":
    main()
