"""Does the DELTA encoding make king-WING buckets affordable too?

The 2026-08-13 pricing built each extra bucket as a full independent decode
block and measured ~134 B/bucket for filler, 4 buckets = 4505 B = 409 OVER.
That pricing never tried storing bucket 1 as `bucket 0 + a coarse delta`,
which is what made the taper fit. Same question, same instrument.

Only the WING form is priced, because it is the only one available: a
per-side OWN-king bucket changes the table when a king moves and invalidates
every carried score in the tree (price_kbucket.py's header states this, and
the pst entry has no accumulator to rebuild -- make_pst_entry excises it).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec, measure, splice          # noqa: E402
import price_taper as pt               # noqa: E402
import price_kbucket as pk             # noqa: E402

base = open(splice.ENTRY).read()
_, cur = measure.pack(base, "cur")
piece, raw = splice.classic_tables()
fill = pt.fillers(raw)

ROOT_KB2 = ('        end = "Q" not in pos.board or "q" not in pos.board\n'
            '        pst.update(pst1 if (pos.board.index("K") % 10 > 5)\n'
            '                   == (pos.board.index("k") % 10 > 5) else EGT)\n'
            '        pst["K"] = K_END if end else K_MID\n'
            '        pst["P"] = P_END if end else P_MID\n')

print("entry as landed: %d bytes, %d spare" % (cur, 4096 - cur))
print("2 king-WING buckets, bucket 1 stored as bucket 0 + a coarse delta.")
print("%-12s %6s %6s %7s %8s  %s" % ("filler/dstep", "levels", "bytes", "spare", "vs entry", "standalone"))
for fname in ("perturbed", "shuffled"):
    for dstep in (8, 16, 25):
        region, lvl = pt.deltaregion(raw, piece, fill[fname], 1, False, dstep)
        src = splice.splice(base, region)
        assert pt.ROOT_OLD in src
        src = src.replace(pt.ROOT_OLD, ROOT_KB2, 1)
        packed, size = measure.pack(src, "kbd")
        bm = measure.standalone(packed)
        print("%-12s %6d %6d %7d %+8d  %s%s" % ("%s/%d" % (fname, dstep), lvl, size,
              4096 - size, size - cur, bm, "" if size <= 4096 else "  OVER"))
