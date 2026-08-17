"""Price the FITTED taper -- the real distilled endgame set, not filler.

filler is a floor, never a ceiling (price_taper's own header says so and the
qseam cross-check measured fitted data 60-75 B DEARER than filler at the same
shape). So the shape prices there do not settle whether this lands; this does.

Two roots (queens-off seam, continuous 24-point blend) x the delta step sweep,
on the mean-CONSTRAINED fit, which is the arm that won held-out.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codec, measure, splice                       # noqa: E402
import price_taper as pt                            # noqa: E402

FIT = sys.argv[1]
ARM = sys.argv[2] if len(sys.argv) > 2 else "constrained"
doc = json.load(open(FIT))[ARM]
piece, raw = splice.classic_tables()
# the fit emits absolute eg values INCLUDING the piece value; price_taper's
# builders want raw tables in classic's frame (piece value NOT included)
eg_raw = {p: [v - piece[p] for v in doc["eg"][p]] for p in "PNBRQ"}
eg_raw["K"] = list(raw["K"])

base = open(splice.ENTRY).read()
_, cur = measure.pack(base, "cur")
print("entry as landed: %d bytes, %d spare" % (cur, 4096 - cur))
print("fit arm: %s   (mg is classic's, byte-identical; only eg is fitted)\n" % ARM)

print("%-9s %-6s %6s %6s %7s %8s  %s"
      % ("root", "dstep", "levels", "bytes", "spare", "vs entry", "standalone"))
for name, rootcode in (("seam", pt.ROOT_SEAM), ("blend", pt.ROOT_BLEND)):
    for dstep in (1, 4, 8, 12, 16, 25, 32):
        region, lvl = pt.deltaregion(raw, piece, eg_raw, 1, False, dstep)
        src = splice.splice(base, region)
        assert pt.ROOT_OLD in src, "landed pend seam not found"
        src = src.replace(pt.ROOT_OLD, rootcode, 1)
        packed, size = measure.pack(src, "fittap")
        bm = measure.standalone(packed)
        flag = "" if size <= 4096 else "   OVER"
        print("%-9s %-6d %6d %6d %7d %+8d  %s%s"
              % (name, dstep, lvl, size, 4096 - size, size - cur, bm, flag))
    print()
