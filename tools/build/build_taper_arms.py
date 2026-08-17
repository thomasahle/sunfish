"""Generate the taper arms from ONE source, at screen time (make_variants' rule).

Arms, all built from nnue_4k/pst_entry.py @ the lane's baseline:
  tap    fitted eg set for PNBRQ, blended on the 24-point phase.
         The KING keeps the landed queens-off cliff, so `tap` vs `ktap`
         attributes the king ramp separately from the table set.
  tapk   the same, plus the king table on the same ramp -- i.e. `tap`
         composed with `ktap`.

BOTH SUBSUME pend. The fitted eg pawn table replaces P_END, so P_MID/P_END
disappear from these arms: that is the subsumption the pend landing wrote
down as an obligation, and it is why the comparison anchor is the SHIPPED
entry (which has pend), not a pend-less strawman.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "eval4k"))
import codec, splice          # noqa: E402
import price_taper as pt      # noqa: E402

FIT, OUTDIR = sys.argv[1], sys.argv[2]
ARM = sys.argv[3] if len(sys.argv) > 3 else "constrained"
DSTEP = int(sys.argv[4]) if len(sys.argv) > 4 else 16

doc = json.load(open(FIT))[ARM]
piece, raw = splice.classic_tables()
eg_raw = {p: [v - piece[p] for v in doc["eg"][p]] for p in "PNBRQ"}
eg_raw["K"] = list(raw["K"])

ROOT_TAPK = '''        ph = min(24, sum(pos.board.count(c) * w
                         for c, w in zip("NnBbRrQq", (1, 1, 1, 1, 2, 2, 4, 4))))
        for k in "PNBRQ":
            pst[k] = tuple(e + (m - e) * ph // 24 for m, e in zip(pst1[k], EGT[k]))
        pst["K"] = tuple(e + (m - e) * ph // 24 for m, e in zip(K_MID, K_END))
'''

base = open(splice.ENTRY).read()
region, lvl = pt.deltaregion(raw, piece, eg_raw, 1, False, DSTEP)
os.makedirs(OUTDIR, exist_ok=True)
for name, rootcode in (("tap", pt.ROOT_BLEND), ("tapk", ROOT_TAPK)):
    src = splice.splice(base, region)
    assert src.count(pt.ROOT_OLD) == 1, "landed pend seam not found -- re-anchor"
    src = src.replace(pt.ROOT_OLD, rootcode, 1)
    # the taper owns the pawn table now; P_MID/P_END are dead names and a dead
    # name that still LOOKS live is how a subsumed term gets silently kept.
    assert "P_MID, P_END" in src
    src = src.replace(pt.PEND_SRC, "")
    assert "P_MID" not in src, "pend survived into a taper arm"
    out = os.path.join(OUTDIR, "e_%s.py" % name)
    open(out, "w").write(src)
    compile(src, out, "exec")
    print("%s  delta step %d, %d levels" % (out, DSTEP, lvl))
