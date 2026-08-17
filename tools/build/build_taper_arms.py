"""Generate the taper arms from ONE source, at screen time (make_variants' rule).

Arms, all built from nnue_4k/pst_entry.py @ the lane's baseline:
  tap    fitted eg set for PNBRQ, blended on the 24-point phase.
         The KING keeps the landed queens-off cliff, so `tap` vs `ktap`
         attributes the king ramp separately from the table set.
  tapk   the same, plus the king table on the same ramp -- i.e. `tap`
         composed with `ktap`.
  tapp   `tap` with pend KEPT UNDER the fitted pawn delta: eg_P = P_END +
         delta_P instead of mg_P + delta_P.

WHY `tapp` EXISTS. The fit is MEAN-CONSTRAINED per piece -- that is what made
it generalise better than the free fit -- and a mean-zero table cannot express
a monotone advancement bonus, which is exactly what pend is. Measured on the
shipped quantised tables, pend gives +50 cp at rank 7 falling to +2 at rank 3,
while the fitted eg pawn delta averages +16 at rank 7 and -14 at rank 4: it
REDISTRIBUTES within the pawn table, it does not push passers. So `tap` and
`tapk` do not subsume pend, they DELETE it and put a mean-zero shape in its
place, and their numbers are net of losing a landed, confirmed +21.31.
`tapp` is the arm that keeps the bonus and adds the shape on top.

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

# pend's endgame pawn table, in the fit's frame (piece value NOT included):
# every pawn is worth (8 - rank)^2 * 2 more with the queens off.
eg_pend = dict(eg_raw)
eg_pend["P"] = [eg_raw["P"][r * 8 + f] + ((8 - (r + 2)) ** 2 * 2 if 1 <= r <= 6 else 0)
                for r in range(8) for f in range(8)]

base = open(splice.ENTRY).read()
region, lvl = pt.deltaregion(raw, piece, eg_raw, 1, False, DSTEP)
region_p, lvl_p = pt.deltaregion(raw, piece, eg_pend, 1, False, DSTEP)
os.makedirs(OUTDIR, exist_ok=True)
for name, rootcode in (("tap", pt.ROOT_BLEND), ("tapk", ROOT_TAPK), ("tapp", pt.ROOT_BLEND)):
    if name == "tapp":
        region, lvl = region_p, lvl_p
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
