"""Convert ANY float training export into the packed representation.

usage: build_ext.py FLOAT.pickle OUT.pickle

Handles every combination the trainer can emit (kind float-kb, float-bil,
float-phase): king-bucketed first layers, bilinear lanes with the odd
tail, and the phase output scale.  The trainer deliberately exports float
weights only; this is the separate build step that quantises them, so the
decision to spend engine-side complexity stays a decision.

Head shift: the largest that satisfies pick_shift for EVERY bucket.
Bilinear bshift: pick_bshift (per-group cap sums and lane excursion).
"""
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pnet

PIDX = {c: i for i, c in enumerate(pnet.PIECES)}


def sq64(i):
    return (i // 10 - 2) * 8 + (i % 10 - 1)


def to_W(E, N, off=0):
    """(768, N) nested list slice at row offset `off` -> W[k][piece][sq120]."""
    W = [{c: [0.0] * 120 for c in pnet.PIECES} for _ in range(N)]
    for c in pnet.PIECES:
        for s in pnet.SQUARES:
            col = E[off + PIDX[c] * 64 + sq64(s)]
            for k in range(N):
                W[k][c][s] = col[k]
    return W


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        d = pickle.load(f)
    kind = d.get("kind")
    if kind not in ("float-kb", "float-bil", "float-phase"):
        raise ValueError("%s is not a float export (kind=%r)" % (src, kind))
    N = d["N"]
    B = d.get("B", 1) or 1
    bias, v, segs = d["bias"], d["v"], tuple(d["segs"])

    rff = None
    if d.get("rff"):
        r = d["rff"]
        rff = {"R": len(r["rw"]), "theta": r["theta"], "phb": r["phb"],
               "rw": r["rw"]}
    bil = None
    if d.get("nb"):
        bil = {"nb": d["nb"], "m": d["m"],
               "Wb": to_W(d["Eb"], d["nb"]),
               "biasb": d["biasb"], "gb": d["gb"], "u": d["u"],
               "tail": d.get("tail")}
        bil["bshift"] = pnet.pick_bshift(bil)
    phase_s = d.get("phase_s")

    if B > 1:
        Ws = [to_W(d["E"], N, off=b * 768) for b in range(B)]
        picks = [pnet.pick_shift(W, bias, v, segs=segs) for W in Ws]
        shift = min(p[0] for p in picks)
        out = pnet.build_kb(Ws, bias, v, shift, clampcp=d["clampcp"],
                            segs=segs, bil=bil, phase_s=phase_s, rff=rff)
    else:
        W = to_W(d["E"], N)
        shift = pnet.pick_shift(W, bias, v, segs=segs)[0]
        out = pnet.build(W, bias, v, shift, clampcp=d["clampcp"],
                         segs=segs, bil=bil, phase_s=phase_s, rff=rff)
    out["train"] = d.get("train")
    out["base_kind"] = d.get("base_kind", "pst")
    pnet.save(dst, out)
    print("built %s: B=%d N=%d shift=%d sum_G=%d excursion=%d nb=%d "
          "bshift=%s phase=%s rff=%d"
          % (dst, B, N, shift, out["sum_G"], out["excursion"],
             d.get("nb", 0), out.get("bshift", "-"),
             len(phase_s) if phase_s else 0, out.get("rff", 0)))


main()
