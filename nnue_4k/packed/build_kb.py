"""Convert a float-kb training export into the packed kb representation.

usage: build_kb.py FLOAT.pickle OUT.pickle

The training side (train_packed.py --kb B) deliberately exports float
weights only; this is the separate build step that quantises them, so the
decision to spend engine-side complexity stays a decision.  The shift is
the largest that satisfies pick_shift for EVERY bucket.
"""
import pickle, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pnet

PIDX = {c: i for i, c in enumerate(pnet.PIECES)}


def sq64(i):
    return (i // 10 - 2) * 8 + (i % 10 - 1)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        d = pickle.load(f)
    if d.get("kind") != "float-kb":
        raise ValueError("%s is not a float-kb export (kind=%r)"
                         % (src, d.get("kind")))
    B, N = d["B"], d["N"]
    E, bias, v = d["E"], d["bias"], d["v"]
    segs = tuple(d["segs"])
    Ws = []
    for b in range(B):
        W = [{c: [0.0] * 120 for c in pnet.PIECES} for _ in range(N)]
        for c in pnet.PIECES:
            for s in pnet.SQUARES:
                col = E[b * 768 + PIDX[c] * 64 + sq64(s)]
                for k in range(N):
                    W[k][c][s] = col[k]
        Ws.append(W)
    picks = [pnet.pick_shift(W, bias, v, segs=segs) for W in Ws]
    shift = min(p[0] for p in picks)
    out = pnet.build_kb(Ws, bias, v, shift, clampcp=d["clampcp"], segs=segs)
    out["train"] = d.get("train")
    pnet.save(dst, out)
    print("built %s: B=%d N=%d shift=%d sum_G=%d excursion=%d"
          % (dst, B, N, shift, out["sum_G"], out["excursion"]))


main()
