"""codes() (scalar, probe path) must equal buckets() (tensor, batch path).

They are the two callers that decide which bucket a position lands in.  When
they were two independent copies, kb2 crashed one and pb was silently ignored
by it -- a probe line that looked green while evaluating the wrong table.
"""
import sys
sys.path.insert(0, ".")
import numpy as np, torch
import data, features

ds = data.load(data.DataCfg(source="pool10m.npz", kind="lambda-npz", cpmax=1000,
                            split="fenkey", split_seed=20260813, limit=3000,
                            workers=4)) if hasattr(data, "DataCfg") else None
if ds is None:
    import config
    ds = data.load(config.DataCfg(source="pool10m.npz", kind="lambda-npz",
                                  cpmax=1000, split="fenkey",
                                  split_seed=20260813, limit=3000, workers=4))

bad = 0
for kb, pb in ((1, 2), (2, 1), (2, 2), (4, 1), (1, 4), (8, 1)):
    ext = features.Extractor(kb, pb)
    if ext.B == 1:
        continue
    w, b = ext.buckets(ds)
    ph = ds.phasec
    n = min(1500, len(ds.y))
    for i in range(n):
        cw, cb = ext.codes(int(ds.kb4[i]), int(ds.kb8[i]), int(ds.kb16[i]),
                           float(ph[i]))
        if cw != int(w[i]) or cb != int(b[i]):
            bad += 1
            if bad < 4:
                print("  MISMATCH kb=%d pb=%d i=%d: codes=(%d,%d) buckets=(%d,%d)"
                      % (kb, pb, i, cw, cb, int(w[i]), int(b[i])))
    print("kb=%d pb=%d B=%-2d  %d positions  %s"
          % (kb, pb, ext.B, n, "AGREE" if bad == 0 else "DISAGREE"))
print("TOTAL MISMATCHES:", bad)
sys.exit(1 if bad else 0)
