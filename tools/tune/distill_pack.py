"""Pack the teacher's labels into the .npz the trainer reads.

Features are built by the SAME six lines as `texel_data.py` -- 6x64
piece-square counts, white minus rank-mirrored black -- because the point of
this dataset is to differ from the Stockfish-labelled one in the LABEL and in
nothing else. A feature change here would make the two sets incomparable and
the whole single-variable argument would be gone.

Positions that the teacher could not put an ordinary number on (mate seen at
the root, or no depth completed inside the budget) are dropped and counted,
never coerced to a score.

usage: distill_pack.py OUT.npz LABELS.jsonl...
"""
import glob
import hashlib
import json
import sys

import chess
import numpy as np

OUT = sys.argv[1]
files = []
for a in sys.argv[2:]:
    files += sorted(glob.glob(a)) if "*" in a else [a]

recs, metas = [], []
for f in files:
    for ln in open(f):
        o = json.loads(ln)
        if "meta" in o: metas.append(o["meta"]); continue
        recs.append(o)
assert recs, "no labelled records in %s" % files
key = [k for k in recs[0] if k.startswith("n")]
assert len(key) == 1, "pack one budget at a time, got %s" % key
key = key[0]
N = int(key[1:])

# Every shard must come from the SAME teacher at the SAME budget. Shards that
# silently disagree are how a set becomes a mixture nobody can interpret.
shas = {m["teacher_sha256"] for m in metas}
buds = {tuple(m["nodes"]) for m in metas}
assert len(shas) == 1 and len(buds) == 1, "shards disagree: %s %s" % (shas, buds)

seen = set()
fens, y, depth, flags = [], [], [], {}
for r in recs:
    f, o = r["fen"], r[key]
    if f in seen: continue                     # shard overlap would double-weight
    seen.add(f)
    if o["cp"] is None or o["flag"]:
        flags[o["flag"] or "nolabel"] = flags.get(o["flag"] or "nolabel", 0) + 1
        continue
    fens.append(f); y.append(o["cp"]); depth.append(o["depth"])
print("records %d | unique %d | kept %d | dropped %s"
      % (len(recs), len(seen), len(fens), flags or "none"))

PIECES = "PNBRQK"
X = np.zeros((len(fens), 384), dtype=np.int8)
for i, fen in enumerate(fens):
    b = chess.Board(fen)
    for sq, pc in b.piece_map().items():
        idx = PIECES.index(pc.symbol().upper())
        if pc.color == chess.WHITE:
            X[i, idx * 64 + sq] += 1
        else:
            X[i, idx * 64 + (sq ^ 56)] -= 1     # mirror rank for black

meta = dict(metas[0])
meta.pop("shard", None)
meta.pop("positions", None)
# The .npz is a TRACKED file and the labeller's meta carries the machine it ran
# on. The bench box's hostname does not go into public artifacts, and a dataset
# committed to the repo is one -- `set20260813.npz` already carries a `host`
# field, which is how this was noticed.
meta.pop("host", None)
meta.update({
    "nodes": N,
    "kept": len(fens),
    "dropped": flags,
    "mean_depth": float(np.mean(depth)),
    "shards": len(metas),
    "source_files": [(f.split("/")[-1], hashlib.sha256(open(f, "rb").read()).hexdigest()[:16])
                     for f in files],
    "features": "6x64 piece-square counts, white minus mirrored black -- identical to texel_data.py",
})
np.savez_compressed(OUT, X=X, y=np.array(y, dtype=np.int16),
                    fens=np.array(fens), meta=json.dumps(meta, indent=1))
print("wrote %s: X %s, mean completed depth %.2f, label sd %.0f cp"
      % (OUT, X.shape, np.mean(depth), np.std(y)))
