#!/usr/bin/env python3
"""tools/tune/data npz corpora -> the lichess-dump jsonl.zst the trainer
reads: npz_to_dump.py in1.npz [in2.npz ...] out.jsonl.zst

Labels are white-POV cp in both conventions (npz meta["label"] and the
dump's), so they pass through unchanged; parse_lines flips for black to
move.  No pv line is emitted, so train these with --quiet 0 (the distill
corpora resolved quiescence in the labelling search itself)."""
import json
import subprocess
import sys

import numpy as np

out = sys.argv[-1]
assert out.endswith(".zst"), "last argument is the output .jsonl.zst"
proc = subprocess.Popen(["zstd", "-q", "-f", "-o", out], stdin=subprocess.PIPE)
n = 0
for path in sys.argv[1:-1]:
    d = np.load(path, allow_pickle=True)
    for fen, y in zip(d["fens"], d["y"]):
        rec = {"fen": str(fen),
               "evals": [{"depth": 1, "pvs": [{"cp": int(y)}]}]}
        proc.stdin.write((json.dumps(rec) + "\n").encode())
        n += 1
proc.stdin.close()
proc.wait()
assert proc.returncode == 0, "zstd failed"
print("wrote %d records -> %s" % (n, out))
