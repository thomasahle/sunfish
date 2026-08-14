#!/bin/bash
# CI GUARD: the committed 4k entry must match what its generator produces.
#
# nnue_4k/pst_entry.py is generated mechanically from sunfish_nnue.py and
# sunfish.py. That is only worth anything if it stays true -- and it did not:
# the committed source drifted 38 lines behind its generator and nothing
# noticed, because the drift was all inside minifier-hide and the packed
# artifact was unchanged. Same class as the stale driver: silent staleness
# that surfaces only when someone happens to rebuild.
#
# Checks BOTH, because they can disagree:
#   1. source: regenerating must reproduce the committed file
#   2. artifact: packing must stay under the 4096-byte ceiling
#
# Deliberately a CEILING, not a pinned size: the packed byte count moves with
# the local pyminify and xz versions (2026-08-14: --no-hoist-literals plus the
# payload shebang strip took the entry 3341 -> 3295), and a pin would turn a
# toolchain bump into a red CI with nothing wrong. Sizes are tracked in
# nnue_4k/MEASUREMENTS.md, where a number can carry the context a pin cannot.
set -euo pipefail
cd "$(dirname "$0")/../.."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
python3 tools/build/make_pst_entry.py "$TMP/gen.py" > /dev/null
if ! diff -q "$TMP/gen.py" nnue_4k/pst_entry.py > /dev/null; then
    echo "FAIL: nnue_4k/pst_entry.py is STALE vs tools/build/make_pst_entry.py"
    diff "$TMP/gen.py" nnue_4k/pst_entry.py | head -20
    echo "fix: python3 tools/build/make_pst_entry.py nnue_4k/pst_entry.py"
    exit 1
fi
bash tools/build/pack.sh nnue_4k/pst_entry.py "$TMP/e.packed" > /dev/null
size=$(wc -c < "$TMP/e.packed" | tr -d ' ')
[ "$size" -le 4096 ] || { echo "FAIL: entry is $size bytes, over the 4096 limit"; exit 1; }
echo "entry OK: source matches generator, packs to $size bytes ($((4096-size)) spare)"
