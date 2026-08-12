#!/bin/bash
# TCEC 4k competition artifact: ONE self-extracting file, engine + weights,
# <= 4096 bytes TOTAL (the weights count -- see nnue_4k/README.md).
#
# Layout:  [head][engine.lzma][weights raw]
#
# The engine arrives through a process substitution and the weights are read
# by the engine FROM THE ARTIFACT ITSELF (the head exports its path and the
# blob length), so the entry creates no temp files and leaves nothing behind
# -- rules: "not leave itself any files lying around". Process substitution
# cannot carry the weights: bash tears the /dev/fd down across `exec`.
#
# Weights are appended RAW. Measured on a 2 KB bit-packed blob, folding them
# into the xz'd source instead costs +156 bytes (base64) or +746 (escaped
# latin-1), because lzma cannot compress already-packed weights but does pay
# for the encoding. Split is right; the historical packer never recorded why.
#
# usage: pack_entry.sh engine.py weights.bin out
set -eu
size() { wc -c < "$1" | tr -d ' '; }
T=$(mktemp)
pyminify --rename-globals --remove-literal-statements \
    <(sed '/# minifier-hide start/,/# minifier-hide end/d' "$1") > "$T"
xz --format=lzma --lzma1=preset=9e,pb=0 -c "$T" > "$T.lzma"
lt=$(size "$T.lzma"); lm=$(size "$2")
lh=100; head=""
while [ $lh != ${#head} ]; do
  lh=${#head}
  head="#!/bin/bash
export SF_A=\"\$0\" SF_N=$lm
exec \$(command -v pypy3||echo python3) <(tail -c+$((lh+1)) \"\$0\"|head -c$lt|xz -d)
"
done
printf '%s' "$head" > "$3"
cat "$T.lzma" >> "$3"; cat "$2" >> "$3"; rm -f "$T" "$T.lzma"; chmod +x "$3"
echo "head $lh + engine $lt + weights $lm = $(size "$3") bytes"
