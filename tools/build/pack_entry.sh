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
# Same indivisible lever pair as tools/build/pack.sh (landed eb8897c), applied
# here after measuring layout B's own consumers -- the bake-off cells -- rather
# than assuming the joint layout's result carried over:
#   * `1{/^#!/d}` drops the source's polyglot `#!/bin/sh`. Dead here for the
#     same reason: the head below execs a NAMED interpreter on a /dev/fd. The
#     source file keeps its header.
#   * `--no-hoist-literals` stops pyminify replacing repeated literals with
#     fresh one-character names, which is the repetition lzma matches for free.
# Layout B, base -> both: b81 bake-off cell 3913 -> 3882 (-31), its elided cell
# 3280 -> 3249 (-31), pst_entry 3380 -> 3334 (-46), classic 3271 -> 3249 (-22),
# sunfish_nnue 3970 -> 3939 (-31), replnet_proto 3880 -> 3851 (-29).
# The shebang strip ALONE is +4 on classic here TOO, so the pair is indivisible
# in this script as well. Only the engine stream moves: the weights are
# appended raw and the head recomputes `head -c$lt` in the same run.
pyminify --rename-globals --remove-literal-statements --no-hoist-literals \
    <(sed -e '1{' -e '/^#!/d' -e '}' \
          -e '/# minifier-hide start/,/# minifier-hide end/d' "$1") > "$T"
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
