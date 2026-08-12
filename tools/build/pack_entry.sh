#!/bin/bash
# Competition artifact = engine + net in ONE file, under 4096 bytes.
# Resurrected from build/pack_nnue.sh @0c0a33a: xz the minified engine,
# append the net RAW, and let a self-extracting header split them.
# usage: pack_entry.sh engine.py net out.packed
set -eu
size() { wc -c < "$1" | tr -d ' '; }
T=$(mktemp)
pyminify --rename-globals --remove-literal-statements "$1" > "$T"
xz -f "$T"
lt=$(size "$T.xz")
lm=$(size "$2")
lh=100; head=""
while [ $lh != ${#head} ]; do
  lh=${#head}
  head="#!/bin/sh
T=\`mktemp\`;M=\`mktemp\`
tail -c +$((lh+1)) \"\$0\"|head -c $lt|xz -d>\$T
tail -c $lm \"\$0\">\$M
(sleep 3;rm \$T \$M)&SF_NET=\$M pypy3 -u \$T
exit
"
done
printf '%s' "$head" > "$3"
cat "$T.xz" >> "$3"; rm -f "$T.xz"
cat "$2" >> "$3"
chmod +x "$3"
echo "head $lh + engine $lt + net $lm = $(size "$3") bytes"
