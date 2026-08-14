#!/bin/bash
echo "Usage: pack.sh engine.py out.packed"

#if [ -f $3 ]; then
#    echo "$3 Already exists."
#    exit 0
#fi

get_file_size() {
   local file="$1"
   local size=$(wc -c < "$file" | awk '{$1=$1};1')
   echo "$size"
}

T=`mktemp`

# Two levers here exist only to feed lzma a better stream, and both were
# measured on a real file per family rather than reasoned about:
#
#  * `1{/^#!/d}` drops the source's polyglot `#!/bin/sh` line. It is DEAD in
#    the artifact -- the head below execs a NAMED interpreter on a /dev/fd, so
#    nothing ever reads the payload's own shebang. Only the copy inside the
#    payload goes; the source file keeps its header, so `./sunfish.py` and
#    every non-packed configuration are untouched.
#
#  * `--no-hoist-literals` turns pyminify's string hoisting OFF. It makes the
#    minified TEXT bigger (+60..+104 chars) and the ARTIFACT smaller: hoisting
#    replaces each repeated literal with a fresh one-character name, which is
#    exactly the repetition lzma would otherwise match for free.
#
# Every family that packs through here, base -> both (2026-08-14):
#   classic 3232 -> 3210 (-22)          sunfish_nnue 3931 -> 3900 (-31)
#   pst_entry 3341 -> 3295 (-46)        replnet proto 3841 -> 3812 (-29)
#   make_variants base/cap/nolmr/khold2  -46 / -52 / -47 / -47
# The shebang strip ALONE is +4 on classic (it lands the stream in a worse
# lzma neighbourhood); it only pays alongside --no-hoist-literals. That is why
# the two are one change and must not be split.
pyminify --rename-globals --remove-literal-statements --no-hoist-literals \
   <(sed -e '1{' -e '/^#!/d' -e '}' \
         -e '/# minifier-hide start/,/# minifier-hide end/d' "$1") \
   > "$T"
# .lzma format, pb=0: ~70 bytes smaller than the xz container on a ~4.5k
# python text stream, and `xz -d` auto-detects it -- the unpack head is
# unchanged.
xz --format=lzma --lzma1=preset=9e,pb=0 "$T"
lt=$(get_file_size "$T.lzma")
echo "Length of script: $lt"

# Process substitution (hence bash, not sh) makes the payload a /dev/fd
# path: no temp file, so no mktemp, no cleanup subshell, and no chmod
# (an interpreter argument never needed the exec bit anyway).
lh=100
head=""
while [ $lh != ${#head} ]
do
   let lh=${#head}
   head="""#!/bin/bash
exec \$(command -v pypy3||echo python3) <(tail -c+$((lh+1)) "\$0"|xz -d)
"""
   echo "Length of head: $lh"
done

printf "$head" > $2

cat $T.lzma >> $2
rm $T.lzma

echo "Total length: $(get_file_size "$2")"

chmod +x $2
