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

pyminify --rename-globals --remove-literal-statements \
   <(sed '/# minifier-hide start/,/# minifier-hide end/d' "$1") \
   > "$T"
# .lzma format, pb=0: ~70 bytes smaller than the xz container on a ~4.5k
# python text stream, and `xz -d` auto-detects it -- the unpack head is
# unchanged.
xz --format=lzma --lzma1=preset=9e,pb=0 "$T"
lt=$(get_file_size "$T.lzma")
echo "Length of script: $lt"

lh=100
head=""
while [ $lh != ${#head} ]
do
   let lh=${#head}
   head="""#!/bin/sh
T=\`mktemp\`
tail -c +$((lh+1)) "\$0"|xz -d>\$T
chmod +x \$T
(sleep 9;rm \$T)&exec \$(command -v pypy3||echo python3) \$T
"""
   echo "Length of head: $lh"
done

printf "$head" > $2

cat $T.lzma >> $2
rm $T.lzma

echo "Total length: $(get_file_size "$2")"

chmod +x $2
