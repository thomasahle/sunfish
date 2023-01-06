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
xz "$T"
lt=$(get_file_size "$T.xz")
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
(sleep 9;rm \$T)&exec \$T
"""
   echo "Length of head: $lh"
done

printf "$head" > $2

cat $T.xz >> $2
rm $T.xz

echo "Total length: $(get_file_size "$2")"

chmod +x $2
