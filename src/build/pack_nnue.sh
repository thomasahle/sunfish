#!/bin/bash
echo "Usage: pack.sh engine.py model.pickle out.packed"

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
pyminify \
   --rename-globals \
   --remove-literal-statements \
   "$1" > "$T"
xz "$T"
lt=$(get_file_size "$T.xz")
echo "Length of script: $lt"

M="$2"
lm=$(get_file_size "$M")
echo "Length of model: $lm"

lh=100
head=""
while [ $lh != ${#head} ]
do
   let lh=${#head}
   head="""#!/bin/sh
T=\`mktemp\`
M=\`mktemp\`
tail -c +$((lh+1)) "\$0"|head -c $lt|xz -d>\$T
tail -c $lm "\$0">\$M
chmod +x \$T
(sleep 9;rm \$T \$M)&exec \$T \$M
"""
   echo "Length of head: $lh"
done

printf "$head" > $3

cat $T.xz >> $3
rm $T.xz

cat $2 >> $3

echo "Total length: $(get_file_size "$3")"

chmod +x $3
