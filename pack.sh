#!/bin/bash
{
   if [ -f $2 ]; then
       echo "$2 Already exists."
       exit 0
   fi
}
T=`mktemp`
pyminify --rename-globals --remove-literal-statements $1 > $T
xz $T
cat pack_head.sh > $2
cat $T.xz >> $2
chmod +x $2
rm $T.xz
