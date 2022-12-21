#!/bin/sh
T=`mktemp`
tail -c +73 "$0"|xz -d>$T
python3 -u $T
rm $T
exit
