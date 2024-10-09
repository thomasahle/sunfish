#~/bin/sh
pyminify --remove-literal-statements \
   <(sed '/# minifier-hide start/,/# minifier-hide end/d' "$1")
