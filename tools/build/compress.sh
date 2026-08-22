#!/bin/bash
# compress.sh engine.py out.py -- the minified PLAIN-PYTHON artifact.
# Same strip+minify as pack.sh's payload (pyminify --rename-globals
# --remove-literal-statements on the minifier-hide-stripped source), but
# no xz and no shell stub: the output is a valid Python file. The
# polyglot header is re-attached by hand because --remove-literal-
# statements strips the """:" ... ":""" interpreter-picking trick.
set -e
T=$(mktemp); trap 'rm -f "$T"' EXIT
pyminify --rename-globals --remove-literal-statements \
   <(sed '/# minifier-hide start/,/# minifier-hide end/d' "$1") > "$T"
{
  printf '#!/bin/sh\n'
  printf '""":"\nfor cmd in pypy3 python3; do command -v "$cmd" > /dev/null && exec "$cmd" "$0" "$@"; done\nexit 1\n":"""\n'
  sed '1{/^#!/d;}' "$T"     # drop pyminify's carried-over shebang line
} > "$2"
chmod +x "$2"
echo "compressed: $(wc -c < "$2" | tr -d ' ') bytes"
