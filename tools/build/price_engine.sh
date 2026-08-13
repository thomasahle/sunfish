#!/bin/bash
# What does the ENGINE cost, with the evaluation data taken out?
#
# The byte budget just changed shape: the eval lane is designing a 1024-1500
# byte evaluation, which leaves ~2500 for everything else. "Everything else"
# is not a number anyone has measured -- it has always been inferred by
# subtracting a table estimate from the entry, and this ledger has been wrong
# every single time it composed a byte figure instead of building one.
#
# So build it. Take the real entry, replace ONLY the numbers inside the `pst`
# literal with zeros (same file, same structure, same everything else), and
# pack it with the real packer. lzma shares one dictionary across the stream,
# so this is not "the tables in isolation" -- it is the honest quantity:
# WHAT THE ENTRY WOULD COST IF ITS EVAL DATA WERE FREE. That is exactly the
# budget the golf mission is spending against.
#
# usage: price_engine.sh [engine.py]     (default: nnue_4k/pst_entry.py)
set -eu
cd "$(dirname "$0")/../.."
SRC=${1:-nnue_4k/pst_entry.py}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

python3 - "$SRC" "$TMP/zero.py" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r"\npst = \{.*?\n\}\n", src, re.S)
assert m, "pst literal not found -- this instrument is reading the wrong file"
body = m.group(0)
# Zero every integer INSIDE the literal, keeping the exact character count of
# the surrounding structure irrelevant: what we want gone is the ENTROPY, not
# the syntax. A field of identical zeros costs lzma almost nothing.
zeroed = re.sub(r"-?\d+", "0", body)
open(sys.argv[2], "w").write(src.replace(body, zeroed, 1))
PY

full=$(bash tools/build/pack.sh "$SRC" "$TMP/full.packed" | tail -1 | grep -o '[0-9]*')
zero=$(bash tools/build/pack.sh "$TMP/zero.py" "$TMP/zero.packed" | tail -1 | grep -o '[0-9]*')

printf '%-34s %5s\n' "entry as shipped" "$full"
printf '%-34s %5s\n' "same file, pst values zeroed" "$zero"
printf '%-34s %5s\n' "  => eval data costs" "$((full - zero))"
printf '%-34s %5s\n' "  => ENGINE-SANS-EVAL" "$zero"
printf '%-34s %5s\n' "target" "2500"
printf '%-34s %5s\n' "  => still to golf" "$((zero - 2500))"
