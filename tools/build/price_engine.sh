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
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "..", "tools/eval4k"))
src = open(sys.argv[1]).read()

# TWO ENTRY FORMS, and reading the wrong one SILENTLY produced a wrong answer.
# The literal form has `pst = {` ... `}` full of decimal numbers. The base-90
# form has `pst = {}` plus a decode loop, and the old regex matched a `\n}`
# hundreds of lines further down, zeroed something unrelated, and reported
# "eval data costs 30, engine 3320" with no complaint. Both numbers were wrong
# and neither looked it.
m = re.search(r"\npst = \{\n.*?\n\}\n", src, re.S)
if m:
    # literal form: kill the ENTROPY, not the syntax -- a field of identical
    # zeros costs lzma almost nothing.
    body = m.group(0)
    out = src.replace(body, re.sub(r"-?\d+", "0", body), 1)
else:
    # encoded form: there is no literal to zero, so replace the whole eval
    # region with a stub that defines the same four names and holds no data.
    import splice
    assert "\npst = {}\n" in src, (
        "neither a pst literal nor a base-90 decode block found -- this "
        "instrument is reading a file it does not understand, and it must "
        "NOT guess: fix the instrument, do not trust a number from it")
    STUB = ('piece = {"P": 100, "N": 280, "B": 320, "R": 479, "Q": 929, "K": 60000}\n'
            'pst = {_k: tuple([0] * 20 + sum(([0] + [piece[_k]] * 8 + [0]\n'
            '                 for _i in range(8)), []) + [0] * 20) for _k in "PNBRQK"}\n'
            'K_MID, K_END = pst["K"], tuple(piece["K"] + 70\n'
            '   - 10 * (abs(2 * (i // 10) - 11) + abs(2 * (i % 10) - 9)) for i in range(120))\n')
    out = splice.splice(src, STUB)
open(sys.argv[2], "w").write(out)
PY

full=$(bash tools/build/pack.sh "$SRC" "$TMP/full.packed" | tail -1 | grep -o '[0-9]*')
zero=$(bash tools/build/pack.sh "$TMP/zero.py" "$TMP/zero.packed" | tail -1 | grep -o '[0-9]*')

printf '%-34s %5s\n' "entry as shipped" "$full"
printf '%-34s %5s\n' "same engine, eval data removed" "$zero"
printf '%-34s %5s\n' "  => eval data costs" "$((full - zero))"
printf '%-34s %5s\n' "  => ENGINE-SANS-EVAL" "$zero"
printf '%-34s %5s\n' "target" "2500"
printf '%-34s %5s\n' "  => still to golf" "$((zero - 2500))"
printf '%-34s %5s\n' "  => EVAL CEILING (4096 - engine)" "$((4096 - zero))"
