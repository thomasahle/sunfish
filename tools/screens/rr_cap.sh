#!/bin/sh
# THE CAP ROUND-ROBIN: one classic-anchored tournament, four questions.
#
# Our null move takes the raw child value where classic clamps it:
#     classic:  score = min(pos.score + EVAL_ROUGHNESS, -bound(...))
#     ours:     score =                                 -bound(...)
# That is looser in two ways -- `score >= gamma` fires more often, AND the
# yielded score becomes this node's returned value, so an inflated pass
# estimate propagates into the tt and into the MTD bisection.
#
# Four A/Bs would answer this in four matches with four separate anchors.
# One round-robin answers all of them off one set of games:
#   entry      vs classic  -- replicate the +19.1 baseline
#   entry_cap  vs classic  -- does the cap move the shipped entry
#   nolmr      vs classic  -- replicate the -46.3 hole
#   nolmr_cap  vs classic  -- IS the uncapped null the hole?
#   entry      vs entry_cap        -- the ship decision (11 bytes)
#   nolmr      vs nolmr_cap        -- the same edit with LMR out of the way
#   entry      vs nolmr            -- LMR, TIMED (fixed-node says +38.9)
#
# TIMED, not fixed-node: classic has no mid-search node cap, so a node-capped
# match hands it a measured 1.70x effort advantage and the comparison is
# confounded. Our-vs-our pairs would tolerate fixed nodes; the classic pairs
# do not, and they share the tournament.
#
# The variants are GENERATED from nnue_4k/pst_entry.py at stage time
# (tools/build/make_variants.py), never accumulated as files -- five
# stale-copy failures in one session all had the shape "the fix went to a new
# path and the old path stayed live".
#
# This script does NOT wait on anything and cannot be started by another
# stage finishing. Producers write their result file on any exit, including
# being killed, so a `while [ ! -f result ]` chain reads "the previous stage
# was stopped" as "start now". Run it deliberately or not at all.
set -u
cd "$(dirname "$0")"
ARENA=$(pwd)
FC=$HOME/sunfish-bench/fastchess-linux-x86-64/fastchess
PY=$HOME/sunfish-bench/pypy3.11-v7.3.20-linux64/bin/pypy3
BOOK=$HOME/sunfish-bench/openings_2k.epd
OUT=$ARENA/RESULT.txt
TC=${TC:-10+0.1}
ROUNDS=${ROUNDS:-200}
CONC=${CONC:-10}

: > "$OUT"
say() { echo "$@" | tee -a "$OUT"; }

say "CAP ROUND-ROBIN  started $(date -u)"
say "arena   $ARENA"
say "tc      $TC   rounds $ROUNDS   concurrency $CONC (= $((CONC*2)) engine processes)"
# 5 engines -> 10 encounters per round, 1 opening per encounter, so ROUNDS*10
# openings are consumed. The book must cover that or fastchess cycles it and
# the repeats quietly shrink the effective sample behind unchanged error bars.
say "book    $BOOK ($(wc -l < "$BOOK") positions; $((ROUNDS*10)) consumed)"
say "games   $((ROUNDS*20)) total, $((ROUNDS*2)) per pairing"
say ""

# ---- GATES ---------------------------------------------------------------
# Legality first, then mates. A mate suite is NOT a legality gate: LMP passed
# 5-vs-5 on the very build that answered `bestmove (none)`. Legality costs
# seconds; a screen launched on an illegal-move build is wasted from game one.
say "--- legality gate (100 positions, 40 of them FORCED: in check, <=2 legal replies)"
for e in e_base e_cap e_nolmr e_nolmrcap; do
    r=$("$PY" "$ARENA/legality_gate.py" "$ARENA/bin/$e.py" 300 2>&1)
    if ! echo "$r" | grep -q "GATE PASSED"; then
        say "LEGALITY GATE FAILED: $e -- this is a correctness bug, not a weak feature"
        echo "$r" | tail -12 | tee -a "$OUT"
        exit 1
    fi
    say "  PASS  $e   $(echo "$r" | grep FORCED)"
done

say "--- mate gate (8 positions, depth <= 4; a variant may not find fewer than base)"
base_m=$("$PY" "$ARENA/mate_gate.py" "$ARENA/bin/e_base.py" "$ARENA/mate1.fen" 4 2>/dev/null \
         | grep -oE "found +[0-9]+" | grep -oE "[0-9]+")
say "  base finds ${base_m:-?}/8"
for e in e_cap e_nolmr e_nolmrcap; do
    v=$("$PY" "$ARENA/mate_gate.py" "$ARENA/bin/$e.py" "$ARENA/mate1.fen" 4 2>/dev/null \
        | grep -oE "found +[0-9]+" | grep -oE "[0-9]+")
    say "  $e finds ${v:-?}/8"
    if [ "${v:-0}" -lt "${base_m:-99}" ]; then
        say "MATE GATE REGRESSION: $e finds ${v} vs base ${base_m} -- recorded, screen continues"
    fi
done
say ""

# ---- THE TOURNAMENT ------------------------------------------------------
# Every engine at the SAME nice level. The box's old r_classic.sh ran classic
# at nice 19 while the variants ran at nice 5; harmless at fixed nodes,
# a systematic gift to our side in a TIMED match.
say "--- round robin, launched $(date -u)"
nice -n 5 "$FC" \
  -engine cmd="$ARENA/r_entry.sh"      name=entry \
  -engine cmd="$ARENA/r_entrycap.sh"   name=entry_cap \
  -engine cmd="$ARENA/r_nolmr.sh"      name=entry_nolmr \
  -engine cmd="$ARENA/r_nolmrcap.sh"   name=entry_nolmr_cap \
  -engine cmd="$ARENA/r_classic.sh"    name=classic \
  -each proto=uci tc="$TC" \
  -tournament roundrobin \
  -openings file="$BOOK" format=epd order=random -srand 20260913 \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" -recover \
  -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=500 \
  -pgnout file="$ARENA/caprr.pgn" > "$ARENA/caprr.log" 2>&1

# ---- HARVEST -------------------------------------------------------------
n=$(grep -c '^\[Result' "$ARENA/caprr.pgn")
say ""
say "--- finished $(date -u)"
say "games played  $n  of $((ROUNDS*20)) scheduled  $([ "$n" -ge $((ROUNDS*20)) ] && echo COMPLETE || echo "SHORT -- do NOT read the Elo until this is explained")"
say "time forfeits $(grep -ci 'time forfeit' "$ARENA/caprr.pgn")"
say "disconnects   $(grep -ci 'disconnect\|stall' "$ARENA/caprr.pgn")"
# ZERO TOLERANCE: any illegal move by any arm is a FAIL naming the game, never
# a count. The structural bestmove floor makes the known class impossible;
# this notices any class, known or new, coming back.
if [ "$(grep -ci 'illegal move' "$ARENA/caprr.pgn")" -gt 0 ]; then
    say "ZERO-TOLERANCE FAIL: illegal move(s) in caprr.pgn -- the offending games:"
    awk '/^\[Round /{r=$0} /^\[White /{w=$0} /^\[Black /{b=$0} /^\[FEN /{f=$0}
         /makes an illegal move/{printf "    %s %s %s %s :: %s\n", r, w, b, f, $0}' \
        "$ARENA/caprr.pgn" | tee -a "$OUT"
    exit 8
fi
say ""
say "--- pairwise Elo, 95% pentanomial intervals (analyzer reproduces fastchess exactly)"
"$PY" "$ARENA/pair_elo.py" "$ARENA/caprr.pgn" 2>&1 | tee -a "$OUT"
say ""
say "--- fastchess ranking table"
grep -A12 "Rank Name" "$ARENA/caprr.log" | tail -20 | tee -a "$OUT"
say "CAP ROUND-ROBIN COMPLETE $(date -u)"
