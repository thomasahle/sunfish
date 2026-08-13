#!/bin/sh
# THE HOLE ROUND-ROBIN: is the ~46 Elo gap to classic a port defect in the EVAL?
#
# Supersedes the cap round-robin, which is deferred. The cap turned out to be
# near-inert (census: binds on 4.3% of null attempts, changes the chosen move
# in 0 of 200 positions at depth 7) AND it reads `pos.score`, so by the
# (feature, eval) rule it has to be re-priced after an eval fix anyway.
#
# What replaced it: the entry does not inherit classic's eval. It pastes
# classic's tables but kept the NNUE engine's king-table phase rule, so it
# evaluates 62.1% of real positions with the middlegame king table. Fixed
# nodes, our-vs-our, that fix measured +47.6 +/- 26.5 -- in the setting that
# UNDERSTATES an endgame defect.
#
# Five arms, one anchor:
#   entry              vs classic  -- re-baseline IN THESE CONDITIONS (was +19.1)
#   entry_nolmr        vs classic  -- reproduce the -46.3 hole as a CONTROL
#   entry_kf           vs classic  -- where the shipping candidate stands
#   entry_nolmr_kf     vs classic  -- IS THE HOLE CLOSED? (the headline)
#   entry_nolmr    vs entry_nolmr_kf -- the fix's timed value, LMR out of the way
#   entry          vs entry_nolmr    -- LMR, timed, on the fixed engine
#
# TIMED, because every classic pair is. Classic has no mid-search node cap and
# sails a measured 1.70x past a node budget, so fixed-node against it rewards
# whichever side prunes less.
#
# The control arm matters: the -46.3 was measured on the laptop, this runs on
# the box. If entry_nolmr does not reproduce near -46 here, the environment
# differs and no other number in this tournament may be compared to the ledger.
#
# No wait-loop. Run it deliberately.
set -u
cd "$(dirname "$0")"
ARENA=$(pwd)
FC=$HOME/sunfish-bench/fastchess-linux-x86-64/fastchess
PY=$HOME/sunfish-bench/pypy3.11-v7.3.20-linux64/bin/pypy3
BOOK=$ARENA/openings_2k.epd
OUT=$ARENA/RESULT_HOLE.txt
TC=${TC:-10+0.1}
ROUNDS=${ROUNDS:-200}
CONC=${CONC:-10}

: > "$OUT"
say() { echo "$@" | tee -a "$OUT"; }
say "HOLE ROUND-ROBIN  started $(date -u)"
say "tc $TC  rounds $ROUNDS  concurrency $CONC (= $((CONC*2)) engine processes)"
say "book $BOOK ($(wc -l < "$BOOK") positions; 5 engines -> 10 encounters/round -> $((ROUNDS*10)) consumed)"
say "games $((ROUNDS*20)) total, $((ROUNDS*2)) per pairing"
say "anchor: classic @b49426b, md5-identical to the build the ledger's +19.1 and -46.3 used"
say ""

say "--- legality gate (100 positions, 40 FORCED)"
for e in e_base e_nolmr e_kendfresh e_nolmrkendfresh; do
    r=$("$PY" "$ARENA/legality_gate.py" "$ARENA/bin/$e.py" 300 2>&1)
    if ! echo "$r" | grep -q "GATE PASSED"; then
        say "LEGALITY GATE FAILED: $e -- correctness bug, no games spent"
        echo "$r" | tail -12 | tee -a "$OUT"; exit 1
    fi
    say "  PASS  $e"
done
say ""

say "--- round robin, launched $(date -u)"
nice -n 5 "$FC" \
  -engine cmd="$ARENA/w_base.sh"           name=entry \
  -engine cmd="$ARENA/w_nolmr.sh"          name=entry_nolmr \
  -engine cmd="$ARENA/w_kendfresh.sh"      name=entry_kf \
  -engine cmd="$ARENA/w_nolmrkendfresh.sh" name=entry_nolmr_kf \
  -engine cmd="$ARENA/w_classic.sh"        name=classic \
  -each proto=uci tc="$TC" \
  -tournament roundrobin \
  -openings file="$BOOK" format=epd order=random -srand 20260931 \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" -recover \
  -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=500 \
  -pgnout file="$ARENA/hole.pgn" > "$ARENA/hole.log" 2>&1

n=$(grep -c '^\[Result' "$ARENA/hole.pgn")
say ""
say "--- finished $(date -u)"
say "games $n of $((ROUNDS*20))  $([ "$n" -ge $((ROUNDS*20)) ] && echo COMPLETE || echo 'SHORT -- explain before reading any Elo')"
say "time forfeits $(grep -ci 'time forfeit' "$ARENA/hole.pgn")   illegal $(grep -ci 'illegal move' "$ARENA/hole.pgn")"
say ""
"$PY" "$ARENA/pair_elo.py" "$ARENA/hole.pgn" 2>&1 | tee -a "$OUT"
say ""
grep -A12 "Rank Name" "$ARENA/hole.log" | tail -20 | tee -a "$OUT"
say "HOLE ROUND-ROBIN COMPLETE $(date -u)"
