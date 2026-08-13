#!/bin/sh
# OUR-VS-OUR fixed-node SPRT screen, for the box.
#
# Fixed nodes is legitimate ONLY between two of our own engines. Against
# classic it is not: classic has no mid-search node cap, so it sails a
# measured 1.70x past the budget and the comparison rewards whichever side
# prunes LESS. Anything anchored on classic has to be timed, and timed needs
# a quiet machine; this screen does not, which is why it can run on a box
# that is sharing with someone else's tournament.
#
# Legality gate first -- seconds, not games. A mate suite is not a legality
# gate: LMP passed one 5-vs-5 on the very build that answered `bestmove
# (none)`.
#
# No wait-loop: this cannot be started by another stage dying. Run it
# deliberately.
#
# usage: ab_fixednode.sh NAME_A ENG_A NAME_B ENG_B TAG SRAND [NODES] [ROUNDS] [CONC]
set -u
cd "$(dirname "$0")"
ARENA=$(pwd)
FC=$HOME/sunfish-bench/fastchess-linux-x86-64/fastchess
PY=$HOME/sunfish-bench/pypy3.11-v7.3.20-linux64/bin/pypy3
BOOK=$ARENA/openings_2k.epd

NA=$1; EA=$2; NB=$3; EB=$4; TAG=$5; SRAND=$6
NODES=${7:-20000}; ROUNDS=${8:-500}; CONC=${9:-10}
OUT=$ARENA/AB_$TAG.txt

: > "$OUT"
say() { echo "$@" | tee -a "$OUT"; }
say "FIXED-NODE AB  $NA vs $NB   started $(date -u)"
say "nodes $NODES  rounds $ROUNDS (cap $((ROUNDS*2)) games)  concurrency $CONC (= $((CONC*2)) procs)"
say "book  $BOOK ($(wc -l < "$BOOK") positions, $ROUNDS consumed)"

for e in "$EA" "$EB"; do
    r=$("$PY" "$ARENA/legality_gate.py" "$ARENA/bin/$e" 300 2>&1)
    if ! echo "$r" | grep -q "GATE PASSED"; then
        say "LEGALITY GATE FAILED: $e -- correctness bug, no games spent"
        echo "$r" | tail -12 | tee -a "$OUT"; exit 1
    fi
    say "  legality PASS  $e   $(echo "$r" | grep FORCED)"
done

nice -n 5 "$FC" \
  -engine cmd="$ARENA/w_$NA.sh" name="$NA" \
  -engine cmd="$ARENA/w_$NB.sh" name="$NB" \
  -each proto=uci nodes="$NODES" \
  -openings file="$BOOK" format=epd order=random -srand "$SRAND" \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" -recover \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=500 \
  -pgnout file="$ARENA/$TAG.pgn" > "$ARENA/$TAG.log" 2>&1

n=$(grep -c '^\[Result' "$ARENA/$TAG.pgn")
say ""
say "games played  $n  $([ "$n" -ge $((ROUNDS*2)) ] && echo 'UNDECIDED-AT-CAP (report as undecided, NOT as a point estimate)' || echo 'stopped early by SPRT')"
say "time forfeits $(grep -ci 'time forfeit' "$ARENA/$TAG.pgn")"
say "illegal moves $(grep -ci 'illegal move' "$ARENA/$TAG.pgn")"
"$PY" "$ARENA/pair_elo.py" "$ARENA/$TAG.pgn" 2>&1 | tee -a "$OUT"
grep -E "^Elo|^Games|SPRT" "$ARENA/$TAG.log" | tail -4 | tee -a "$OUT"
say "NOTE: SPRT's terminal Elo is biased away from zero. A PASS means positive,"
say "      not this big; winners earn their number from a fixed-N confirmation."
say "AB $TAG COMPLETE $(date -u)"
