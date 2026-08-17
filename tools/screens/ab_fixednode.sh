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
# THE CLOCK IS PINNED ON PURPOSE (tc=6000+0), and removing it re-opens a
# defect that has already voided results. `nodes=N` alone sends no clock, so
# the engine's own UCI loop defaults wtime to 60000, computes
# think = min(60000/40, 60000/2-1000) = 1500 ms, and sets a deadline BEFORE
# the node-cap block ever runs -- so a "fixed-node" game silently becomes a
# 1.5-second-per-move timed game, and the comparison rewards whichever arm is
# FASTER rather than whichever searches better. A prior sweep found 16.82% of
# one match's moves hitting that deadline, 1.51x more often on the slower arm.
# tc=6000+0 gives wtime 6,000,000 ms -> think 150 s, which the node cap always
# reaches first, so the deadline is unreachable and the screen is speed-free.
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
    # Gate at the budget this screen actually PLAYS at. The gate that cleared
    # the mirrored build only ever sent `go movetime`, so the fixed-node path
    # it was guarding was never exercised.
    r=$("$PY" "$ARENA/legality_gate.py" "$ARENA/bin/$e" 300 --nodes="$NODES" 2>&1)
    if ! echo "$r" | grep -q "GATE PASSED"; then
        say "LEGALITY GATE FAILED: $e -- correctness bug, no games spent"
        echo "$r" | tail -12 | tee -a "$OUT"; exit 1
    fi
    say "  legality PASS  $e   $(echo "$r" | grep FORCED)"
    say "  starvation     $e   $(echo "$r" | grep FIRST-YIELD)"
done

nice -n 5 "$FC" \
  -engine cmd="$ARENA/w_$NA.sh" name="$NA" \
  -engine cmd="$ARENA/w_$NB.sh" name="$NB" \
  -each proto=uci nodes="$NODES" tc=6000+0 \
  -openings file="$BOOK" format=epd order=random -srand "$SRAND" \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" -recover \
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=500 \
  -pgnout file="$ARENA/$TAG.pgn" > "$ARENA/$TAG.log" 2>&1

n=$(grep -c '^\[Result' "$ARENA/$TAG.pgn")
say ""
say "games played  $n  $([ "$n" -ge $((ROUNDS*2)) ] && echo 'UNDECIDED-AT-CAP (report as undecided, NOT as a point estimate)' || echo 'stopped early by SPRT')"
say "time forfeits $(grep -ci 'time forfeit' "$ARENA/$TAG.pgn")"
# ZERO TOLERANCE: any illegal move by any arm is a FAIL naming the game.
if [ "$(grep -ci 'illegal move' "$ARENA/$TAG.pgn")" -gt 0 ]; then
    say "ZERO-TOLERANCE FAIL: illegal move(s) in $TAG.pgn -- the offending games:"
    awk '/^\[Round /{r=$0} /^\[White /{w=$0} /^\[Black /{b=$0} /^\[FEN /{f=$0}
         /makes an illegal move/{printf "    %s %s %s %s :: %s\n", r, w, b, f, $0}' \
        "$ARENA/$TAG.pgn" | tee -a "$OUT"
    exit 8
fi
# DORMANCY GATE, relative to the pinned deadline. With tc=6000+0 the engine's
# own deadline is ~150 s, so any move at or past 150/10 = 15 s means the node
# cap did NOT bind and that move was shaped by something else (a stall, or the
# clock after all). Void rather than average it away: this is the check that
# would have caught the clock coupling before it cost a campaign of results.
mt=$(grep -oE '[0-9]+\.[0-9]+s' "$ARENA/$TAG.pgn" | tr -d 's' | sort -rn | head -1)
if [ -z "$mt" ]; then
    say "DORMANCY GATE: no per-move times in the pgn -- gate could NOT run."
    say "  Not a pass. Re-run with move times enabled before trusting this screen."
else
    slow=$(grep -oE '[0-9]+\.[0-9]+s' "$ARENA/$TAG.pgn" | tr -d 's' \
           | awk '$1 >= 15 {n++} END {print n+0}')
    say "dormancy gate  slowest move ${mt}s, moves >=15s: $slow"
    if [ "$slow" -gt 0 ]; then
        say "DORMANCY VOID: $slow move(s) at or past deadline/10 -- the node cap did"
        say "  not bind on those moves. This screen is VOID, not a weak result."
        exit 9
    fi
fi
"$PY" "$ARENA/pair_elo.py" "$ARENA/$TAG.pgn" 2>&1 | tee -a "$OUT"
grep -E "^Elo|^Games|SPRT" "$ARENA/$TAG.log" | tail -4 | tee -a "$OUT"
say "NOTE: SPRT's terminal Elo is biased away from zero. A PASS means positive,"
say "      not this big; winners earn their number from a fixed-N confirmation."
say "AB $TAG COMPLETE $(date -u)"
