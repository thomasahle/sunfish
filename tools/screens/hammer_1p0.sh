#!/bin/bash
# THE 1+0 HAMMER: the zero-tolerance legality confirmation for the structural
# bestmove floor.
#
# 1+0 sudden death is the regime that produced 19 illegal-move forfeits in 400
# games (seedtimed 2026-08-14): the driver budget min(wtime/12, wtime/2 - 1)
# goes NEGATIVE below 2 s of clock, so the in-search deadline is already past
# when `go` arrives, and any position whose first root fail-high needs more
# than 2,048 nodes -- the poll granularity of both stop conditions -- used to
# answer `bestmove (none)`. The floor (driver v3 / the entry's builtin
# fallback) claims that class is now impossible BY CONSTRUCTION. This script
# is the confirmation: engine vs itself at 1+0, and the REQUIRED result is
# ZERO illegal moves. Any illegal move fails the run and names the game.
#
# STAGED, NOT ARMED. Checks its GO marker ONCE and exits if absent. It never
# waits and never chains itself: create the marker when an arena slot is free.
# This is a CORRECTNESS run, not a strength measurement -- self-play at 1+0
# says nothing about Elo and no number from it may be quoted as one.
#
# usage: hammer_1p0.sh GOFILE ARENA [WRAPPER] [GAMES] [CONC] [SRAND]
#   WRAPPER defaults to $ARENA/w_base.sh (the current entry build's wrapper).
set -u
GOFILE=${1:?GOFILE required}
ARENA=${2:?ARENA required}
WRAPPER=${3:-$ARENA/w_base.sh}
GAMES=${4:-100}
CONC=${5:-8}
SRAND=${6:-20260814}

FC=$HOME/sunfish-bench/fastchess-linux-x86-64/fastchess
BOOK=$ARENA/openings_2k.epd
RESULT=$ARENA/RESULT_hammer.txt
TAG=hammer1p0

finish() {
    code=$?
    { echo "exit_code    $code"
      echo "verdict      $([ $code -eq 0 ] && echo ZERO-ILLEGAL-CONFIRMED || echo FAILED)"
      echo "finished     $(date -u +%FT%TZ)"
      echo "games        $(grep -c '^\[Result' "$ARENA/$TAG.pgn" 2>/dev/null || echo 0)"
    } >> "$RESULT"
    exit $code
}
trap finish EXIT

mkdir -p "$ARENA"
: > "$RESULT"
say() { echo "$@" | tee -a "$RESULT"; }
say "1+0 HAMMER  started $(date -u +%FT%TZ)"

if [ ! -f "$GOFILE" ]; then
    say "NOT ARMED: no GO marker at $GOFILE"
    say "This script never waits. Create the marker when a slot is granted."
    exit 2
fi
say "go_marker    $GOFILE"
for f in "$FC" "$BOOK" "$WRAPPER"; do
    [ -e "$f" ] || { say "MISSING: $f"; exit 3; }
done
say "wrapper      $WRAPPER"
say "cotenancy    $(uptime)"

nice -n 5 "$FC" \
  -engine cmd="$WRAPPER" name=hammerA \
  -engine cmd="$WRAPPER" name=hammerB \
  -each proto=uci tc=1+0 \
  -openings file="$BOOK" format=epd order=random -srand "$SRAND" \
  -rounds $((GAMES / 2)) -games 2 -repeat -concurrency "$CONC" -recover \
  -pgnout file="$ARENA/$TAG.pgn" > "$ARENA/$TAG.log" 2>&1

n=$(grep -c '^\[Result' "$ARENA/$TAG.pgn")
say "games        $n of $GAMES"
say "none-emits   $(grep -c '(none)' "$ARENA/$TAG.log")"
say "forfeits     $(grep -ci 'time forfeit' "$ARENA/$TAG.pgn")   (time forfeits are losses, not violations)"
[ "$n" -ge "$GAMES" ] || { say "SHORT RUN: only $n games -- not a confirmation"; exit 9; }

n_ill=$(grep -ci 'illegal move' "$ARENA/$TAG.pgn")
if [ "$n_ill" -gt 0 ]; then
    say "ZERO-TOLERANCE FAIL: $n_ill illegal-move game(s) -- the offending games:"
    awk '/^\[Round /{r=$0} /^\[White /{w=$0} /^\[Black /{b=$0} /^\[FEN /{f=$0}
         /makes an illegal move/{printf "    %s %s %s %s :: %s\n", r, w, b, f, $0}' \
        "$ARENA/$TAG.pgn" | tee -a "$RESULT"
    exit 8
fi
say "ZERO illegal moves in $n games at 1+0 -- floor confirmed"
say "1+0 HAMMER COMPLETE $(date -u +%FT%TZ)"
