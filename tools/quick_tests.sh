#!/bin/bash
set -e -u -o pipefail
# Run with `py command` or `py command --debug`
TOOLS=$(dirname "$0")
echo "$TOOLS"
T="python3 $TOOLS/tester.py"

echo "Mate in 1..."
#$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate1.fen --depth 2
$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate1.fen --movetime 10000
echo

# Stockfish finds this at around depth 14 with normal search, but faster
# if using "go mate". Currently it's too deep for sunfish to find.
# $T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/nullmove_mates.fen --depth 12

# These mates should be findable at depth=4, but because of null-move
# We need to go to depth=6.
echo "Mate in 2..."
#$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate2.fen --mate-depth 4 --limit 20
$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate2.fen --movetime 10000 --limit 20
echo

echo "Mate in 3..."
$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate3.fen --movetime 10000 --limit 5
echo

echo "Stalemate in 0..."
$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate0.fen --movetime 10000
echo

echo "Stalemate in 1..."
$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate1.fen --movetime 10000
echo

echo "Stalemate in 2+"
# Regression floor at fixed depth (fully deterministic): fail if the
# count drops below the current baseline; raise the floor when a change
# genuinely improves it. Historical note: the old "(Should be about
# 85/130)" was an artifact of a pre-89d6741 FEN-loader bug that built
# test positions with score=0, making nearly every quiet line count as
# a found draw; the honest baseline at depth 4 is below.
STALE2_FLOOR=10
stale2_out=$($T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate2.fen --depth 4)
echo "$stale2_out"
stale2_n=$(echo "$stale2_out" | grep -o "Succeeded in [0-9]*" | grep -o "[0-9]*$" | tail -1)
if [ "${stale2_n:-0}" -lt "$STALE2_FLOOR" ]; then
    echo "FAIL: stalemate2 regression: got ${stale2_n:-none}/130, floor is $STALE2_FLOOR/130"
    exit 1
fi
echo

echo "Other puzzles..."
$T "$1" ${2:-"--quiet"} best $TOOLS/test_files/win_at_chess_test.epd --movetime 100
echo

echo "Perft"
$T "$1" ${2:-"--quiet"} perft $TOOLS/test_files/perft.epd --depth 2
echo

echo "Self play"
$T "$1" ${2:-"--quiet"} self-play --time 1000 --inc 100
echo


