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
# Regression floor (fully deterministic at fixed depth). Historical note:
# the old "(Should be about 85/130)" was an artifact of a pre-89d6741
# FEN-loader bug (positions built with score=0) that made nearly every
# quiet line count as a found draw. Raise the floor when a change
# genuinely improves it.
# 15 -> 13: two of the old successes were false positives - transient
# exact-0 info lines produced by crossed TT bounds (settled scores were
# -266/-246); see tests/test_tt_consistency.py.  13 -> 17: the kcx
# rework's exact terminal reporting recovers four genuine detections
# the masked bounds had been hiding.
$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate2.fen --depth 4 --floor 17
echo

echo "Regression floors (fixed depth, deterministic; raise when improved)..."
$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate1.fen --depth 2 --floor 8
$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate2.fen --depth 6 --limit 20 --floor 20
$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate3.fen --depth 8 --limit 5 --floor 5
$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate4.fen --depth 8 --limit 10 --floor 1
$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate0.fen --depth 3 --floor 4
$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate1.fen --depth 4 --floor 2
$T "$1" ${2:-"--quiet"} best $TOOLS/test_files/win_at_chess_test.epd --depth 3 --floor 87
$T "$1" ${2:-"--quiet"} best $TOOLS/test_files/bratko_kopec_test.epd --depth 4 --floor 4
$T "$1" ${2:-"--quiet"} best $TOOLS/test_files/3fold.epd --depth 4 --floor 2
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


