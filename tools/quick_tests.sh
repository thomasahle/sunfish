#!/bin/bash
set -e -u -o pipefail
# Run with `py command` or `py command --debug`
TOOLS=$(dirname "$0")
TESTF="$(dirname "$TOOLS")/tests/files"
echo "$TOOLS"
T="python3 $TOOLS/tester.py"

echo "Terminal and eventual-mate correctness..."
# Every real edge costs at most C=3. A mate proof of k plies is therefore
# armed at D >= C*k+4; mate-in-n has k=2*n-1. These curated witnesses keep
# the theorem-derived depths affordable in Python CI.
$T "$1" ${2:-"--quiet"} draw $TESTF/stalemate0.fen --depth 1 --floor 4
$T "$1" ${2:-"--quiet"} mate $TESTF/mate1.fen --depth 7 --floor 8
$T "$1" ${2:-"--quiet"} mate $TESTF/mate2_eventual.fen --depth 13 --floor 5
$T "$1" ${2:-"--quiet"} mate $TESTF/mate3_eventual.fen --depth 19 --floor 2
echo

echo "Tactical strength regressions..."
$T "$1" ${2:-"--quiet"} best $TESTF/win_at_chess_test.epd --depth 8 --floor 207
$T "$1" ${2:-"--quiet"} best $TESTF/bratko_kopec_test.epd --depth 8 --floor 11
$T "$1" ${2:-"--quiet"} best $TESTF/3fold.epd --depth 4 --floor 2
echo

echo "Perft"
$T "$1" ${2:-"--quiet"} perft $TESTF/perft.epd --depth 2
echo

echo "Self play"
$T "$1" ${2:-"--quiet"} self-play --time 1000 --inc 100
echo
