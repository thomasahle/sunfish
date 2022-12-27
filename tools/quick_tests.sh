#!/bin/bash
set -e -u -o pipefail
# Run with `py command` or `py command --debug`
TOOLS=$(dirname "$0")
echo "$TOOLS"
T="python3 $TOOLS/tester.py"

$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate1.fen --depth 2

# Stockfish finds this at around depth 14 with normal search, but faster
# if using "go mate". Currently it's too deep for sunfish to find.
#$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/nullmove_mates.fen --depth 12

# These mates should be findable at depth=4, but because of null-move
# We need to go to depth=6.
$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate2.fen --depth 6 --quick

$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate2.fen --depth 4
echo "Should be about 73/130"
