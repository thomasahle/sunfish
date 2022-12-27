#!/bin/bash
set -e -u -o pipefail
# Run with `py command` or `py command --debug`
TOOLS=$(dirname "$0")/tools
T="python3 $TOOLS/tester.py"
$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate1.fen --depth 4
# Stockfish finds this at around depth 14 with normal search, but faster
# if using "go mate". Currently it's too deep for sunfish to find.
#$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/nullmove_mates.fen --depth 12
#$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/mate2.fen --depth 6 --quick
$T "$1" ${2:-"--quiet"} draw $TOOLS/test_files/stalemate2.fen --depth 4
#$T "$1" ${2:-"--quiet"} mate $TOOLS/test_files/statemate2.fen --depth 6 --quick
