#!/bin/sh
# COPRIMALITY PRE-FLIGHT. Run BEFORE a multi-pairing round-robin; refuses to
# launch a schedule that cannot deal distinct openings.
#
# fastchess deals openings by the FLATTENED pair index -- opening =
# (round*NPAIR + pairing) mod NOPENINGS, NOPENINGS = -rounds -- so each pairing
# sees exactly  rounds / gcd(rounds, pairings)  distinct openings. When that
# gcd is not 1 the tournament silently plays fewer openings than it reports
# games, every existing gate passes (legality, count, forfeits, dormancy: none
# of them asks whether the games DIFFER), and the error bars read replays as
# independent pairs.
#
# This lane shipped that defect twice on 2026-08-17: two 900-game round-robins
# at -rounds 150 with 3 engines, gcd 3, 50 openings per pairing instead of 150.
# Neither was void -- the replays were not byte-identical, so they carried
# partial information -- but the intervals had to be recomputed clustered on
# the opening, and one of them went from 1.87 sigma to 1.08.
#
# usage: coprime_preflight.sh ROUNDS N_ENGINES
set -u
R=$1; E=$2
P=$(( E * (E - 1) / 2 ))
a=$R; b=$P
while [ "$b" -ne 0 ]; do t=$b; b=$((a % b)); a=$t; done
G=$a
PER=$((R / G))
echo "rounds $R, engines $E -> $P pairing(s), gcd(rounds,pairings) = $G"
# Each ROUND plays every pairing twice (colour-swapped), so a pairing gets
# R*2 games drawn from R/gcd openings -- a reuse factor of exactly 2*gcd.
echo "each pairing plays $((R * 2)) games drawn from $PER openings = ${G}x2 = $((2 * G))x reuse"
if [ "$G" -ne 1 ]; then
    echo "PRE-FLIGHT FAIL: gcd is $G, not 1 -- each pairing replays every opening"
    echo "  $G times. Choose -rounds coprime to $P (e.g. $((R + 1)) or $((R - 1)))"
    echo "  or run the pairings as separate two-engine matches."
    exit 1
fi
echo "PRE-FLIGHT PASS: schedule deals $PER distinct openings per pairing"
