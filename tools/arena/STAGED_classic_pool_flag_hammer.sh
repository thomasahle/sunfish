#!/bin/bash
# STAGED, NOT ARMED -- the sudden-death safety question the classic pool's
# landing leaves open, and the one the surrogate explicitly cannot answer
# ("the surrogate reproduces mechanisms; it does not certify flag safety",
# tools/ctwin/README.md).
#
#   arm    the classic packed artifact carrying the POOL
#   base   the same artifact carrying min40_4 (master before 2026-08-17)
#   TC     1+0 -- and 1 s is BELOW the pool's own knee, (M+2)*O = 8.4 s, so
#          P == 0 for the whole game: t_soft collapses to the 0.05 s floor and
#          the quarter-clock clamp is unreachable.  This is the regime the
#          landing DISCLOSED and did not fix, and the driver's own arm scored
#          -209.91 +/- 60.11 in it.
#
# THIS IS NOT AN ELO ARM AND MUST NEVER BE READ AS ONE.  A large negative
# score with zero forfeits PASSES: the disclosed cost is that the pool plays
# a sudden-death endgame shallower, and the question here is only whether it
# stays LEGAL and NEVER FLAGS while doing so.  One forfeit fails it however
# good the score.  The Elo column exists to be ignored.
#
#   PASS   zero time forfeits AND zero illegal moves AND zero `(none)`
#          on the POOL arm.  The base is allowed to forfeit -- if it does,
#          that is the control working and it does not excuse the arm.
#
# Why it is staged rather than run: the fixed-N confirmation at 30+1 owned the
# venue when this was written, and a second timed match cotenant with it is
# exactly the contamination that voided two earlier runs.  It does not
# self-launch.
#
#   ./STAGED_classic_pool_flag_hammer.sh          # prints the plan, exits 0
#   GO=1 ./STAGED_classic_pool_flag_hammer.sh     # runs it
#
set -euo pipefail

TC=${TC:-1+0}
ROUNDS=${ROUNDS:-50}          # x2 games with -repeat
CONC=${CONC:-4}               # deliberately low: a flag test must not be
                              # starved by its own concurrency
BOOK=${BOOK:-book3k.pgn}      # PGN: the packed loop parses only
                              # "position startpos moves ..."
ARM=${ARM:-pool.packed}
BASE=${BASE:-min40.packed}
OUT=${OUT:-classic_pool_hammer}

cat <<PLAN
STAGED flag hammer -- classic builtin clock, POOL vs min40_4
  engine1 : $ARM    soft = min(P/40, A/4), wall = min(5*soft, A/2)
  engine2 : $BASE   min(wtime/40 + 0.9*winc, wtime/4)
  TC      : $TC   rounds $ROUNDS x2 (-repeat), concurrency $CONC
  book    : $BOOK (PGN), adjudication NONE, NO SPRT
  PASS    : zero time forfeits AND zero illegal AND zero (none) on $ARM
  NOT an Elo arm.  The score is reported and must not be acted on.
  readings: forfeits per arm, illegal per arm, (none) per arm, end-clock
            median/min per arm, and the move at which each arm first spends
            the 0.05s floor (the pool is predicted to do so from move 1)
PLAN

if [ "${GO:-}" != "1" ]; then
    echo
    echo "NOT ARMED.  Re-run with GO=1 once the box is free.  Exiting 0."
    exit 0
fi

for f in "$ARM" "$BASE" "$BOOK"; do
    [ -r "$f" ] || { echo "missing: $f" >&2; exit 2; }
done

# Presence marker, not exclusivity: an owner file that says so in writing and
# invites any lane that needs the window to reclaim it.
if mkdir .boxlock 2>/dev/null; then
    echo "$$ classic-pool-hammer $(date -u +%FT%TZ) PRESENCE ONLY -- reclaim freely" \
        > .boxlock/owner
    trap 'rm -rf .boxlock' EXIT INT TERM
else
    echo "box busy ($(cat .boxlock/owner 2>/dev/null)); queueing behind it" >&2
    echo "re-run when it clears -- this script does not preempt." >&2
    exit 3
fi

fastchess \
    -engine cmd="./$ARM" name=pool \
    -engine cmd="./$BASE" name=min40_4 \
    -each proto=uci tc="$TC" restart=on \
    -openings file="$BOOK" format=pgn order=random \
    -rounds "$ROUNDS" -games 2 -repeat -recover \
    -concurrency "$CONC" \
    -pgnout "file=$OUT.pgn" \
    | tee "$OUT.log"

# Bare `grep -c`: never `|| echo 0`, which prints 0 AND exits nonzero and so
# silently disarms the check.
echo
echo "games        $(grep -c '^\[Result' "$OUT.pgn")"
echo "ILLEGAL      $(grep -ci 'illegal move' "$OUT.pgn")"
echo "FORFEITS     $(grep -ci 'loses on time' "$OUT.pgn")"
grep -o '\[Termination "[a-z ]*"\]' "$OUT.pgn" | sort | uniq -c
echo "Read the tripwires FIRST and the score never."
