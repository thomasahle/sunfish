#!/bin/bash
# STAGED, NOT ARMED -- the flag hammer for the classic builtin clock.
#
# This is NOT an Elo match and must never be read as one.  min40_4 parks
# LOWEST of every policy measured (0.22 s at 60+0.1 on the surrogate, below
# even the incumbent's 2.11 s): it wastes almost no clock, and the price of
# that is the thinnest flag margin in the field.  The surrogate cannot answer
# it -- it reproduces mechanisms, not flag safety, and its own README says so
# ("the surrogate reproduces mechanisms; it does not certify flag safety").
#
# So this arm asks one question at the most hostile TC available:
#
#     does it ever lose on time, or play an illegal move, at 1+0?
#
# Pass is ZERO time forfeits and ZERO illegal moves.  Elo is not a reading
# here; a large negative Elo at 1+0 with zero forfeits still PASSES, and a
# single forfeit fails it however good the score is.
#
#   ./STAGED_tm_min40_4_flag_hammer.sh          # prints the plan and exits 0
#   GO=1 ./STAGED_tm_min40_4_flag_hammer.sh     # runs it
#
set -euo pipefail

TC=1+0                    # the hostile end: ~25 moves before the floor knee
ROUNDS=150                # x2; enough that a knife-edge shows, cheap at 1+0
CAP=400
BOOK=${BOOK:-book3k.pgn}
ARM=${ARM:-min40_4.packed}
BASE=${BASE:-base.packed}
OUT=${OUT:-tm_min40_4_hammer}

cat <<PLAN
STAGED flag hammer -- classic builtin clock (NOT an Elo arm)
  engine1 : $ARM       min(wtime/40 + 0.9*winc, wtime/4)
  engine2 : $BASE      the incumbent, as a same-conditions control
  TC      : $TC   rounds $ROUNDS x2 (-repeat), cap $CAP
  SPRT    : none -- this is a safety screen, not a hypothesis test
  PASS    : zero time forfeits AND zero illegal moves, on the min40_4 arm
  FAIL    : any single forfeit or illegal move, reported naming the game
  note    : the incumbent is expected to forfeit here (its budget goes
            negative under a 2 s clock); that is the control working, and
            it does NOT excuse a forfeit on the arm
PLAN

if [ "${GO:-}" != "1" ]; then
    echo
    echo "NOT ARMED.  Re-run with GO=1 once a slot frees.  Exiting 0."
    exit 0
fi

for f in "$ARM" "$BASE" "$BOOK"; do
    [ -r "$f" ] || { echo "missing: $f" >&2; exit 2; }
done

if mkdir .boxlock 2>/dev/null; then
    echo "$$ classic-tm $(date -u +%FT%TZ) PRESENCE ONLY -- reclaim freely" \
        > .boxlock/owner
    trap 'rm -rf .boxlock' EXIT INT TERM
else
    echo "box busy ($(cat .boxlock/owner 2>/dev/null)); queueing behind it" >&2
    exit 3
fi

fastchess \
    -engine cmd="./$ARM" name=min40_4 \
    -engine cmd="./$BASE" name=legacy12 \
    -each proto=uci tc="$TC" restart=on \
    -openings file="$BOOK" format=pgn order=random \
    -rounds "$ROUNDS" -games 2 -repeat -maxmoves "$CAP" -recover \
    -concurrency "${CONC:-8}" \
    -pgnout "file=$OUT.pgn" \
    | tee "$OUT.log"

echo
echo "=== VERDICT (read these two lines, not the Elo) ==="
grep -ci "illegal" "$OUT.log" | sed 's/^/illegal-move mentions: /'
grep -c "on time" "$OUT.pgn" | sed 's/^/time forfeits (both arms): /'
echo "per-arm forfeits must be split by hand from $OUT.pgn before ledgering."
