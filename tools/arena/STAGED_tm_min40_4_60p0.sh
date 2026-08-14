#!/bin/bash
# STAGED, NOT ARMED -- the confirmation SPRT for the classic builtin clock.
#
#   arm      min40_4 = min(wtime / 40 + 0.9 * winc, wtime / 4)
#   vs       the classic incumbent, min(wtime/12 + 0.9*winc, wtime/2 - 1000)
#   venue    packed CLASSIC artifact (pack.sh strips the minifier-hide block,
#            so this loop is the one the artifact runs; a checkout reaches
#            sunfish_ui/uci.py instead and is untouched by this change)
#
# RULE THIS SCRIPT EXISTS TO OBEY: the surrogate ranks, the real clock
# confirms.  min40_4 already won the surrogate's classic-builtin venue
# (tools/ctwin/README.md, ranking of 2026-08-15) -- this is the ONE real-clock
# arm that pre-registration bought, and it is a CONFIRMATION, not a search.
# Nothing here may be retuned on the strength of its own result: the single
# permitted remedy is fixed in nnue_4k/MEASUREMENTS.md and needs a rerun.
#
# It does not self-launch.  It runs only when a human passes GO, and only
# when the box is actually free -- it queues behind the resident matches
# rather than claiming the box out from under them.
#
#   ./STAGED_tm_min40_4_60p0.sh            # prints the plan and exits 0
#   GO=1 ./STAGED_tm_min40_4_60p0.sh       # runs it
#
set -euo pipefail

TC=60+0
ROUNDS=200                # x2 games with -repeat
CAP=400
ELO0=0                    # engine1's frame: min40_4 is engine1
ELO1=20
BOOK=${BOOK:-book3k.pgn}  # PGN, not EPD: the packed loop parses only
                          # "position startpos moves ..."
ARM=${ARM:-min40_4.packed}
BASE=${BASE:-base.packed}
OUT=${OUT:-tm_min40_4_60p0}

cat <<PLAN
STAGED confirmation arm -- classic builtin clock
  engine1 : $ARM       min(wtime/40 + 0.9*winc, wtime/4)
  engine2 : $BASE      min(wtime/12 + 0.9*winc, wtime/2 - 1000)
  TC      : $TC   rounds $ROUNDS x2 (-repeat), cap $CAP
  SPRT    : elo0=$ELO0 elo1=$ELO1 alpha=0.05 beta=0.05 model=normalized
  book    : $BOOK (PGN)
  adjudication: NONE -- a drained clock kills long level endgames, which is
                the exact class -draw would delete before the defect shows
  readings: W/L/D + pentanomial Elo, LLR/LOS, ILLEGAL MOVES (zero tolerance),
            time forfeits per arm, end-clock median/min and games under 2 s,
            move at which the clock first falls under 2.4 s and how many
            follow, per-arm median and max move time, and the realized park
            altitude (min40_4 is predicted to park LOWEST -- that is the
            number this arm exists to put a real clock behind)
PLAN

if [ "${GO:-}" != "1" ]; then
    echo
    echo "NOT ARMED.  Re-run with GO=1 once a slot frees.  Exiting 0."
    exit 0
fi

for f in "$ARM" "$BASE" "$BOOK"; do
    [ -r "$f" ] || { echo "missing: $f" >&2; exit 2; }
done

# Presence marker, not a claim: an owner file that says so in writing and
# invites any lane that needs the window to reclaim it.
if mkdir .boxlock 2>/dev/null; then
    echo "$$ classic-tm $(date -u +%FT%TZ) PRESENCE ONLY -- reclaim freely" \
        > .boxlock/owner
    trap 'rm -rf .boxlock' EXIT INT TERM
else
    echo "box busy ($(cat .boxlock/owner 2>/dev/null)); queueing behind it" >&2
    echo "re-run when it clears -- this script does not preempt." >&2
    exit 3
fi

exec fastchess \
    -engine cmd="./$ARM" name=min40_4 \
    -engine cmd="./$BASE" name=legacy12 \
    -each proto=uci tc="$TC" restart=on \
    -openings file="$BOOK" format=pgn order=random \
    -rounds "$ROUNDS" -games 2 -repeat -maxmoves "$CAP" -recover \
    -sprt elo0=$ELO0 elo1=$ELO1 alpha=0.05 beta=0.05 model=normalized \
    -concurrency "${CONC:-8}" \
    -pgnout "file=$OUT.pgn" \
    | tee "$OUT.log"
