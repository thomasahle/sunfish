#!/bin/bash
# Enlarge the corpus and label the new positions with OUR OWN SEARCH at 160k.
#
# STAGED, NOT ARMED. This script does not wait for anything and does not chain
# itself. It checks its GO marker ONCE and exits if the marker is absent, so
# the only way it runs is that somebody with a machine slot deliberately
# creates the marker and starts it. Four self-chaining waiters were killed on
# this box in one night; this is not a fifth.
#
# WHY IT EXISTS. The laptop corpus is EXHAUSTED: the sampling rule admits
# 19,689 unique positions across all 4,482 games and set20260813 already spent
# 19,491 of them, leaving 198. The phase-balanced experiment that is running
# now works around that by re-balancing labels we already have, which caps a
# flat draw at 2,198 per band (8,792 total) against d1's 19,434. Restoring the
# size means new GAMES, new POSITIONS, and new LABELS -- in that order.
#
# HOW MUCH IS NEEDED, measured not guessed. A flat draw at d1's N needs 4,858
# per band. Present supply is 8,374 / 5,952 / 2,198 / 2,910, so the shortfall
# is entirely in the two thin bands:
#
#     phase 12-17   need 2,660 more
#     phase 18-24   need 1,948 more
#
# Those bands are 11.3% and 15.0% of what the sampler yields, so ~23,500 fresh
# sampled positions -- on the order of 5,000 games -- covers both. The box
# arenas hold far more than that; they have to be pulled to wherever this runs.
#
# COST. Labelling ran at 0.58 pos/s per worker at 160,000 nodes. 4,608 new
# positions on 8 workers is ~17 minutes; on 4 workers, ~33. The teacher is
# unchanged and its sha is checked below, because a label set that mixes two
# engines is a mixture nobody can interpret.
#
# usage: label_corpus.sh GOFILE PGNDIR OUTDIR TARGET [NSHARDS] [NICE]
#
#   GOFILE   must already exist. Create it deliberately when a slot is granted.
#   PGNDIR   directory of *.pgn to harvest (rsync the box arenas here first)
#   OUTDIR   shards, result file and the packed .npz land here
#   TARGET   phase_balance.py target spec, e.g. "12-17=2660,18-24=1948,0-5=0,6-11=0"
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
GOFILE=${1:?GOFILE required}
PGNDIR=${2:?PGNDIR required}
OUTDIR=${3:?OUTDIR required}
TARGET=${4:?TARGET required}
NSHARDS=${5:-8}
NICEVAL=${6:-15}

TEACHER="$REPO/nnue_4k/pst_entry.py"
TEACHER_SHA=f2f0bdc87cd1a9e13737ead5b1fb8f19fe3d5c5a0ddf2c3c01291143a40c52b7
NODES=160000
RESULT="$OUTDIR/RESULT.txt"
mkdir -p "$OUTDIR"

# THE PRODUCER WRITES A RESULT ON ANY EXIT. A consumer that polls for a result
# file must never be able to wait forever on a job that died: every exit path,
# including a crash, a bad argument and a kill, leaves a verdict behind. This
# lane has been burned by silence that looked like progress.
finish() {
    code=$?
    { echo "exit_code   $code"
      echo "verdict     $([ $code -eq 0 ] && echo COMPLETE || echo FAILED)"
      echo "finished    $(date -u +%FT%TZ)"
      echo "shards      $(ls "$OUTDIR"/new160k_s*.jsonl 2>/dev/null | wc -l | tr -d ' ')"
      echo "records     $(cat "$OUTDIR"/new160k_s*.jsonl 2>/dev/null | grep -v '"meta"' | wc -l | tr -d ' ')"
      echo "npz         $([ -f "$OUTDIR/new160k.npz" ] && echo "$OUTDIR/new160k.npz" || echo none)"
    } >> "$RESULT"
    exit $code
}
trap finish EXIT

: > "$RESULT"
echo "started     $(date -u +%FT%TZ)"          >> "$RESULT"
echo "host        (bench box or laptop)"        >> "$RESULT"
echo "teacher     nnue_4k/pst_entry.py @ $NODES nodes" >> "$RESULT"

# ---- the GO gate: checked ONCE, never waited on -----------------------------
if [ ! -f "$GOFILE" ]; then
    echo "NOT ARMED: no GO marker at $GOFILE" | tee -a "$RESULT"
    echo "This script never waits. Create the marker when a slot is granted." >> "$RESULT"
    exit 2
fi
echo "go_marker   $GOFILE" >> "$RESULT"

# ---- the teacher must be the one that made the existing labels --------------
# A MISSING teacher is not a CHANGED teacher. The first version of this check
# reported "TEACHER CHANGED: != deadbeef" when the file was simply not there,
# which sends whoever reads the result file looking for a commit that never
# happened. Distinguish the two before comparing anything.
if [ ! -f "$TEACHER" ]; then
    echo "TEACHER NOT FOUND at $TEACHER (REPO resolved to $REPO)" | tee -a "$RESULT"
    echo "Run this script from its committed location, or fix REPO." >> "$RESULT"
    exit 4
fi
have=$(shasum -a 256 "$TEACHER" 2>/dev/null | cut -d' ' -f1)
[ -n "$have" ] || have=$(sha256sum "$TEACHER" | cut -d' ' -f1)
if [ "$have" != "$TEACHER_SHA" ]; then
    echo "TEACHER CHANGED: $have != $TEACHER_SHA" | tee -a "$RESULT"
    echo "Labels from a different engine are not comparable with distill160k."  >> "$RESULT"
    exit 3
fi
echo "teacher_sha $have  (matches distill160k)" >> "$RESULT"

# ---- census, then select the top-up -----------------------------------------
# Both existing sets are excluded: a position that is already labelled must not
# be relabelled into a second row of the same corpus.
EXCL="$REPO/tools/tune/data/set20260813.npz $REPO/tools/tune/data/distill160k.npz"
nice -n "$NICEVAL" python3 "$REPO/tools/tune/phase_balance.py" census "$PGNDIR" $EXCL \
    2>&1 | tee -a "$RESULT"
nice -n "$NICEVAL" python3 "$REPO/tools/tune/phase_balance.py" select "$PGNDIR" \
    "$OUTDIR/topup" "$TARGET" $EXCL 2>&1 | tee -a "$RESULT"

FENS="$OUTDIR/topup.fen"
n=$(wc -l < "$FENS" | tr -d ' ')
echo "selected    $n positions -> $FENS" >> "$RESULT"

# ---- label, sharded, nice'd --------------------------------------------------
# One worker per shard, every worker at the same nice level. Cotenancy is
# logged rather than assumed, and labelling never gates a match.
echo "cotenancy   $(uptime)" >> "$RESULT"
for s in $(seq 0 $((NSHARDS - 1))); do
    nice -n "$NICEVAL" python3 "$REPO/tools/tune/distill_label.py" \
        "$FENS" "$OUTDIR/new160k_s$s.jsonl" "$NODES" "$s" "$NSHARDS" \
        > "$OUTDIR/new160k_s$s.log" 2>&1 &
done
wait

# ---- pack, and only then declare a dataset ----------------------------------
nice -n "$NICEVAL" python3 "$REPO/tools/tune/distill_pack.py" \
    "$OUTDIR/new160k.npz" "$OUTDIR/new160k_s*.jsonl" 2>&1 | tee -a "$RESULT"

echo "NEXT (not chained, run deliberately):" >> "$RESULT"
echo "  merge new160k.npz with distill160k.npz, then" >> "$RESULT"
echo "  phase_balance.py select MERGED.npz bal_full flat:4858" >> "$RESULT"
