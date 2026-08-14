#!/bin/bash
# The gamma seed's NON-INFERIORITY screen, plus the timed correctness check
# that fixed-node play structurally cannot perform.
#
# STAGED, NOT ARMED. Checks its GO marker ONCE and exits if absent. It never
# waits, never polls for a quiet machine and never chains itself: four
# self-chaining waiters were killed on the box in one night and this is not a
# fifth. Somebody with a slot creates the marker and starts it.
#
# WHAT IS BEING BOUGHT. `search()` seeds every search at gamma = 0 and the root
# stores a move only on a fail-high, so a build whose first root fail-high needs
# more than 2,048 nodes -- the poll granularity of BOTH stop conditions --
# answers `bestmove (none)`. Seeding at `pos.score - 150` makes the first probe
# cheap and one-sided. Measured over 505 positions: base 780 -> 171, d1 1,896 ->
# 537, d8 2,059 FAIL -> 478, b1 2,011 -> 728, b8 2,433 FAIL -> 394. Price: +5
# bytes on every base, 1.0000x nodes to complete depth 8, same move 40/40.
#
# THIS IS A SEARCH CHANGE. If it passes it lands on nnue-4k in sunfish-packed,
# not on eval-decode-track.
#
# STAGE 1, SPRT, NON-INFERIORITY. engine1 = seed, engine2 = base, elo0 = -10,
# elo1 = 0. fastchess states the bounds in ENGINE1's frame -- verified against
# our own C2 record, where base ran first with elo0=0 elo1=10 and "H1 was
# accepted" meant BASE was better. So:
#     H1 accepted -> seed is not inferior by 10 Elo -> LAND
#     H0 accepted -> seed is worse than -10          -> DROP
#     cap reached -> UNDECIDED, reported as undecided, never as a point estimate
#
# STAGE 2, THE TIMED CHECK, and why it is NOT 10+0.1. No abort can land before
# node 2,048, so any build with max first yield <= 2,048 is immune to the
# `(none)` class at EVERY time control -- base (780) and seed (171) both are.
# The driver budget is think = min(wtime/12 + 0.9*winc, wtime/2 - 1) s, and at
# the measured 42,473 nps 2,048 nodes is ~48 ms. At 10+0.1 the 0.9*winc term
# alone floors think at 90 ms (~3,800 nodes), so nothing is ever aborted early
# and the check would return zero for both arms whatever is true -- a gate that
# passes everything. At 1+0, think = wtime/12 falls under 48 ms whenever
# wtime < 0.58 s, which happens in the endgame of essentially every game.
# So: 1+0, and the arm under test is b8 (max first yield 2,433, ABOVE the
# cliff) against b8seed, with base-vs-seed as the control pair that shows the
# time control alone does not manufacture the failure.
# This was a CORRECTNESS COUNT until 2026-08-14 (it measured b8 15/200 and
# b8seed 4/200 illegal-move forfeits, all `bestmove (none)` after an abort
# beat the first root fail-high). The structural bestmove floor closed that
# class by construction, so stage 2 is now a LEGALITY STRESS with a zero-
# tolerance verdict: any illegal move from any arm fails the whole screen.
# 1+0 on a shared box is far too noisy for a strength claim and none is made.
#
# usage: seed_screen.sh GOFILE ARENA [ROUNDS] [CONC] [SRAND] [TIMED_GAMES]
set -u
GOFILE=${1:?GOFILE required}
ARENA=${2:?ARENA required}
ROUNDS=${3:-500}
CONC=${4:-8}
SRAND=${5:-20260816}
TGAMES=${6:-200}

FC=$HOME/sunfish-bench/fastchess-linux-x86-64/fastchess
PY=$HOME/sunfish-bench/pypy3.11-v7.3.20-linux64/bin/pypy3
BOOK=$ARENA/openings_2k.epd
RESULT=$ARENA/RESULT_seedscreen.txt
NODES=20000

# THE PRODUCER WRITES A RESULT ON ANY EXIT, including a crash or a kill. A
# consumer polling for a verdict must never be able to wait forever on a job
# that died.
finish() {
    code=$?
    { echo "exit_code    $code"
      echo "verdict      $([ $code -eq 0 ] && echo COMPLETE || echo FAILED)"
      echo "finished     $(date -u +%FT%TZ)"
      echo "sprt_games   $(grep -c '^\[Result' "$ARENA/seedscreen.pgn" 2>/dev/null || echo 0)"
      echo "timed_games  $(grep -c '^\[Result' "$ARENA/seedtimed.pgn" 2>/dev/null || echo 0)"
      echo "control_games $(grep -c '^\[Result' "$ARENA/seedctl.pgn" 2>/dev/null || echo 0)"
    } >> "$RESULT"
    exit $code
}
trap finish EXIT

mkdir -p "$ARENA"
: > "$RESULT"
say() { echo "$@" | tee -a "$RESULT"; }
say "SEED SCREEN  started $(date -u +%FT%TZ)"

# ---- ZERO TOLERANCE ---------------------------------------------------------
# "We should never accept illegal moves. 15 is too much, so is 4." Any illegal
# move by any of our arms, in any run of this screen, is a FAIL that names the
# game -- never a count to be reported. The structural bestmove floor (driver
# v3 / the entry's builtin fallback) makes the known "(none)" class impossible
# by construction; this gate is what notices if any class, known or new, ever
# comes back.
require_zero_illegal() {
    pgn=$1
    n_ill=$(grep -ci 'illegal move' "$pgn" 2>/dev/null || true)
    if [ "${n_ill:-0}" -gt 0 ]; then
        say "ZERO-TOLERANCE FAIL: illegal move(s) in $pgn -- the offending games:"
        awk '/^\[Round /{r=$0} /^\[White /{w=$0} /^\[Black /{b=$0} /^\[FEN /{f=$0}
             /makes an illegal move/{printf "    %s %s %s %s :: %s\n", r, w, b, f, $0}' \
            "$pgn" | tee -a "$RESULT"
        exit 8
    fi
}

# ---- the GO gate: checked ONCE, never waited on -----------------------------
if [ ! -f "$GOFILE" ]; then
    say "NOT ARMED: no GO marker at $GOFILE"
    say "This script never waits. Create the marker when a slot is granted."
    exit 2
fi
say "go_marker    $GOFILE"

for f in "$FC" "$PY" "$BOOK"; do
    [ -e "$f" ] || { say "MISSING: $f"; exit 3; }
done
for e in e_base e_seed e_b8 e_b8seed; do
    [ -f "$ARENA/bin/$e.py" ] || { say "MISSING ARM: $ARENA/bin/$e.py"; exit 4; }
    [ -x "$ARENA/w_${e#e_}.sh" ] || { say "MISSING WRAPPER: $ARENA/w_${e#e_}.sh"; exit 5; }
done
say "cotenancy    $(uptime)"
say "other users  $(ps -eo user,args --sort=-pcpu | awk 'NR>1 && $1 !~ /^'"$USER"'/' | head -3 | cut -c1-90 | tr '\n' ';')"

# ---- gates before games -----------------------------------------------------
# A screen that spends games on a build that cannot always produce a legal move
# is spending them to rediscover that. Seconds, not games.
for e in e_base e_seed e_b8 e_b8seed; do
    r=$("$PY" "$ARENA/legality_gate.py" "$ARENA/bin/$e.py" 300 2>&1)
    echo "$r" | grep -q "GATE PASSED" || { say "LEGALITY FAILED: $e"; echo "$r" | tail -8 >> "$RESULT"; exit 6; }
    y=$("$PY" "$ARENA/first_yield_gate.py" "$ARENA/bin/$e.py" "$ARENA/first_yield_fens.fen" 2>&1 \
        | grep -oE "MAX [0-9]+" | grep -oE "[0-9]+")
    say "  gate PASS  $e   legality 100/100   first-yield max ${y:-?}"
done
# b8 MUST still sit above the 2048-node first-yield cliff -- it is the stress
# control for stage 2: the arm guaranteed to hit aborts before its first root
# fail-high, i.e. the arm that actually exercises the structural bestmove
# floor. If a rebuild silently lowers it under the cliff, stage 2 stresses
# nothing and must not be reported as a clean result.
y8=$("$PY" "$ARENA/first_yield_gate.py" "$ARENA/bin/e_b8.py" "$ARENA/first_yield_fens.fen" 2>&1 \
     | grep -oE "MAX [0-9]+" | grep -oE "[0-9]+")
[ "${y8:-0}" -gt 2048 ] || { say "STAGE-2 CONTROL VOID: e_b8 first yield ${y8:-?} <= 2048, nothing to catch"; exit 7; }

# ---- STAGE 1: fixed-node non-inferiority SPRT -------------------------------
say ""
say "STAGE 1  SPRT non-inferiority   engine1=seed engine2=base  elo0=-10 elo1=0"
say "         H1 accepted => seed NOT inferior by 10 => LAND; H0 => DROP"
say "         nodes $NODES  rounds $ROUNDS  conc $CONC  srand $SRAND"
nice -n 5 "$FC" \
  -engine cmd="$ARENA/w_seed.sh" name=seed \
  -engine cmd="$ARENA/w_base.sh" name=base \
  -each proto=uci nodes="$NODES" \
  -openings file="$BOOK" format=epd order=random -srand "$SRAND" \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" -recover \
  -sprt elo0=-10 elo1=0 alpha=0.05 beta=0.05 \
  -draw movenumber=40 movecount=8 score=10 -resign movecount=4 score=500 \
  -pgnout file="$ARENA/seedscreen.pgn" > "$ARENA/seedscreen.log" 2>&1
n=$(grep -c '^\[Result' "$ARENA/seedscreen.pgn")
say "  games $n  $([ "$n" -ge $((ROUNDS*2)) ] && echo 'UNDECIDED-AT-CAP (report as undecided)' || echo 'stopped early by SPRT')"
say "  time forfeits $(grep -ci 'time forfeit' "$ARENA/seedscreen.pgn")"
require_zero_illegal "$ARENA/seedscreen.pgn"
"$PY" "$ARENA/pair_elo.py" "$ARENA/seedscreen.pgn" 2>&1 | tee -a "$RESULT"
grep -E "^Elo|^Games|SPRT" "$ARENA/seedscreen.log" | tail -4 | tee -a "$RESULT"

# ---- STAGE 2: the timed legality stress -------------------------------------
# Runs regardless of stage 1's verdict. This used to be a CORRECTNESS COUNT
# ("does the seed reduce the (none) class") -- that class is now closed by
# construction (the structural bestmove floor), so the count question is dead:
# the only acceptable count is zero, enforced by require_zero_illegal, and the
# seed's pair rule is DEPTH-QUALITY only. What stage 2 still buys is the
# stress itself: 1+0 is the regime that produced 19 forfeits in 400 games, and
# the floor's claim of impossibility is checked here on every run.
timed() {
    tag=$1; a=$2; b=$3
    nice -n 5 "$FC" \
      -engine cmd="$ARENA/w_$a.sh" name="$a" \
      -engine cmd="$ARENA/w_$b.sh" name="$b" \
      -each proto=uci tc=1+0 \
      -openings file="$BOOK" format=epd order=random -srand "$SRAND" \
      -rounds $((TGAMES / 2)) -games 2 -repeat -concurrency "$CONC" -recover \
      -pgnout file="$ARENA/$tag.pgn" > "$ARENA/$tag.log" 2>&1
    say "  $tag ($a vs $b, 1+0): games $(grep -c '^\[Result' "$ARENA/$tag.pgn")" \
        " none=$(grep -c '(none)' "$ARENA/$tag.log")" \
        " forfeit=$(grep -ci 'time forfeit' "$ARENA/$tag.pgn")"
    require_zero_illegal "$ARENA/$tag.pgn"
}
say ""
say "STAGE 2  timed legality stress at 1+0 -- NOT an Elo measurement"
say "         REQUIRED: zero illegal moves from EVERY arm. b8 sits above the"
say "         2048-node first-yield cliff on purpose: it is the arm that MUST"
say "         exercise the structural bestmove floor, and it must survive it."
timed seedtimed b8 b8seed
timed seedctl   base seed
say ""
say "SEED SCREEN COMPLETE $(date -u +%FT%TZ)"
