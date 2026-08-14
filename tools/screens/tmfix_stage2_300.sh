#!/bin/sh
# STAGE 2 -- SPECCED, NOT ARMED. Running this without ARMED=1 prints the spec
# and exits, on purpose: it costs ~6-9 h of a bench-box slot and needs a slot
# decision from the coordinator, not a lane's unilateral launch.
#
#     ARMED=1 sh stage2_300.sh          # only after the stage-1 PASS
#
# PRECONDITION (pre-registered in nnue_4k/MEASUREMENTS.md before stage 1 ran):
# stage 1 PASSED, i.e. its SPRT accepted H1 or the 95% lower bound of
# (tmfix - oldtm) was > 0 at cap, AND tmfix's time-forfeit count was <= 1% of
# its games and strictly below oldtm's. If stage 1 failed or came out neutral,
# this run is NOT justified -- do not arm it.
#
# WHAT IT BUYS, and what it does not: the appendix's DIRECT bar (tmfix - oldtm
# at the benchmark TC). It does NOT recover the ANCHORED bar
# ((tmfix - classic) - (entry - classic)), which needs classic in the same
# tournament -- a round-robin at ~1.5x the games. That is a coordinator call.
#
#
# PROVENANCE: a copy of this script was shipped to the bench-box arena
# ~/sunfish-bench/tmfix60-20260814/stage2_300.sh at stage-1 launch and THAT is
# the copy that would run. They were identical when committed; if you edit one,
# reship it -- an edited repo copy beside a live arena copy is the stale-copy
# failure this project has paid for five times.
# Same arms, same book, same bounds as stage 1; only the clock changes.
set -u
A=$HOME/sunfish-bench/tmfix60-20260814
FC=$HOME/sunfish-bench/fastchess-linux-x86-64/fastchess
TC=300+0
ROUNDS=200          # x2 games, -repeat  =  400 game cap
CONC=8
SEED=20260815
OUT=$A/stage2

if [ "${ARMED:-0}" != "1" ]; then
    echo "STAGE 2 IS NOT ARMED.  spec:"
    echo "  arms      $A/bin/e_tmfix.packed (engine1) vs $A/bin/e_oldtm.packed"
    echo "  tc        $TC sudden death, no adjudication"
    echo "  sprt      elo0=0 elo1=20 alpha=0.05 beta=0.05, cap $((ROUNDS*2)) games"
    echo "  book      $A/book3k.pgn (pgn; the artifact parses only startpos+moves)"
    echo "  cost      ~6-9 h at concurrency $CONC, and LONGER if the fix works,"
    echo "            because the arm that stops flagging plays longer games"
    echo "  arm with  ARMED=1 sh $0"
    exit 0
fi

mkdir -p "$OUT"
. "$HOME/sunfish-bench/boxlock.sh"
box_acquire tmfix300-stage2
nice -n 10 "$FC" \
  -engine cmd="$A/w_tmfix.sh" name=tmfix \
  -engine cmd="$A/w_oldtm.sh" name=oldtm \
  -each proto=uci tc="$TC" \
  -openings file="$A/book3k.pgn" format=pgn order=random -srand "$SEED" \
  -rounds "$ROUNDS" -games 2 -repeat -concurrency "$CONC" -recover \
  -sprt elo0=0 elo1=20 alpha=0.05 beta=0.05 model=normalized \
  -ratinginterval 10 -autosaveinterval 10 \
  -pgnout file="$OUT/match.pgn" timeleft=true \
  > "$OUT/match.log" 2>&1
"$HOME/sunfish-bench/pypy3.11-v7.3.20-linux64/bin/pypy3" "$A/tally.py" "$OUT/match.pgn"
"$HOME/sunfish-bench/pypy3.11-v7.3.20-linux64/bin/pypy3" "$A/pair_elo.py" "$OUT/match.pgn"
