# Testing sunfish changes

Functional tests (`tools/quick_tests.sh`, `pytest`, CI) tell you the engine is
*correct*. They tell you nothing about whether a change makes sunfish *stronger
or weaker*. Any PR that touches search, evaluation, tables, or time management
— whether written by a human or an AI — must be validated with an ELO
tournament between the old and new version. This document is the required
methodology; it exists because shortcuts here produce confident, wrong numbers.

## TL;DR recipe

```bash
# 1. Freeze BOTH engines completely (code + tools, no shared files!)
mkdir -p /tmp/elo/old /tmp/elo/new
git archive master     sunfish.py tools | tar -x -C /tmp/elo/old
git archive my-change  sunfish.py tools | tar -x -C /tmp/elo/new
chmod +x /tmp/elo/old/sunfish.py /tmp/elo/new/sunfish.py

# 2. Get fastchess and an opening book
#    https://github.com/Disservin/fastchess/releases  (the release tar also
#    contains a usable book at app/tests/data/openings.epd)

# 3. Run the match — then LEAVE THE MACHINE ALONE until it finishes
fastchess \
  -engine cmd=/tmp/elo/new/sunfish.py name=new \
  -engine cmd=/tmp/elo/old/sunfish.py name=old \
  -each proto=uci tc=4+0.04 \
  -openings file=openings.epd format=epd order=random \
  -rounds 150 -games 2 -concurrency 6 \
  -draw movenumber=40 movecount=8 score=10 \
  -resign movecount=4 score=500
```

## The rules, and why each one exists

1. **Freeze both engines with `git archive`, including `tools/`.**
   `sunfish.py` imports `tools/uci.py`, so an engine run from the working tree
   picks up whatever is in the tree *at process start*. If both engines share a
   live checkout — or you edit any imported file mid-match — you are no longer
   testing the diff you think you are testing. This is not hypothetical: a
   contaminated run in this repo once measured **-60 ELO (LOS 0.02%)** for a
   change that was later shown to search bit-identical node counts.

2. **Do nothing else on the machine while the match runs.**
   Sunfish plays on a real clock. Compiles, test suites, or other engine
   processes steal CPU from whichever game happens to be running, adding
   timeouts and noise that error bars do not account for.

3. **Always use an opening book with color-swapped pairs (`-games 2`).**
   Sunfish is deterministic: without varied openings, every game pair is the
   same two games repeated N times, and the error bars fastchess prints become
   fiction. Random order + paired colors gives valid pentanomial statistics.

4. **Size the match to the effect you are hunting.**
   ~300 games at fast TC gives roughly ±30 ELO (95%). That detects blunders,
   not refinements. A few-percent speedup is worth a few ELO and needs
   thousands of games — or better, use SPRT so the match stops itself:
   `-sprt elo0=0 elo1=10 alpha=0.05 beta=0.05` (accepts/rejects "at least
   10 ELO better" with 5% error rates).

5. **Time control: short but with an increment.**
   Sunfish is slow; sudden-death blitz makes every game a timeout lottery.
   `tc=4+0.04` works well for regression tests; use longer (e.g. `tc=10+0.1`)
   when the change affects time management or pondering. A handful of time
   losses over hundreds of games is normal; dozens mean the TC is too fast.

6. **Adjudicate finished games** (`-draw`/`-resign` flags above) — sunfish has
   no resign logic and weak endgames, so unadjudicated games drag on and waste
   most of the wall time on decided positions.

7. **Distrust surprising results — verify before believing.**
   Before accepting a big ELO swing:
   - **Run an old-vs-old control match** under the same conditions. It must
     come out near 0 within the error bars; if it doesn't, the harness or the
     environment is biased and every other number is meaningless.
   - **Check behavior-neutrality with a depth-limited lockstep**, not a
     clock-based one: feed both engines the same `position` + `go depth N`
     sequences and compare moves *and node counts*. Time-based searches are
     nondeterministic (the wall-clock cutoff lands on different internal
     iterations run to run), so clock-based comparisons cannot distinguish a
     logic change from timing jitter. At fixed depth sunfish is fully
     deterministic: any node-count difference is a real logic change.
   - Check the match log for `disconnect`, `illegal`, and timeout counts per
     engine, and look at *how* games ended (adjudication vs mate vs time).
   If anything was contaminated, rerun — never average a dirty match into a
   clean one.

8. **Know that tournament managers reuse engine processes across games.**
   fastchess (and cutechess by default) starts an engine once per match slot
   and plays dozens of games in that one process, sending `ucinewgame`
   between them. Engine state that survives `ucinewgame` — like sunfish's
   transposition tables — then carries knowledge *between games*: with a
   deterministic engine and a repeating opening book this acts as invisible
   cross-game learning and can swing a match by dozens of ELO. A change that
   merely alters what persists across games (caching, table size, eviction)
   can therefore look like a large strength change when the per-game search
   is provably identical. Check the trace (`-log file=x level=trace
   engine=true`) to see what your engines actually receive.

9. **Report the numbers, not the vibe.** A PR claiming a strength change must
   quote: games, TC, book, ELO ± error bounds, and time-loss/crash counts —
   plus the exact commits of both frozen engines.

## Why not cutechess-cli?

Same methodology applies, and the flags are nearly identical — but the
cutechess project publishes no Linux or macOS CLI binaries, so CI and most
contributors use fastchess (the cutechess-compatible successor used by
Stockfish testing). If you have cutechess-cli locally, it is fine.

## Testing pondering

fastchess does not support pondering. Use a python-chess harness that calls
`engine.play(board, limit, ponder=True)` for one side and manages both clocks,
pitting ponder-on against ponder-off of the same engine. Keep concurrency at
most `cores / 2` — a pondering game keeps two engines computing
simultaneously.
