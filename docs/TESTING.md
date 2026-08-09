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
  -rounds 150 -games 2 -concurrency 6 -recover \
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

5. **Time-management changes need validation at MULTIPLE time controls.**
   Fast-TC matches are structurally blind to long-TC time bugs: a change
   that mis-spends the clock at 15+3 can be behavior-identical at 4+0.04
   (this happened: an opening think-time ramp accidentally governed entire
   long games and sailed through 300-game fast-TC matches). For any change
   to think-time computation, additionally verify the per-move time curve
   directly at a long TC (drive the engine over a scripted game and assert
   the budget is actually reached mid-game).

6. **Time control: short but with an increment.**
   Sunfish is slow; sudden-death blitz makes every game a timeout lottery.
   `tc=4+0.04` works well for regression tests; use longer (e.g. `tc=10+0.1`)
   when the change affects time management or pondering. A handful of time
   losses over hundreds of games is normal; dozens mean the TC is too fast.

7. **Adjudicate finished games** (`-draw`/`-resign` flags above) — sunfish has
   no resign logic and weak endgames, so unadjudicated games drag on and waste
   most of the wall time on decided positions.

8. **Distrust surprising results — verify before believing.**
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

9. **Know that tournament managers reuse engine processes across games.**
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

10. **Report the numbers, not the vibe.** A PR claiming a strength change must
   quote: games, TC, book, ELO ± error bounds, and time-loss/crash counts —
   plus the exact commits of both frozen engines.

11. **Always pass `-recover`, and check the finished game count.**
   Without it, the *first* engine stall aborts the entire tournament — the
   harvest still prints a perfectly well-formed Elo line, just for however
   many games happened before the stall. Nothing warns you. Four matches in
   one 2026-08 session died this way and silently lost ~40% of their
   scheduled games (165/500, 172/500, 551/750, 275/600); an earlier
   confirmation run that had to be restarted three times, and was written up
   as `part1/part2/part3`, was the same bug misdiagnosed. `-recover` restarts
   a stalled engine instead of killing the match. Belt and braces: after the
   match, assert the reported game count equals what you asked for, and never
   pool separately-launched runs to reach a target — a pooled total hides
   exactly this failure.

12. **Fixed depth is a screening instrument, not a verdict.** A fixed-depth
   match holds search *effort* constant, so it measures evaluation accuracy
   with the clock switched off. But a sharper evaluation usually makes the
   search work harder per depth, and a real game charges for that. This has
   now flipped the *sign* of a result twice in this repo: a tempo term
   measured +61 [+13, +109] at fixed depth and −37 against the same engine at
   60+1; a passed-pawn term measured +132 [+22, +278] at 33 games and decayed
   to +34 [−34, +105] by 92. Time-to-depth is the hidden variable. Screen
   with fixed depth if you like, but only a wall-clock match decides.

## Testing the packed artifact

`build/pack.sh` inlines a minimal UCI loop that handles `position startpos
moves ...` only — **`position fen` is deliberately unsupported**, because the
tournaments the packed build targets never send it and every byte counts.
That is fine for the artifact and fatal for a careless harness: fastchess
delivers an **EPD** book as `position fen ...`, which the packed engine
silently ignores, so it plays on from the initial board and then emits moves
that are illegal in the actual game. It looks like a catastrophic engine bug
(0/10, "makes an illegal move") and is really a book-format mismatch.

So: test packed artifacts with a book fastchess delivers as `position
startpos moves ...` (a PGN book), or measure the unpacked engine — which
runs through `tools/uci.py` and does parse FEN — and cover the packed build
with a separate startpos-only smoke. Either way, confirm which form your
book actually produces by dumping the UCI trace on a two-game run rather
than assuming.

## Why not cutechess-cli?

Same methodology applies, and the flags are nearly identical — but recent
cutechess releases ship no CLI binary for Linux or macOS, so CI and most
contributors use fastchess (the cutechess-compatible successor used by
Stockfish testing). cutechess-cli is still worth keeping around for the one
thing fastchess cannot do — see pondering below; release 1.3.1 is the last
one with a Linux CLI build.

## Testing pondering

fastchess does not support pondering, so use cutechess-cli (1.3.1), which
does: give one side `ponder` in its `-engine` block and pit it against the
same engine without. Two caveats. Keep concurrency at most `cores / 2` — a
pondering game keeps two engines computing simultaneously, so the process
budget is double what the game count suggests. And verify the plumbing on a
2-game run at a fast TC before committing to a long match: dump the UCI
traffic and confirm you actually see `go ponder` and `ponderhit` (`sunfish.py`
advertises `option name Ponder` and handles the protocol, so no engine change
is needed — but a silently non-pondering match measures nothing).

Result on record (2026-08, 300 games at 60+1, one core per engine): pondering
is worth **+81.4 ± 36.9 Elo, LOS 100%**, with zero time losses. Note this is
self-play, so it measures the value of thinking on the opponent's clock;
against an instant-moving opponent there is nothing to think during. It also
doubles CPU demand per game, which is why it was a net *loss* on a
burst-credit cloud VM: the credits drained mid-game, the engine was throttled,
and a deadline-less ponder search starved the bot process into a time
forfeit. Ponder on hardware with a core to spare, not on an e2-micro.
