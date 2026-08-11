# Testing sunfish changes

Functional tests (`tools/quick_tests.sh`, `pytest`, CI) tell you the engine is
*correct*. They tell you nothing about whether a change makes sunfish *stronger
or weaker*. Any PR that touches search, evaluation, tables, or time management
— whether written by a human or an AI — must be validated with an ELO
tournament between the old and new version. This document is the required
methodology; it exists because shortcuts here produce confident, wrong numbers.

## TL;DR recipe

```bash
# 1. Freeze BOTH engines completely (code + interface, no shared files!)
mkdir -p /tmp/elo/old /tmp/elo/new
git archive master     sunfish.py sunfish_tools | tar -x -C /tmp/elo/old
git archive my-change  sunfish.py sunfish_tools | tar -x -C /tmp/elo/new
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

1. **Freeze both engines with `git archive`, including `sunfish_tools/`.**
   `sunfish.py` imports `sunfish_tools/uci.py`, so an engine run from the working tree
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
   **The book must have at least as many positions as the match has rounds**
   (rounds = games/2), or fastchess cycles it and repeated openings quietly
   shrink the effective sample. The 99-position fastchess test book is too
   small for 150+ round matches; use a 2000+ position book (e.g. sampled
   from the lichess eval dump: early-middlegame, both queens on, ≥26 men,
   |eval| ≤ 80cp, deduplicated by the first four FEN fields).

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

6. **Time control: 30+1 minimum for anything decision-grade.**
   Sunfish is slow; sudden-death blitz makes every game a timeout lottery,
   and very fast TCs measure interpreter overhead as much as chess (two
   changes on record flipped sign between fast TC and 60+1). `tc=4+0.04`
   is for regression tests and lockstep sanity only. Any match whose
   result feeds a merge/decline decision runs at `tc=30+1` or slower;
   final confirmation of a winner stays at 60+1. A handful of time losses
   over hundreds of games is normal; dozens mean the TC is too fast.

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
runs through `sunfish_tools/uci.py` and does parse FEN — and cover the packed build
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

## KCX landing evidence (2026-08)

Reproducible benchmarks recorded outside CI gates. CI enforces the semantic
tests (`tests/test_tt_consistency.py`, `tests/test_terminal_bench.py`), the
mate/stalemate floors, and the model-audit drift guard; the numbers below are
hardware- and version-sensitive and are documentation, not gates.

- **Behavioral**: bound-level equivalence with the proof-first reference
  implementation (an eager king-capture guard at node entry) over 9,600 probes
  — 65 positions x depths 0-4 x an 8-gamma ladder x both probe orders x cold
  and warm tables: **zero value mismatches**.
- **Invariant**: the king-capture contract is **stratified by depth** — at
  depth 0 a capturable node must FAIL HIGH, at depth >= 1 it must report
  MATE_UPPER exactly. Contract sweep over 250 generated king-capturable
  positions x 23 gammas x depths 0-3, cold and warm — 91,952 probes, **zero
  violations**: every depth-0 return >= gamma, every depth >= 1 return exactly
  MATE_UPPER. (The earlier 9,600-probe sweep over 200 positions asserted
  exactness at every depth; that is the pre-stratification claim and no longer
  describes the shipped depth-0 code.) A compact both-halves version of this
  runs in ordinary CI: `test_qs_stratified_contract`.
- **Legality oracle**: `Position.king_capture()` agrees with python-chess on
  500/500 generated positions. In CI, `test_legality_oracle_vs_python_chess`
  asserts the board predicate directly (the search probe is kept as a
  secondary assertion), and `test_king_capture_special_rules` pins 19
  deterministic special-rule cases — castling through / into / out of check
  (the `kp` rule), en passant uncovering a rook or a bishop, pins,
  king-next-to-king, promotion captures. The castling cases matter: python-
  chess does not consider an illegal castling even *pseudo*-legal, so the
  playout-based differential test skips them and can never reach `kp`.
- **Killer invariants**: `test_killer_invariants_over_corpus` audits the
  8,653 `tp_move` entries a probe sweep over the 148-position bench leaves
  (5,086 of them at king-capturable nodes) for the three properties the null
  fast path reads them as: the stored move is generated, at a capturable node
  it IS the capture, and otherwise it is legal.
- **Consistency**: ladder and full-driver crossing scans over 35+ positions,
  **0 crossed table entries** (master: 28 driver / 35 ladder on the same set).
- **Suites**: terminal bench 148/148 (master 130); stalemate2 17/130, floor
  raised 13 -> 17; mate1 8/8, mate2 20/20, mate3 5/5.
- **Cost**: +5.3% wall on a 32-position depth-5 battery. Note the node count
  reads 99.6% of master, which is *not* a cost saving: the legality oracle is
  now a board predicate rather than a depth-0 search, so its work no longer
  increments the node counter. When a change moves work between counted and
  uncounted code, wall time is the only honest measure — see rule 12.
- **Elo**: -10.4 +/- 28.7 over 300 games at 60+1, zero time losses (114W-123L
  -63D). Measured on the kcx production build *before* the subsequent golf,
  full-scan revert, board-predicate and driver-bracket commits; those were
  validated as value-identical by the equivalence battery above rather than by
  a fresh match.
- **Rejected alternative**: a "licensed null" prototype (require mobility
  before admitting any virtual option — structurally equivalent guarantees)
  measured +3.6% nodes and **-27.9 +/- 33.2 Elo** over 200 games at 60+1
  (71W-87L-42D, zero time losses): dominated, not landed.
- **Rejected micro-optimisation**: moving `killer = self.tp_move.get(pos)`
  below the null and stand-pat yields in `moves()`, so a stand-pat cutoff pays
  for no hash lookup. Looks free — neither virtual yield reads it — but it is
  **not behaviour-preserving**. `tp_move` is keyed by position alone (not by
  `(pos, depth)`), and QS below the null is not ply-limited, so the null
  subtree can transpose back to the same board with the same side to move and
  store a killer at this key; and once `tp_move` reaches `TABLE_SIZE` it can
  evict the key instead. Instrumented discriminator reading at both sites over
  a 4.76M-node battery to depth 10: **10 disagreements in 3,015,876 reads**
  (9 of them `None` -> a move, i.e. a store, not an eviction). Deep driver
  equivalence on the same battery: every depth/gamma/score/move line
  identical, node counts differ on **2 of 25** positions. One dict lookup on a
  stand-pat cutoff does not buy a behaviour change: not landed. The reasoning
  is recorded in a comment at the read site so it is not re-proposed.
- **Rejected micro-optimisation**: caching the terminal scan with a walrus
  (`dead := all(pos.move(m).king_capture() ...)`) so a positive null fail-high
  does not rescan in the correction. Instrumented census: on a 48-position
  depth-5 battery (614,499 nodes) the fold scan ran **13** times, returned
  true **0** times, and the walrus would have saved **0** of 8,786 correction
  scans; on the whole terminal bench under ladders + drivers (727,717 nodes)
  it would have saved **4** of **43,006** (0.009%). Wall time on the latter,
  best of two interleaved runs: 14.81s plain vs 14.88s walrus — noise. A
  variable in the densest part of the patch for a saving that does not exist:
  not landed.
