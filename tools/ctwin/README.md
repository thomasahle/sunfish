# ctwin — a node-identical C twin of classic sunfish

**Lab instrument, never ships.** Its one job is to make fixed-node tuning
games cheap while searching the *exact same tree* as the Python reference.

## Fidelity contract

NODE-IDENTITY is the definition of done: same position in, same chosen
move, same node count, same score out — verified **differentially**, not
assumed. `difftest.py` drives the real `sunfish.py` (repo root, under
pypy3, via `pyref.py`) and `sunfish_c` over one protocol and compares
every MTD-bi probe byte for byte: `(depth, gamma, score, killer move,
cumulative node count)`, plus `gen_moves()` order and `value()` of every
move at each test position and one ply below it.

Never claim identity that has not been measured; the harness prints the
exact coverage (positions × depths × probes) of each run. If a divergence
appears, report the first divergent probe and fix the twin — never
approximate around it.

**Reference:** `sunfish.py` at the repo root of the checkout the harness
runs in — the twin's defaults reproduce that engine (capped null below
depth 6, intrinsic LMR from depth 6, mate-distance scoring, and no IID).
The reference is imported
live by `pyref.py`, so drift in the Python file shows up as a harness
failure, not silent staleness — re-pass the gate, re-tune the flavor
knob defaults, and re-pin variants.py's drift hashes when the search
changes. Lab-only search variants must remain differentially testable
against the live reference.

### Where clones silently diverge (all handled, all tested)

- **Floor division.** Python `//` floors; C `/` truncates toward zero.
  Every division that can see a negative operand (`gamma = (lower +
  upper + 1) // 2` with mate-band bounds, `render`) goes through
  `pyfloordiv`/`pymod`.
- **Sort order.** `sorted(((v, m) …), reverse=True)` orders full
  `(val, Move)` tuples; keys are unique, so the order is total —
  descending `(val, i, j, prom)` with `prom` compared as a byte
  (`'\0' < 'B' < 'N' < 'Q' < 'R'`). No stability question remains, and
  `qsort`'s instability is harmless.
- **Dict semantics.** Both tables keep insertion order; updates keep
  their original slot; `tp_score` FIFO-evicts the oldest entry; the
  `tp_move` eviction skips the search root. Keys compare like the
  namedtuples do — score included (a position rebuilt under a different
  K-table is a different key in Python too).
- **Generator laziness.** `bound()`'s move phases run in Python's exact
  order: the killer is read before null search, capture substitution scans
  the board directly, king captures resolve before recursion, the sorted
  list is never built if the killer cuts, and a sub-window cap answers for
  its move (and, in the twin's counted form, for the whole sorted tail at
  once) without a child search, so a prefix cutoff skips it on both sides.
- **Node counting.** `nodes` increments at exactly one site: `bound()`
  entry, including driver probes and TT-hit returns.
- **Module state.** `pst["K"]` swaps to the endgame table per search and
  *stays* swapped for subsequent position parsing, exactly like the
  Python module globals. `reset` reproduces a fresh interpreter.

## Files

- `sunfish.c` — the twin. Board, movegen, `value`, `move`, `bound`,
  MTD-bi driver transcribed from `sunfish.py`; speed comes from C, not
  from a redesign.
- `gen_tables.py` — dumps the reference's padded PST tables (+ `K_END`
  + piece values) into `tables_classic.txt`. Eval variants from any
  generator become C engines by dumping their tables here; the binary
  never needs recompiling.
- `pyref.py` — protocol server around the real `sunfish.py` (plus the
  movegen-call counter and the battery knob plumbing).
- `difftest.py` — the differential harness (see contract above).
- `variants.py` — drift-guarded Python reference for the tp_move
  replacement-policy battery (`EVICT_POLICY`, `KILLER_COUNT`); refuses to
  run if `sunfish.py`'s search changed under it (pinned source hashes),
  and its transcription is itself difftest-proven (`USE_VARIANT=1` at
  default knobs must match the real engine byte for byte).
- `battery.json`, `nodescreen.py`, `match.py` — the battery matrix, the
  C-only node/movegen screen over it, and the paired-openings fixed-node
  match driver (python-chess arbiter, trinomial SPRT, zero tolerance for
  illegal moves).
- `../tune/logistic_gp/` — the general asynchronous game-result tuner.
  Its Sunfish example uses this twin for fast experiments and a deterministic
  mate-safety gate before games are allocated.
- `tmlib.py`, `tmsim.py`, `vmatch.py`, `tmmatrix.py`, `npsprofile.py`,
  `npsmodel.json` — the TIME-MANAGEMENT surrogate (see below): the formula
  mirrors, the stage-0 trajectory simulator, the virtual-clock match driver,
  the concurrent knob-matrix runner, and the measured node-rate profile the
  virtual clock converts seconds with.

## Measured status (2026-08-14, this laptop)

- Identity, standing gate (`make gate`, PGO binary): 27 positions
  (startpos + openings, Bratko-Kopec, WAC, mates, stalemates, null-move
  mates, perft set, KQK) × depth 1..6 with the movegen walk — 818 MTD-bi
  probes and 901 movegen lists byte-identical; depth 1..7 on 6 positions,
  235 probes; two tuned-knob sweeps (QS/QS_A/EVAL_ROUGHNESS on both
  sides) × depth 1..5, 1396 probes; an LMR sweep × depth 1..6, 818 probes;
  eviction sweeps (`TABLE_SIZE` 500 and 50) × depth 1..6, 1828 probes.
  Movegen *call counts* are compared in
  every `done` line since the battery landed.
- Identity, battery cells: USE_VARIANT transcription proof (wide + walk,
  and `TABLE_SIZE` 500) plus 8 cells (policies 1/2/3, killers 2/3, two
  combinations) × (wide, `TABLE_SIZE` 500, `TABLE_SIZE` 50) at depth
  1..5 — 32 suites total in the full battery gate, 0 mismatches.
- Speed: ~22x sunfish.py under warm pypy3 at identical node counts
  (1,155,634 nodes, depth-7 battery; ctwin 1.421s -> 0.669s across the
  optimization rounds, measured ratios 21.6-27.3x as the pyref side
  swings with host load; `make pgo && make bench` reproduces).
- **Interpreter cross-check (2026-08-16): `make gate` is INTERPRETER-
  INVARIANT on this corpus.** `difftest.py` has always driven the
  reference side under `pypy3`; lean-surfaces uses a separately pinned
  CPython (3.9.19) as its own oracle and had never been checked against
  this harness's choice of interpreter. Ran the full 7-line gate twice
  on the same rebased `master` commit, same positions/config both times,
  swapping only the reference interpreter (a one-line, env-gated,
  **uncommitted** change to `difftest.py`'s `Engine(["pypy3", ...])`
  call): every one of the 7 lines produced byte-identical coverage
  (positions x depth, probe counts, movegen-list counts) and 0
  mismatches under CPython 3.9.19, matching the pypy3 baseline exactly.
  The square's edge is settled: this harness's node-identity claim does
  not depend on which interpreter drives the reference side.
  **Not runtime-free, so the default stays `pypy3`:** on a small smoke
  config CPython took ~3.2x longer (37.9s vs 11.9s, `--n 5 --depth 7`);
  on the gate's `QS=0 EVAL_ROUGHNESS=40` sweep specifically — the one
  line that disables quiescence delta-pruning and thereby explodes raw
  node count — CPython took on the order of two hours against pypy3's
  share of a 351s total-gate run, at least an order of magnitude worse
  than the smoke-config ratio. Mechanism: CPython's per-op interpreter
  overhead multiplies against the exploded node count from disabled
  pruning, while pypy3's JIT benefit grows with how hot the inner loop
  runs — so the two effects compound in opposite directions specifically
  where pruning is weakest. `pypy3` remains the shipped default
  reference interpreter; this entry is the record that the choice was
  checked, not assumed.

## Use

```sh
make            # build + regenerate tables
make test       # quick identity probe (~seconds)
make test-full  # wide sweep, depth 6
make gate       # the FULL fidelity gate: wide sweep + walk, depth 7,
                # knob sweeps, eviction sweeps.  Required after ANY
                # change to sunfish.c or sunfish.py (TESTING.md rule 14).
make bench      # C-vs-PyPy wall-time ratio at identical nodes
```

Tuning knobs (no recompile): UCI `setoption name NAME value VALUE`, lab
`set NAME VALUE`, `SF_NAME=` env, or `NAME=VALUE` argv after the table path —
`QS QS_A LMR EVAL_ROUGHNESS TABLE_SIZE NULL_CAP_MARGIN NULL_MIN_DEPTH
NULL_LIMIT NULL_CUT_RED LMR_MIN_DEPTH IID_MIN_DEPTH IID_RED FUT_MAX FUT_CAP
FUT_CAP_DEPTH MATE_DIST FEN_HIST` (`NULL_CAP_MARGIN=-1` follows
`EVAL_ROUGHNESS`; `NULL_CUT_RED` controls the shallow null probe;
`LMR_MIN_DEPTH` is where shallow null ends and intrinsic LMR begins).
`FUT_CAP` selects no shallow cap, the current
ordinary-move cap, or the simpler negative-`value()` cap;
`FUT_CAP_DEPTH` selects its horizon. `FEN_HIST=0` restores the pre-2026-08-16
one-ply `position fen` history; `1` (default) is the driver's two-ply
construction for black-to-move FENs, which is what a match actually plays. `VALUE_N VALUE_B
VALUE_R VALUE_Q` tune material, while
`PST_P PST_N PST_B PST_R PST_Q PST_K PST_KE` scale the positional component
of each loaded table. The tp_move battery adds `EVICT_POLICY` (0 master
root-guarded FIFO, 1 unguarded
evict-before-insert, 2 depth-stored bounded scan with `EVICT_SCAN_K`,
3 hash-slot two-tier replace-if-deeper), `KILLER_COUNT` (1..3 most recent
distinct killers), `USE_VARIANT` (Python-side transcription proof; no-op
in C). Unknown or out-of-range knobs are hard errors on every input path.

The complete optimizer documentation and the Sunfish joint-parameter example
now live in [`../tune/logistic_gp/`](../tune/logistic_gp/README.md). Keeping the
tournament model outside `ctwin` makes explicit that it works with any UCI
engine; the C twin is only Sunfish's high-throughput engine under test.

Game use: `position startpos moves …` / `position fen …`, then
`go nodes N` (primary — clock-free surrogate games), `go depth D`,
`go movetime T`, or the standard `wtime` / `btime` / increment controls.
The `go nodes` consumer transcribes `sunfish_ui/uci.py`'s
`go_loop`: the probe that crosses the cap always finishes and its yield
counts, the cap is checked between probes at depth > 1, candidates commit
when their depth completes, and bestmove has the structural floor (never
"(none)" while a legal move exists). This matters: an earlier mid-probe
abort made the twin play one depth staler than pypy at stop points — the
calibration match caught it at -54 ± 43 and the run was voided.

## Fidelity and transfer calibration

Node identity proves exact classic semantics at fixed depth and node count.
It does not make C and Python runtimes identical: a timed match prices C
operations, and Python-only allocation or interface changes are invisible.
For a node-identical classic search or table-eval diff, `docs/TESTING.md` uses
the timed C `3+0.1` match as the primary decision instrument. Rerun these
transfer checks after material harness changes and periodically to catch drift:

1. **Sanity match:** ctwin vs `sunfish.py` under pypy3, both at the same
   fixed node budget, standard book, 200+ paired games. Expected ~50%
   (they play identical moves at identical depths). Any significant
   deviation is a harness bug, not a result — and the first run proved
   the point: it measured -54 ± 43 and caught the go_game mid-probe abort
   (see Game use above); that run was voided, the driver fixed.
   **Result (fixed driver): 300 games, 124W-125L-51D = 49.83%, Elo
   -1.16 ± 12.23, 92.7% of pairs move-identical, 0 illegal moves.**
2. **Known-Elo pair A (search flavor):** branch flavor vs master flavor
   (`IID_MIN_DEPTH 2`, `MATE_DIST 0`), both sides ctwin, fixed nodes.
   Must reproduce the sign and rough magnitude of the pypy-measured gap
   between those flavors at fixed effort.
3. **Known-Elo pair B (eval/knob gap):** a knob detuning with a known
   real-match gap (e.g. `QS`/`EVAL_ROUGHNESS` shifted to a previously
   measured losing setting). Same acceptance test.

Stage 1 passed on 2026-08-14; stages 2-3 remain useful known-effect checks.
Rule 12 of `docs/TESTING.md` still makes fixed-node matches screens rather than
timed Elo decisions.

## Time management on a virtual clock

TM used to be the one thing the twin could not accelerate. A faster engine
does not make a timed game cheaper — a 60+0 game burns two minutes of wall
clock whichever engine plays it — so every TM question cost real hours on
sunfish.py, and the twin excluded clocks by design.

The surrogate removes the wall clock from the loop instead of the engine.
**Nothing in `tmsim.py` or `vmatch.py` reads a clock** (a unit test asserts
that neither file so much as imports `time`), and the funnel spends effort in
the order of what it can rule out:

```
stage 0   tmsim.py    no games at all.  Walks a clock through a game of a
                      given length under each manager and SOLVES for the
                      pathologies: the negative-cap threshold, the fixed
                      point a budget parks the clock at, the clock where the
                      pool stops spending.  Milliseconds per manager.
stage 1   vmatch.py   virtual-clock games on the twin.  ~4 s per 60+0 game
                      against ~2 min of real clock -- ~30x on top of the
                      twin's 22x, because the surrogate skips the parts of
                      a timed game that are pure waiting.
stage 2   the real thing.  ONE wall-clock match on sunfish.py for the
                      candidate the surrogate ranked first, plus the 1+0
                      hammer for flag safety.
```

**How a virtual move works.** `tmlib` turns the virtual clock into
`(soft, hard)`; `npsmodel.json` turns `hard` into a node budget (how far the
*Python* engine would have got in that many seconds); the twin runs
`go nodes` and emits its whole probe trace; `vmatch.replay` walks that trace
with `elapsed = nodes / nps` substituted for the clock and applies the arm's
own stop rule, which yields both the move played and the spend; the clock is
charged `spend + overhead` and credited the increment.

Replay lives in the driver because that is where the stop rules live in every
shipped engine (`uci.py`'s `go_loop`, the packed entry's inlined loop) — and
it keeps `sunfish.c` untouched, which matters because any edit there must
re-pass the full node-identity gate before any twin number counts. It is also
*more* faithful than the twin alone: the twin's node cap is a yield-boundary
rule, while the real deadline aborts inside `bound()`, so replay models the
wall as a mid-probe abort at exactly `hard`.

**The managers are mirrors, not re-derivations.** `tmlib.PINNED` holds the
exact source text of every shipped budget — classic's `/12`, the packed
`oldtm`/`steptm`/`smooth` arms, the pool, and the `min40_4` candidate — with
its repo, file and commit. `python3 tmlib.py` re-evaluates each literal on a
21×9 clock/increment grid and asserts the mirror agrees to 1e-12 (2,646
values), then greps the live source and **fails on drift**. Sources it cannot
find are reported UNCHECKED per manager rather than silently skipped, and a
manager listed in `CANDIDATES` reports "not landed yet" instead — a missing
pin is news for those, not a failure. `TM_UCI=<path>` additionally checks the
pool mirror against a checked-out `uci.py` that has `pool_budget`: 45,520
values identical to branch `tm-pool-manager` at 7e8e1ff.

### Calibration gate (2026-08-14, this laptop)

A surrogate that cannot reproduce results we already paid for in real hours
may not rank anything. Each row is a real fastchess match on the packed
artifact, re-run on the virtual clock at 50 ms/move of charged overhead.

| target | real (fastchess, packed) | surrogate (virtual, twin) |
|---|---|---|
| (a) `steptm` vs `oldtm` @ 60+0 | **+235.5 ± 65.4**, H1 in 100 g | **+227.6 [+154, +325]**, H1 in 80 g |
| (b) `smooth` vs `steptm` @ 60+0.1 | **+40.6 ± 25.6**, H1 in 438 g | **+84.9 [+42, +131]**, H1 in 192 g |
| (c) `pool` vs `smooth` @ 60+0 | **+119.9 ± 36.4**, H1 in 274 g | **+112.3 [+48, +185]**, 80 g |

Mechanism telemetry, which is the part that matters more than the Elo:

| reading | real | surrogate |
|---|---|---|
| `oldtm` blind moves @ 60+0 | 22.3% | 26.2% |
| `oldtm` crosses 2.4 s at move | 42 (median) | 40-43 (solved, `tmsim`) |
| `steptm` minimum clock @ 60+0.1 | 2.0 s | **2.00 s** |
| starved-move RATIO, step : smooth | 8.6x (34.3% / 4.0%, ≤0.15 s) | 8.5x (12.8% / 1.5%, ≤0.05 s) |
| time forfeits, both arms @ 60+0.1 | 0 | 0 |
| `pool` median spend / `smooth`'s | 0.79x | 0.55x |
| `pool` max spend / `smooth`'s | 3.3x | 4.3x |

(The starved-move *thresholds* differ — the ledger's metric is ≤0.15 s, the
surrogate counts moves whose budget was at the 0.05 s floor — so the levels
are not comparable and only the ratio is.)

All three gates pass on sign and rough magnitude, two of them within 8 Elo.
The pool arm additionally reports WHICH RULE ended each search: **75.6% of
its moves stopped on the MTD-bracket soft limit and 0% on the depth-transition
backstop**, which is the design's own load-bearing claim (reading the limit at
a new depth arrives one full probe late) measured rather than assumed.

The three arithmetic **signatures** are solved rather than simulated, and
are invariant to *every* modelled parameter (node rate, iteration ladder,
branching): `oldtm`'s negative-cap threshold **2.400 s**, the step budget's
parking equilibrium **2.109 s** at 60+0.1 (2.2 s with no overhead charge —
the measurement's 2.1 s median is what pins the charge at ~50 ms), and the
pool's floor knee **(M+2)·O = 8.400 s**. All three are unit tests.

**Where it fails, stated plainly.** (b) overstates the smooth edge ~2x, and
`tmsim` says why: its parked step arm spends a median 0.057 s over its last
twenty moves where the real one spent 0.115 s, so the surrogate's step arm is
*twice as starved* as reality and the gap it loses by widens to match. The
direction of that error is knowable in advance from the stage-0 trace, which
is the useful part. And the `oldtm` FLAG count is a knife-edge in the modelled
overhead — never within 80 moves at 5 ms, move 61 at 50 ms — while the
floor-crawl that causes it is robust across the whole range. The surrogate
reproduces mechanisms; it does not certify flag safety.

### First full ranking pass (2026-08-15)

Virtual clock, 50 ms/move charge, 60 games/cell (80 where noted, 16 at
300+3). Elo is A-minus-B; `st<2.4` counts moves made with under 2.4 s left.

| arm | vs | 60+0 | 60+0.1 | 30+1 | 60+1 | 300+3 |
|---|---|---|---|---|---|---|
| `pool` | `smooth` | **+112** [+48,+185] | — | — | — | — |
| `min40_4` | `legacy12` | **+147** [+86,+219] | — | — | — | — |
| `min40_4` | `smooth` | ≡ identical | +4 [−62,+71] | −31 [−98,+35] | +26 [−43,+97] | −44 [−194,+92] |
| `min40_4` | `pool` | — | **−114** [−208,−34] | **−134** [−218,−62] | **−114** [−198,−41] | — |
| `onemax` | `legacy12` | — | −6 [−85,+73] | −47 [−128,+30] | −41 [−115,+30] | — |
| `onemax` | `min40_4` | — | **−89** [−170,−16] | −0 [−80,+80] | +23 [−54,+103] | — |
| `onemax` | `pool` | — | **−215** [−328,−132] | **−114** [−201,−39] | **−120** [−206,−47] | — |

**The pool wins every increment TC against both classic candidates, by
+114 to +215** — and it is the only arm that spends its clock down: at 30+1
it makes 2,400+ moves under 2.4 s and bottoms out at 1.3 s, where `min40_4`
and `onemax` make **zero** such moves and never go below 2.8 s. That is the
whole trade, and the surrogate is the wrong instrument for one half of it.

Two identities do a lot of work here and are checked in `tmlib`, not assumed:
`legacy12` **is** `oldtm` (same expression, other units — and classic is the
worse of the two below 2.4 s, where it has no floor at all and the budget
goes negative), and `min40_4` **is** `smooth`/`steptm` at `winc == 0`. So
calibration (a) already priced the classic pairing at sudden death, and the
direct cell agrees: +147 vs a +228 calibration analogue.

**The instrument reports its own bias in that cell.** `legacy12`'s budget
goes negative below a 2 s clock, so 594 of its moves hit the structural-floor
path where the surrogate substitutes the twin's bestmove — a *better* move
than the real engine would have played. +147 is therefore a floor on the
classic gap, not an estimate of it; the packed calibration, whose loser
floors at 0.05 s and needed **zero** substitutions, read +228.

Pool knob stage 1 @ 60+1 (baseline `pool` s=1.0, O=200 ms, M=40), **3 of 6
cells — partial**: `s=0.8` +35 [−33,+105], `s=1.2` −12 [−89,+64],
`O=100 ms` −29 [−106,+45]. Nothing separates yet; `O=300 ms`, phase-M and
the dynamic target had not finished. Stage-0 had already pruned `s=1.2` on
shape (it is the only cell that introduces floor moves at 60+0).

### The 2x2: budget x stop rule (2026-08-17)

The ranking pass above left one question unanswered and the classic landing had
to answer it: the pool is TWO changes, a budget and a stop rule, and classic
pays for them separately — the budget is three statements and the bracket rule
four more, which on a ~150-line engine is a real question and not a rhetorical
one. (It was answered twice over: the arms below priced the Elo, and the
landing put the soft check at `Searcher.search()`'s existing bracket boundary,
making the whole port ZERO minified lines. The cheap half would therefore have
saved nothing in the end either.) Two arms priced them.
**Both DELEGATE their parent's numbers** rather than restating the arithmetic,
so a cell meant to isolate one change cannot be measuring two
(`test_tm_surrogate.py` asserts exactly that):

```
poolyield   the pool's (soft, hard), read by CLASSIC's break-at-any-yield rule
            -- frac is soft/hard, because no fixed fraction of a 5x wall names
            the soft limit
min40_4c    min40_4's (soft, hard), read by the pool's bracket-converged rule
```

Virtual clock, 50 ms/move charge, α=β=1e-30 so no cell can stop itself early.
Elo is A-minus-B; read points were fixed before harvest.

| | classic's break-at-any-yield | the pool's bracket-converged break |
|---|---|---|
| **min40_4's numbers** | *shipped reference* | `min40_4c` **+64.4** [+8.1, +124.3] |
| **the pool's numbers** | `poolyield` **+40.7** [−41.7, +128.0] | `pool` **+223.3** [+136.6, +345.5] |

**IT IS THE PAIR THAT PAYS.** Each single change is modest — +40.7 and +64.4,
summing to ~+105 — and the pair is +223.3. The interaction is about as large as
both main effects together, and the reason is arithmetic rather than
mysterious: the bracket rule's entire effect is to let an unsettled search run
past the soft limit toward the wall, so `hard/soft` bounds what it can buy.
That ratio is **1.25x for min40_4**, which derives its target as 0.8 of its own
wall, against **5x for the pool**. A cheap classic port of the budget alone
would have thrown the whole mechanism away for about half the bytes -- and the
full port landed line-neutral on the minified engine anyway, so the cheap half
would have bought nothing at all.

Per-TC, budget alone against the shipped `min40_4` (`poolyield` minus
`min40_4`, 120 games unless noted):

| TC | games | Elo |
|---|---|---|
| 30+1 | 60 (read point) | +40.7 [−41.7, +128.0] |
| 30+1 | 200 (ran on) | +41.9 [−0.4, +85.5] |
| 60+1 | 120 | +37.8 [−15.5, +92.9] |
| 60+0.1 | 120 | +31.9 [−22.6, +88.1] |
| 60+0 | 120 | +52.5 [+0.3, +107.2] |

The 60+0 cell carries the one safety reading in the set, and it is against the
INCUMBENT: `min40_4` flagged **3 of 120** modelled games and reached a −0.08 s
clock, where the pool budget flagged none and never went below 3.80 s. The
surrogate does not certify flag safety and this does not either — but it is the
recorded cost of parking lowest in the field showing up as flags.

The bracket rule on top of the pool budget, as its own cell: **+76.5**
[+22.0, +135.0] at 30+1 (120 g), and **+117.2** [+64.1, +176.2] at 60+1 (120 g).

**Non-transitivity, stated rather than smoothed.** The same quantity read two
ways disagrees: the direct cell says +223.3, while
(`pool` − `poolyield`) + (`poolyield` − `min40_4`) = +76.5 + 40.7 = **+117.2**,
and the ranking pass above read **+134** [+62, +218] on its own 60 games. A
60-game cell at a 78% score rate has an unstable Elo scale, so the surrogate's
estimate of this gap is best read as *somewhere in +117 to +223, direction
certain*. That is the standing rule working, not failing: the surrogate ranks
and one real-clock match validates. The real clock read **+96.19 ± 33.81** over a fixed 300 at 30+1 —
direction and mechanism confirmed, altitude 1.2x to 2.3x lower than the
screen's range. Logged as a calibration datum: this instrument's cells are
RANKS, and quoting one as a magnitude overstates by that much.

**Telemetry, which matters more than the Elo.** In the decision cell the two
arms have the SAME median spend — 1.218 s pool against 1.236 s `min40_4` — so
the pool is not simply spending more. Max spend is 7.414 s against 1.650 s, and
**29% of `min40_4`'s moves end at the WALL** (934 `deadline` stops of 3180)
against **10% of the pool's**, which ends 90% of its searches on the soft limit.
`floorbk` is 0 on both arms, so this cell carries none of the structural-floor
bias the +147 legacy cell had to disclose.

**Cost note, honestly:** the surrogate's speedup comes from skipping
*waiting*, not *searching*, so it shrinks as the TC grows. A 60+0 cell is
minutes; a 300+3 cell is nearly an hour, because the node budget scales with
the clock. 300+3 is where stage-0 earns its keep: it answers the reserve
question exactly and for free (`min40_4` and `onemax` both end 300+3 holding
73-76 s unspent, against `smooth`'s 11 s — the underspend risk, confirmed).

**What the surrogate cannot see, and never will:** real lag variance, PyPy
warmup and JIT deoptimisation, OS scheduling and cotenancy, and the true
per-position node rate (the shipped profile explains ~57% of the nps
variance; the residual is ±18%, and it is recorded in `npsmodel.json`
rather than smoothed away). **The standing rule is therefore: THE SURROGATE
RANKS, ONE REAL-CLOCK MATCH VALIDATES** — plus a 1+0 hammer, because flag
safety is exactly the property a modelled overhead cannot certify.

```sh
python3 tmlib.py                                   # mirror + drift gate
git show origin/tm-pool-manager:sunfish_ui/uci.py > /tmp/uci_pool.py
TM_UCI=/tmp/uci_pool.py python3 tmlib.py           # ...incl. the live pool
python3 tmsim.py --tc 60+0.1 --plies 63            # stage 0, one table
python3 vmatch.py --arm-a pool --arm-b smooth --tc 60+0 --rounds 100
pypy3 npsprofile.py measure && python3 npsprofile.py fit   # re-measure nps
```

## Caveats (deliberate, documented)

- The shipping clock manager is *not* cloned. Standard UCI clocks use the
  fixed search-only budget below for timed search matches; `vmatch.py` drives
  `go nodes` when the time manager itself is under test. `go` without limits
  runs to depth 999, so always pass a limit or a clock.
- Standard UCI clock games use a fixed `time/12 + 0.9*increment` budget and
  half-clock cap for calibrated search-only screens. This is deliberately
  not the shipping time manager; compare time policies through `vmatch.py`.
  `go` without limits runs to depth 999, so always pass a limit or clock.
- `go nodes` semantics transcribe `sunfish_ui/uci.py`'s go_loop (see Game
  use above); node-identity claims are made at fixed depth, where both
  sides are exactly classic. The twin-vs-pypy sanity match at fixed nodes
  is the driver's regression test.
- Repetition handling equals classic's: `history` = positions of the
  game line only, compared with score included; the known classic quirk
  that a K-table swap hides old history entries is reproduced, not fixed.
