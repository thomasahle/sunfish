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
depth 6, fuel-oracle null from depth 6, intrinsic LMR, mate-distance
scoring, and no IID). The reference is imported
live by `pyref.py`, so drift in the Python file shows up as a harness
failure, not silent staleness — re-pass the gate, re-tune the flavor
knob defaults, and re-pin variants.py's drift hashes when the search
changes (done for #192). Historical flavors stay reachable by knob:
`set FUEL_NULL 0` for the pre-#192 deep-null cutoff, or `2` to spend two
depth units on a hot node; `set IID_MIN_DEPTH
2` + `set MATE_DIST 0` for pre-capped-null master — knob-off settings
are no longer difftest-provable against the live reference; their
identity was proven against the reference of their day and is archived
in the git history of this file.

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
  the board directly, king captures resolve before recursion, and the sorted
  list is never built if the killer cuts.
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
- `adaptive_gp.py`, `logistic_gp.py`, `all_parameters.json` — an
  asynchronous logistic-GP game tuner and its mixed search/evaluation space.
  `sunfish_gate.py` prevents policies that lose the curated eventual-mate
  guarantees from consuming games.
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
`QS QS_A LMR EVAL_ROUGHNESS TABLE_SIZE NULL_CAP_MARGIN NULL_MARGIN
NULL_MIN_DEPTH NULL_LIMIT NULL_CUT_RED IID_MIN_DEPTH IID_RED FUT_MAX FUT_CAP FUT_CAP_DEPTH
MATE_DIST FUEL_NULL FUEL_MIN_DEPTH FEN_HIST` (`NULL_CAP_MARGIN=-1` follows
`EVAL_ROUGHNESS`, `NULL_MARGIN` is the fuel-probe target margin, and
`NULL_CUT_RED` controls the shallow probe; `FUEL_NULL` controls
the hot node's extra depth cost, while zero skips the probe but retains the
static intrinsic-LMR guard). `FUT_CAP` selects no shallow cap, the current
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

For long joint studies, one color-swapped opening pair is one posterior
update. Forty engine processes means twenty scheduler slots:

`all_parameters.json` covers every live search/evaluation constant, including
the null-oracle fuel amount. It excludes
`TABLE_SIZE` (a memory budget) and
the historical or PR-only flavor selectors above; those belong in separate
ablation matches, not in the production-parameter posterior. Its default
point is current master, including `NULL_LIMIT=750`. Every proposed challenger
must pass the mate-floor correctness gate before games are spent.
The numeric search domains cover the source's declared tuning ranges, except
that `QS_A=0` is excluded because it would permanently filter moves instead
of eventually widening the real tree. Evaluation ranges are limited by the
corner-checked mate-band, promotion, and nonnegative-table invariants; a
posterior maximum on one of those boundaries is a proof constraint, not an
invitation to sample an invalid table.

```sh
python3 adaptive_gp.py \
  --fastchess /path/to/fastchess \
  --engine ./sunfish_c --engine-args ./tables_classic.txt \
  --baseline-options default \
  --space all_parameters.json --openings openings.fen \
  --gate "python3 sunfish_gate.py" --gate-design --gate-workers 20 \
  --cycle-openings \
  --slots 20 --queue-batches 60 --refill-batches 20 \
  --pairs 1 --initial-design 256 --inducing 128 --update-batches 8 \
  --explore-start .5 --explore-floor .2 --duel-fraction .3 \
  --wall-time 3d --batches 1000000
```

`--baseline-options default` pins the exact default point to zero. Opening
reuse is balanced by a fresh deterministic shuffle per epoch, and independent
books remain mandatory for final confirmation. A fixed inducing basis permits
online Laplace updates without rebuilding a quadratic comparison matrix. The
optimizer keeps its small matrix operations single-threaded so its 128-site
model does not compete with the 20 game lanes. Its 2,048-point global design is
used for the initial design, acquisition restarts, and inducing sites. It
reserves 512 points for the default, every one-axis setting, and nearby two-axis
combinations; the rest retain broad global coverage. Coordinate refinements
are gated on demand. The pure-variance arm explores finite-design policies and
validated coordinate refinements. It covers the unseen, gate-safe global
design before revisiting a policy; afterward, statistically dominated policies
leave the pure arm. UCB is free to replicate promising policies throughout.
Proposals pass the correctness gate before games are spent. Rejected policies
consume neither games nor allocation credit.
Three reserved pairs per lane,
replenished while two remain, hide that latency. `--gate-all` instead
prevalidates the finite design and confines every acquisition to it; use that
for a broad census, not the final joint refinement. `--gate-design` rejects
unsafe design points up front while still allowing gated coordinate
refinements. Results append to a JSONL journal and compact every 1,000 pairs,
avoiding quadratic checkpoint I/O while remaining restartable. At the
wall-time limit, the scheduler finishes every reserved color pair before its
final checkpoint. `--seed-state` replays that journal and inherits its
allocation clock; pass `--seed-selections 0` only to restart exploration.

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

## Calibration plan (stage 1 PASSED 2026-08-14; stages 2-3 staged)

Node-identity proves the twin searches classic's tree; it does not yet
prove that *match results* from the twin transfer. Before any twin number
feeds a merge/decline decision (docs/TESTING.md rule 14), run, in order:

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

Only after all three: twin grids/SPSA results are decision-grade at
*fixed effort* — rule 12 of docs/TESTING.md still applies before any
wall-clock claim.

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

- The clock-management branch of classic's `main()` (wtime/winc budgets)
  is *not* cloned — the twin itself is for clock-free games. `go` without
  limits runs to depth 999; always pass `nodes`, `depth` or `movetime`.
  Clocked play is supplied *around* the twin by `vmatch.py`'s virtual
  clock (above), which drives the same `go nodes` path.
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
