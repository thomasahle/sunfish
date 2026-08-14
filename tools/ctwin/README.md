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
runs in — the twin lives on master and that is master's engine (capped
null move below depth 6, fuel-oracle null from depth 6 since #192,
mate-distance scoring, IID at `depth > 3`). The reference is imported
live by `pyref.py`, so drift in the Python file shows up as a harness
failure, not silent staleness — re-pass the gate, re-tune the flavor
knob defaults, and re-pin variants.py's drift hashes when the search
changes (done for #192). Historical flavors stay reachable by knob:
`set FUEL_NULL 0` for the pre-#192 deep-null cutoff, `set IID_MIN_DEPTH
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
  order — killer read *before* the null-move search, the null proof
  re-reading the table, IID only when the early read found nothing, the
  killer re-searched before the sorted list exists, the sorted list
  never built if the killer cuts.
- **Node counting.** `nodes` increments at exactly one site: `bound()`
  entry, including driver probes, IID probes and TT-hit returns.
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
  replacement-policy battery (`EVICT_POLICY`, `KILLER_COUNT`) and the
  frozen-guide battery (below); refuses to run if `sunfish.py`'s search
  changed under it (pinned source hashes), and its transcription is
  itself difftest-proven (`USE_VARIANT=1` at default knobs must match
  the real engine byte for byte).
- `battery.json`, `guide_battery.json`, `nodescreen.py`, `match.py` —
  the battery matrices, the C-only node/movegen screen over them, and
  the paired-openings fixed-node match driver (python-chess arbiter,
  trinomial SPRT, `--jobs` for concurrent pairs, `--counters` to dump
  the instrumentation, zero tolerance for illegal moves).

### Cross-search identity (`difftest --repeat N`)

A single `go depth D` per position only proves the *first* search of a
fresh engine. Anything whose state crosses a search call — the move
table that survives `search()`, or a guide frozen once per search — is
untested by it, and a per-search guide is EMPTY on that first search, so
a one-shot probe would prove only the guideless path. `--repeat 3` runs
three consecutive searches per position and compares all three. It
immediately earned its place: the twin's `go_depth` was promoting a
guide after the final bracket, where the Python harness consumer breaks
out and abandons the generator before it can. Single-shot probes could
never see it; the second search diverged on its first probe.

## Measured status (2026-08-14, this laptop)

- Identity, standing gate (`make gate`, PGO binary): 27 positions
  (startpos + openings, Bratko-Kopec, WAC, mates, stalemates, null-move
  mates, perft set, KQK) × depth 1..6 with the movegen walk — 830 MTD-bi
  probes and 901 movegen lists byte-identical; depth 1..7 on 6 positions,
  264 probes; two tuned-knob sweeps (QS/QS_A/EVAL_ROUGHNESS on both
  sides) × depth 1..5, 1285 probes; eviction sweeps (`TABLE_SIZE` 500 and
  50) × depth 1..6, 1691 probes. Movegen *call counts* are compared in
  every `done` line since the battery landed.
- Identity, battery cells: USE_VARIANT transcription proof (wide + walk,
  and `TABLE_SIZE` 500) plus 8 cells (policies 1/2/3, killers 2/3, two
  combinations) × (wide, `TABLE_SIZE` 500, `TABLE_SIZE` 50) at depth
  1..5 — 32 suites total in the full battery gate, 0 mismatches.
- Identity, frozen-guide battery: 22 knob settings (every cell of
  `guide_battery.json`, the baseline, and the `USE_VARIANT`
  transcription proof) through the four standing suites — 58,146 probes
  and 21,142 movegen lists — and each one again through two `--repeat 3`
  cross-search suites, 110,664 probes, because a per-search guide is
  empty on a fresh engine's first search. 0 mismatches in all 44 suites.
  The cross-search suites are not ceremony: they caught `go_depth`
  promoting a guide after the final bracket, where the Python consumer
  breaks out and the reference never promotes at all.
- Speed: ~22x sunfish.py under warm pypy3 at identical node counts
  (1,155,634 nodes, depth-7 battery; ctwin 1.421s -> 0.669s across the
  optimization rounds, measured ratios 21.6-27.3x as the pyref side
  swings with host load; `make pgo && make bench` reproduces).

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

Tuning knobs (no recompile): `set NAME VALUE` on stdin, `SF_NAME=` env, or
`NAME=VALUE` argv after the table path (for match harnesses) —
`QS QS_A EVAL_ROUGHNESS TABLE_SIZE NULL_MARGIN NULL_MIN_DEPTH NULL_LIMIT
NULL_RED IID_MIN_DEPTH IID_RED FUT_MAX MATE_DIST FUEL_NULL
FUEL_MIN_DEPTH` (`NULL_MARGIN` is the fuel-probe target margin, master's
own knob since #192, independent of `EVAL_ROUGHNESS`, which still caps
the classic sub-depth-6 null), plus the
tp_move battery: `EVICT_POLICY` (0 master root-guarded FIFO, 1 unguarded
evict-before-insert, 2 depth-stored bounded scan with `EVICT_SCAN_K`,
3 hash-slot two-tier replace-if-deeper), `KILLER_COUNT` (1..3 most recent
distinct killers), `USE_VARIANT` (Python-side transcription proof; no-op
in C), plus the frozen-guide battery below. Unknown or out-of-range
knobs are hard errors on every input path.

### Frozen-guide battery (`guide_battery.json`)

Design: Thomas Ahle. Two generations of the move table. `tp_old` is a
FROZEN GUIDE — a value-bearing policy held constant for one epoch — and
`tp_move` stays the MUTABLE CURRENT table, allowed to affect ordering
and the returned move and nothing else. That split is what lets a
guide-dependent reduction keep classic's invariant: the value at a
`tp_score` key stays a function of the key and the epoch, never of
mutable ordering state.

**The epoch rule**: a `tp_score` interval is valid for exactly one
guide, so every promotion clears `tp_score` in the same breath.

| knob | meaning |
|---|---|
| `SCORE_EPOCH` | isolated control: clear `tp_score` per ID iteration, change nothing else |
| `GUIDE_MODE` | 0 off; 1 promote after each COMPLETED ID bracket; 2 freeze one guide per `search()` call |
| `GUIDE_COPY` | 0 `tp_old, tp_move = tp_move, {}` (as designed); 1 `tp_old = dict(tp_move)` — freeze without emptying |
| `GUIDE_MIN_DEPTH` | guide lookups run at `depth >` this (3 = IID's gate; 0 opens them to depth 1) |
| `TWO_KILLERS` | search the guide as a second killer when distinct |
| `KILLER_DEDUP` | skip already-searched killers in the sorted list, *after* the futility test |
| `GUIDE_IIR` | replace the recursive IID probe with a one-ply reduction on guideless nodes |
| `GUIDE_INJECT` | admit the guide at positive depth regardless of `val_lower` |
| `GUIDE_PV` | guide keeps the full child depth, alternatives lose a ply |

A partial ID iteration is never promoted: promotion sits after the
bracket, so a mid-depth stop keeps the previous completed guide. Guide
lookups are delayed until after the null phases, so a null cutoff pays
nothing. Under `GUIDE_IIR` the root is never reduced and NOMINAL depth
still keys `tp_score`, sets `val_lower`, scores mates, classifies
terminals and gates every eligibility test — only the real-child
recursion shortens. `GUIDE_MODE` with `EVICT_POLICY 3` is a hard error
(policy 3 replaces `tp_move` with a slot table, so a promotion would
freeze an empty guide).

`counters` on stdin dumps the cumulative instrumentation — guide
lookups/hits, current-vs-guide agreement, cutoffs from the second
killer, IID probes, IIR activations, `tp_score` cutoffs, dedup skips,
promotions and peak table sizes. `match.py --counters` reads it at the
end of a run.

Game use: `position startpos moves …` / `position fen …`, then
`go nodes N` (primary — clock-free surrogate games), `go depth D`,
`go movetime T`. The `go nodes` consumer transcribes `sunfish_ui/uci.py`'s
`go_loop`: the probe that crosses the cap always finishes and its yield
counts, the cap is checked between probes at depth > 1, candidates commit
when their depth completes, and bestmove has the structural floor (never
"(none)" while a legal move exists). This matters: an earlier mid-probe
abort made the twin play one depth staler than pypy at stop points — the
calibration match caught it at -54 ± 43 and the run was voided.

## Measured verdicts

### Frozen guide (2026-08-14): DECLINED

Fixed-node (20k) C-vs-C against the master twin, paired openings, 334
pairs max. `mode 1` promotes per completed ID bracket, `mode 2` freezes
one guide per `search()` call.

| cell | games | Elo [95%] | SPRT |
|---|---|---|---|
| `epoch` (score epoch alone) | 668 | -17.7 [-41.5, +5.9] | undecided at book end |
| `guideonly` (mode 1, anchor) | 548 | -59.5 [-86.6, -33.2] | H0 (<= -10) |
| `k2guide` (mode 1 + 2 killers) | 384 | -69.7 [-101.6, -38.9] | H0 (<= 0) |
| `iir` (mode 1 + IIR) | 296 | -90.0 [-126.7, -55.2] | H0 (<= 0) |
| `composite` (mode 1, all) | 464 | -59.0 [-88.5, -30.2] | H0 (<= 0) |
| `pv` (mode 1 + frozen PV) | 240 | -118.8 [-161.8, -79.0] | H0 (<= 0) |
| `guideonly2` (mode 2, anchor) | 668 | -11.4 [-34.9, +11.9] | undecided at book end |
| `iir2` (mode 2 + IIR) | 668 | -12.0 [-35.3, +11.3] | undecided at book end |
| `k2guide2` (mode 2 + 2 killers) | 668 | -2.6 [-26.0, +20.7] | undecided at book end |
| `composite2` (mode 2, all) | 668 | -10.9 [-34.4, +12.4] | undecided at book end |
| `copy2` (mode 2 + snapshot, all) | 668 | -14.6 [-38.2, +9.0] | undecided at book end |
| `copy2_k2` (mode 2 + snapshot, 2 killers) | 668 | -2.1 [-25.5, +21.3] | undecided at book end |

Nothing beats master. The mode-1 numbers are dominated by the regime,
not by the search changes: promotion is `tp_old, tp_move = tp_move, {}`,
so the current table restarts empty every ID iteration and classic loses
its accumulated killers at each depth (`iid` 36402 vs master's 28586,
peak `tp_move` 369022 -> ~30000). Under mode 2 the regime is nearly free
and every composite collapses onto its own anchor: the guide machinery
is cheap there and simply worth nothing. `GUIDE_COPY 1` (freeze without
emptying) was added because the anchors point at the emptying, and it
does not rescue the family either.

`GUIDE_INJECT` is a **structural no-op in classic**, measured rather than
assumed: `val_lower = QS - depth*QS_A` is -520 at depth 4 and -100 at
depth 1, and a guide is by construction a move that already fail-highed,
so over 469000 guide hits at every lookup depth the `guide_below` counter
is 0. A_G = A identically, and the cell is byte-identical to its control.

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

## Caveats (deliberate, documented)

- The clock-management branch of classic's `main()` (wtime/winc budgets)
  is *not* cloned — the twin is for clock-free games. `go` without
  limits runs to depth 999; always pass `nodes`, `depth` or `movetime`.
- `go nodes` semantics transcribe `sunfish_ui/uci.py`'s go_loop (see Game
  use above); node-identity claims are made at fixed depth, where both
  sides are exactly classic. The twin-vs-pypy sanity match at fixed nodes
  is the driver's regression test.
- Repetition handling equals classic's: `history` = positions of the
  game line only, compared with score included; the known classic quirk
  that a K-table swap hides old history entries is reproduced, not fixed.
