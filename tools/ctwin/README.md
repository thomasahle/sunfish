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

**Reference:** `sunfish.py` at the repo root of branch `nnue-4k`
(capped null move, mate-distance scoring, IID at `depth > 3`). The
reference is imported live by `pyref.py`, so drift in the Python file
shows up as a harness failure, not silent staleness. Master-flavor
behavior is reachable by knob: `set IID_MIN_DEPTH 2`, `set MATE_DIST 0`.

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
- `pyref.py` — protocol server around the real `sunfish.py`.
- `difftest.py` — the differential harness (see contract above).

## Measured status (2026-08-14, this laptop)

- Identity: 27 positions (startpos + openings, Bratko-Kopec, WAC, mates,
  stalemates, null-move mates, perft set, KQK) × depth 1..6, 830 MTD-bi
  probes and 901 movegen lists byte-identical; 5 positions × depth 1..7,
  223 probes identical; two tuned-knob sweeps (QS/QS_A/EVAL_ROUGHNESS
  changed on both sides) × depth 1..5, 746 probes identical.
- Speed: 8-10x faster than sunfish.py under pypy3 at identical node
  counts (871k nodes, depth 7, JIT warm; `make bench` reproduces).

## Use

```sh
make            # build + regenerate tables
make test       # quick identity probe (~seconds)
make test-full  # wide sweep, depth 6
make bench      # C-vs-PyPy wall-time ratio at identical nodes
```

Tuning knobs (no recompile): `set NAME VALUE` on stdin or `SF_NAME=` env —
`QS QS_A EVAL_ROUGHNESS TABLE_SIZE NULL_MARGIN NULL_MIN_DEPTH NULL_LIMIT
NULL_RED IID_MIN_DEPTH IID_RED FUT_MAX MATE_DIST` (`NULL_MARGIN -1`
tracks `EVAL_ROUGHNESS`, which is classic's actual coupling).

Game use: `position startpos moves …` / `position fen …`, then
`go nodes N` (primary — clock-free surrogate games), `go depth D`,
`go movetime T`. The `go nodes` loop mirrors the packed dev build: cap
checked every 2048 nodes inside the search, candidates committed only
when their depth completes.

## Caveats (deliberate, documented)

- The clock-management branch of classic's `main()` (wtime/winc budgets)
  is *not* cloned — the twin is for clock-free games. `go` without
  limits runs to depth 999; always pass `nodes`, `depth` or `movetime`.
- `go nodes` semantics match the packed dev build's node cap, which the
  classic *reference* does not have; node-identity claims are made at
  fixed depth, where both sides are exactly classic.
- Repetition handling equals classic's: `history` = positions of the
  game line only, compared with score included; the known classic quirk
  that a K-table swap hides old history entries is reproduced, not fixed.
