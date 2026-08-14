# PACKED128 revival audit (2026-08-14, branch `packed128-revival`)

Scope: reconstruct the packed big-int NNUE line's +96 artifact, price its
composition with today's entry, settle comparability, and stage (not arm) a
retrain recipe. No matches were played; one ~1s stdin smoke of the prototype
artifact and one 2.3s trainer smoke were the only processes run.

## 1. Archaeology: the +96 artifact

**Identity.** "packed128 v1" = `sunfish_packed.py` at commit **`92c4746`**
("Round the net residual towards zero", 2026-08-09 12:47) + the v1 width-128
net (today at `nnue_4k/packed/net128.pickle`: N=128, shift 6, clampcp 600,
plain clipped ReLU, sum_G 47973, excursion 13309; val 0.01025 in the width
sweep, `7ec71af`).

**The measurement.** Classic-anchored 60+1 gauntlet over widths 64/128/256,
160 games per pair, 480 games total, bench box, marker `PACKED_CLOCK.txt`,
verdict 2026-08-10 midday:

| arm | vs classic @60+1 |
|---|---|
| packed256 v1 | +100.4 ± 53.6 |
| **packed128 v1** | **+95.7 ± 54.8** |
| packed64 v1 | +48.1 ± 46.2 |

Width decision: **128** (`7ec71af`: "Width 128 chosen"; 256 cost 17% nps for
an overlapping interval). **Ledger gap, recorded honestly:** this verdict has
NO in-repo ledger entry — it predates `MEASUREMENTS.md` and the backfill
(`af95d18`) covered commit messages only; the verdict lived in the box marker
and the workstream memory. The nearest in-repo record is `951c441` (10+0.1,
+107.5 ± 65.3 for N=256, 120g, LOS 99.98%), which announces the queued 60+1
gauntlet that produced the +96.

**Eval architecture** (founding commit `a9d2334`, quoted): whole accumulator
and whole head in ONE Python int, 2N = 256 sixteen-bit lanes.

> "Output weights are FOLDED INTO the input tables, so lane k already holds
> G_k * a_k ... Clipped ReLU then becomes a clamp of lane k to [0, G_k] ...
> Horizontal sum by MODULAR REDUCTION: 2^16 == 1 (mod 2^16-1), so a block's
> residue IS its lane sum. ... sum(G) <= 65534 is asserted at build time."

`score = pst(pos) + clip(nn(pos), ±600)`; `value(move)` stays classic's exact
pst delta, so ordering, the QS gate and futility are classic's. 19 big-int ops
per eval; measured 87-96% of classic's nps at width 128. Quantisation: 16-bit
lanes, output in 1/64 cp (shift 6), round-toward-zero for exact antisymmetry
(`92c4746`). Training: lichess SF-eval dump, quiet-filtered, sigmoid-MSE at
K=400.

**Search it was paired with:** the 2026-08-09 classic-parity MTD-bi
(classic-exact search predicates, `8ce2626`). Pre-everything that defines
today's entry: pre-KCX (`b307710`, 08-11), pre-go-loop-correctness
(`4446ad6`), pre-MTD-guards/LMR (08-12), pre-IIR/kend-fix/capped-null (08-13),
pre-TM formula (`d3f7f12`). The baseline classic was 2026-08-09 classic, which
per the goal-line ledger "gained ~+130 during the campaign" by 08-12.

**Bytes.** Reproduced today with that tree's own `build/pack.sh`:
**engine 3952 B** (matches the budget ledger "92c4746 v1 engine 3952 ✓") —
plus an EXTERNAL 400,563 B net pickle. Under the later accounting rule
(`b267a19`: "the net counts toward the 4096 bytes") the +96 artifact was never
4k-legal; its honest size is ~404 KB.

**The recorded next steps — and a correction to the revival premise.** The
coordinator sequencing of 2026-08-10 ("retrain@128 → king buckets → bilinear
m=4 fold + NARROW tail") was **not left unexecuted**. It ran, same week:
retrain/stack-v2 (`71b2751`, `5ec78b5`), king buckets kb4/kb8 (`2535a9b`,
`e39fed1`, `7044c3e`; kb8 became the play king, +96 pairwise over kb4 — a
different "+96"), bilinear m=4 + odd-symmetrized narrow tail (`e2ef111`,
`6da5edd`/`bf46a60`; best-val kbbil COLLAPSED −118 in play, `7602d7b`), width
256 (+52.5 ± 43.6 over kb8@128), rff, rehab. The line's true high-water is the
**GOAL-LINE VERDICT: +187.0 ± 49.7 vs classic @60+1, 272 games (2026-08-12)**
with 256kb8@100M — from an engine verified to have NO LMR/guards. The lane
then pivoted on two findings: "SPEED IS ELO: ~100 Elo/doubling" (08-11) and
the byte accounting ("705B of NNUE machinery, and the 4k arithmetic may kill
the thesis", `f80ad33`), which produced today's distilled-PST entry.

## 2. Composition audit: packed128 eval + today's search

Today's entry: `nnue_4k/pst_entry.py` at `1ca26e4` (nnue-4k HEAD), packs to
**3357 B** (reproduced), 739 spare.

The generators do not share an eval seam (the entry is generated pst-only;
packed128 was a hand-written engine), but the Position skeletons are siblings
— today's entry's `rotate()` docstring still carries the accumulator comment.
So an **honest hand-graft prototype** was built and is committed as
`nnue_4k/packed128_compose_proto.py` (loudly marked NOT AN ENTRY): today's
entry + the exact v1 hot loop (`nn_cp`, acc/ps/pf threading through
`Position`/`move`/`rotate`/`from_board`) with STUB net data.

| build | pack.sh bytes | delta |
|---|---|---|
| entry @ HEAD | 3357 | — |
| + packed128 machinery (stub data) | **3799** | **+442** |
| remaining for net data | 297 | |
| the +96 net's data, lzma −9e | **126,076** | raw rows 738,720 |

Correctness of the graft (no engine matches): 40-ply random-walk with live
random rows — incremental accumulator == from-scratch, score identity, and
exact antisymmetry all PASS; the packed artifact answers UCI and produces a
sane bestmove in a 1s stdin smoke.

**Verdict: structurally impossible at width 128.** The gap is not
engineering: 768×128 = 98,304 int16 first-layer weights against a ≤ ~740 B
data budget — over by ~126 KB, a factor of ~420. A real build would also add
two 4096-bit literals (MGP, ACC_BASE) the stub elides.

**Smallest bridge.** Not golf — width. An embeddable packed residual needs
N ≤ 4-8 with ternary base-3+lzma weights (`4850894`: b3 and lzma COMPOSE):
N=4 ≈ 3,072 trits ≈ 615 B raw base-3, ~400-500 B packed → 3357 + ~442 + ~450
≈ **4249 B, i.e. ~150 B over even at N=4**. The bridge therefore requires
both the small net AND ~150-200 B of machinery golf (drop the MTS loop,
fuse constants). Marginal, but the only shape that fits. Critically, this
capacity region (3k params, nonlinear) sits BETWEEN the five dead 384-param
linear fits (Texel, C1 −57.7, C2 −93.8, d1 −76.0, b1 −182.6) and the
98k-param net that measured +96/+187 in play — it is genuinely untested.

*Correction, same night (Section 6): the estimate above was optimistic on
both terms. Measured by building, the golfed N=4 composition is 4512 B —
416 over — and the composed-figures-miss-measured-ones rule claims its
6th victim. See the golf ledger.*

## 3. Staleness check

The +96 is **superseded, not additive**:

1. **Dominated within its own line**: the same design, retrained and
   king-bucketed, measured **+187 ± 50 @60+1** on 2026-08-12 against a
   classic ~+130 stronger than the one packed128 v1 beat. packed128 v1 is a
   strictly dominated ancestor.
2. **Baseline drift**: today's entry's +100-130 is against current classic.
   Naive frame-shifting puts packed128 v1 + its 08-09 search near or below
   parity with current classic — likely BELOW today's entry.
3. **What survives**: the packed-residual DESIGN (residual on top of an
   untouched pst, `value(move)` exact) is the only fitted-eval form with a
   positive in-play record on this project, and its +224-equivalent eval
   edge (the +400 decomposition, `2b2bf18`-era) was real at 98k params. The
   five fitted-table failures are a capacity/objective failure, not a
   refutation of the form.

**The measurement that would answer it** (pre-registered, NOT armed): once an
embeddable composed build exists —

- Arms: `packed128_compose` (small-net residual) vs `pst_entry` @ HEAD, both
  as packed artifacts.
- Instrument: the standard fixed-node screen — fresh `sunfish_ui` driver
  (DRIVER_VERSION + `max_nodes` verified at startup), `go nodes 20000`,
  PGN-book openings per the artifact rule (`openings` book, srand
  pre-registered before launch), `-recover`, 0 forfeits tolerated, SPRT cap
  1000 games.
- LAND bar, pre-registered: **95% LB > 0** at fixed nodes, AND a timed
  confirmation LB > 0 at 30+1 before landing — fixed nodes alone hides the
  ~19-op/eval speed tax, and this lane's ledger shows fixed-node/timed sign
  flips are its house specialty.

## 4. Retrain prep (staged, NOT armed — no GO marker exists)

**Pipeline located and smoked.** Trainer `nnue_4k/packed/train_packed.py`
(torch; full flag set incl. `--kb 8` king buckets, `--satpen`, `--clampcp`,
`--losspow`, `--factor`, bilinear `--nb/--bm`, `--rff`), packed build/verify
in `pnet.py` + `build_kb.py` + `verify.py`. Data source: lichess SF-eval dump
(jsonl.zst; the 21.7 GB dump and 30M/100M parsed caches live on the frozen
bench box). Labels: dump Stockfish evals, quiet-filtered, `--cpmax 1000`.

Smoke (allowed, training tooling): 1 epoch ≈ 10 steps of the kb8 width-128
recipe, batch 512, on 4,999 real positions (the committed
`tools/tune/data/set20260813.npz` converted to dump format), CPU `nice -15`:
**2.3 s wall**, parse → train → export all green. Smoke artifacts stayed in
the scratchpad; nothing committed.

**Recipe (per the recorded designs, resized to what can ship):**

- Arch: packed residual, **N=4 (fallback N=8)**, `--kb 8` king-bucketed
  features per `e2ef111`/`7044c3e` — note kb multiplies FLOAT rows, not
  shipped bytes, only if buckets are folded at export to shared ternary rows;
  otherwise kb8 ×8s the data and must be dropped to kb=1. Decide by pricing
  both exports through pack.sh before any games.
- Regularisation: satpen 0.03 @ 480, clampcp 800, phasecap 2.0 (the rehab
  recipe — SATPEN IS DEFAULT), plus strong ternary sparsity pressure (the
  byte cost is the loss term that matters at this size).
- Data: 2-8M quiet positions re-parsed from a dump slice (a small net does
  not need the 100M cache), **flat phase-balanced mix** (`a46a801`: the mix
  is the mechanism; but note `3ee1262`: b1's balanced 384-param fit still
  died — mix helps fitting, capacity decides play).
- Where: laptop CPU, `nice -15` (minutes per epoch at this size; the box is
  frozen and not needed).
- Gate ladder before any game: val vs pst anchor, shapecheck, pack.sh ≤ 4096
  on the REAL composed entry, then the Section 3 screen.

## 5. Recommendation

**Neither "screen as-is" nor "retrain@128".** There is no as-is composition
inside 4096 (over by ~126 KB), and as an oversized dev build the question is
already answered by dominance (+187 supersedes +96; both against weaker
classics than today's). Retraining at width 128 reproduces an artifact that
cannot ship.

The one live move this audit supports: **small-net retrain under the
packed-residual form** (N=4-8 ternary, recipe above) — the only fitted-eval
design with a positive game-measured record — **then** the pre-registered
fixed-node + timed screen against today's entry. Before arming it, spend one
session on the ~150-200 B machinery golf, because at N=4 the composition is
~150 B over budget even with data that fits; if the golf does not close, the
lane stays closed and the +96 stays what it is: the ancestor of a +187 line
that the 4096-byte accounting, not play, killed.

## 6. Machinery golf (2026-08-14, coordinator-directed)

Target: composed artifact (machinery + N=4 ternary data) ≤ 4096 against the
current entry at 3357. Every step below ran the invariant suite
(`nnue_4k/packed/proto_check.py`: mirror identity, 40-ply walk with
incremental==from-scratch acc/ps/score, exact antisymmetry, net-fires) and
`tools/build/pack.sh`; regressing steps were REVERTED, not patched around.
The N=4 payload is a real-shaped random ternary net (`make_proto_payload.py`:
768 chars through the entry's own base-90 codec, one char = one feature's 4
trits, char-aligned so lzma sees the sparsity; 55% zeros = the middle of the
ledger's measured 42/55/66% real-weight sparsities, `4850894`).

### Golf ledger

| step | change | packed B | delta | invariants |
|---|---|---|---|---|
| G0 | audit prototype (N=128 machinery, stub data) | 3799 | — | green |
| G1 | re-baseline: N=4 + real-shaped 777-char payload, naive machinery | 4619 | (baseline) | green |
| G2 | drop the dead MTS segment loop (the +96 net shipped plain crelu) | 4591 | −28 | green |
| G3 | "." zero-row to collapse move()'s capture branch | 4597 | +6 → **REVERTED** | green |
| G4 | shared `_dec` codec function for pst + payload | 4607 | +16 → **REVERTED** (lzma already dedups the twin loops; the abstraction adds unique bytes) | green |
| G5 | replicator masks: `_U = (2^128−1)//(2^16−1)`, kill `_rep` | 4573 | −18 | green |
| G6 | fold gains into weights at decode; `_lane` = pure shift-sum | 4568 | −5 | green |
| G7 | fuse from_board's ps and acc into one pass | 4555 | −13 | green |
| G8 | micros, bisected: keep `ACC_BASE = MLO + …` (−1); min/max clamp (+2) and extraction-loop merge (+27) **REVERTED** | 4554 | −1 | green |
| G9 | fuse payload decode with half-row build (one char = one feature, build order = extraction order; `_W`, `_lane`, and the square-validity test all disappear) | 4527 | −27 | green |
| G10 | inline single-use BIAS and MASKLO | 4517 | −10 | green |
| G11 | both-block replication `*(1 | 1<<HALF)` for MGP/ACC_BASE; drop NLANE and the `_b` list | 4514 | −3 | green |
| G12 | tuple-assign constant folding | 4512 | −2 | green |

Golfed total: **4619 → 4512 (−107)**. Final artifact answers UCI and plays
(1s stdin smoke, `bestmove g1f3`).

### Where the bytes are (measured on the golfed tree)

| component | B |
|---|---|
| entry @ HEAD | 3357 |
| machinery (payload elided, repacked) | **532** |
| N=4 payload @ 55% zeros, in-context | **623** |
| **total** | **4512** |
| **gap to 4096** | **−416** |

Sensitivity: at 66% zeros (the ledger's sparsest measured real-weight point)
the payload prices at 576 B in-context → total **4465, still 369 over**.
Considered and declined: moving the payload out of base-90 source text into
pack.sh's joint-lzma raw-blob stream (the `4850894` mechanism) — 5 trits/byte
saves ~70 B of payload but costs ~90 B of self-read machinery; net negative
at this payload size.

### Extrapolation across N (machinery is width-invariant)

| N | data est. | total est. | verdict |
|---|---|---|---|
| 4 | 623 | 4512 measured | over by 416 |
| 2 | ~310 | ~4200 | over by ~100 |
| 1 | ~160 | ~4050 | fits — but a 1-hidden-unit "net" is not the packed-residual design |

### Verdict: the ternary retrain is NOT GO-able

The pre-registered kill-condition fires: golf cannot close the gap at N=4
against the current entry — machinery alone (532 B) is within 200 B of the
739-byte spare, before any weights. No meaningful width closes: even N=2 is
~100 over. What WOULD reopen the lane, for the record:

1. **Entry-side shrink ≥ ~420 B** (golf against 3357 was the instruction;
   unhatched savings not counted — but if a future entry lands near ~2940,
   which the golf ledger `296fd55` once measured as reachable, N=4 fits).
2. A machinery breakthrough halving the 532 B — nothing in tonight's twelve
   steps suggests one; the hot loop, decode, and threading are each near
   their floor, and the two abstraction attempts (G3, G4) both LOST bytes
   because lzma already shares repeated code.

Until one of those exists, the lane closes per the plan of record, and the
staged retrain recipe stays unarmed.
