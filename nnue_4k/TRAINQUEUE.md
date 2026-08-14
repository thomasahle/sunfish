# TRAINQUEUE — the trainer never idles (standing rule, Thomas 2026-08-14)

Priority-ordered next nets for the replacement-net (replnet) lane. The
moment a run finishes, the top entry starts (labeller-class on the box:
nice 19, ≤8 workers/threads, forfeit tripwire on live matches). Gates,
screens, and landings stay coordinator-dispatched and play-gated — this
queue is about TRAINING only. Re-order freely as results land; never
empty. Provenance pinned on every run.

## THE FAMILY OBJECTIVE (Thomas, 2026-08-14)

"The goal is to have an nnue that's general enough that it can capture
and learn all the things we need: end games, king protection, midgame,
pawn structure, mobility, etc." — and NOT to write custom code per
weakness: hand terms spend bytes on code that weights should learn.

Knowledge-class → capacity-axis mapping (probes.py scores every export
by these classes):

| knowledge class | capacity axis | queue arm |
|---|---|---|
| endgames / phase | phase axis | c1024-phase |
| king safety | king buckets + count nonlinearity | c1024-kb4, replnet_ml2 |
| pawn structure, mobility | second-order capacity — a LINEAR net over ps768 cannot represent pairwise relations (passer-vs-blocker, shelter-vs-attacker, mobility) AT ALL; they need products | replnet_ml2, bilt, rff |
| midgame | base + all axes | c1024-cal onward |

Affordability thesis: generality at 1024 B comes from STRUCTURED SHARING
(low-rank + trained-codebook parametrizations, in-loop — the extension
lane is building them in train/structures.py) + RUNTIME PRODUCTS (ml2
certified lanes) — never from enumerated feature crosses.

**SUBSUMPTION RULE (standing):** every hand term that lands (pend passed
its screen 2026-08-14, +37 B) carries a standing ablation obligation —
when a phase-capable net candidate reaches screening, the screen matrix
includes net-vs-net+term; a term the net subsumes is DELETED and its
bytes refunded. Hand terms are stopgaps, not accumulation.

## Context

2026-08-14: payload target 1024 B (Thomas; golf lane opening the code
side, exact capacity confirmation pending). Budget-617-era numbers: v1
winner l1=0.001 val 0.01385 @59.6% zeros; v1c 0.01389 @65.2%. At 1024
the 3,072-trit ps768 payload fits at any sparsity — sparsity is a
capacity dial now. Encoder: compress/ bake-off winner becomes export
default when its table lands; arms re-size to the weight capacity it
buys. Probe suite (train/probes.py) runs at every export; scores are
ledgered per net (.probes.json), diagnostics never gates.

## Queue

1. **c1024-cal — capacity calibration at the new budget** (winner recipe,
   N=4 ps768, sparsity released: l1 ∈ {0, 0.0003}, τ 0.6; ~35-50% zeros
   ≈ 1.6-2.0k nonzeros). Cheapest capacity arm, calibration point for
   everything below. PRICE per export (pack.sh + bake-off winner).
2. **c1024-phase — phase capacity IN WEIGHTS** — the arm that can SUBSUME
   the hand phase terms (K_END swap, khold2, pend-class knowledge): if
   phase×feature products are representable, the net can learn "passers
   grow with phase" and "king activity flips sign" on its own — the
   probes' phase class is its scoreboard. Three candidate forms, pick by
   CERTIFIED PRICE at the 1024 target before training the winner:
   (a) tapered two-phase tables — every feature gets MID/END trits,
       root-phase blend (K_MID/K_END generalized): ~2× table ≈ ~1.2 KB
       at v1 sparsity — likely over; price via a real-shaped double
       payload through the bake-off before dismissing;
   (b) phase-bucketed features — 2-3 phase buckets × ps768 through the
       ternshared shared+delta fold (buckets keyed on root piece count,
       not king square); price the delta sparsity the fold buys;
   (c) phase-as-input through the ml2 certified line — products learned
       at runtime, no table doubling; likely the byte-efficient form;
       coordinate with replnet_ml2 (its certificate prices the machinery).
3. **c1024-kb4 — king buckets at 1024, PRICE-FIRST** (kb4 × 3,072 trits
   ≈ mid-2k B at old rates — likely still over at 1024; the ternshared
   route — shared rows + ternary per-bucket DELTAS — is the form to
   price. kb8fold's fold quality feeds this call.)
4. **replnet_ml2 — certified multi-layer, PRICE-FIRST** (packed_layers
   line; certificate.json beside the run). Carries the second-order
   burden for pawn structure/mobility AND is c1024-phase form (c)'s
   machinery. Val probe + field-budget certificate before any packed
   build.
5. **c1024-n8 — wider hidden N=8 at ps768** (~6.1k trits, two chars per
   feature — codec seam is the GOLF LANE's; agree it before training).
7. **replnet_ratecal — rate-aware retrain** (compression lane's yaml,
   `train/queue/85_replnet_ratecal.yaml`; natural slot beside #1 — the
   rate term is the capacity family's native loss once calibrated).
8. **replnet_tau — threshold sweep** (τ ∈ {0.6, 1.1}; low side subsumed
   by #1).
9. **replnet_bilt — bilinear m=4 + odd tail, PRICE-FIRST, behind the
   capacity family.**
10. **replnet_rff — rff64 at tiny width, VAL probe only.**
11. **replnet_clamp — CLAMP/satpen interaction** (400 vs 600, byte-free).
12. **c1024-general — THE TERMINAL ARM: the composed net** (king buckets
    × phase × second-order, through whichever sharing structures price
    in). Explicitly GATED on the individual axes landing first: cal →
    phase → kb4 → ml2/lowrank/codebook results all feed its form.
    PRICE-FIRST, certified, probe-suite-scored per knowledge class at
    export. This is where the subsumption rule points: when c1024-general
    screens, the matrix includes it vs entry+hand-terms — the goal state
    is the net winning that comparison INCLUDING its nps tax.

### APPENDED 2026-08-14 by the pipeline-extension lane — trained structure

Single-writer discipline: appended as its own block, nothing above
renumbered. Natural slot for both is **beside c1024-cal** (entry 1) —
they are the same capacity dial seen from the payload side — and both are
**VAL-PROBE-FIRST behind it**, since cal is the denominator they are
read against. Re-order freely. Configs are written and certified:
`train/queue/90_c1024_cb.yaml`, `train/queue/91_c1024_lr.yaml`.

- **c1024-cb — trained CODEBOOK at 1024, PRICE-FIRST, VAL probe.**
  `arch: cb`, K ∈ {32, 64} at block 8, ternary entries. The post-hoc
  version (cb8) priced +141 B on v1 because a net trained for a dense
  trit stream has no reason to repeat blocks; `structures.CodebookWeight`
  puts the book and the assignment stream in the graph (hard-argmin
  forward, identity to the shadow + softmax-weighted to the book
  backward). **Expected payload composition, pre-codec, K=32/block 8:
  book 32×8 = 256 base-90 chars + indices 96×log2(32) = 480 bits = 74
  chars → 332 chars + 9 header**, against cal's dense 768-char stream;
  K=64 → 598 chars. Two measured taxes stated up front: the codebook
  decoder costs **68 B** (layout A) and lzma finds block repeats for
  free — on a 22-codeword smoke net the explicit book still lost by
  **+47 B**. The reason to run it anyway is the operating point: the book
  size is sparsity-independent while the dense stream loses its zero-run
  advantage at cal's 35-50% zeros. NO COMPOSED TOTALS — the artifact
  number is `compress/bakeoff.py`'s (arm `trained_cb`, control `cb8`).

- **c1024-lr — trained LOW-RANK + RESIDUAL at 1024, PRICE-FIRST, VAL
  probe.** `arch: lowrank`, rank ∈ {1, 2}, W = clip(U@V + R, ±1), V
  zero-init so epoch 0 IS the plain net. The post-hoc version (lr_svd)
  priced +326 B; measured on the smoke net, the same day, same net:
  post-hoc SVD left **89.4 %** of the residual nonzero, the trained
  factorization **9.2 %** (a further 1762 entries dropped because the
  clip makes them invisible) — 5492 B vs 4220 B artifacts. **Expected
  payload composition, pre-codec, rank 1: U 768 trits = 188 chars + V 1
  char + residual at 5 % of 3072 = 154 nz × (log2 R + 1) ≈ 166 chars →
  ~355 chars + 5 header**; rank 2 adds another 188 chars of U (U costs
  ~152 B raw PER RANK UNIT against a 3072-trit dense table, so rank is
  the expensive knob and rank 1 is the honest first arm). Decoder tax
  measured at **136 B** — twice the codebook's. Bar: bytes at equal val
  or val at equal bytes, through the measured table.

**Next end-to-end candidates** (coordinator, Thomas's standing principle
"we should train everything end-to-end" — NOT built, for the training
lane to schedule after the c1024 family; both PRICE-FIRST and
VAL-probe-first):

- **trainable flat-material BASE.** The replacement design's base is a
  fixed constant vector; make it jointly trainable with the net on the
  certified grid and export it through the same codec. Cheapest possible
  end-to-end win if it moves val at all, since the values already exist
  in the artifact.
- **trainable K_MID / K_END seam tables.** Currently hand-kept classic
  tables. Same treatment: in the graph, on the grid, through the codec.

## Log (newest first)

- 2026-08-15 ~23:0x UTC prev-day: **c1024_phase_ml2 val 0.01286 — phase
  knowledge measured IN WEIGHTS** (king-activity flip +16 from +0,
  pawn-advance +4..+8; MEASUREMENTS has the full probe rows). Critical
  path → ml2 engine machinery price (golf lane seam). 80_ml2 requeued
  (deployment path), queue fed by extension lane (90_cb, 91_lr).

- 2026-08-15 ~02:5x UTC: 13h IDLE VIOLATION (session-bound watcher died
  with the session) — liveness moved box-side: queue_runner.py detached,
  own lock/tripwire; monitors are now advisory only. c1024-cal verdict:
  density buys nothing (0.01421 / 0.01388 vs 0.01378) — structure
  binding. c1024-phase form (c) phase-through-ml2 RUNNING under the
  runner (certificate green). Screen arms rebuilt at HEAD (entry
  5d7d0d1 3308 B / candidate 3536 B), full quick-ladder PASS; ×5
  conversion stands (0/5 vs 1/5) — coordinator holds the launch call.

- 2026-08-14 ~12:5x UTC: chain2 done. **8Mv: val 0.01378** (comparable
  split) — beats v1 0.01385: data scale pays; candidate packs 3536 B
  (whole-feature sparsity). **kb8fold verdict: NEGATIVE as shipped** —
  training-form 0.01267 (buckets carry real signal) vs shipped folded
  0.01428 (the fold loses it; worse than plain v1). Consequence for
  c1024-kb4: shared-rows-only folding is dead; only per-bucket DELTAS
  (bytes) or ml2/phase products can carry king knowledge. Gate-stability
  ×5 (MEASUREMENTS): entry 0/5, v1 5/5, 8Mv 1/5 — v1 withdrawn from the
  staged screen, 8Mv staged NOT-READY, kqk-mid is the phase arm's
  scoreboard. Started c1024-cal (τ 0.6, l1 {0, 0.0003}, 8M+valn).

- 2026-08-14 ~12:2x UTC: TRAINER IDLE INCIDENT, ~70 min — the chain's
  wait loop (`pgrep -f "replnet_8M.pickle"`) was self-matched by my own
  severed ssh launcher wrapper, whose cmdline embeds the script text.
  Killed the wrapper; 8Mv started immediately (comparable-val confirmed:
  anchors identical to the 4M split). Root-cause rule for future chains:
  wait on `pgrep -f "train_packed.py.*<out>"` (the interpreter line),
  never on a bare string that launcher wrappers also carry.

- 2026-08-14 ~11:3x UTC: Thomas objective + phase directive via
  coordinator — header rewritten around THE FAMILY OBJECTIVE, c1024-phase
  added (#2, three forms, price-first), replnet_ml2 given its explicit
  entry (#4), c1024-general terminal arm added (#12), subsumption rule
  recorded (pend's +37 B carries the first ablation obligation).
  probes.py landed and wired into export.py: every export now ledgers
  per-class knowledge scores (.probes.json). 1-epoch wiring smoke showed
  exactly the expected signature: mobility arriving (+16 knight
  centralization), phase absent (+1/+0), passers still wrong-signed (−5).

- 2026-08-14 ~11:1x UTC: Thomas directive via coordinator — payload
  budget 1024 B. CAPACITY-1024 family added at the top (calibration arm
  first), speculative arms demoted behind it. Current chain (8M → 8Mv →
  kb8fold) runs to its natural boundary, then c1024-cal starts.

- 2026-08-14 ~11:0x UTC: caught an instrument slip in the 8M launch — no
  --valn, so its val split is not the 4M runs' and its numbers are
  unreadable against them. Chained replnet_8Mv (--valn 4027406 = the 4M
  val ids by construction) then replnet_kb8fold (ternshared landed at
  567e4ef) to start the moment the current run exits. The no-valn 8M run
  completes and is recorded, but only 8Mv's val counts for the queue.

- 2026-08-14 ~10:46 UTC: v1c finished — val 0.01389 @65.2% zeros (does
  not beat v1's 0.01385 in val, but sits mid-band on bytes; both are
  screen-eligible, v1 stays the staged candidate). Started replnet_8M
  (queue swap clause: kb8fold needs its fold-at-export step written
  first).

- 2026-08-14 ~10:00 UTC: v1 arms finished (l1=0.001 → 0.01385 @59.6%;
  l1=0.002 → 0.01404 @73.7%); winner l1=0.001 by the pinned rule;
  started replnet_v1c.
