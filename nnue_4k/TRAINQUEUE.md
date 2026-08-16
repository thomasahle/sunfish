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

**EXPORTED-FIDELITY RULE (standing, 2026-08-15, coordinator-authorized):**
a net trains under the resolution its ARTIFACT has, from step 1 — every
tensor the payload rounds is rounded inside forward by STE, never rounded
for the first time at export. The ternary weights always had this; `u2`
and the gain/bias digits did not, and the cost was measured: the ml2
family's entire val win over the linear net was 0.00098, and forcing `u2`
onto its certified grid gave back 0.00082 of it (run 60). Mechanism is
`model.gridste` (gains + biases) with `model.u2grid` (layer-2 read-out);
both default off, so historical configs reproduce bit-for-bit. Any new
arm in this family carries them unless it states why not.

## Context

2026-08-14: payload target 1024 B (Thomas; golf lane opening the code
side, exact capacity confirmation pending). Budget-617-era numbers: v1
winner l1=0.001 val 0.01385 @59.6% zeros; v1c 0.01389 @65.2%. At 1024
the 3,072-trit ps768 payload fits at any sparsity — sparsity is a
capacity dial now. Encoder: compress/ bake-off winner becomes export
default when its table lands; arms re-size to the weight capacity it
buys. Probe suite (train/probes.py) runs at every export; scores are
ledgered per net (.probes.json), diagnostics never gates.

## SELECTOR SPEC (standing, 2026-08-15, coordinator-authorized)

**The problem it solves.** `val` ranks arms **inside** a family and
**inverts across** family boundaries — measured three times, ledgered
under the anti-predictive finding. Eight fitted evals died in play behind
that inversion. Selection therefore has to be **play-anchored**, and the
question is only how few games that costs. Measured retrospectively on the
three surviving screens' PGNs (MEASUREMENTS.md 2026-08-15, "TRUNCATION
VERDICT"), no new games spent.

### The rule

> **Candidate selection = a fixed-node mini-match, N\* = 50 games, against
> the pinned current entry, fresh srand per candidate, ranked by score%.**

| field | value |
|---|---|
| N\* | **50 games** = 25 colour-swapped pairs. Always whole pairs — a mini-match that stops mid-pair is biased |
| opponent | the **pinned current entry**, one common base for every candidate in a cohort; never candidate-vs-candidate, never a moving base |
| limit | fixed nodes 20000 (the ledger's standard screen budget), **sources under pypy3** — packed artifacts ignore `go nodes` |
| srand | **fresh per candidate**, recorded |
| book | `book3k.pgn`, order=random |
| statistic | **score%** (equivalently pentanomial Elo). The pentanomial interval is reported but is not the decision |
| gate | zero illegal moves. One ends the mini-match and voids it — same tolerance as a screen |
| cost | ~10 min at concurrency 8 (from the ml2 screen's own rate, 195 games in 41 min) |

### What it returns, and what it does NOT return

**It returns a TOP PICK. It does not return a ranking, and it never
returns an Elo for the ledger.** The measured calibration (bootstrap,
4000 resamples, on true gaps of 127 and 66 Elo):

| decision at N=50 | accuracy |
|---|---|
| **pick the best arm** | **96.8%** |
| resolve a ~127 Elo cross-family gap | 96.0% |
| **complete 3-way ordering** | **72.4%** |
| resolve a ~66 Elo adjacent gap | 76.2% |

So: promote the winner, and treat everything below the winner as
**unordered**. Reading positions 2 and 3 off a 50-game mini-match is
reading noise about a quarter of the time.

### Promotion to a full screen

A candidate is promoted to a registered screen when it **wins its cohort's
mini-match**, i.e. tops the score% table against the pinned base. If two
candidates finish within **10 percentage points** of each other (the
measured resolution floor: 50 games resolve ~101 Elo at z = 1.645, so
anything closer is a tie), the mini-match has **not** separated them —
either extend that pair to **N\*\* = 150 games** (where a ~127 Elo gap is
99.9% and a ~66 Elo gap 90.8%) or promote both and let the screens decide.
Never break a mini-match tie by val: that is the inversion this whole spec
exists to route around.

### Honesty caveats, attached permanently

- **The selector RANKS; only a registered screen produces a verdict.** No
  mini-match number is ever quoted as Elo in MEASUREMENTS.md, and a
  mini-match win is not evidence a candidate clears any bar. Screens
  decide landing; this decides queue order.
- **The source data is SPRT-truncated.** All three screens stopped early
  and their full-N numbers are biased away from zero. The study used them
  for *ordering*, which truncation does not distort, never for altitude.
- **The three screens are not mutually calibrated.** Bases 3308 B vs
  3405 B, srands 20260814/20260814/20260815, `openings_2k.epd` vs
  `book3k.pgn`, and adjudication **active on 8mv (229/320) but inert on
  both ml2 screens**. N\* is fitted under that heterogeneity, which is why
  the spec pins ONE base and a fresh srand — the configuration the number
  was fitted for is the configuration it is valid in.
- **N\* = 50 sits exactly on the crossing point**, with zero margin: one
  pair earlier the top two tie exactly, and below that the ranking
  inverts. 50 is where it first becomes right and stays right on *these*
  trajectories — the bootstrap's 5th–95th percentile for that crossing
  spans 10 to 158 games, and 7.6% of resamples never stabilised at all.
  This is why the rule is "top pick at 96.8%", not "ranking at N\*".
- **The bootstrap covers outcome noise only** — not srand, base, book, or
  a genuinely different candidate's true strength. It is a lower bound on
  the uncertainty.
- **Gaps under ~60 Elo are out of reach** at any affordable mini-match
  size: the two ml2 arms (66 Elo apart) never separated even with all 370
  games their two screens played.

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

## PHASE A FEASIBILITY — Leela data: located, licensed, and SMALLER than asked for (2026-08-16)

**Nothing downloaded yet.** This is the scoped check, reported first as ordered.

### Where the data actually is

| route | host | format | license found |
|---|---|---|---|
| **A — official Lc0 runs** | `storage.lczero.org/files/training_data/` — `run3`, `test30/40/60/71/75/78/79/80/90/91`, tars ~15 MB–2.1 GB | Lc0 **V6 records** (fixed-size, includes full policy planes) | **ODbL 1.0 + DbCL 1.0**, stated in `LICENSE.txt` at that path |
| **B — SF-community converted packs** | HuggingFace `linrock/test78`, `test79`, `test80-2022/2023/2024`; Leela93/95/96/99 on Kaggle | **SF `binpack`** (`.min-v2.v6.binpack.zst`) | **no license tag on the HF datasets** — recorded as unstated |

Concrete sizes, `linrock/test80-2024`: 19 files, **~353 GB** total; monthly
binpacks **6.9–12.3 GB compressed**, the raw `.tar.zst` 15–34 GB.

### The number that reframes the ask

**Route A is infeasible at any useful position count and Route B is far
cheaper than the tasking assumed.** V6 records carry a 1858-wide policy
vector, so they run ~8 KB/position — 10 M positions would be ~80 GB. Binpack
runs closer to **~7 bytes/position**, so the same 10 M is **tens of MB of
stream**, not a few GB.

And then the real point: **our student has ~3,072 trainable trits.** A 10–50 M
position slice is 3–16 thousand positions per parameter. **Data volume is not
the binding constraint at this scale — label quality and distribution are.**
So the slice should be sized by *coverage*, not by matching reference-recipe
volume: **1–5 M positions (~10–40 MB of binpack stream) is already generous**,
and it makes the download a non-event. Recommend we take the smallest
convenient slice and spend the saved effort on the decoder and the
distribution question.

### The real cost: the decoder

`binpack` is a chained format, so a clean-room implementation from the format
documentation is genuine work — and it is the only lawful route, since the SF
tooling that reads it is GPL and **we transplant no code**. Two mitigations
worth pricing before committing: (i) a *prefix* of a `.zst` binpack should
decode up to the last complete chunk, which is all we need for a few million
positions; (ii) `linrock/test78`/`test79` are older and smaller than the 2024
packs. Route A's V6 is the *easier* decode (fixed-size records) and is the
**properly licensed** one — it loses only on size, which is why it stays the
fallback rather than the plan.

### Updated Phase A matrix — 2-D, small nets, selector-judged

| | **label: teacher value→cp** | **label: win-prob + WDL blend (λ)** | **label: twin own-search** |
|---|---|---|---|
| **positions: Leela slice** | ✔ favored teacher arm | ✔ | — (teacher positions, own labels is incoherent) |
| **positions: our 93k-game archive** | — (no teacher labels for these) | — | ✔ cheap on-distribution arm (~22× twin, nearly free) |
| **positions: blend** | ✔ | ✔ | ✔ |

- **λ semantics are a live trap**, flagged from the study: bmdanielsson's
  `--wdl` has **1.0 = pure game outcome**, while the nnue-pytorch lineage's
  `lambda_` is conventionally the *opposite* orientation. Whichever we
  implement, the direction gets asserted by a unit test before any run.
- Scaling constants from the study: win-prob via `600/361`, label via
  `/410`; our house `sigK` is 400, so the conversion is a small, checkable
  change rather than a new mechanism.
- **The on-distribution risk is the scientific point, not a formality.**
  Leela positions come from superhuman self-play; our 3376-byte student meets
  classic-level opposition at shallow depth. That mismatch is exactly what the
  archive arm exists to measure, and it is why Leela-data is the *favored*
  teacher arm rather than an axiom.
- Everything already approved stands: fenkey split, `perm_seed` pinned,
  **two val draws from day one**, Zobrist/FEN dedup, quiet filter per the
  study, and **label depth decided by calibration on the SELECTOR** — because
  this campaign has already measured that val does not predict play across
  families.

**Not built, not downloaded — awaiting the mix/label confirm.**

## TRAINER STUDY — bmdanielsson/nnue-trainer (NNUE-V2 Phase A, 2026-08-16)

**LICENSE FIRST, and it is the strict case.** The repo has **no LICENSE file**
and GitHub's API reports **`license: null`** — i.e. no grant at all, so it is
"all rights reserved" by default, which is *more* restrictive than the GPLv3
lineage it descends from (its README says "Training code is based on
glinscott/nnue-pytorch"). **IDEAS ONLY, independent reimplementation, zero
transplanted code** — our standing rule for rival engines, and here it is also
the only lawful reading. Nothing below is copied; these are recipe facts read
off a public repo (main branch, last pushed 2024-05-25).

### The recipe, as read

**1 — Data: they GENERATE, they do not harvest.**

| knob | their value |
|---|---|
| source | self-play, two engine instances over UCI |
| per-move budget | `--depth` **or** `--nodes` (mutually exclusive), `MAX_TIME = 30` s cap |
| opening diversity | `RANDOM_PLIES = [16]` random plies before engine play (a `2moves_v2.epd` book also ships) |
| target scale | `--npositions` default **100,000,000** |
| **quiet filter** | keep only if **not promotion, not capture, not en-passant, not in check, and the move does not give check** |
| decisiveness cutoff | `EVAL_LIMIT = 10000` adjudicates the game out |
| game caps | `MAX_PLY = 400`; draw adjudication `DRAW_SCORE 10` / `DRAW_COUNT 10` after `MIN_DRAW_PLY 80` |
| **dedup** | Zobrist-hash set, duplicates skipped |

**2 — Labels: interpolated eval↔outcome, in win-probability space.**
`wdl_value_target = wdl_eval_target*(1−wdl) + outcome*wdl`, with **`--wdl`
default `1.0`** — and their own help text says `0.0` trains on evaluations
while `1.0` trains on **game results**. Scaling: model output → win prob via
`pred*600.0/361`, label via `score/410`. A separate
`rescore_training_data.py` **re-labels existing positions with a fresh search
at `--depth` default 8**, touching the eval field only and leaving outcomes
alone.

**3 — Loss + schedule.** Squared error **in win-prob space**
(`|target − pred|²`, sigmoids on both sides — not a centipawn loss).
**RAdam** lr 1e-3, betas (.95, .999), eps 1e-5, weight_decay 0; **batch
16384**; **`ReduceLROnPlateau`** factor 0.3, patience 1, min_lr 1e-6; **no
fixed epoch count — `while True`**, the plateau schedule ends the run.

**4 — Quantization: post-hoc, and SILENT.** `model.py` contains **no weight
clipping and no quantization constants**; `quantize.py` runs **after**
training on a saved checkpoint (`NNUE2SCORE 600`, `MAX_QUANTIZED_ACTIVATION
127`, `WEIGHT_SCALE_BITS 6`, `OUTPUT_SCALE 16`; input layer → int16, output
weights → int8). Output weights are **clamped to ≈±1.68 with no warning and
no assert** when they exceed the representable range.

**5 — Architecture.** `NUM_INPUTS = 64*12 = 768` (plain piece-square, not
HalfKP), **L1 = 1024** per perspective, output layer over `2*L1 = 2048`,
activation `clamp(0,1)` (clipped ReLU), one output neuron.

### What this changes for V2 — follow, or state the departure

**FOLLOW (evidence-backed, and it answers the mix question):**

- **Generate, don't harvest.** Their entire dataset is self-play; our 92,912
  archived games (~10.2 M plies pre-filter, and far fewer after quiet-filter
  and dedup) are two-plus orders of magnitude short of their 1e8 target and
  are correlated with the very engines we test. **Proposed mix: generated
  self-play as the training set; the match archive held out as a
  distribution check**, not as training data — it is the right set for asking
  "is the net off-distribution for the engines we actually play?".
- **Their exact quiet filter and Zobrist dedup**, both of which our
  `data.py`/config already have analogues for (`quiet`, `cpmax`, fenkey).
- **Depth-8 re-scoring** as the prior for our approved label-depth
  calibration — it brackets the range to sweep rather than guessing it.
- **Plateau LR scheduling instead of a fixed epoch count.** Our own
  `dense60` finding (60 epochs buys a noise-level minimum) is what a fixed
  epoch budget looks like without a plateau rule.

**DEPART, with the reason:**

- **Width.** Their accumulator alone is 768×1024 int16 ≈ 3 MB. Our entire
  artifact is 4096 B and our payload ~800 B — a ~4000× gap. Every
  architectural choice they make scales down catastrophically; ours stays
  ps768 × N=4. This is the departure that defines the project.
- **Quantization — and here we are genuinely ahead.** They clip silently
  post-hoc with no assert. We hit exactly that failure and diagnosed it: the
  float-ml2 net trained a `u2` its container could not express and the export
  rounded it to **[0,0,0,0]**. Our `gridste`/`u2grid` snap **inside forward**
  by STE so the loss sees the real resolution, and `export.py` **refuses** an
  all-zero read-out and prints the price sheet. Keep ours; do not adopt
  theirs. (Fair note: the upstream `nnue-pytorch` lineage *does* clip weights
  to quantization ranges — this derivative appears to have dropped it. The
  claim is about this trainer, not about NNUE practice generally.)

**THE ONE THAT MAY EXPLAIN OUR CAMPAIGN — ingredient 2 revised.** We train
**purely on centipawn eval agreement**; we have no WDL term at all. Their
default is the opposite extreme: **`wdl = 1.0`, pure game outcome.** Our
headline finding this week is that **val (eval-agreement) is anti-predictive
of play across family boundaries** — linear val 0.01378 played −107, better-val
float-ml2 0.01280 played −234. A loss that optimises agreement with a cp
oracle is not optimising the thing we score, and this trainer's default says
the reference practice trains toward **results**. **Proposal: V2 carries a
`wdl` interpolation term as a first-class dial, swept early** — it is cheap,
it is the sharpest available lever on the anti-predictive problem, and it
costs zero bytes.

## Log (newest first)

- 2026-08-16 02:10 UTC: **THE FORFEIT TRIPWIRE FIRED, CORRECTLY — and the
  pause it caused cost 7h19m of training, which is the next structural gap.**

  `queue_runner` SIGSTOPped `99_tail016` when forfeits rose **34 → 38**.
  Attribution, done before anything was unpaused:

  | source | forfeits | verdict |
  |---|---|---|
  | `nnue-match/*.pgn` (10 files, mtimes 08-07…08-09) | **exactly 34** | the baseline itself — static for a week, entirely historical |
  | **`guide-lmr-20260815/match.pgn`** | **the +4** | `[White "guide-lmr"] [Black "master"]` — an **owner-side CLASSIC search experiment**, same family as `elo-frontier-lmr` / `elo-intrinsic-lmr` / `elo-capped-lmr` and PR #202/#205's intrinsic LMR. **Out of scope.** |
  | `replnet-20260814`, `tmpool-20260814`, `tmsmooth-20260814`, `tmfix60-20260814`, `meter-20260815` | **0, all five** | none of our dirs forfeited anything |
  | **`entry-consts-20260816`** | **0** across 11 PGNs / 1119 games | the flagged anomaly — a fixed-node sweep forfeiting would mean it ran timed or the driver fell through. It did not, and it **is** inside the runner's globs, so it was genuinely watched |

  That the static files sum to **exactly** the baseline is the check that
  makes the attribution airtight: the 34 was never live traffic, so the
  entire delta is one identified out-of-scope match.

  **The tripwire proved its worth.** It is the instrument that would catch a
  real regression — our training starving a live match — and it fired on the
  first genuine rise it ever saw, paused rather than guessed, and named the
  pid and the numbers in a report. Pause-and-report is the correct design and
  it behaved exactly as written.

  **Re-baseline arithmetic — and a correction to the instruction.** The
  tasking said re-arm at 38. That would have tripped again within seconds:
  `guide-lmr` kept forfeiting for ~6 minutes after the trip (4 → **32**), so
  the live count is **66 = 34 static + 32 guide-lmr**. The runner's own
  operator path handles this — deleting `PAUSED_REPORT.txt` SIGCONTs *and*
  re-baselines to the current count — so the correct action was to delete the
  report and let it compute 66 itself. Confirmed:
  `[queue] resumed 99_tail016.yaml (new baseline 66)`, process state `RNl`.

  **The cost, stated plainly: the run sat SIGSTOPped for 7 h 19 m**
  (18:48:48Z → 02:07Z). The tail was built so an empty queue could not idle
  the trainer without a human; it does that. But a **paused** run idles the
  trainer just as completely, and pause-and-report needs a reader — so the
  always-training rule broke again through a door the tail does not cover.
  Three earlier gaps today were 38, 31 and 50 minutes; this one was longer
  than all three combined. **Naming it, not fixing it unilaterally:** the
  obvious candidates (auto-resume after attribution, or narrowing the globs
  to exclude owner-side dirs) both weaken a safety instrument that just
  demonstrated it works, and that trade is a coordinator call, not mine.


- 2026-08-15 16:15 UTC: **PROGRAM-WIDE COMPLETE-PENDING-DIRECTION — the
  training program is formally handed to the owner's decision.**

  ### Instrument verdict — the draw × seed 2×2 (one-layer, 30-epoch budget)

  | | n | mean | sd | range |
  |---|---|---|---|---|
  | **draw A** (`perm_seed 0`) | 10 | **0.013747** | 3.43e-5 | [0.01368, 0.01378] |
  | **draw B** (`perm_seed 1`) | 3 | **0.013900** | 3.46e-5 | [0.01388, 0.01394] |

  **The draw effect IS separable from seed luck, decisively.** Offset
  **1.53e-4**, t = 6.7, and — the robust statement that needs no
  distributional assumption — **the two sets do not overlap at all**: draw
  A's worst point (0.01378) is better than draw B's best (0.01388). Three
  independent draw-B seeds, ten draw-A seeds, complete separation.

  **The consequence is the sharpest instrument finding of the campaign:**
  the draw offset is **4.5σ of the seed sd**, which makes *which 200 000
  rows you validate on* a **larger effect than any dial this family ever
  measured**. Val numbers are comparable only *within* a draw, and the whole
  campaign ran on one draw — so its ladder was measuring a draw-specific
  quantity all along. That is a property of the instrument, not of any net.

  ### The tail's own question, answered by the tail

  Seven firings, `tail001`–`tail007`, each an ordinary logged-and-archived
  entry with the seed rotated (4, 5, 6, …), exactly as designed. Six
  completed points extend the draw-A census from n=4 to n=10:

  | seed | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|
  | val | 0.01377 | 0.01377 | 0.01373 | 0.01378 | 0.01376 | 0.01376 |
  | packed | 3569 B | 3571 B | 3577 B | 3564 B | 3574 B | 3577 B |

  **Long-run seed variance does NOT accumulate or drift — it TIGHTENS:**
  sd **4.43e-5 at n=4 → 3.43e-5 at n=10**. The process is stationary and
  n=4 was mildly over-estimating it. The one-layer noise floor is now a
  solid **3.4e-5 in val**, and **20 B (sd) over a 54 B range** in bytes.
  (`tail007`, seed 10, was still mid-run at this reading and is excluded.)

  With the firmer floor, the retroactive readings settle:

  | claim | gap | verdict at n=10 |
  |---|---|---|
  | "data scale pays": 8Mv vs v1 | 7e-5 | **2.0σ — still not resolved**, and n=1 per side |
  | c1024-cal 0.01388 vs 0.01378 | 1.0e-4 | 2.9σ — credible but single-seed |
  | τ 1.1 vs 0.85 | <1e-5 | 0.3σ — tie, confirmed for the third time |

  ### PROGRAM STATE: COMPLETE-PENDING-DIRECTION (both families)

  Every dial either family can turn without a seam is measured — **l1 ×
  clamp × τ × satpen × ride × seed × draw × fidelity-flags** — across ml2
  and the one-layer replacement net. **Campaign-wide, exactly three effects
  exceed their own noise floor:**

  1. **The exported-fidelity rule** — float ml2 0.01280 → grid ml2 0.01344,
     **6.4e-4 ≈ 5σ**. Real, and it *costs* val: honest training is more
     expensive, not cheaper.
  2. **satpen off breaks the run** — not a val delta but a divergence,
     caught by the trainer's own anchor tripwire at epoch 21.
  3. **The play numbers** — linear **−107**, float-ml2 **−234.18 ± 55.08**,
     grid-ml2 **−300.56 ± 71.33**, pentanomial, gaps above 100 Elo.

  Plus one instrument fact that is not about any net: **the val draw is
  worth 1.53e-4**, more than any dial.

  **Everything else this campaign ranked on sits inside its own noise.**
  Selection therefore rests entirely on the **SELECTOR SPEC**'s
  play-anchored 50-game mini-match — not as an improvement over val, but as
  the only instrument either family has left.

  **What remains needs the owner, not a run.** The two registered arms that
  could still move capacity (#5 n8, #3 kb4) both require an **entry-decoder
  seam** this lane declined on direction grounds — priced at 4199 B and
  ~95%-sparsity-dependent respectively — and #12 is owner-gated. The
  standing question put to Thomas is unchanged and now fully evidenced:
  **should the replacement family spend anything further on capacity before
  it has one net that plays?** Every capacity increase measured so far has
  played worse, and the export-faithful net — the one with every technical
  excuse removed — was the worst of the three.

  **Queue state: tail-backed indefinitely.** `tail007` is running and
  `tail.yaml` persists; the queue cannot go idle again without a human
  choosing to remove it. **Nothing else is queued, deliberately** — there is
  no remaining question either family can answer by itself, and queueing
  work to look busy is what the tail exists to make unnecessary. Each future
  firing buys another census point for the only family that ever beat a coin
  flip in play.

- 2026-08-15 13:20 UTC: **THE ONE-LAYER INSTRUMENT STORY CLOSES, and the
  always-training rule stops depending on me remembering.**

  ### Instrument verdict — one-layer seed spread (n=4, draw A, 30 ep)

  | seed | val | packed |
  |---|---|---|
  | 0 (`71`) | 0.01378 | 3570 B |
  | 1 (`78`) | 0.01374 | 3575 B |
  | 2 (`84`) | 0.01368 | 3618 B |
  | 3 (`85`) | 0.01370 | 3618 B |
  | | mean **0.013725**, sd **4.4e-5**, range **1.0e-4** | range **48 B** |

  **The two-point estimate HELD.** 78's 4e-5 became 4.4e-5 at n=4 — unlike
  ml2, whose 1.8e-4 two-point became a 1.25e-4 sd over a 3.0e-4 range. **The
  one-layer family is 2.8× quieter on val and its byte spread is 48 B
  against ml2's 70 B.** That asymmetry is itself a finding: the family with
  the second layer is the noisier one on every axis measured.

  ### Instrument verdict — one-layer val draw

  Matched 30-epoch budget, seed 0: **draw A 0.01378 → draw B 0.01388 =
  1.0e-4** (`83`'s best-of-first-30; its first 30 epochs are identical to a
  30-epoch run's). Against ml2's 3.6e-4, again **3.6× quieter**. Same shape
  of answer as ml2, smaller magnitude: **the draw moves this family's val by
  about one seed-range.**

  **Retroactively, for the family that matters most.** Taking seed sd and
  draw shift together, a one-layer val gap needs roughly **1.5e-4** before
  it means anything. The campaign's one-layer claims:

  | claim | gap | verdict |
  |---|---|---|
  | "data scale pays": 8Mv 0.01378 vs v1 0.01385 | 7e-5 | **1.6σ — NOT resolved** |
  | c1024-cal 0.01388 vs 0.01378 | 1.0e-4 | 2.3σ — marginal |
  | τ 1.1 vs 0.85 (`71`/`72`) | <1e-5 | tie, confirmed |

  Even the quiet family cannot support "data scale pays" as stated. And
  `83`'s own 60-epoch ride bought 0.01388 → 0.01381 = 7e-5, inside the seed
  sd — the `dense60` lesson holding for a third family.

  **Program-level state: COMPLETE-PENDING-DIRECTION, now for BOTH families.**
  ml2 carried this already; the one-layer family now joins it. Every dial
  either family can turn without a seam has been measured — l1 × clamp × τ ×
  satpen × ride × seed × draw × fidelity — and **only three effects in the
  whole campaign exceed their own noise floor**: the fidelity rule (5σ, and
  it costs val), satpen-off (which breaks the run), and the play numbers.
  Selection therefore rests entirely on the **SELECTOR SPEC**'s
  play-anchored mini-match. The next move is a direction call, not a run.

  ### The structural fix: the queue no longer depends on a human

  Three gaps today — 38 min, 31 min, and **50 min (12:22→13:12Z)** — all the
  same cause: refill was manual and I was the manual part. `queue_runner`
  now fires a **TAIL** config when the queue is genuinely empty, instead of
  nagging for ten minutes.

  - `spawn_tail()` copies `train/tail.yaml` into the queue as an ordinary
    entry with **the seed rotated** by the number of tail runs already
    archived — so firings accumulate a census (seed 4, 5, 6 …) instead of
    recomputing one run, and it is logged, run and archived like anything
    else. `tail.yaml` is never consumed. A tail with no rotatable `seed: N`
    is **REFUSED**, not run unrotated.
  - Tested on a **dummy** queue dir before deployment, never the live one:
    rotation across firings, `perm_seed`/`split_seed` correctly untouched,
    refusal path, and a full end-to-end cycle —
    `EMPTY -- firing the TAIL tail.yaml -> 99_tail001.yaml` → started →
    logged to LOG.md → archived to `done/`, with `tail.yaml` surviving.
  - **The tail's own named question** (it is maintenance, not busywork):
    continue the one-layer seed census — does sd hold at 4.4e-5 as n grows,
    in val *and* in bytes? Every idle window now buys a census point for the
    only family that ever beat a coin flip in play.

  **Restart, done by the book.** The runner was **idle — 0 `train.py`
  children, verified before the signal** — and was this workstream's own
  process (`start_runner.sh`, `setsid nohup`, PPID 1). Per
  `box-systemd-queue-incident` I checked `systemctl --user` first: no unit
  owns it (the only sunfish unit present is an unrelated **failed**
  `pawn-debt-handoff` service). Old `runner.log` preserved to
  `runner.log.20260815T1310Z`; the stale `.runlock` (SIGTERM skips the
  `finally`) removed deliberately — the code says a human removes it and
  this restart was the authorization. **Downtime 13:11→13:12Z ≈ 70 s.** New
  pid holds the lock from 13:12:12 and picked up `86_plain_drawB_s1`
  immediately.

  **Cotenancy recorded at launch, per `box-cotenancy-ban`'s amendment.**
  Thomas's own joint-eval tuning campaign (`adaptive_gp.py`, PR #202, 3+0.1,
  10 slots) started 12:59Z and is live at ~1750% CPU. Box is **96 cores,
  load 35** — ample headroom, and Thomas's 2026-08-14 amendment authorizes
  capacity sharing when no other human needs the box. The runner's **forfeit
  tripwire** is armed against exactly this (baseline 34): any rise in "loses
  on time" SIGSTOPs our training and reports rather than guessing.

  **Queue:** `86_plain_drawB_s1` (running) and `87_plain_drawB_s2` complete
  the draw × seed 2×2 — with draw B at n=1, one point cannot separate a draw
  effect from an unlucky seed, and this is the last instrument question the
  family can answer by itself. The tail backs them.

- 2026-08-15 11:15 UTC: **BOTH INSTRUMENT VERDICTS, and the family's dial
  space is SPANNED — COMPLETE-PENDING-DIRECTION.**

  ### Verdict A — the ml2 seed spread (n=4, l1 .001, 30 ep, draw A)

  | seed | val | zeros | **packed** |
  |---|---|---|---|
  | 0 (`74`) | 0.01347 | 72.9% | 3740 B |
  | 1 (`82x`) | 0.01359 | 73.2% | **3697 B** |
  | 2 (`79`) | 0.01329 | 70.0% | 3766 B |
  | 4 (`82y`) | 0.01341 | 72.5% | **3767 B** |
  | | mean **0.01344**, sd **1.25e-4**, range **3.0e-4** | | mean 3742, **range 70 B** |

  **The two-point estimate held and got worse.** Every ml2 val distinction
  this campaign drew — density .0005 vs .001 (6e-5), bm4 vs bm2 (5e-5),
  grid .0005 vs .0003 (7e-5), u2grid-only vs full grid (1.5e-4) — is
  **inside one standard deviation.** The ml2 val ladder cannot resolve any
  of them.

  **And a correction to my own entry of two hours ago.** I wrote that the
  byte results were unaffected because "`pack.sh` is deterministic, no seed
  involved". That conflated *the measurement* being deterministic with *the
  quantity* being stable. The net depends on the seed, so the byte outcome
  of a **recipe** does too — and it moves **70 B across seeds at fixed
  recipe**, which is **twice** the 35 B I ledgered as l1 .001's "free
  bytes". **That claim does not survive either**: 3775 → 3740 was n=1 per
  arm inside a 70 B spread. l1 .001 is not established as cheaper; it is
  established as *not worse*, on val and bytes both.

  ### Verdict B — the val draw

  Same seed, same recipe, different 200 000-row draw: **`74` 0.01347
  (draw A) vs `82_valdraw_ml2` 0.01383 (draw B) = 3.6e-4** — *larger than
  the entire seed range*. So the ladder's differences were **draw noise as
  well as seed noise**. This is an instrument finding, not a finding about
  any net: a val gap under ~4e-4 on this family means nothing unless it is
  replicated across seeds *and* draws, and nothing in this campaign was.

  ### What SURVIVES both verdicts, stated so this is not over-read

  - **The fidelity effect is real**: float ml2 0.01280 → grid ml2 mean
    0.01344 = **6.4e-4 ≈ 5σ** of the measured seed sd. Training under the
    grid genuinely costs val; that was never noise. (Caveat: the float
    number is n=1 and its own noise is unmeasured.)
  - **The ⅓ recovery, restated properly rather than retracted.** Against
    the one-layer mean 0.01376, recovery is
    (0.01376−0.01344)/(0.01376−0.01280) = **33%, at ≈2.6σ**. Two hours ago
    I called it "1.7× a single-seed observation, not resolved"; with n=4 it
    is *better* than that — resolved at about 2.6σ. The correction runs in
    both directions and this is the direction it ran.
  - **Every play number**: −107, −234.18 ± 55.08, −300.56 ± 71.33 are
    pentanomial with >100 Elo gaps. **The anti-predictive finding is
    untouched and is now the only load-bearing selection evidence the
    family has.**
  - **The attractor**, which lives in the weights and never touched a val
    set.

  ### Arm 11's satpen half — CLOSED, and not by a val comparison

  `81_satpen_off` did not finish: the trainer's own anchor tripwire killed
  it at epoch 21.

  > `EARLY-KILL: val 0.01894 vs anchors zero 0.02001 / mat 0.01616 at epoch
  > 21 -- the run is broken, not merely weak. Stopping.`

  With `satpen: 0` the run **diverged** — clip-saturation 0.02% → **4.87%**
  in one epoch and val worse than the material-only anchor. Its 0.01329
  "best" is a pre-blowup epoch and is not a result; its payload packs to
  **3814 B**, the worst of any grid-era ml2 net (zeros fell to 60.7%).
  **satpen 0.03 stays**, the kbbil lesson holds, and **arm 11 is now closed
  on both halves** (CLAMP by run 61, satpen here). The instrument caught it
  and stopped — the anchors earned their keep.

  ### Attractor census — FINAL

  **14 finished ml2 runs, 56 components**, spanning seeds {0,1,2,3,4},
  l1 {.0003,.0005,.001}, clamp {400,600}, rides {30,60}, satpen {0,0.03}
  and **both val draws**. Max deviation over *all 56* components:
  **1.85e-2**. `80_seed3_l1_001_long` — seed 3 × 60 epochs, the last free
  combination — landed at **7.63e-08**, the tightest of the campaign, and
  vindicates refusing to read its epoch-3 intermediate (which sat at 0.12
  only because `u2` climbs from zero). The attractor is not a seed, ride,
  dial or draw artifact.

  ### State: COMPLETE-PENDING-DIRECTION

  Every dial the family header licenses without a seam is now measured:
  **l1 × clamp × τ × satpen × ride × seed × draw × fidelity-flags.** Only
  two effects exceed the noise floor: **the fidelity rule** (5σ, and it
  costs val) and **satpen off** (which breaks the run). Everything else
  this campaign ranked on is unresolved at its own measurement precision.
  The remaining registered arms (#5 n8, #3 kb4) need a seam this lane
  declined on direction grounds, and #12 is owner-gated. **The family has
  no next question it can answer by itself — the next move is a direction
  call.**

  Queued meanwhile (depth 3, real questions only): `83_valdraw_plain_long`
  (running — the one-layer draw answer) and the **one-layer seed census**
  `84_plain_seedcensus_s2` / `85_plain_seedcensus_s3`. That family is the
  only one that ever beat a coin flip in play, its noise floor rests on
  n=2, and the ml2 census just demonstrated what n=2 hides — in bytes as
  much as in val.

  (Context: Thomas retitled #202, "Jointly tune null shaping and intrinsic
  LMR" — his campaign is consolidating. No training effect; this lane cites
  nnue-4k revs throughout.)

- 2026-08-15 09:30 UTC: **THE ml2 VAL LADDER CANNOT RESOLVE WHAT THIS
  CAMPAIGN USED IT TO DECIDE: seed alone moves it 1.8e-4.**

  **`79_seed2_l1_001_pinned`** — the `perm_seed` fix's first ml2 use, and it
  worked (`val_sha 0239a7b84ec6ba2f`, on the ladder). Its named question was
  "is the l1 .001 free-byte result seed-stable?" The answer splits:

  | | `74` (seed 0) | `79` (seed 2) |
  |---|---|---|
  | best val | 0.01347 | **0.01329** |
  | zeros | 72.9% | 70.0% |
  | read-out | [1, 0, 1, 0] | [1, 0, 1, 0] |

  **Sparsity is stable (2.9 points); val is not — 1.8e-4 from
  initialisation alone.** That is **4.5× the one-layer family's 4e-5**
  (run 78) and confirms, as a measurement, the caveat that entry stated in
  advance: ml2 is noisier, not quieter.

  **What 1.8e-4 does to the ledger.** It is larger than essentially every
  ml2 val distinction this campaign has drawn:

  | distinction | gap | vs 1.8e-4 |
  |---|---|---|
  | float ml2 l1 .0005 (0.01280) vs .001 (0.01286) — "density pays" | 6e-5 | **0.3×** |
  | float ml2 bm4 (0.01286) vs bm2 (0.01291) | 5e-5 | **0.3×** |
  | grid l1 .0005 (0.01347) vs .0003 (0.01354) | 7e-5 | **0.4×** |
  | u2grid-only (0.01362) vs full grid (0.01347) | 1.5e-4 | **0.8×** |
  | **my own "recovers ⅓ of the win"**: linear 0.01378 vs grid ml2 0.01347 | 3.1e-4 | 1.7× |

  **Every one of those is at or under the noise, including my own headline
  from this morning.** The "⅓ recovery" figure is 1.7× a single-seed
  observation — the least unresolved of the set and still not resolved. I
  wrote it as a result; it is a direction at best, and this entry corrects
  it rather than leaving it to be quoted.

  **What is NOT affected, stated so the correction is not over-read.**
  (1) **The play numbers**: −107, −234.18 ± 55.08, −300.56 ± 71.33 carry
  pentanomial intervals and gaps above 100 Elo — the anti-predictive
  finding stands untouched, and this entry strengthens its motivation.
  (2) **The byte measurements**: 3740 / 3775 / 3794 B and the +1 B τ
  result are deterministic through `pack.sh`, no seed involved — l1 .001's
  **35 free bytes stand**, and sparsity's stability across seeds (72.9% vs
  70.0%) is now evidence for it. (3) **The attractor**: it is a property of
  trained weights, not of a val set.

  **Attractor census, recounted properly** (nine FINISHED ml2 runs, spanning
  seeds 0/1/2/3, l1 {.0003,.0005,.001}, clamp {400,600}, rides {30,60}):
  **36 components, 29 within 1e-4 of the tie, all 36 within 1.9e-2.**
  Tightest are `74` and `79` at max 1.4e-6 and 1.8e-6.

  **`80` is NOT in that census and is not being read**: it is mid-flight at
  epoch 3, and its intermediate checkpoint has `u2·scale ≈ 0.12` simply
  because `u2` starts at zero and climbs. Reading an in-progress checkpoint
  as "the attractor broke" is exactly the mistake this ledger keeps
  catching; it waits for the run to finish.

  **Queued to depth 6, 60-epoch tail still last:** `81_satpen_off`,
  `82_valdraw_ml2`, **`82x_seedcensus_s1`** and **`82y_seedcensus_s4`** (new
  — n=2 gives a difference, n=4 gives a spread that can be quoted beside
  every ml2 number in this file), `83_valdraw_plain_long`. If the spread
  lands near 2e-4, the honest conclusion is that the ml2 val ladder cannot
  decide anything this campaign used it for, and the **SELECTOR SPEC**'s
  play-anchored mini-match is not an improvement on val — it is the only
  instrument the family has.

- 2026-08-15 09:10 UTC: **THE ONE-LAYER NOISE FLOOR IS ~4e-5 — and it is
  the same size as gaps this ledger has been ranking on.**

  **`78_plain_seed1`** (seed 1, split PINNED — `val_sha 0239a7b84ec6ba2f`,
  the `perm_seed` fix's first real use and it worked): best val **0.01374**,
  last-10 mean 0.01376, converged as the one-layer family always is.

  | run | seed | τ | best val | last-10 mean |
  |---|---|---|---|---|
  | `71_gridste_plain` | 0 | 0.85 | 0.01378 | 0.01379 |
  | `72_replnet_tau11` | 0 | 1.1 | 0.01378 | 0.01379 |
  | **`78_plain_seed1`** | **1** | 0.85 | **0.01374** | 0.01376 |

  **What the 71/72 tie actually was.** It was never evidence of a "floor" —
  they shared **seed 0**, and the one-layer family is extremely stable
  *given a seed*. Changing τ moved val by **< 1e-5** (identical to five
  decimals); changing the **seed** moved it by **4e-5**. So the τ effect is
  four times below the seed effect, which strengthens rather than weakens
  arm 8's verdict: **τ 1.1 pays nothing, and its effect is below the noise
  the measurement can even resolve.**

  **The retroactive part, which is the uncomfortable one.** A 4e-5 seed
  sensitivity is the same order as several differences this ledger has
  reported as results:

  | claim | gap | vs the 4e-5 observation |
  |---|---|---|
  | "data scale pays": 8Mv 0.01378 vs v1 0.01385 | 7e-5 | 1.8× |
  | c1024-cal 0.01388 vs 0.01378 | 1.0e-4 | 2.5× |
  | ml2 bm4 0.01286 vs bm2 0.01291 | 5e-5 | 1.3× |
  | grid l1 .0005 0.01347 vs .0003 0.01354 | 7e-5 | 1.8× |

  **This does not say any of those claims is wrong.** It says they are
  **unresolved at the precision they were stated**, because each side is a
  single seed and the one measured seed change is a comparable size. The
  honest reading is that the campaign has been ranking arms on differences
  it never showed to exceed its own noise — and that is exactly the failure
  mode the SELECTOR SPEC above exists to end.

  **Two caveats stated so this is not over-read in the other direction.**
  (1) `n = 2` seeds: one difference of 4e-5 is a single *observation* of
  seed sensitivity, **not a variance estimate** — no σ, no interval, and
  the next seed could read anything. (2) The ml2 family is **noisier**, not
  quieter: its best-vs-last-10 spreads run 5e-4 to 8e-4 against the
  one-layer family's 2e-5, so an ml2 seed sensitivity is likely larger. The
  runs now queued turn both caveats into measurements.

  **A third finding, free with the first: the export shift is NOT a
  property of the recipe.** `78` landed on **shift 3** (gains
  [43, 42, 48, 45]) where `71`/`72` landed on **shift 4** (gains
  [83, 87, 88, 82]) — same recipe, same data, different seed. Harmless for
  a one-layer net. **For ml2 it is the whole ballgame**: the shift sets the
  layer-2 read-out scale, and shift 4 versus 3 is the difference between
  the dead read-out of runs 60/61 and a live one. A candidate whose second
  layer survives export is therefore partly a **draw**, not purely a
  recipe — which is one more reason nothing in this family should be
  promoted on val.

  **Promotion path, referencing the lane above rather than inventing one:**
  any grid-era candidate that earns a look goes through the **SELECTOR
  SPEC**'s 50-game fixed-node mini-match against the pinned entry first —
  never straight to a screen, and never on val alone.

  **Re-baselining, from the measurement lane's stage-1 gap:** the entry
  measures **−1.74 ± 27.93 vs classic at equal nodes (~+60
  bias-corrected)**. So a replacement net's target is not classic, it is
  **the entry** — and the best net this campaign has produced is **107 Elo
  short of it** (linear −107; the two ml2 arms −234 and −300). That is the
  size of the hole, stated in the units that matter.

  **Queued (depth 5, longest last), each with its question named:**
  `81_satpen_off` — registered arm 11's **satpen half**, never varied since
  the kbbil lesson made 0.03 default-on; byte-free, so pure val ·
  `82_valdraw_ml2` and `83_valdraw_plain_long` — **is the ladder itself
  stable?** Every val in this campaign comes from ONE 200 000-row draw, and
  until `perm_seed` existed nobody could ask what a different draw says.
  These ask it for both families: if the draw moves val by less than the
  gaps we read, the ladder is sound; if by more, the campaign's val
  differences are draw noise and the instrument, not any net, is the
  finding.

- 2026-08-15 08:25 UTC: **THE SEED SWEEP WAS MEASURING ITSELF: `opt.seed`
  moves the val SPLIT, not just the initialisation — caught by `val_sha`,
  fixed, damage contained, and the attractor answer survives it.**

  **`76_gridste_seed1_long` (seed 1, 60 epochs)** finished: best-of-60
  **0.01340**, best-of-first-30 0.01396, last-10 mean 0.01484 — and
  **`val_sha 96e37345a39624ce`**, against the `0239a7b84ec6ba2f` that every
  other run in this campaign carries. **Its val number is withdrawn from
  the ladder**: it was not measured on the same validation set, so it
  cannot be compared to 0.01347 or to anything else here.

  **The defect.** Under `legacy-perm`, `train.py` drew the split from the
  same `random.Random(cfg.opt.seed)` that seeds initialisation, so a seed
  replicate changes the validation set underneath the experiment. Measured
  directly on the split logic: **the seed-0 and seed-1 val sets overlap
  5.00%** — which is exactly the overlap of two independent 200 000-row
  draws from 4 027 406, i.e. they are effectively disjoint samples. My
  refill asked "how much of a val difference is seed noise?" with configs
  that changed the measuring stick along with the thing being measured.
  **The instrument caught it, and that is the only reason this entry is not
  a wrong result:** `val_sha` exists precisely so that "same val set" is
  checkable rather than asserted, and it was checked.

  **Damage is contained, and the containment is the important part.** Every
  run this campaign has compared — v1, 8Mv, the c1024 arms, 40/41, 50, 60,
  61, 70–75 — ran at **seed 0** and therefore shares one val set. **No
  existing comparison in this ledger is affected.** Exactly two runs sit
  off the ladder: `76` (seed 1) and `77` (seed 2, running now, queued
  before the fix).

  **The fix** (`data.perm_seed`, training-side only, landed with this
  entry): the split may be seeded independently of `opt.seed`.
  `perm_seed: -1` is the default and reproduces history bit for bit — one
  Random draws split then epoch shuffles, in `train_packed.py`'s original
  order. Verified before deploying: default at seed 0 reproduces the
  campaign's val ids exactly, and `perm_seed: 0` reproduces them for
  `opt.seed` ∈ {1, 2, 3} while the old path does not.

  **The attractor answer SURVIVES all of this**, because the parking is a
  property of the trained weights and needs no val set at all. At **seed 1**
  the read-out lands at `u2·scale` =
  **[0.500031, 0.4999669, 0.5000135, 0.5078581]** — three of four within
  **3e-5** of the tie. **The attractor is not a seed-0 artifact.** With
  `76` the count is seven runs and twenty-eight components; `77` (seed 2)
  makes it eight, and `80` (seed 3, 60 epochs) closes seed × ride.

  **Requeued with the split pinned**, longest last: `78_plain_seed1`
  (`ebaa41819de7`) one-layer val variance, now actually answerable ·
  `79_seed2_l1_001_pinned` (`41a93306f637`) redoes `77`'s void val leg —
  is the l1 .001 free-byte result seed-stable? · `80_seed3_l1_001_long`
  (`559e3432f3b0`) the 60-epoch tail. `77` is left running: its attractor
  leg is valid and costs nothing, its val leg is void and is not being
  read.

  **One consequence worth recording once:** `config_hash` covers the whole
  config dict, so *adding a field rehashes every config*. The hashes quoted
  in entries written before `gridste`, `u2grid` and now `perm_seed` existed
  will not recompute under current code. Each run's own
  `PROVENANCE.json` records the hash as computed at run time and remains
  self-consistent — that file, not a quoted hash in prose, is the
  authority.

- 2026-08-15 08:10 UTC: **THE DENSITY DIAL REVERSES under the fidelity rule
  — 35 free bytes — and the n8/kb4 seams are PRICED and DECLINED as a
  direction call, not queued.**

  **`74` / `75`: the l1 sweep, measured end to end** (val on the pinned
  split, bytes through `pack.sh` on the real ml2 entry):

  | run | l1 | best val | zeros | **packed** |
  |---|---|---|---|---|
  | `75_gridste_l1_0003` | .0003 | 0.01354 | 39.3% | 3794 B |
  | `70_gridste_ml2` | .0005 | **0.01347** | 57.1% | 3775 B |
  | **`74_gridste_l1_001`** | **.001** | **0.01347** | **72.9%** | **3740 B** |

  `74`'s pre-registration asked exactly this and the answer is the one it
  named: **the float era's "density pays under products" line was an
  artifact of the free-float read-out.** In the float era l1 .0005 beat
  .001 (0.01280 vs 0.01286) and this ledger wrote density down as a win.
  Under the exported-fidelity rule the two are a **dead heat on val** and
  .001 buys **15.8 points more sparsity for 35 fewer bytes** — strictly
  dominant, free. `75`'s denser point is worse on both axes, so the curve
  is monotone and the direction is clear. **The family's ml2 operating
  point moves to l1 = .001**; `77`/`79` below re-measure it on fresh seeds
  before anything is called a default.

  **The attractor is now six runs and twenty-four components**, spanning
  l1 ∈ {.0003, .0005, .001}, clamp ∈ {400, 600}, rides ∈ {30, 60} — and
  every single component still sits at half a grid step. `74` is the
  tightest yet: `u2·scale` = [0.5000001, 0.4999989, 0.5000002, 0.4999986],
  all four within **1.4e-6**. Only the seed remains unvaried (all six are
  seed 0); `76` (running) and `77`/`79` close that.

  **THE SEAM DECISION — priced first, then declined.** The header's two
  remaining registered arms both need a seam agreement, so I priced them
  before asking anyone anything (`make_proto_payload.py` real-shaped
  streams through `pack.sh`, current code floor):

  | arm | payload | at 72.9% zeros | 85% | 95% | 98% |
  |---|---|---|---|---|---|
  | reference (N=4, 768) | 778 ch | 3737 B | — | — | — |
  | **#5 n8** (N=8, 768) | 1554 ch | **4199 — OVER by 103** | 4009 (87 spare) | 3596 | — |
  | **#3 kb4** ternshared (shared + 3 delta blocks) | 3082 ch | — | — | 3952 (144 spare) | 3617 |

  So n8 needs **≥ ~85%** payload sparsity to fit and kb4 needs **≥ ~95%**,
  against the 72.9% the best trained net actually reaches. (Honest caveat
  in the pessimistic direction: these are RANDOM-shaped stand-ins, and this
  ledger has measured trained payloads compressing well below them — v1:
  612 random → 382 trained on the same 777 chars. So n8 is plausibly
  affordable; plausibly is not measured, which is the whole point of
  price-first.)

  **Decision: NOT queued — this is a direction call, not a technical one.**
  `gridste` was resolvable by this lane because it was entirely
  training-side: `gridste: 0` is the bit-exact identity and the shipped
  artifact never changed. n8 and kb4 are the opposite — both change the
  **entry's decoder**, which is (a) explicitly another lane's surface in
  this very header ("codec seam is the GOLF LANE's; agree it before
  training") and (b) a spend of the one resource that decides the
  competition. That is owner/direction territory and this lane does not
  take it.

  **The seam question, stated crisply for the direction call** — and it is
  bigger than "does it fit":

  > Both remaining registered arms buy CAPACITY, and capacity is the axis
  > the play record has already ruled out twice. The ledger's own line
  > after the linear screens was "capacity was NOT the missing ingredient";
  > the non-linear family then screened **worse** (−234, −300 vs the linear
  > −107), and the export-faithful net — the one with every technical
  > excuse removed — was the **worst of the three**. Spending the last
  > ~350 B of headroom and a decoder change on a wider or bucketed net asks
  > the campaign to double down on the losing axis. **The question for
  > Thomas/the coordinator is not "may I have the n8 seam" but "should the
  > replacement family spend anything further on capacity before it has one
  > net that plays".** If the answer is yes, n8 is the cheaper of the two
  > and needs the golf lane's byte savings plus a trained ≥85%-sparse
  > stream; kb4 needs 95% and should wait behind it.

  **Refill (depth 3 behind `76`, longest last), every question named in its
  own registration:** `77_seed2_l1_001` (`4737ba6ee662`) attractor point 3 +
  seed-stability of the free-byte result · `78_plain_seed1`
  (`0007671c5d0f`) one-layer val variance — `71` and `72` both landed on
  0.01378 to five decimals, and nobody has measured the seed noise that
  every one-layer gap this ledger quotes would have to clear ·
  `79_seed3_l1_001_long` (`79ce56feb412`) the 60-epoch tail, attractor
  point 4 (seed and epochs together).

  Context: Thomas merged #204, so `pool` TM is the classic driver default
  on master. No training effect; noted because this lane's screens pin
  **nnue-4k** revs (`c5534cd`, `5f16bae`) and never "current master", so
  nothing here needs a rev correction.

- 2026-08-15 06:55 UTC: **THE GRID ERA, all four runs — and the knife-edge
  is a HARD ATTRACTOR with a mechanism, not a curiosity.**

  | run | arch | dial | best val | read-out | packed |
  |---|---|---|---|---|---|
  | `70_gridste_ml2` | ml2 | l1 .0005 | **0.01347** | [1, 1, 1, 0] | **3775 B** |
  | `71_gridste_plain` | 1-layer | τ .85 | **0.01378** | n/a | **3570 B** |
  | `72_replnet_tau11` | 1-layer | τ 1.1 | **0.01378** | n/a | **3571 B** |
  | `73_gridste_ml2_long` | ml2 | l1 .0005, 60 ep | 0.01331 (see below) | [1, 1, 1, 0] | — |

  All on the pinned split (`val_sha 0239a7b84ec6ba2f`, n_train 7 827 406).

  **`71` — the control answers cleanly: the defect was ml2's alone.**
  0.01378 is the linear 8Mv reference **exactly**, and the run is
  *converged* (last-10 mean 0.01379). Snapping gains and biases onto the
  payload's integer digits costs the one-layer family **nothing**: it is
  fully grid-representable, and every historical linear val in this ledger
  stands as written. Pre-registered criterion met.

  **`70` — honest two-layer training recovers about a third.** 0.01347
  against the linear 0.01378 and the float-ml2 0.01280: the ml2 family's
  whole val win was 0.00098 and 0.00031 of it survives contact with the
  grid — **31.6%**. Criterion met (non-zero read-out, val under the
  u2-only run's 0.01362). **Caveat that must travel with the number:** the
  two-layer grid runs *thrash* — `70` best 0.01347 against a last-10 mean
  of **0.01430**, `60` 0.01362 against 0.01411 — while `71`/`72` sit still
  (0.01378 / 0.01379). "Full grid beats u2-only" **reverses** on last-10
  mean and is NOT established.

  **`72` — registered arm 8 CLOSED by measurement.** τ 1.1 gives val
  **0.01378**, identical to τ 0.85, and sparsity moves 46.1% → 46.7%. Its
  pre-registration said bytes decide, and bytes were measured through the
  real pack path: **3571 B vs 3570 B — plus one byte for nothing.** The
  high side pays nothing; τ stays 0.85.

  **`73` — the tail's question is answered, and it is the strong answer.**
  Pre-registered as "does doubling the ride move `u2` off the tie?"
  **No.** At 60 epochs all four components land within **2e-6** of the
  boundary — [0.5000015, 0.5000011, 0.5000008, 0.4999979] — *tighter* than
  the 30-epoch run's. And its 0.01331 is a best-of-60 selection artifact,
  the `dense60` lesson repeating exactly: **best-of-first-30 is 0.01381**,
  worse than `70`'s 0.01347 at matched budget, with a last-10 mean of
  0.01436. Longer riding buys a lower minimum and nothing else.

  **The attractor, across every grid-era ml2 run** (`u2 · scale`, where 1.0
  is one grid step):

  | run | components | digits |
  |---|---|---|
  | `60` | 0.5000003, 0.4967398, 0.5000005, 0.4996593 | [1, 0, 1, 0] |
  | `61` | 0.5000057, 0.4954618, 0.5000003, 0.4992411 | [1, 0, 1, 0] |
  | `70` | 0.5000051, 0.5074217, 0.5000017, 0.4999114 | [1, 1, 1, 0] |
  | `73` | 0.5000015, 0.5000011, 0.5000008, 0.4999979 | [1, 1, 1, 0] |

  Sixteen components, four runs, two clamps, two ride lengths — **every one
  parked at half a grid step.** The mechanism is STE boundary chatter and
  it is not mysterious: the forward is a step function, so above 0.5 the
  digit is 1 and the layer fires at full strength (loss pushes down), below
  0.5 the digit is 0 and the layer vanishes (loss pushes up). The optimizer
  oscillates about the threshold instead of converging, and **the val
  thrashing above is the same phenomenon read off the loss.** The net is
  not refusing a second layer — **it is asking for a read-out strength
  between 0 and 1 grid units, and the container cannot express one.** At
  shift 4 the read-out has ~2 usable settings; the export sweep showed
  shift 3 gives it [2.0, 2.03, 2.0, 2.0] and shift 2 gives [8, 8.1, 8, 8],
  i.e. real resolution. **That is a container-design decision and it is the
  coordinator's, not this lane's — recorded, not taken.**

  **Play context, so none of these vals are misread:** `70` screened at
  **−300.56 ± 71.33** (MEASUREMENTS, same date). Val ranks arms inside a
  family and inverts across family boundaries — the linear net has the
  worst val of the three and the best play.

  Queue held at **depth 3** behind this (`74` l1 .001, `75` l1 .0003, `76`
  seed-1 60-epoch tail). `76` is the seed replicate — `73` was **seed 0**,
  so seed-dependence of the attractor is still open and `76` answers it.

- 2026-08-15 05:12 UTC: **CLAMP 400 LOSES, the knife-edge REPLICATES, and
  the queue emptied a SECOND time — 31 minutes, same manual-refill cause.**

  **Correction first, because my last report got it wrong.** I reported
  "one config remains behind it" after `61_replnet_clamp` started. There
  was no third config. `queue_runner` moves a yaml to `done/` only when it
  FINISHES, so the `ls queue/*.yaml | wc -l` that returned 1 was counting
  61 **itself**, still sitting in the queue while running. I queued exactly
  two (60, 61); both ran; `queue/done/` holds exactly those two and
  `runs/` gained exactly those two. The count was real, my reading of it
  was not — a self-referential-instrument error of the same class as the
  `grep -c` gate that lied and the `pgrep -f` wait loop that matched its
  own launcher. **Reading a queue depth while you are the thing in the
  queue needs the same care as reading a process list while you are the
  process.**

  **`61_replnet_clamp` (registered arm 11, CLAMP 400 vs 600) — 400 LOSES.**
  val **0.01380** against run 60's **0.01362** on the identical recipe,
  split and val set (`val_sha 0239a7b84ec6ba2f`, n_train 7 827 406); one
  field apart, so the comparison is clean. **CLAMP stays 600**; the arm is
  answered and closed. Mechanism visible in the log: clip-saturation rose
  to 0.14–0.52% in late epochs against 60's 0.00–0.08%, i.e. the tighter
  clip started biting exactly where the ml2 second layer lives (that term
  alone is worth mean 60.50 cp with a 186.59 cp maximum). The probe suite
  agrees it costs knowledge, not just loss: king-activity **14.83 → 0.00**,
  passed-vs-opposed **25.53 → 0.00**, rook-open-file **14.44 → 0.00**.
  Byte-free was the arm's whole appeal and it is not free in val.

  **The knife-edge REPLICATES.** 61 parked `u2` at
  **[12.50014, 12.38654, 12.50001, 12.48103]** — the same 12.5 boundary as
  run 60 ([12.50001, 12.41849, 12.50001, 12.49148]), under a different
  clamp, reaching the same read-out digits **[1, 0, 1, 0]**. Two
  independent runs park at half a grid step. That is now a property of the
  recipe, not an accident of one run — and it is why the queue below
  carries a 60-epoch tail asking whether the ride length changes it.

  **Refill: 4 configs, ~100 minutes of runway, longest last.**
  `70_gridste_ml2` (`19ea4a95c764`) full exported fidelity on ml2 ·
  `71_gridste_plain` (`10413c06ccc6`) the same on the PLAIN one-layer net,
  the control that says whether the defect is ml2's or the family's ·
  `72_replnet_tau11` (`e525cdd862f6`) registered arm 8's owed high side ·
  `73_gridste_ml2_long` (`de0446fa9a34`) the 60-epoch TAIL. Each pre-registers
  its reading in its own header. `model.gridste` was verified before
  queueing: at v = [130.63, 134.15, 135.54, 127.47] it reproduces the
  exporter's own digits exactly (g [65, 67, 68, 64], bd [40, 41, 41, 39]),
  gradients pass 1.0, and `gridste: 0` is the bit-exact identity.

  **Structural fix, and its limit.** `queue_runner` has **no** low-water
  hook or tail mechanism — its entire CLI is `--queue-dir / --once /
  --pgn-globs`, and on empty it nags and sleeps 600 s. So the only lever
  available without touching a running process is **depth plus a long tail
  entry placed last**, which is what this refill is. A real fix (refuse to
  idle below a configured depth, or run a designated tail config on empty)
  is a `queue_runner` change that only takes effect at the next restart,
  and restarting the runner is not this lane's call — **proposed here, not
  taken.** Until it lands, every lane that drains the queue owes it a
  refill in the same breath.

- 2026-08-15 03:59 UTC: **QUEUE REFILLED after a 38-MINUTE IDLE GAP — a
  standing-rule violation, recorded rather than glossed.** `50_dense60`
  finished 03:21; the queue sat empty until 03:59 while this lane was
  harvesting and exporting, and `queue_runner` nagged `[queue] EMPTY` four
  times into `runner.log` — the instrument worked, the operator (me) was
  the failure. In my export report I described the empty queue as benign
  because the runner was healthy; that was wrong, and the coordinator
  corrected it. **The trainer never idles: a finishing run is a REFILL
  TRIGGER, not a status update.** Two entries queued, both validated
  box-side before drop (`config.load` + `config_hash`):
  - `60_ml2_u2grid.yaml` (hash `d5a8b66bc26c`) — **the fix for the export
    blocker**: `21_phase_ml2_dense`'s recipe with `u2` snapped onto the
    certified integer read-out grid INSIDE forward, STE at the export
    scale (`model.Ml2Net._u2`, `model.u2grid: 1`). Verified on the box
    before queueing: the trained free-float `u2` [4.27, 2.10, 3.39, 2.69]
    snaps to **[0, 0, 0, 0] inside forward** — the defect is now visible
    to the loss instead of only to the exporter — the grid step is 25.0 at
    shift 4, the STE gradient passes (1.0), and `u2grid: 0` (the default)
    is bit-identical to before, so no queued or historical config moves.
  - `61_replnet_clamp.yaml` (hash `c0ab623b4fd5`) — registered family arm
    **#11 `replnet_clamp`** (CLAMP 400 vs the 600 default, byte-free), on
    the same `u2grid` recipe so the pair differs in exactly one field. No
    new seam, no new invention. It is more interesting now than when it
    was registered: the ml2 second layer's own contribution is mean 60.50
    cp with a 186.59 cp maximum, so the residual reaching the clip is
    materially bigger than it was when 600 was chosen.

  Pre-registered success criterion for `60`, stated before the run: a
  **non-zero integer read-out at export** (`export.py` prints it beside
  every export and refuses to price an all-zero one) and val at or under
  0.01280, the free-float number being an upper bound on what a
  grid-constrained net can reach. A second `[0,0,0,0]` is a real answer —
  it would say the capacity is not reachable at this scale.

- 2026-08-15 ~05:3x UTC: **ml2 EXPORT BLOCKED — the trained layer-2
  read-out quantizes to ZERO on the certified grid**, and the export now
  refuses to price a silent second layer (`train/export.py`). Harvest:
  `40_c1024_cb` (trained codebook K=32) val **0.01415**, `41_c1024_lr`
  (trained low-rank rank=1) val **0.01384**, both certificates green.
  Same 200 k val as the whole family (`val_sha 0239a7b84ec6ba2f`) but at
  4 M data, so the honest control is the ml2-at-4M run `80_replnet_ml2`
  at 0.01283 — **NEITHER structured arm beats the 0.01280 incumbent**
  (+8.1 % / +10.5 % relative). Export candidate stays `21_phase_ml2_dense`
  (ml2, l1 .0005, 8 M). `50_dense60` finished at 0.01274, which is a
  **best-of-60 selection artifact, not a gain**: at matched budget its
  best-of-first-30 is 0.01282 (21's is 0.01280), its last-10 mean is
  0.01302 (21's 0.01292), and epoch-to-epoch val noise is ±0.0002. Sixty
  epochs buy nothing on this recipe. Blocker detail and the whole price
  sheet in MEASUREMENTS.md, same date; the retrain it implies —
  **u2 quantized on the certified grid INSIDE forward, at the export
  scale, the way the ternary weights already are** — is the natural next
  ml2 queue entry and is NOT queued unilaterally.

- 2026-08-15 morning: **SCREEN: H0, −107.06 ± 35.84** (318 games, penta
  [51,27,58,12,11], zero illegal). Linear-family generalization sharpens
  (six linear play-failures, both capacity ends); phase-ml2 inherits
  skepticism only. ml2 val ladder: 0.01280 new best (l1 .0005 — density
  pays under products). 90/91 path-fixed → requeued 40/41; 50_dense60
  queued; own-labels 28k probe flips king geometry (retest at scale).

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
