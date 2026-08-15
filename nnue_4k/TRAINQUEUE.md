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

## Log (newest first)

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
