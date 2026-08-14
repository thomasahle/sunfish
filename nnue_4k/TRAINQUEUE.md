# TRAINQUEUE — the trainer never idles (standing rule, Thomas 2026-08-14)

Priority-ordered next nets for the replacement-net (replnet) lane. The
moment a run finishes, the top entry starts (labeller-class on the box:
nice 19, ≤8 workers/threads, forfeit tripwire on live matches). Gates,
screens, and landings stay coordinator-dispatched and play-gated — this
queue is about TRAINING only. Re-order freely as results land; never
empty. Provenance pinned on every run (commit, seed, data sha in
PROVENANCE.txt beside the run).

Context (2026-08-14, updated ~11:1x UTC): **Thomas directive — payload
target is now 1024 B** (was 617; a golf lane is opening the code side to
keep total ≤4096, exact capacity confirmation pending). The ≥58%-zeros
hard gate applied to the 617 budget; at 1024 the 3,072-trit ps768 payload
fits at ANY sparsity, so sparsity becomes a capacity dial, not a fit
gate. Coordinate: golf lane owns generator/machinery; the bake-off lane
(nnue_4k/train/compress/) owns the encoder — when its table lands, the
best encoder becomes the export default and arms re-size to the WEIGHT
capacity it buys. v1 arms: l1=0.001 →
val 0.01385 @59.6% zeros (winner, thin margin); l1=0.002 → 0.01404
@73.7%. Sparsity is nearly free (~+0.0002 val per +14% zeros).

## Queue

1. **c1024-cal — capacity calibration at the new budget** (winner recipe,
   N=4 ps768, sparsity pressure released: l1 ∈ {0, 0.0003}, τ 0.6; target
   ~35-50% zeros ≈ 1.6-2.0k nonzeros through the same codec). Cheapest
   capacity arm and the calibration point for everything below. PRICE the
   payload through pack.sh at each sparsity as it trains.
2. **c1024-kb4 — king buckets at 1024, PRICE-FIRST** (kb4 × 3,072 trits =
   12,288 raw ≈ mid-2k B by the old rates — likely still over even at
   1024; recheck the arithmetic with real exports, incl. the ternshared
   route: shared ternary rows + small ternary per-bucket DELTAS, which
   the 567e4ef fold makes buildable. Train only what prices.)
3. **c1024-n8 — wider hidden N=8 at ps768** (~6.1k trits, two chars per
   feature = 4+4 trits — a small decode change the GOLF LANE owns; agree
   the codec seam before training. 8-9k-weight family target.)
4. **replnet_kb8fold — in-flight** (chained after 8Mv; runs to completion
   — its fold quality is direct evidence for c1024-kb4's pricing call).
5. **replnet_tau — threshold sweep** (τ ∈ {0.6, 1.1} at the winner's l1)
   — subsumed partly by c1024-cal's τ 0.6; keep for the high-τ side.
6. **replnet_bilt — bilinear m=4 + odd tail, PRICE-FIRST, behind the
   capacity family** (--nb 4 --bm 4 [--tailw 4]; ext machinery unpriced).
7. **replnet_rff — rff64 at tiny width, VAL probe only** (--rff 64).
8. **replnet_clamp — CLAMP/satpen interaction** (clampcp 400 vs 600).
9. **replnet_ratecal — rate-aware retrain of the winner recipe**
   (`train/queue/85_replnet_ratecal.yaml`; APPENDED BY THE COMPRESSION
   LANE, re-order freely — natural slot is beside c1024-cal since both
   turn the capacity dial). Swaps l1 for `loss.rate`: the differentiable
   order-0 payload-BYTE estimator (constraints.rate_penalty), which
   matched the zoo's rc_o0 coder 518.3 vs 519 B on v1. Two arms, rate ∈
   {2e-6, 4e-6} (calibrated to v1's l1 pressure in the yaml). VAL PROBE;
   export prices per-net through compress/bakeoff.py's measured winner.

## Log (newest first)

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
