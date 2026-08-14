# TRAINQUEUE — the trainer never idles (standing rule, Thomas 2026-08-14)

Priority-ordered next nets for the replacement-net (replnet) lane. The
moment a run finishes, the top entry starts (labeller-class on the box:
nice 19, ≤8 workers/threads, forfeit tripwire on live matches). Gates,
screens, and landings stay coordinator-dispatched and play-gated — this
queue is about TRAINING only. Re-order freely as results land; never
empty. Provenance pinned on every run (commit, seed, data sha in
PROVENANCE.txt beside the run).

Context (2026-08-14): budget = engine-sans-eval 2871 + machinery 578;
payload must land ≥58% zeros (hard), 60-66% target. v1 arms: l1=0.001 →
val 0.01385 @59.6% zeros (winner, thin margin); l1=0.002 → 0.01404
@73.7%. Sparsity is nearly free (~+0.0002 val per +14% zeros).

## Queue

1. **replnet_kb8fold — king-bucketed training, folded export** (--kb 8
   --factor 1, fold the B buckets to shared ternary rows at export; new
   export step, small). kb multiplies FLOAT rows, not shipped bytes, iff
   folded; the factorizer may find better shared rows than kb=1 training
   does. If folding needs new code time, swap with 3.
2. **replnet_tau — threshold sweep** (τ ∈ {0.6, 1.1} at the winner's l1).
   τ and l1 are two knobs on one sparsity mechanism; v1 only moved l1.
3. **replnet_bilt — bilinear m=4 + odd tail, PRICE-FIRST** (--nb 4 --bm 4
   [--tailw 4]). Bilt history says the tail carries signal, but the ext
   machinery is float/dev-only: this run is a VAL probe; shipping needs
   ≥66% zeros (71 B spare) plus new priced machinery. Train, price, then
   decide.
4. **replnet_rff — rff64 at tiny width, VAL probe only** (--rff 64).
   Proven −3.9% val at width 128; unknown at N=4, machinery unpriced.
5. **replnet_clamp — CLAMP/satpen interaction** (clampcp 400 vs 600 at
   the winner's sparsity). In replacement mode the clip bounds ALL
   positional signal; 600 was inherited from the residual era, never
   measured here. Byte-neutral.

## Log (newest first)

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
