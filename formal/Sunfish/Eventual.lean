/-
EVENTUAL CLASSIFICATION: how much of the trichotomy's premise ledger the
"for all large enough depths" weakening can pay off.

`Classification.lean`'s `eventual_classification` states the trichotomy
at EVERY depth and spends four premises: `ValFloor` and `EvalQuiet`
(fidelity, tables), `NoZugzwang` (chess, layer 2) and
`NoMaskedMobility` (chess, layer 2, the model-side stand-in for the
engine fix in #171).  The question this module answers is whether
weakening the conclusion from "at every depth" to "at every depth from
some `D0` on" -- which is all the driver's `range(1, 1000)` deepening
ever needs -- lets the last of those be dropped.

**The hope, and why it was reasonable.**  Masking is a FRONTIER effect.
The QS val-filter admits `val >= QS - depth * QS_A`, so at remaining
depth 2 the threshold is already -240, below the shipped tables' move
value floor of -192, and every legal move survives
(`filter_identity_off_frontier`).  Masking therefore exists ONLY at
remaining depth 1 -- roughly `D` plies from the root.  A phantom found
there is `D` negations away from the root, so it looked like it should
either be absorbed as `D` grows (`CexF`'s phantom is: it dissolves at
depth 3) or arrive with nothing left to claim.

**Why the hope fails, in one sentence.**  A masked node does not report
a shallow mate; it reports the OFF-LADDER sentinel
(`maskedFrontier_value`: the filtered fold has no admitted legal move,
so nothing displaces the initial `LOSS` accumulator and the parent
negates it to `MATE_UPPER`).  The mate ladder decays one
`EVAL_ROUGHNESS` rung per ply of unspent depth; `MATE_UPPER` decays not
at all.  So a phantom's contribution survives an arbitrary number of
negations undiminished, and the frontier RENEWS it at every horizon: at
depth `D` the phantom sits at ply `D - 1`, at depth `D + 2` at ply
`D + 1`.  Absorbing the old one buys nothing when a new one arrives.

`CexE` (section 3) is that argument turned into a machine-checked
countermodel, and it is not a near miss: the root is a draw whose
declared value is `MATE_UPPER` at every even depth and `-MATE_UPPER` at
every odd one.  The classification does not merely fail to settle -- it
oscillates between the two false extremes forever.  Neither acyclicity
(the model is a strictly increasing chain, no position ever repeats)
nor a read-time clamp at `MATE_UPPER - 1` touches it.  The frontier-tail
variant of `Classification.lean` Part B values the same root at an
honest `0` at every depth, so what retires `NoMaskedMobility` is the
engine change and nothing else.

**What the weakening does buy, and it is not nothing.**  Both
COMPLETENESS arms -- a real forced mate reaches the band, a real forced
loss reaches the mated band -- never needed the frontier premise at all
(`eventual_completeness_without_frontier`).  `NoMaskedMobility` pays
for exactly one thing: HONESTY, the "neither" arm.  A phantom can only
invent a mate that is not there; it can never hide one that is.

**And `NearMaximalChoice` is free at this granularity** (section 2).
The driver stops bisecting at `upper - lower <= EVAL_ROUGHNESS = 15`,
so the shipped root can settle for a value within 15 of the best.  The
declared value's range avoids the whole corridor between the static
score range and the band edge (`nullValueD2_offCorridor`: an induction
whose only cases are the sentinel, the terminal ladder, the static
eval, and closure of the corridor complement under negation and `max`),
and for the shipped tables that corridor is 32,486 points wide --
`shipped_band_gap_wide`, more than two THOUSAND stopping tolerances.
So no 15-point slack can move a value across a band edge
(`bandOf_eq_of_slack`), and a near-maximal choice lands in the same
band as an exact argmax (`nearMaximal_band_exact`).  This is the band
level counterpart of `Shortest.lean`'s rung-level result: there parity
refunds the tolerance one rung at a time, here the tolerance is simply
three orders of magnitude too small to matter.  Both readings rest on
the same decision -- one `EVAL_ROUGHNESS` of mate tempo per ply (#172).

Nothing here changes `sunfish.py`; `CexE` is a model, and the
frontier-tail variant it vindicates is the proof-first `negamaxD2t`
that Part B of `Classification.lean` already carries.
-/

import Sunfish.Classification
import Sunfish.Shortest

namespace Sunfish

/-! ## 1. The frontier is local; the sentinel it prints is not -/

/-- Every child of a node with no king capture still has its own king:
the parent's `hasKingCapture` scan is exactly that test. -/
theorem eval_child_live (G : QSGame) {p m : G.Pos}
    (hcap : hasKingCapture G.toNullGame.toGame p = false) (hm : m ∈ G.moves p) :
    ¬ (G.eval m ≤ -MATE_LOWER) := by
  intro h
  have hk : hasKingCapture G.toNullGame.toGame p = true :=
    (hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, h⟩
  rw [hcap] at hk
  exact Bool.noConfusion hk

/-- **(a) Masking is confined to remaining depth 1.**  `val_lower 2 =
40 - 2 * 140 = -240` is already below the tables' move-value floor of
-192, so from remaining depth 2 up the QS filter is the identity on the
legal move list and no move can be masked.  A restatement of
`movesAbove_all` at the constants, named because the whole question of
this module is where masking can live. -/
theorem filter_identity_off_frontier (G : QSGame) (hF : ValFloor G 192)
    {d : Nat} (hd : 2 ≤ d) (p : G.Pos) :
    movesAbove G (val_lower d) p = G.moves p :=
  movesAbove_all G d p (allAboveB_of_floor G hF d p (val_lower_le_neg_floor d hd))

/-- **The phantom is the sentinel, exactly.**  At a node that HAS a
legal move but whose every depth-1-admitted move is illegal, the fold
runs over refuted moves only: each contributes `-(MATE_UPPER) = LOSS`,
the pass term is `LOSS` (the null option is off below remaining depth
3), and nothing displaces the initial accumulator.  So the node reports
`-MATE_UPPER` and its parent reports `MATE_UPPER` -- a value strictly
above every rung the mate ladder can produce, and one that does not
decay with distance from the horizon.

This is why the depth-decay argument that retires a phantom mate
cannot retire a phantom sentinel: there is no rung to outrank. -/
theorem maskedFrontier_value (G : QSGame) (guard : G.Pos → Bool) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = false)
    (hai : allIllegalB G p = false)
    (hmask : ∀ m ∈ movesAbove G (val_lower 1) p,
      hasKingCapture G.toNullGame.toGame m = true) :
    nullValueD2 G guard 1 p = -MATE_UPPER := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hcap' : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by rw [hcap]; simp
  show nullValueD2 G guard (0 + 1) p = -MATE_UPPER
  rw [nullValueD2_of_fold G guard 0 p hkg hcap' hai, show nullTermD2 G guard 0 p = LOSS from by
    simp [nullTermD2]]
  have hall : ∀ m ∈ movesAbove G (val_lower (0 + 1)) p, -(nullValueD2 G guard 0 m) ≤ LOSS := by
    intro m hm
    have hmm := movesAbove_subset G _ p m hm
    have hcm := hmask m hm
    rw [nullValueD2_of_capture G guard 0 m (eval_child_live G hcap hmm) hcm]
    omega
  have h1 := foldMax_le (fun m => -(nullValueD2 G guard 0 m))
    (movesAbove G (val_lower (0 + 1)) p) LOSS hall (Int.le_refl _)
  have h2 := foldMax_ge_init (fun m => -(nullValueD2 G guard 0 m))
    (movesAbove G (val_lower (0 + 1)) p) LOSS
  omega

/-- **(b) Completeness never needed the frontier premise.**  Both
"finding" arms of `eventual_classification` -- a forced mate in `k` puts
the declared value in the band from depth `k + 1` on, a forced loss puts
it at or below `-MATE_LOWER` from depth `k + 2` on -- go through
`forcedMate_complete_band` and `forcedlyMated_negamaxD2_band`, which
spend only `ValFloor` and `NoZugzwang`.  A phantom adds spurious HIGH
values to the side that finds it; it can invent a mate that is not
there, never hide one that is.

So the premise ledger is sharper than the trichotomy's statement
suggests: `NoMaskedMobility` buys the honesty arm and nothing else. -/
theorem eventual_completeness_without_frontier (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hZ : NoZugzwang G guard) (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false) :
    (∀ k, ForcedMate G k p → ∀ D, k + 1 ≤ D → MATE_LOWER ≤ nullValueD2 G guard D p) ∧
    (∀ k, ForcedlyMated G k p → ∀ D, k + 2 ≤ D → nullValueD2 G guard D p ≤ -MATE_LOWER) := by
  refine ⟨fun k hFM D hD => forcedMate_complete_band G guard hF hZ hFM D hD, fun k hFMd D hD => ?_⟩
  rw [nullValue_eq_realValue_of_noZugzwang G guard hZ D p]
  exact forcedlyMated_negamaxD2_band G hF hcapf hFMd D hD

end Sunfish
