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

/-! ## 2. The driver's stopping tolerance is free at band granularity

`NearMaximalChoice` is the honest form of the engine's move rule:
`search` bisects only until `upper - lower <= EVAL_ROUGHNESS`, so the
move left in `tp_move` is guaranteed only to be within 15 of the best.
`Shortest.lean` pays for that slack one RUNG at a time, and needs
parity to refund it.  At CLASSIFICATION granularity no refund is
needed, because the question is only which of three bands the value
falls in and the bands are three orders of magnitude apart.

The statement that makes this a theorem rather than an appeal to
magnitudes is that the declared value's range avoids the corridor
between the static score range and the band edge entirely.  That is an
induction with four cases and no chess content: the sentinel, the
terminal ladder, the static eval, and the fact that the corridor's
complement is closed under negation and `max`.
-/

/-- **EvalBand B**: outside the king-gone zone the static score stays
within `±B`.  The shipped tables give `B = EvalBounds.evalBound =
15437`, the same table arithmetic `EvalQuiet` reads one-sidedly; the
link from board strings to tables is unmodelled exactly as for
`Bounded`. -/
def EvalBand (G : Game) (B : Int) : Prop :=
  ∀ p, ¬ (G.eval p ≤ -MATE_LOWER) → -B ≤ G.eval p ∧ G.eval p ≤ B

/-- **OffCorridor B v**: `v` is not in the no-man's-land between the
static range and either band edge.  Every value the search can declare
is off-corridor (`nullValueD2_offCorridor`), and that is what makes a
15-point tolerance unable to move a value across a band edge. -/
def OffCorridor (B v : Int) : Prop :=
  v ≤ -MATE_LOWER ∨ (-B ≤ v ∧ v ≤ B) ∨ MATE_LOWER ≤ v

theorem offCorridor_neg {B v : Int} (h : OffCorridor B v) : OffCorridor B (-v) := by
  rcases h with h | ⟨h1, h2⟩ | h
  · exact Or.inr (Or.inr (by omega))
  · exact Or.inr (Or.inl ⟨by omega, by omega⟩)
  · exact Or.inl (by omega)

theorem offCorridor_max {B v w : Int} (hv : OffCorridor B v) (hw : OffCorridor B w) :
    OffCorridor B (max v w) := by
  rcases Int.le_total v w with h | h
  · rw [show max v w = w from by omega]; exact hw
  · rw [show max v w = v from by omega]; exact hv

theorem offCorridor_LOSS {B : Int} : OffCorridor B LOSS := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  exact Or.inl (by omega)

theorem foldMax_offCorridor {α : Type _} (w : α → Int) {B : Int} :
    ∀ (ms : List α) (acc : Int), (∀ m ∈ ms, OffCorridor B (w m)) → OffCorridor B acc →
      OffCorridor B (foldMax w ms acc) := by
  intro ms
  induction ms with
  | nil => intro acc _ hacc; exact hacc
  | cons a ms ih =>
    intro acc hall hacc
    exact ih (max acc (w a)) (fun x hx => hall x (List.mem_cons_of_mem a hx))
      (offCorridor_max hacc (hall a (List.mem_cons_self a ms)))

/-- The terminal correction is off-corridor at every depth: checkmate
lands on the ladder at or below `-MATE_LOWER` (the clamp keeps it above
the illegal-move sentinel), stalemate is exactly `0`. -/
theorem terminalValue_offCorridor (G : QSGame) {B : Int} (hB : 0 ≤ B) (d : Nat) (p : G.Pos) :
    OffCorridor B (terminalValue G d p) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hnn : (0 : Int) ≤ (d : Int) := Int.ofNat_nonneg d
  have hmul : (0 : Int) ≤ (d : Int) * EVAL_ROUGHNESS :=
    Int.mul_nonneg hnn (by decide)
  unfold terminalValue
  split
  · exact Or.inl (by omega)
  · exact Or.inr (Or.inl ⟨by omega, hB⟩)

/-- **The declared value never enters the corridor.**  Four cases, none
of them about chess: the two sentinel branches are `±MATE_UPPER`, the
terminal branch is the ladder or `0`, depth 0 is the static eval, and
the fold is a `max` of negated off-corridor values seeded with an
off-corridor pass term.  The corridor's complement is closed under both
operations, so the induction is immediate. -/
theorem nullValueD2_offCorridor (G : QSGame) (guard : G.Pos → Bool) {B : Int}
    (hB : 0 ≤ B) (hE : EvalBand G.toNullGame.toGame B) :
    ∀ (d : Nat) (p : G.Pos), OffCorridor B (nullValueD2 G guard d p) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [nullValueD2_kingGone G guard d p hkg]
      unfold OffCorridor; omega
    · by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [nullValueD2_of_capture G guard d p hkg hcap]
        unfold OffCorridor; omega
      · cases d with
        | zero =>
          have hval : nullValueD2 G guard 0 p = G.eval p := by
            simp only [nullValueD2]; rw [if_neg hkg, if_neg hcap]
          rw [hval]
          have := hE p hkg
          unfold OffCorridor; omega
        | succ d' =>
          by_cases hai : allIllegalB G p = true
          · rw [nullValueD2_of_allIllegal G guard d' p hkg hcap hai]
            exact terminalValue_offCorridor G hB (d' + 1) p
          · have hai' : allIllegalB G p = false := by
              cases h : allIllegalB G p with
              | false => rfl
              | true => exact absurd h hai
            rw [nullValueD2_of_fold G guard d' p hkg hcap hai']
            refine foldMax_offCorridor _ _ _
              (fun m _ => offCorridor_neg (ih d' (by omega) m)) ?_
            unfold nullTermD2
            split
            · split
              · exact offCorridor_max offCorridor_LOSS
                  (offCorridor_neg (ih (d' + 1 - 3) (by omega) (G.pass p)))
              · exact offCorridor_LOSS
            · exact offCorridor_LOSS

/-- The classification a value carries: `-1` mated, `0` neither, `1`
mating.  Exactly the three arms of `eventual_classification`, read off
the value by its band. -/
def bandOf (v : Int) : Int :=
  if v ≤ -MATE_LOWER then -1 else if MATE_LOWER ≤ v then 1 else 0

/-- **The band-granularity lemma.**  If `v` is at most `w` plus the
driver's whole stopping tolerance, and neither sits in the corridor,
then `v`'s band is at most `w`'s.  The engine MINIMISES the value it
moves to, so this is the direction that matters: the near-maximal
choice is never classified worse than the alternative it may have
displaced. -/
theorem bandOf_mono_of_slack {B v w : Int} (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    (hv : OffCorridor B v) (hw : OffCorridor B w) (h : v ≤ w + EVAL_ROUGHNESS) :
    bandOf v ≤ bandOf w := by
  have hML : MATE_LOWER = 47923 := rfl
  have hER : EVAL_ROUGHNESS = 15 := rfl
  unfold OffCorridor at hv hw
  unfold bandOf
  split <;> (try split) <;> (try split) <;> (try split) <;> omega

/-- Two values the driver's tolerance can conflate lie in the same
band. -/
theorem bandOf_eq_of_slack {B v w : Int} (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    (hv : OffCorridor B v) (hw : OffCorridor B w)
    (h1 : v ≤ w + EVAL_ROUGHNESS) (h2 : w ≤ v + EVAL_ROUGHNESS) :
    bandOf v = bandOf w :=
  Int.le_antisymm (bandOf_mono_of_slack hgap hv hw h1) (bandOf_mono_of_slack hgap hw hv h2)

section
set_option maxRecDepth 4096

/-- The shipped tables clear the gap condition. -/
theorem shipped_band_gap : EvalBounds.evalBound + EVAL_ROUGHNESS < MATE_LOWER := by decide

/-- ... and not narrowly: the corridor is 32,486 points wide, more than
two thousand stopping tolerances.  This is the quantitative content of
"`NearMaximalChoice` is irrelevant at classification level". -/
theorem shipped_band_gap_wide :
    2000 * EVAL_ROUGHNESS < MATE_LOWER - EvalBounds.evalBound := by decide

end

/-- **`NearMaximalChoice` is exact at band granularity.**  The move the
shipped root leaves in `tp_move` is classified no worse than any
admitted alternative -- so replacing the idealised exact argmax by the
driver's actual stopping behaviour cannot change which of the three
classes the chosen move belongs to.  No parity, no rungs, no chess
premise; only the width of the corridor. -/
theorem nearMaximal_band_exact (G : QSGame) (guard : G.Pos → Bool) {B : Int}
    (hB : 0 ≤ B) (hE : EvalBand G.toNullGame.toGame B)
    (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    {d : Nat} {ch : G.Pos → G.Pos} (hch : NearMaximalChoice G guard d ch)
    {p m : G.Pos} (hai : allIllegalB G p = false)
    (hm : m ∈ movesAbove G (val_lower (d + 1)) p) :
    bandOf (nullValueD2 G guard d (ch p)) ≤ bandOf (nullValueD2 G guard d m) :=
  bandOf_mono_of_slack hgap (nullValueD2_offCorridor G guard hB hE d (ch p))
    (nullValueD2_offCorridor G guard hB hE d m) ((hch p hai).2 m hm)

/-- The reading that matters for play: a near-maximal choice never
misses a mate.  If some admitted move reaches a child valued at or
below `-MATE_LOWER`, so does the move the driver actually keeps. -/
theorem nearMaximal_keeps_mate (G : QSGame) (guard : G.Pos → Bool) {B : Int}
    (hB : 0 ≤ B) (hE : EvalBand G.toNullGame.toGame B)
    (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    {d : Nat} {ch : G.Pos → G.Pos} (hch : NearMaximalChoice G guard d ch)
    {p m : G.Pos} (hai : allIllegalB G p = false)
    (hm : m ∈ movesAbove G (val_lower (d + 1)) p)
    (hmate : nullValueD2 G guard d m ≤ -MATE_LOWER) :
    nullValueD2 G guard d (ch p) ≤ -MATE_LOWER := by
  have hb := nearMaximal_band_exact G guard hB hE hgap hch hai hm
  have hoc := nullValueD2_offCorridor G guard hB hE d (ch p)
  unfold bandOf at hb
  rw [if_pos hmate] at hb
  by_cases h : nullValueD2 G guard d (ch p) ≤ -MATE_LOWER
  · exact h
  · rw [if_neg h] at hb
    split at hb <;> omega

/-- **The driver's own stopping rule, read at band granularity.**
`search` returns when `upper - lower <= EVAL_ROUGHNESS`; both ends
bracket the same off-corridor value, so the converged bracket cannot
straddle a band edge.  Whatever the bisection settles on, the
classification it reports is the one the value has. -/
theorem driver_stop_band_stable {B lower upper : Int}
    (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    (hl : OffCorridor B lower) (hu : OffCorridor B upper)
    (hle : lower ≤ upper) (hstop : upper - lower ≤ EVAL_ROUGHNESS) :
    bandOf lower = bandOf upper := by
  have hER : (0 : Int) ≤ EVAL_ROUGHNESS := by decide
  exact bandOf_eq_of_slack hgap hl hu (by omega) (by omega)

end Sunfish
