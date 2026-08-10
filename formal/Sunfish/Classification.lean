/-
Milestone 3: EVENTUAL CLASSIFICATION -- the trichotomy for the game
sunfish actually plays -- and the frontier-tail variant that makes the
honesty arm unconditional.

# Part A: `eventual_classification`

The composition of everything the liveness arc has proven, stated once:
for a legally-reached position (`hasKingCapture` false, king on the
board), the declared value `nullValueD2` -- the function the shipped
search provably brackets (`bound_null_spec` / `boundKCX_null_spec`,
layer 1, no chess premise) -- eventually classifies the position:

* **win**: a forced mate in ≤ k plies puts the value in the mate band
  at every depth `D ≥ k + 1` (`forcedMate_complete`);
* **loss**: the mated dual, at every depth `D ≥ k + 2`
  (`forcedlyMated_negamaxD2` through the layer-2 transfer);
* **neither**: with no forced mate for either side at ANY horizon, the
  value stays strictly inside the band at EVERY depth -- the
  contrapositives of no-false-mates (`forcedMate_of_nullValueD2`) and
  its dual (`forcedlyMated_of_nullValueD2`).

**The premise ledger** (who pays for which arm):

| premise | kind | arm |
|---|---|---|
| `ValFloor G 192` | fidelity (tables) | all three (spine + inversions) |
| `NoZugzwang` | chess, layer 2 | win + loss (the transfer to the declared function; the finding side) |
| `EvalQuiet` | fidelity (tables) | neither (depth-0 and frontier inversions) |
| `NoMaskedMobility` | chess, layer 2 | neither (the honesty side; required per `CexF`) |
| `Bounded`, `KillerLegal`, `KingCaptureValHigh`, `CaptureFirst` | fidelity / theorem | only the probe/driver corollaries, through layer 1 |

The two recorded discharge options, NEITHER implemented in the engine:

* for `NoZugzwang` (finding side): the depth-decaying null guard
  (`abs(score) < 500 - 10*depth`) -- switches the pass off at large
  remaining depth and makes completeness unconditional at `D ≥ k + 52`;
  recorded in formal/README.md, deliberately not implemented (Thomas's
  decision: give the layer-2 assumption the exercise instead).
* for `NoMaskedMobility` (honesty side): the FRONTIER TAIL SEARCH --
  Part B of this file.  Verify-on-suspicion applied to the QS filter:
  when a mate-band conclusion is forming at a depth where filtering was
  active, unfilter.  For the t-variant defined below the honesty arm is
  a THEOREM under fidelity premises alone -- `NoMaskedMobility` is not
  assumed, its role is discharged by construction -- and `CexF` becomes
  a positive test.  Proof-first: NO code change shipped; the t-model
  sits alongside like `boundKCX''` did, so the decision can be made
  with theorems (and later an Elo screen) in hand.

**What "draw" means here, honestly.**  The game being classified is the
game sunfish plays: chess WITHOUT draw rules.  "Neither" =
no-forced-mate for either side in the ruleless game.  FIDE draws --
50-move, threefold repetition, insufficient material -- are NOT
detected as 0: in K+B vs K the defender is never mated and never
mates, so the value converges to the sub-band arm (a small material
score), not to 0.  The classification says the report never enters the
mate band there -- it does not say the engine calls it a draw.

Axioms, checked with `#print axioms`: `classification_exclusive` (pure
spine composition) is `propext`/`Quot.sound` only;
`eventual_classification`, the two `iff`s and the driver corollaries
additionally inherit `Classical.choice` -- through the landed
no-false-mates theorems (`forcedMate_of_nullValueD2` carries it since
milestone 2C) and, for the driver corollaries, through layer 1
(`bound_null_spec` / `boundKCX_null_spec`) -- same sets as milestone 2,
nothing new.
-/

import Sunfish.Liveness

namespace Sunfish

/-! ## Part A: the trichotomy -/

/-- **Eventual classification** -- the trichotomy theorem for the game
sunfish plays, at a legally-reached position with the king on the
board.  One statement, three arms:

* **win**: `ForcedMate G k p` puts the declared value in the mate band
  at every `D ≥ k + 1`;
* **loss**: `ForcedlyMated G k p` puts it at or below `-MATE_LOWER` at
  every `D ≥ k + 2`;
* **neither**: no forced mate at any horizon, for either side, keeps
  it strictly inside `(-MATE_LOWER, MATE_LOWER)` at EVERY depth -- no
  floor needed: depth 0 is the quiet static eval, and every deeper
  band value would produce the forbidden witness.

Premise ledger in the module comment above.  The three arms are
mutually exclusive (`classification_exclusive`); the driver sees the
classification through `classification_visible` below and the landed
probe corollaries (`forcedMate_probe_failsHigh`, `mate_report_honest`,
`mated_report_honest`, and their `_kcx` twins). -/
theorem eventual_classification (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    (∀ k, ForcedMate G k p →
      ∀ D, k + 1 ≤ D → MATE_LOWER ≤ nullValueD2 G guard D p) ∧
    (∀ k, ForcedlyMated G k p →
      ∀ D, k + 2 ≤ D → nullValueD2 G guard D p ≤ -MATE_LOWER) ∧
    ((∀ k, ¬ ForcedMate G k p) → (∀ k, ¬ ForcedlyMated G k p) →
      ∀ D, -MATE_LOWER < nullValueD2 G guard D p ∧
        nullValueD2 G guard D p < MATE_LOWER) := by
  have hML : MATE_LOWER = 47923 := rfl
  refine ⟨?_, ?_, ?_⟩
  · intro k hFM D hD
    exact forcedMate_complete G guard hF hZ hFM D hD
  · intro k hFMd D hD
    rw [nullValue_eq_realValue_of_noZugzwang G guard hZ D p]
    exact forcedlyMated_negamaxD2 G hF hcapf hFMd D hD
  · intro hnFM hnFMd D
    constructor
    · by_cases hlo : nullValueD2 G guard D p ≤ -MATE_LOWER
      · exfalso
        cases D with
        | zero =>
          have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by
            simp [hcapf]
          have hval : nullValueD2 G guard 0 p = G.eval p := by
            simp only [nullValueD2]
            rw [if_neg hkg, if_neg hcap]
          rw [hval] at hlo
          exact hkg hlo
        | succ D' =>
          exact hnFMd D'
            (forcedlyMated_of_nullValueD2 G guard hF hQ hNM D' p hcapf hkg hlo)
      · omega
    · by_cases hhi : MATE_LOWER ≤ nullValueD2 G guard D p
      · exact absurd (forcedMate_of_nullValueD2 G guard hF hQ hNM D p hcapf hhi)
          (hnFM D)
      · omega

/-- The win and loss arms are mutually exclusive -- directly on the
real-move spine, no zugzwang or honesty premise: at a common deep
horizon the two bounds contradict. -/
theorem classification_exclusive (G : QSGame) (hF : ValFloor G 192)
    {k k' : Nat} {p : G.Pos}
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hFM : ForcedMate G k p) (hFMd : ForcedlyMated G k' p) : False := by
  have hML : MATE_LOWER = 47923 := rfl
  have h1 := forcedMate_negamaxD2 G hF hFM (k + k' + 2) (by omega)
  have h2 := forcedlyMated_negamaxD2 G hF hcapf hFMd (k + k' + 2) (by omega)
  omega

/-- The "eventual" reading, win side: a forced mate exists at SOME
horizon iff the declared value reaches the mate band at SOME depth --
completeness one way, no-false-mates the other. -/
theorem eventual_mate_iff (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (p : G.Pos) (hcapf : hasKingCapture G.toNullGame.toGame p = false) :
    (∃ k, ForcedMate G k p) ↔ ∃ D, MATE_LOWER ≤ nullValueD2 G guard D p := by
  constructor
  · rintro ⟨k, hFM⟩
    exact ⟨k + 1, forcedMate_complete G guard hF hZ hFM (k + 1) (Nat.le_refl _)⟩
  · rintro ⟨D, hD⟩
    exact ⟨D, forcedMate_of_nullValueD2 G guard hF hQ hNM D p hcapf hD⟩

/-- The "eventual" reading, loss side.  The king-on-the-board premise
is genuinely needed: a kingless root sits at `-MATE_UPPER` forever
without being "mated" in the spec's sense. -/
theorem eventual_mated_iff (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (p : G.Pos) (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    (∃ k, ForcedlyMated G k p) ↔ ∃ D, nullValueD2 G guard D p ≤ -MATE_LOWER := by
  constructor
  · rintro ⟨k, hFMd⟩
    refine ⟨k + 2, ?_⟩
    rw [nullValue_eq_realValue_of_noZugzwang G guard hZ]
    exact forcedlyMated_negamaxD2 G hF hcapf hFMd (k + 2) (Nat.le_refl _)
  · rintro ⟨D, hD⟩
    cases D with
    | zero =>
      exfalso
      have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by
        simp [hcapf]
      have hval : nullValueD2 G guard 0 p = G.eval p := by
        simp only [nullValueD2]
        rw [if_neg hkg, if_neg hcap]
      rw [hval] at hD
      exact hkg hD
    | succ D' =>
      exact ⟨D', forcedlyMated_of_nullValueD2 G guard hF hQ hNM D' p hcapf hkg hD⟩

/-! ## The classification is visible at the driver

Composing the trichotomy with the driver package (milestone 2 A:
`search_inner_loop_converges`): after the 15-probe budget at depth `D`,
the converged bracket `[lower, upper]` REPORTS the classification --
the bracket lands against the band exactly as the arms dictate, up to
the driver's own `EVAL_ROUGHNESS` slop on the certified side:

* **win** (mate within the depth): `upper` is in the band and `lower`
  is at worst `EVAL_ROUGHNESS` below it;
* **loss**: dually, `lower ≤ -MATE_LOWER` and `upper` at worst
  `EVAL_ROUGHNESS` above;
* **neither**: BOTH ends stay strictly off the band edges -- the
  bracket can never certify a mate that is not there (this direction
  has no slop: it is no-false-mates seen through the invariant).

The one-ply and roughness asymmetries are honest artifacts of the "at
most k" spec indexing and of `EVAL_ROUGHNESS`; the slop-free per-probe
statements are the landed `forcedMate_probe_failsHigh` /
`mate_report_honest` / `mated_report_honest` (+ `_kcx`). -/

/-- **The driver sees the trichotomy** (reference consumer).  Premises
are the trichotomy's plus layer 1's (`Bounded`, `KillerLegal`); axioms
include `Classical.choice` through `bound_null_spec`, as everywhere
downstream of layer 1. -/
theorem classification_visible (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hK : KillerLegal G kill)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (D : Nat) (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (carried : Int) (hc1 : -MATE_UPPER < carried) (hc2 : carried ≤ MATE_UPPER) :
    ((∃ k, k + 1 ≤ D ∧ ForcedMate G k p) →
      MATE_LOWER - EVAL_ROUGHNESS
          ≤ (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).lower ∧
        MATE_LOWER
          ≤ (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).upper) ∧
    ((∃ k, k + 2 ≤ D ∧ ForcedlyMated G k p) →
      (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).lower
          ≤ -MATE_LOWER ∧
        (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).upper
          ≤ -MATE_LOWER + EVAL_ROUGHNESS) ∧
    ((∀ k, ¬ ForcedMate G k p) → (∀ k, ¬ ForcedlyMated G k p) →
      (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).lower
          < MATE_LOWER ∧
        -MATE_LOWER
          < (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).upper) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hE : EVAL_ROUGHNESS = 15 := rfl
  obtain ⟨hw, hlow, hup⟩ :=
    search_inner_loop_converges G guard kill hB hK D p carried hc1 hc2
  obtain ⟨harmW, harmL, harmN⟩ :=
    eventual_classification G guard hF hQ hNM hZ p hcapf hkg
  refine ⟨?_, ?_, ?_⟩
  · rintro ⟨k, hkD, hFM⟩
    have hband := harmW k hFM D hkD
    omega
  · rintro ⟨k, hkD, hFMd⟩
    have hband := harmL k hFMd D hkD
    omega
  · intro hnFM hnFMd
    have hband := harmN hnFM hnFMd D
    omega

/-- **The driver sees the trichotomy** (production consumer,
`boundKCX`), through `boundKCX_null_spec`'s premises. -/
theorem classification_visible_kcx (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G)
    (hK : KillerLegal G kill)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) (hZ : NoZugzwang G guard)
    (D : Nat) (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (carried : Int) (hc1 : -MATE_UPPER < carried) (hc2 : carried ≤ MATE_UPPER) :
    ((∃ k, k + 1 ≤ D ∧ ForcedMate G k p) →
      MATE_LOWER - EVAL_ROUGHNESS
          ≤ (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).lower ∧
        MATE_LOWER
          ≤ (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).upper) ∧
    ((∃ k, k + 2 ≤ D ∧ ForcedlyMated G k p) →
      (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).lower
          ≤ -MATE_LOWER ∧
        (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).upper
          ≤ -MATE_LOWER + EVAL_ROUGHNESS) ∧
    ((∀ k, ¬ ForcedMate G k p) → (∀ k, ¬ ForcedlyMated G k p) →
      (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).lower
          < MATE_LOWER ∧
        -MATE_LOWER
          < (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).upper) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hE : EVAL_ROUGHNESS = 15 := rfl
  obtain ⟨hw, hlow, hup⟩ :=
    search_inner_loop_converges_kcx G guard kill hB hV hCF hK D p carried hc1 hc2
  obtain ⟨harmW, harmL, harmN⟩ :=
    eventual_classification G guard hF hQ hNM hZ p hcapf hkg
  refine ⟨?_, ?_, ?_⟩
  · rintro ⟨k, hkD, hFM⟩
    have hband := harmW k hFM D hkD
    omega
  · rintro ⟨k, hkD, hFMd⟩
    have hband := harmL k hFMd D hkD
    omega
  · intro hnFM hnFMd
    have hband := harmN hnFM hnFMd D
    omega

end Sunfish
