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

Axioms, checked with `#print axioms`: every VALUE-level theorem here
(`eventual_classification`, `classification_exclusive`, the two
`iff`s, and the whole t-variant of Part B) is `propext`/`Quot.sound`
only.  A finding en route: milestone 2C's no-false-mates theorems had
been carrying `Classical.choice` from a single cosmetic step -- an
`omega` closing a NON-arithmetic goal (`ForcedMate ...`) from
contradictory hypotheses routes through `Classical.byContradiction` --
replaced by `absurd` in this milestone, so the entire value level is
now choice-free.  Only the probe/driver corollaries inherit
`Classical.choice`, genuinely, through layer 1 (`bound_null_spec` /
`boundKCX_null_spec`).
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

/-! ## Part B: the frontier-tail variant (`negamaxD2t` / `nullValueD2t`)

**The design** (Thomas: "don't set QS_A=232, there must be another
way" -- this is the other way).  Verify-on-suspicion applied to the QS
val-filter itself: when a mate-band conclusion is forming at a depth
where filtering was active, UNFILTER -- fold the filtered-out tail
too.  Proof-first: NO code change is shipped; the t-model sits
alongside the shipped model exactly as `boundKCX''` did, so the
decision can be made with theorems (and later an Elo screen) in hand.

**The trigger shape, adjusted and RECORDED.**  The task's first-draft
trigger was "depth-1 fold ≤ -MATE_LOWER".  The shape proven here is:
**every ADMITTED move is illegal** (`allAdmittedIllegalB` -- the
QS-filtered move list, scanned with the same board predicate
`king_capture()` the correction already uses), evaluated at every
interior depth.  Three reasons, all load-bearing:

* *Search-observability.*  The fold-value trigger is not observable
  through the engine's `best`: the FUTILITY species prices an illegal
  admitted move at its child stand-pat (`pos.score + val`), which can
  hold `best` strictly above `-MATE_LOWER` while every true admitted
  contribution is sub-band -- the search would fail to unfilter
  exactly where the declared value did, an unsoundness on the fail-low
  side.  The admitted-legality scan is species-blind: futility cannot
  mask it.
* *It is the premise's own shape.*  `NoMaskedMobility`'s hypothesis is
  literally "every admitted move is illegal"; triggering on that
  predicate is verify-on-suspicion applied to the premise itself --
  the search checks the assumption's antecedent and, where it fires,
  searches the moves whose absence the premise asserted.
* *It implements the intended trigger where it matters.*
  `trigger_shapes_agree_frontier` below: at the frontier (defender
  remaining depth 1, the only depth the shipped tables let the filter
  bite -- `ValFloor` vs `val_lower 2`), the admitted-scan trigger and
  the fold-value trigger are PROVABLY equivalent under `EvalQuiet`.
  At depth ≥ 2 under `ValFloor` nothing is filtered, so both triggers
  change nothing.

**What is proven** (all sorry-free, value level):

* `forcedMate_of_nullValueD2t` / `forcedlyMated_of_nullValueD2t` --
  **no-false-mates for the t-variant needs NO chess premise**:
  `ValFloor` + `EvalQuiet` (fidelity, tables) and root legality only.
  `NoMaskedMobility` does not appear -- the CexF channel closes by
  construction: any legal escape enters the fold exactly when its
  absence would fabricate a band value.  The frontier lemmas
  (`masked_frontier_escape_t`, `admitted_frontier_escape_t`) replace
  `frontier_filtered_escape` premise-free.
* `forcedMate_negamaxD2t` / `forcedlyMated_negamaxD2t` -- the
  completeness spine SURVIVES, same `ValFloor` premise, same `k + 1` /
  `k + 2` depth bounds: the tail only ADDS defender options, and on
  the defender side extra options only lower the loss claim toward
  honesty (the attacker's witness is admitted, so the mask provably
  never fires at attacker nodes).
* `nullValueD2t_eq_realValue_of_noZugzwangT` + `forcedMate_completeT`
  -- the two-layer structure transfers: under `NoZugzwangT` (the same
  statement, aimed at the t-functions) the declared t-value collapses
  onto the real-move t-value.
* `eventual_classification_t` -- the trichotomy again, with the
  honesty arm now paid for by fidelity alone.
* `cexF_t_positive` -- **CexF is now a positive test**: on the very
  countermodel where `nullValueD2` fabricates the depth-2
  `MATE_UPPER` (`cexF_bandValue`) and `NoMaskedMobility` was proven
  required, the t-variant computes the honest draw 0 at depth 2 (and
  3).

**What the code change would be** (for the decision, not shipped): in
`bound`'s move loop the QS break stays; at the post-loop correction
site -- where `not live` already triggers the full `king_capture()`
scan -- when the scan finds legal moves but every ADMITTED move was
illegal (equivalently: every legal move the scan found sits below
`val_lower`), search those found legal moves at `depth - 1` with the
same window and fold the yields into `best`/`live` before the
store/return.  Cost: only at not-live fail-low nodes, where the scan
already runs.  The tail yields are REAL yields -- they set `live`, so
a tail escape disproves terminality exactly as the scan does, and the
`termFix` interaction stays coherent (the correction still fires only
when the scan certifies NO legal move anywhere).  A tail fail-high is
a real fail-high: it may store `tp_move`, and the `storedMoveLegal`
certificate applies to it verbatim.  The engine-side residue shared
with everything downstream of the QS leaf: depth-0 reports are
fail-soft, not exact, so the depth-1 alignment argument leans on the
same QS-as-eval leaf abstraction (and its known sentinel-masking
channel) as every layer-1 theorem -- nothing new is assumed, and
nothing about this variant widens that channel.

Axioms: the t-value theorems are `propext`/`Quot.sound` only (checked;
nothing here routes through layer 1, and the omega-vs-`absurd` point
from the module comment is applied here too). -/

/-- The admitted-move legality scan: every move that SURVIVES the QS
val-filter leaves the mover's king capturable.  `NoMaskedMobility`'s
antecedent as a computable, `(pos, depth)`-determined, gamma-free
predicate -- and the t-variant's unfilter trigger. -/
def allAdmittedIllegalB (G : QSGame) (d : Nat) (p : G.Pos) : Bool :=
  (movesAbove G (val_lower d) p).all (fun m => hasKingCapture G.toNullGame.toGame m)

theorem allAdmittedIllegalB_true_iff {G : QSGame} {d : Nat} {p : G.Pos} :
    allAdmittedIllegalB G d p = true
      ↔ ∀ m ∈ movesAbove G (val_lower d) p,
          hasKingCapture G.toNullGame.toGame m = true := by
  simp [allAdmittedIllegalB, List.all_eq_true]

theorem allAdmittedIllegalB_false_of_legal {G : QSGame} {d : Nat} {p m : G.Pos}
    (hm : m ∈ movesAbove G (val_lower d) p)
    (hleg : hasKingCapture G.toNullGame.toGame m = false) :
    allAdmittedIllegalB G d p = false := by
  cases h : allAdmittedIllegalB G d p with
  | false => rfl
  | true =>
    have := allAdmittedIllegalB_true_iff.mp h m hm
    rw [hleg] at this
    exact Bool.noConfusion this

/-- A failed `all` names its witness (list form, constructive). -/
theorem exists_legal_of_not_all {G : QSGame} :
    ∀ l : List G.Pos,
      l.all (fun m => hasKingCapture G.toNullGame.toGame m) = false →
      ∃ m ∈ l, hasKingCapture G.toNullGame.toGame m = false := by
  intro l
  induction l with
  | nil => intro h; exact Bool.noConfusion h
  | cons a l ih =>
    intro h
    rw [List.all_cons] at h
    cases hc : hasKingCapture G.toNullGame.toGame a with
    | false => exact ⟨a, List.mem_cons_self a l, hc⟩
    | true =>
      rw [hc, Bool.true_and] at h
      obtain ⟨m, hm, hmc⟩ := ih h
      exact ⟨m, List.mem_cons_of_mem a hm, hmc⟩

/-- A false scan names a legal admitted move. -/
theorem exists_legal_admitted {G : QSGame} {d : Nat} {p : G.Pos}
    (h : allAdmittedIllegalB G d p = false) :
    ∃ m ∈ movesAbove G (val_lower d) p,
      hasKingCapture G.toNullGame.toGame m = false := by
  simp only [allAdmittedIllegalB] at h
  exact exists_legal_of_not_all _ h

/-- A fold over a filtered list never exceeds the fold over the full
list (same initial accumulator) -- the tail only ADDS options. -/
theorem foldMax_filter_le {α : Type _} (w : α → Int) (f : α → Bool)
    (l : List α) (init : Int) :
    foldMax w (l.filter f) init ≤ foldMax w l init := by
  rw [foldMax_filter_split w f l init]
  omega

/-- **The real-move t-value**: `negamaxD2` with the unfilter trigger --
when every admitted move is illegal (and the position is not
oracle-terminal), the fold runs over the FULL move list instead of the
filtered one.  `(pos, depth)`-determined, gamma-free. -/
def negamaxD2t (G : QSGame) : Nat → G.Pos → Int
  | 0, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if allIllegalB G p = true then terminalValue G p
    else
      foldMax (fun m => -(negamaxD2t G d m))
        (if allAdmittedIllegalB G (d + 1) p = true then G.moves p
         else movesAbove G (val_lower (d + 1)) p) LOSS

/-- **The declared (null-inclusive) t-value**: `nullValueD2` with the
same unfilter trigger; the pass term (the fold's initial accumulator,
with the A1 suppression baked in) is unchanged.  Still
`(pos, depth)`-determined and window-free. -/
def nullValueD2t (G : QSGame) (guard : G.Pos → Bool) : Nat → G.Pos → Int
  | 0, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if allIllegalB G p = true then terminalValue G p
    else
      foldMax (fun m => -(nullValueD2t G guard d m))
        (if allAdmittedIllegalB G (d + 1) p = true then G.moves p
         else movesAbove G (val_lower (d + 1)) p)
        (if guard p = true ∧ 2 < d + 1 then
          (if -(nullValueD2t G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER then
            max LOSS (-(nullValueD2t G guard (d + 1 - 3) (G.pass p)))
          else LOSS)
        else LOSS)
termination_by d _ => d
decreasing_by all_goals omega

/-- The t-pass term, named (same shape as `nullTermD2`). -/
def nullTermD2t (G : QSGame) (guard : G.Pos → Bool) (d : Nat) (p : G.Pos) : Int :=
  if guard p = true ∧ 2 < d + 1 then
    (if -(nullValueD2t G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER then
      max LOSS (-(nullValueD2t G guard (d + 1 - 3) (G.pass p)))
    else LOSS)
  else LOSS

/-! ### Branch lemmas -/

theorem negamaxD2t_kingGone (G : QSGame) (d : Nat) (p : G.Pos)
    (h : G.eval p ≤ -MATE_LOWER) : negamaxD2t G d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [negamaxD2t]; rw [if_pos h]
  | succ d => simp only [negamaxD2t]; rw [if_pos h]

theorem negamaxD2t_of_capture (G : QSGame) (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    negamaxD2t G d p = MATE_UPPER := by
  cases d with
  | zero => simp only [negamaxD2t]; rw [if_neg hkg, if_pos hcap]
  | succ d => simp only [negamaxD2t]; rw [if_neg hkg, if_pos hcap]

theorem negamaxD2t_of_allIllegal (G : QSGame) (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true) :
    negamaxD2t G (d + 1) p = terminalValue G p := by
  simp only [negamaxD2t]
  rw [if_neg hkg, if_neg hcap, if_pos hai]

theorem negamaxD2t_of_masked (G : QSGame) (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hmask : allAdmittedIllegalB G (d + 1) p = true) :
    negamaxD2t G (d + 1) p
      = foldMax (fun m => -(negamaxD2t G d m)) (G.moves p) LOSS := by
  simp only [negamaxD2t]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_pos hmask]

theorem negamaxD2t_of_fold (G : QSGame) (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hmask : allAdmittedIllegalB G (d + 1) p = false) :
    negamaxD2t G (d + 1) p
      = foldMax (fun m => -(negamaxD2t G d m))
          (movesAbove G (val_lower (d + 1)) p) LOSS := by
  simp only [negamaxD2t]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_neg (by simp [hmask])]

theorem nullValueD2t_kingGone (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (h : G.eval p ≤ -MATE_LOWER) :
    nullValueD2t G guard d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [nullValueD2t]; rw [if_pos h]
  | succ d => simp only [nullValueD2t]; rw [if_pos h]

theorem nullValueD2t_of_capture (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    nullValueD2t G guard d p = MATE_UPPER := by
  cases d with
  | zero => simp only [nullValueD2t]; rw [if_neg hkg, if_pos hcap]
  | succ d => simp only [nullValueD2t]; rw [if_neg hkg, if_pos hcap]

theorem nullValueD2t_of_allIllegal (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true) :
    nullValueD2t G guard (d + 1) p = terminalValue G p := by
  simp only [nullValueD2t]
  rw [if_neg hkg, if_neg hcap, if_pos hai]

theorem nullValueD2t_of_masked (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hmask : allAdmittedIllegalB G (d + 1) p = true) :
    nullValueD2t G guard (d + 1) p
      = foldMax (fun m => -(nullValueD2t G guard d m)) (G.moves p)
          (nullTermD2t G guard d p) := by
  simp only [nullValueD2t]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_pos hmask]
  rfl

theorem nullValueD2t_of_fold (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hmask : allAdmittedIllegalB G (d + 1) p = false) :
    nullValueD2t G guard (d + 1) p
      = foldMax (fun m => -(nullValueD2t G guard d m))
          (movesAbove G (val_lower (d + 1)) p) (nullTermD2t G guard d p) := by
  simp only [nullValueD2t]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_neg (by simp [hmask])]
  rfl

theorem nullTermD2t_ge_LOSS (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) : LOSS ≤ nullTermD2t G guard d p := by
  simp only [nullTermD2t]
  by_cases h1 : guard p = true ∧ 2 < d + 1
  · rw [if_pos h1]
    by_cases h2 : -(nullValueD2t G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
    · rw [if_pos h2]; omega
    · rw [if_neg h2]; omega
  · rw [if_neg h1]; omega

/-- The A1 suppression is baked into the t-pass term too: it can never
reach the mate band, so a band-value fold still names a real move. -/
theorem nullTermD2t_lt_ML (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) : nullTermD2t G guard d p < MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  simp only [nullTermD2t]
  by_cases h1 : guard p = true ∧ 2 < d + 1
  · rw [if_pos h1]
    by_cases h2 : -(nullValueD2t G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
    · rw [if_pos h2]
      omega
    · rw [if_neg h2]
      omega
  · rw [if_neg h1]
    omega

/-- **The two trigger shapes agree at the frontier** -- the record that
the adjusted (admitted-scan) trigger implements the intended
(fold-value) one exactly where the shipped tables let the filter bite:
at defender remaining depth 1, under `EvalQuiet`, "every admitted move
illegal" and "the admitted depth-1 fold sits at or below `-MATE_LOWER`"
are equivalent.  (At depth ≥ 2 under `ValFloor` nothing is filtered
and both triggers are inert.) -/
theorem trigger_shapes_agree_frontier (G : QSGame) (guard : G.Pos → Bool)
    (hQ : EvalQuiet G.toNullGame.toGame)
    (p : G.Pos)
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true)) :
    allAdmittedIllegalB G 1 p = true
      ↔ foldMax (fun m => -(nullValueD2t G guard 0 m))
          (movesAbove G (val_lower 1) p) LOSS ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  constructor
  · intro h
    refine foldMax_le _ _ _ (fun m hm => ?_) (by omega)
    have hcm := allAdmittedIllegalB_true_iff.mp h m hm
    have hmm := movesAbove_subset G _ p m hm
    have hmkg : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
      hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hmm, hh⟩)
    show -(nullValueD2t G guard 0 m) ≤ -MATE_LOWER
    rw [nullValueD2t_of_capture G guard 0 m hmkg hcm]
    omega
  · intro h
    rw [allAdmittedIllegalB_true_iff]
    intro m hm
    have hle : -(nullValueD2t G guard 0 m)
        ≤ foldMax (fun m => -(nullValueD2t G guard 0 m))
            (movesAbove G (val_lower 1) p) LOSS :=
      foldMax_le_of_mem _ _ _ m hm
    have hmm := movesAbove_subset G _ p m hm
    have hmkg : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
      hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hmm, hh⟩)
    cases hcm : hasKingCapture G.toNullGame.toGame m with
    | true => rfl
    | false =>
      exfalso
      have hval : nullValueD2t G guard 0 m = G.eval m := by
        simp only [nullValueD2t]
        rw [if_neg hmkg, if_neg (by simp [hcm])]
      rw [hval] at hle
      have := hQ m hmkg
      omega

/-! ### The completeness spine survives -/

/-- Any-branch defender bound: whichever list the trigger selects, its
members come from the full move list, so a uniform member bound closes
the fold.  The tail can only LOWER a defender's loss claim toward
honesty -- this is that sentence as a lemma. -/
theorem negamaxD2t_defender_le (G : QSGame) {d : Nat} {m : G.Pos}
    (hkgm : ¬ (G.eval m ≤ -MATE_LOWER))
    (hcapm : ¬ (hasKingCapture G.toNullGame.toGame m = true))
    (hnt : allIllegalB G m = false)
    (hall : ∀ m' ∈ G.moves m, -(negamaxD2t G d m') ≤ -MATE_LOWER) :
    negamaxD2t G (d + 1) m ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  by_cases hmask : allAdmittedIllegalB G (d + 1) m = true
  · rw [negamaxD2t_of_masked G d m hkgm hcapm hnt hmask]
    exact foldMax_le _ _ _ hall (by omega)
  · have hmask' : allAdmittedIllegalB G (d + 1) m = false := by
      cases h : allAdmittedIllegalB G (d + 1) m with
      | false => rfl
      | true => exact absurd h hmask
    rw [negamaxD2t_of_fold G d m hkgm hcapm hnt hmask']
    exact foldMax_le _ _ _
      (fun m' hm' => hall m' (movesAbove_subset G _ m m' hm')) (by omega)

/-- **Mate-in-k completeness for the t-variant, real-move layer** --
same premise (`ValFloor G 192`), same bound (`D ≥ k + 1`) as
`forcedMate_negamaxD2`.  The attacker's witness is admitted, so the
unfilter trigger provably never fires at attacker nodes; at defender
nodes the tail only adds options that the `ForcedMate` derivation
already refutes. -/
theorem forcedMate_negamaxD2t (G : QSGame) (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, k + 1 ≤ D → MATE_LOWER ≤ negamaxD2t G D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  induction hFM with
  | @mate k p m hkg hm hleg hmate =>
    intro D hD
    cases D with
    | zero => omega
    | succ d =>
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [negamaxD2t_of_capture G (d + 1) p hkg hcap]; omega
      · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
        have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
        have hmask : allAdmittedIllegalB G (d + 1) p = false :=
          allAdmittedIllegalB_false_of_legal hmem hleg
        rw [negamaxD2t_of_fold G d p hkg hcap hai hmask]
        have hchild : negamaxD2t G d m ≤ -MATE_LOWER := by
          cases d with
          | zero => omega
          | succ d' =>
            by_cases hkgm : G.eval m ≤ -MATE_LOWER
            · rw [negamaxD2t_kingGone G (d' + 1) m hkgm]; omega
            · rw [negamaxD2t_of_allIllegal G d' m hkgm (by simp [hleg]) hmate.1]
              simp only [terminalValue]
              rw [if_pos hmate.2]
              omega
        have hfold : -(negamaxD2t G d m)
            ≤ foldMax (fun x => -(negamaxD2t G d x))
                (movesAbove G (val_lower (d + 1)) p) LOSS :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega
  | @step k p m hkg hm hleg hnt _hreply ih =>
    intro D hD
    cases D with
    | zero => omega
    | succ d =>
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [negamaxD2t_of_capture G (d + 1) p hkg hcap]; omega
      · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
        have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
        have hmask : allAdmittedIllegalB G (d + 1) p = false :=
          allAdmittedIllegalB_false_of_legal hmem hleg
        rw [negamaxD2t_of_fold G d p hkg hcap hai hmask]
        have hchild : negamaxD2t G d m ≤ -MATE_LOWER := by
          cases d with
          | zero => omega
          | succ d' =>
            by_cases hkgm : G.eval m ≤ -MATE_LOWER
            · rw [negamaxD2t_kingGone G (d' + 1) m hkgm]; omega
            · refine negamaxD2t_defender_le G hkgm (by simp [hleg]) hnt
                (fun m' hm' => ?_)
              have hkgm' : ¬ (G.eval m' ≤ -MATE_LOWER) := by
                intro hle
                have hc : hasKingCapture G.toNullGame.toGame m = true :=
                  (hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm', hle⟩
                rw [hleg] at hc
                exact Bool.noConfusion hc
              cases hcm : hasKingCapture G.toNullGame.toGame m' with
              | true =>
                rw [negamaxD2t_of_capture G d' m' hkgm' hcm]; omega
              | false =>
                have := ih m' hm' hcm d' (by omega)
                omega
        have hfold : -(negamaxD2t G d m)
            ≤ foldMax (fun x => -(negamaxD2t G d x))
                (movesAbove G (val_lower (d + 1)) p) LOSS :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega

/-- The mated-side dual for the t-variant: same premise, same
`D ≥ k + 2` bound as `forcedlyMated_negamaxD2`. -/
theorem forcedlyMated_negamaxD2t (G : QSGame) (hF : ValFloor G 192)
    {k : Nat} {q : G.Pos}
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hFL : ForcedlyMated G k q) :
    ∀ D : Nat, k + 2 ≤ D → negamaxD2t G D q ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro D hD
  cases D with
  | zero => omega
  | succ d =>
    by_cases hkg : G.eval q ≤ -MATE_LOWER
    · rw [negamaxD2t_kingGone G (d + 1) q hkg]; omega
    · have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by
        simp [hcapq]
      cases hFL with
      | inl hcm =>
        rw [negamaxD2t_of_allIllegal G d q hkg hcapq' hcm.1]
        simp only [terminalValue]
        rw [if_pos hcm.2]
        omega
      | inr h =>
        obtain ⟨hai, hall⟩ := h
        refine negamaxD2t_defender_le G hkg hcapq' hai (fun m hm => ?_)
        cases hcm : hasKingCapture G.toNullGame.toGame m with
        | true =>
          have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := by
            intro hle
            have hc : hasKingCapture G.toNullGame.toGame q = true :=
              (hasKingCapture_iff G.toNullGame.toGame q).mpr ⟨m, hm, hle⟩
            rw [hcapq] at hc
            exact Bool.noConfusion hc
          rw [negamaxD2t_of_capture G d m hkgm hcm]; omega
        | false =>
          have := forcedMate_negamaxD2t G hF (hall m hm hcm) d (by omega)
          omega

/-! ### The two-layer transfer survives -/

/-- `NoZugzwang`, aimed at the t-functions: the raw pass term never
strictly beats the ADMITTED real-move fold.  (Beating the admitted
fold is enough: the full fold only grows, `foldMax_filter_le`.) -/
def NoZugzwangT (G : QSGame) (guard : G.Pos → Bool) : Prop :=
  ∀ (d : Nat) (p : G.Pos),
    ¬ (G.eval p ≤ -MATE_LOWER) →
    ¬ (hasKingCapture G.toNullGame.toGame p = true) →
    allIllegalB G p = false → guard p = true → 2 < d + 1 →
    -(nullValueD2t G guard (d + 1 - 3) (G.pass p))
      ≤ foldMax (fun m => -(nullValueD2t G guard d m))
          (movesAbove G (val_lower (d + 1)) p) LOSS

/-- **Layer 2 for the t-variant**: under `NoZugzwangT` the declared
t-value collapses onto the real-move t-value -- the mirror of
`nullValue_eq_realValue_of_noZugzwang`, with the unfilter trigger
handled by the fact that it is position-intrinsic (the SAME branch is
taken on both sides) and the pass term dominated through the admitted
fold. -/
theorem nullValueD2t_eq_realValue_of_noZugzwangT (G : QSGame)
    (guard : G.Pos → Bool) (hZ : NoZugzwangT G guard) :
    ∀ (d : Nat) (p : G.Pos), nullValueD2t G guard d p = negamaxD2t G d p := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p
    cases d with
    | zero =>
      simp only [nullValueD2t, negamaxD2t]
    | succ d =>
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [nullValueD2t_kingGone G guard (d + 1) p hkg,
          negamaxD2t_kingGone G (d + 1) p hkg]
      · by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [nullValueD2t_of_capture G guard (d + 1) p hkg hcap,
            negamaxD2t_of_capture G (d + 1) p hkg hcap]
        · cases hai : allIllegalB G p with
          | true =>
            rw [nullValueD2t_of_allIllegal G guard d p hkg hcap hai,
              negamaxD2t_of_allIllegal G d p hkg hcap hai]
          | false =>
            have hTle : nullTermD2t G guard d p
                ≤ foldMax (fun m => -(nullValueD2t G guard d m))
                    (movesAbove G (val_lower (d + 1)) p) LOSS := by
              have hfl := foldMax_ge_init (fun m => -(nullValueD2t G guard d m))
                (movesAbove G (val_lower (d + 1)) p) LOSS
              by_cases hen : guard p = true ∧ 2 < d + 1
              · by_cases hml : -(nullValueD2t G guard (d + 1 - 3) (G.pass p))
                    < MATE_LOWER
                · have hT' : nullTermD2t G guard d p
                      = max LOSS (-(nullValueD2t G guard (d + 1 - 3) (G.pass p))) := by
                    simp only [nullTermD2t]
                    rw [if_pos hen, if_pos hml]
                  have hZ' := hZ d p hkg hcap hai hen.1 hen.2
                  rw [hT']
                  omega
                · have hT' : nullTermD2t G guard d p = LOSS := by
                    simp only [nullTermD2t]
                    rw [if_pos hen, if_neg hml]
                  rw [hT']
                  omega
              · have hT' : nullTermD2t G guard d p = LOSS := by
                  simp only [nullTermD2t]
                  rw [if_neg hen]
                rw [hT']
                omega
            by_cases hmask : allAdmittedIllegalB G (d + 1) p = true
            · rw [nullValueD2t_of_masked G guard d p hkg hcap hai hmask,
                negamaxD2t_of_masked G d p hkg hcap hai hmask]
              have hcongr := foldMax_congr (fun m => -(nullValueD2t G guard d m))
                (fun m => -(negamaxD2t G d m)) (G.moves p) LOSS
                (fun m _ => by
                  show -(nullValueD2t G guard d m) = -(negamaxD2t G d m)
                  rw [ih d (by omega) m])
              have hsub : foldMax (fun m => -(nullValueD2t G guard d m))
                  (movesAbove G (val_lower (d + 1)) p) LOSS
                  ≤ foldMax (fun m => -(nullValueD2t G guard d m)) (G.moves p) LOSS := by
                simp only [movesAbove]
                exact foldMax_filter_le _ _ _ _
              have hsplit := foldMax_init_split
                (fun m => -(nullValueD2t G guard d m)) (G.moves p)
                (nullTermD2t G guard d p) (nullTermD2t_ge_LOSS G guard d p)
              omega
            · have hmask' : allAdmittedIllegalB G (d + 1) p = false := by
                cases h : allAdmittedIllegalB G (d + 1) p with
                | false => rfl
                | true => exact absurd h hmask
              rw [nullValueD2t_of_fold G guard d p hkg hcap hai hmask',
                negamaxD2t_of_fold G d p hkg hcap hai hmask']
              have hcongr := foldMax_congr (fun m => -(nullValueD2t G guard d m))
                (fun m => -(negamaxD2t G d m))
                (movesAbove G (val_lower (d + 1)) p) LOSS
                (fun m _ => by
                  show -(nullValueD2t G guard d m) = -(negamaxD2t G d m)
                  rw [ih d (by omega) m])
              have hsplit := foldMax_init_split
                (fun m => -(nullValueD2t G guard d m))
                (movesAbove G (val_lower (d + 1)) p)
                (nullTermD2t G guard d p) (nullTermD2t_ge_LOSS G guard d p)
              omega

/-- Mate-in-k completeness for the declared t-value: the transfer,
under `NoZugzwangT`. -/
theorem forcedMate_completeT (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hZ : NoZugzwangT G guard)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, k + 1 ≤ D → MATE_LOWER ≤ nullValueD2t G guard D p := by
  intro D hD
  rw [nullValueD2t_eq_realValue_of_noZugzwangT G guard hZ D p]
  exact forcedMate_negamaxD2t G hF hFM D hD

/-! ### No false mates for the t-variant: NO chess premise

The CexF channel closes by construction.  A sub-band defender value at
the frontier now means the fold ran over EVERY move -- either the mask
fired (and the fold was the full list) or a legal admitted move exists
(whose quiet depth-0 value refutes the band claim directly).  The two
frontier lemmas below carry `EvalQuiet` only; `NoMaskedMobility`
appears nowhere in this section. -/

/-- Frontier, masked side: a full-list depth-0 reply set all in the
mate band forces every reply illegal -- contradicting non-terminality.
Premise-free beyond `EvalQuiet`. -/
theorem masked_frontier_escape_t (G : QSGame) (guard : G.Pos → Bool)
    (hQ : EvalQuiet G.toNullGame.toGame)
    {m : G.Pos}
    (hcapm : ¬ (hasKingCapture G.toNullGame.toGame m = true))
    (hai : allIllegalB G m = false)
    (hrep : ∀ m' ∈ G.moves m, MATE_LOWER ≤ nullValueD2t G guard 0 m') :
    False := by
  have hML : MATE_LOWER = 47923 := rfl
  have hallcap : ∀ m' ∈ G.moves m,
      hasKingCapture G.toNullGame.toGame m' = true := by
    intro m' hm'
    have hv := hrep m' hm'
    by_cases hkgm' : G.eval m' ≤ -MATE_LOWER
    · exact absurd
        ((hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm', hkgm'⟩) hcapm
    · cases hcm' : hasKingCapture G.toNullGame.toGame m' with
      | true => rfl
      | false =>
        exfalso
        have hval : nullValueD2t G guard 0 m' = G.eval m' := by
          simp only [nullValueD2t]
          rw [if_neg hkgm', if_neg (by simp [hcm'])]
        rw [hval] at hv
        have := hQ m' hkgm'
        omega
  rw [allIllegalB_true_iff.mpr hallcap] at hai
  exact Bool.noConfusion hai

/-- Frontier, unmasked side: a legal ADMITTED reply exists, and its
quiet depth-0 value refutes the band claim on the spot. -/
theorem admitted_frontier_escape_t (G : QSGame) (guard : G.Pos → Bool)
    (hQ : EvalQuiet G.toNullGame.toGame)
    {m : G.Pos}
    (hcapm : ¬ (hasKingCapture G.toNullGame.toGame m = true))
    (hmask : allAdmittedIllegalB G 1 m = false)
    (hrep : ∀ m' ∈ movesAbove G (val_lower 1) m,
      MATE_LOWER ≤ nullValueD2t G guard 0 m') :
    False := by
  have hML : MATE_LOWER = 47923 := rfl
  obtain ⟨m0, hm0, hleg0⟩ := exists_legal_admitted hmask
  have hv := hrep m0 hm0
  have hm0m : m0 ∈ G.moves m := movesAbove_subset G _ m m0 hm0
  have hkg0 : ¬ (G.eval m0 ≤ -MATE_LOWER) := fun hh =>
    hcapm ((hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m0, hm0m, hh⟩)
  have hval : nullValueD2t G guard 0 m0 = G.eval m0 := by
    simp only [nullValueD2t]
    rw [if_neg hkg0, if_neg (by simp [hleg0])]
  rw [hval] at hv
  have := hQ m0 hkg0
  omega

/-- **No false mates, t-variant -- unconditional over chess**: a
mate-band declared t-value at a legally-reached root IS a forced mate
within the probed depth, under FIDELITY premises alone
(`ValFloor G 192` + `EvalQuiet`).  `NoMaskedMobility` -- required for
the shipped `nullValueD2` (`cexF_masked_mobility`) -- is absent: where
the shipped value's defender fold could silently drop the escape, the
t-fold provably contains it. -/
theorem forcedMate_of_nullValueD2t (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame) :
    ∀ (D : Nat) (p : G.Pos),
      hasKingCapture G.toNullGame.toGame p = false →
      MATE_LOWER ≤ nullValueD2t G guard D p →
      ForcedMate G D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro D
  induction D using Nat.strongRecOn with
  | _ D ih =>
    intro p hcapf hband
    have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by
      simp [hcapf]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · -- (`absurd`, not a bare `omega`: an omega closing a
      -- non-arithmetic goal routes through `Classical.byContradiction`,
      -- and this theorem stays choice-free without it.)
      rw [nullValueD2t_kingGone G guard D p hkg] at hband
      exact absurd hband (by omega)
    cases D with
    | zero =>
      exfalso
      have hval : nullValueD2t G guard 0 p = G.eval p := by
        simp only [nullValueD2t]
        rw [if_neg hkg, if_neg hcap]
      rw [hval] at hband
      have := hQ p hkg
      omega
    | succ d =>
      cases hai : allIllegalB G p with
      | true =>
        exfalso
        rw [nullValueD2t_of_allIllegal G guard d p hkg hcap hai] at hband
        simp only [terminalValue] at hband
        by_cases hic : inCheckB G.toNullGame p = true
        · rw [if_pos hic] at hband
          omega
        · rw [if_neg hic] at hband
          omega
      | false =>
        have hwit : ∃ m ∈ G.moves p,
            MATE_LOWER ≤ -(nullValueD2t G guard d m) := by
          by_cases hmask : allAdmittedIllegalB G (d + 1) p = true
          · rw [nullValueD2t_of_masked G guard d p hkg hcap hai hmask] at hband
            obtain ⟨m, hmem, hmv⟩ :=
              foldMax_failHigh_witness (fun x => -(nullValueD2t G guard d x))
                (G.moves p) (nullTermD2t G guard d p)
                (nullTermD2t_lt_ML G guard d p) hband
            exact ⟨m, hmem, hmv⟩
          · have hmask' : allAdmittedIllegalB G (d + 1) p = false := by
              cases h : allAdmittedIllegalB G (d + 1) p with
              | false => rfl
              | true => exact absurd h hmask
            rw [nullValueD2t_of_fold G guard d p hkg hcap hai hmask'] at hband
            obtain ⟨m, hmem, hmv⟩ :=
              foldMax_failHigh_witness (fun x => -(nullValueD2t G guard d x))
                (movesAbove G (val_lower (d + 1)) p) (nullTermD2t G guard d p)
                (nullTermD2t_lt_ML G guard d p) hband
            exact ⟨m, movesAbove_subset G _ p m hmem, hmv⟩
        obtain ⟨m, hm, hmv⟩ := hwit
        have hchild : nullValueD2t G guard d m ≤ -MATE_LOWER := by omega
        have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
          hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, hh⟩)
        have hlegm : hasKingCapture G.toNullGame.toGame m = false := by
          cases hcm : hasKingCapture G.toNullGame.toGame m with
          | false => rfl
          | true =>
            exfalso
            rw [nullValueD2t_of_capture G guard d m hkgm hcm] at hchild
            omega
        have hcapm : ¬ (hasKingCapture G.toNullGame.toGame m = true) := by
          simp [hlegm]
        cases d with
        | zero =>
          exfalso
          have hval : nullValueD2t G guard 0 m = G.eval m := by
            simp only [nullValueD2t]
            rw [if_neg hkgm, if_neg hcapm]
          rw [hval] at hchild
          exact hkgm hchild
        | succ d' =>
          cases hai' : allIllegalB G m with
          | true =>
            rw [nullValueD2t_of_allIllegal G guard d' m hkgm hcapm hai'] at hchild
            by_cases hic : inCheckB G.toNullGame m = true
            · exact ForcedMate.mate (k := d' + 1) hkg hm hlegm ⟨hai', hic⟩
            · exfalso
              simp only [terminalValue] at hchild
              rw [if_neg hic] at hchild
              omega
          | false =>
            by_cases hmaskm : allAdmittedIllegalB G (d' + 1) m = true
            · rw [nullValueD2t_of_masked G guard d' m hkgm hcapm hai' hmaskm]
                at hchild
              have hrep : ∀ m' ∈ G.moves m,
                  MATE_LOWER ≤ nullValueD2t G guard d' m' := by
                intro m' hm'
                have hle : -(nullValueD2t G guard d' m')
                    ≤ foldMax (fun x => -(nullValueD2t G guard d' x))
                        (G.moves m) (nullTermD2t G guard d' m) :=
                  foldMax_le_of_mem _ _ _ m' hm'
                omega
              cases d' with
              | zero =>
                exact (masked_frontier_escape_t G guard hQ hcapm hai' hrep).elim
              | succ d'' =>
                refine ForcedMate.step (k := d'' + 1) hkg hm hlegm hai' ?_
                intro m' hm' hleg'
                exact ih (d'' + 1) (by omega) m' hleg' (hrep m' hm')
            · have hmaskm' : allAdmittedIllegalB G (d' + 1) m = false := by
                cases h : allAdmittedIllegalB G (d' + 1) m with
                | false => rfl
                | true => exact absurd h hmaskm
              rw [nullValueD2t_of_fold G guard d' m hkgm hcapm hai' hmaskm']
                at hchild
              have hrep : ∀ m' ∈ movesAbove G (val_lower (d' + 1)) m,
                  MATE_LOWER ≤ nullValueD2t G guard d' m' := by
                intro m' hm'
                have hle : -(nullValueD2t G guard d' m')
                    ≤ foldMax (fun x => -(nullValueD2t G guard d' x))
                        (movesAbove G (val_lower (d' + 1)) m)
                        (nullTermD2t G guard d' m) :=
                  foldMax_le_of_mem _ _ _ m' hm'
                omega
              cases d' with
              | zero =>
                exact (admitted_frontier_escape_t G guard hQ hcapm hmaskm' hrep).elim
              | succ d'' =>
                refine ForcedMate.step (k := d'' + 1) hkg hm hlegm hai' ?_
                intro m' hm' hleg'
                have hmem' : m' ∈ movesAbove G (val_lower (d'' + 1 + 1)) m :=
                  mem_movesAbove_of_floor G hF (d := d'' + 1 + 1) (by omega) hm'
                exact ih (d'' + 1) (by omega) m' hleg' (hrep m' hmem')

/-- The mated-side dual, t-variant: fidelity premises only. -/
theorem forcedlyMated_of_nullValueD2t (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (D : Nat) (q : G.Pos)
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hkgq : ¬ (G.eval q ≤ -MATE_LOWER))
    (hlo : nullValueD2t G guard (D + 1) q ≤ -MATE_LOWER) :
    ForcedlyMated G D q := by
  have hML : MATE_LOWER = 47923 := rfl
  have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by
    simp [hcapq]
  cases hai : allIllegalB G q with
  | true =>
    rw [nullValueD2t_of_allIllegal G guard D q hkgq hcapq' hai] at hlo
    by_cases hic : inCheckB G.toNullGame q = true
    · exact Or.inl ⟨hai, hic⟩
    · exfalso
      simp only [terminalValue] at hlo
      rw [if_neg hic] at hlo
      omega
  | false =>
    by_cases hmask : allAdmittedIllegalB G (D + 1) q = true
    · rw [nullValueD2t_of_masked G guard D q hkgq hcapq' hai hmask] at hlo
      have hrep : ∀ m' ∈ G.moves q,
          MATE_LOWER ≤ nullValueD2t G guard D m' := by
        intro m' hm'
        have hle : -(nullValueD2t G guard D m')
            ≤ foldMax (fun x => -(nullValueD2t G guard D x))
                (G.moves q) (nullTermD2t G guard D q) :=
          foldMax_le_of_mem _ _ _ m' hm'
        omega
      cases D with
      | zero =>
        exact (masked_frontier_escape_t G guard hQ hcapq' hai hrep).elim
      | succ D' =>
        refine Or.inr ⟨hai, fun m' hm' hleg' => ?_⟩
        exact forcedMate_of_nullValueD2t G guard hF hQ (D' + 1) m'
          hleg' (hrep m' hm')
    · have hmask' : allAdmittedIllegalB G (D + 1) q = false := by
        cases h : allAdmittedIllegalB G (D + 1) q with
        | false => rfl
        | true => exact absurd h hmask
      rw [nullValueD2t_of_fold G guard D q hkgq hcapq' hai hmask'] at hlo
      have hrep : ∀ m' ∈ movesAbove G (val_lower (D + 1)) q,
          MATE_LOWER ≤ nullValueD2t G guard D m' := by
        intro m' hm'
        have hle : -(nullValueD2t G guard D m')
            ≤ foldMax (fun x => -(nullValueD2t G guard D x))
                (movesAbove G (val_lower (D + 1)) q)
                (nullTermD2t G guard D q) :=
          foldMax_le_of_mem _ _ _ m' hm'
        omega
      cases D with
      | zero =>
        exact (admitted_frontier_escape_t G guard hQ hcapq' hmask' hrep).elim
      | succ D' =>
        refine Or.inr ⟨hai, fun m' hm' hleg' => ?_⟩
        have hmem' : m' ∈ movesAbove G (val_lower (D' + 1 + 1)) q :=
          mem_movesAbove_of_floor G hF (d := D' + 1 + 1) (by omega) hm'
        exact forcedMate_of_nullValueD2t G guard hF hQ (D' + 1) m'
          hleg' (hrep m' hmem')

/-! ### The t-trichotomy: the honesty arm goes unconditional -/

/-- **Eventual classification for the t-variant.**  Same three arms as
`eventual_classification`; the premise ledger improves exactly where
the design aimed: the NEITHER arm consumes `ValFloor` + `EvalQuiet`
(fidelity) ONLY -- no `NoMaskedMobility`, no `NoZugzwang*` -- while the
win/loss arms consume `ValFloor` + `NoZugzwangT` as before (the
finding side's premise was never the target here; its recorded
discharge option remains the depth-decaying guard). -/
theorem eventual_classification_t (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hZ : NoZugzwangT G guard)
    (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    (∀ k, ForcedMate G k p →
      ∀ D, k + 1 ≤ D → MATE_LOWER ≤ nullValueD2t G guard D p) ∧
    (∀ k, ForcedlyMated G k p →
      ∀ D, k + 2 ≤ D → nullValueD2t G guard D p ≤ -MATE_LOWER) ∧
    ((∀ k, ¬ ForcedMate G k p) → (∀ k, ¬ ForcedlyMated G k p) →
      ∀ D, -MATE_LOWER < nullValueD2t G guard D p ∧
        nullValueD2t G guard D p < MATE_LOWER) := by
  have hML : MATE_LOWER = 47923 := rfl
  refine ⟨?_, ?_, ?_⟩
  · intro k hFM D hD
    exact forcedMate_completeT G guard hF hZ hFM D hD
  · intro k hFMd D hD
    rw [nullValueD2t_eq_realValue_of_noZugzwangT G guard hZ D p]
    exact forcedlyMated_negamaxD2t G hF hcapf hFMd D hD
  · intro hnFM hnFMd D
    constructor
    · by_cases hlo : nullValueD2t G guard D p ≤ -MATE_LOWER
      · exfalso
        cases D with
        | zero =>
          have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by
            simp [hcapf]
          have hval : nullValueD2t G guard 0 p = G.eval p := by
            simp only [nullValueD2t]
            rw [if_neg hkg, if_neg hcap]
          rw [hval] at hlo
          exact hkg hlo
        | succ D' =>
          exact hnFMd D'
            (forcedlyMated_of_nullValueD2t G guard hF hQ D' p hcapf hkg hlo)
      · omega
    · by_cases hhi : MATE_LOWER ≤ nullValueD2t G guard D p
      · exact absurd (forcedMate_of_nullValueD2t G guard hF hQ D p hcapf hhi)
          (hnFM D)
      · omega

/-! ### CexF becomes a positive test -/

/-- The masked defender node of `CexF`, under the t-value: the trigger
fires (the only admitted reply is the illegal `I`), the fold runs over
BOTH replies, and the legal escape `E` lifts the depth-1 value to the
honest 0 -- where `nullValueD2` returned the raw `-MATE_UPPER`
(`cexF_M_depth1`). -/
theorem cexF_t_M_depth1 :
    nullValueD2t CexF (fun _ => false) 1 FPos.M = 0 := by
  rw [nullValueD2t_of_masked CexF (fun _ => false) 0 FPos.M (by decide)
    (by decide) (by decide) (by decide)]
  have hmv : CexF.moves FPos.M = [FPos.I, FPos.E] := rfl
  have hT : nullTermD2t CexF (fun _ => false) 0 FPos.M = LOSS := by
    simp only [nullTermD2t]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
  have hI : nullValueD2t CexF (fun _ => false) 0 FPos.I = MATE_UPPER :=
    nullValueD2t_of_capture CexF (fun _ => false) 0 FPos.I (by decide) (by decide)
  have hE : nullValueD2t CexF (fun _ => false) 0 FPos.E = 0 := by
    simp only [nullValueD2t]
    rw [if_neg (by decide), if_neg (by decide)]
    decide
  rw [hmv, hT]
  simp only [foldMax]
  rw [hI, hE]
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  omega

/-- **The positive test, bundled**: on the countermodel that proved
`NoMaskedMobility` REQUIRED for the shipped value (`cexF_bandValue`:
depth-2 declared value the full `MATE_UPPER`; `cexF_no_forcedMate`: no
mate exists), the t-variant computes the honest draw 0 at depth 2
already -- and stays there at depth 3. -/
theorem cexF_t_positive :
    nullValueD2t CexF (fun _ => false) 2 FPos.R = 0 ∧
    nullValueD2t CexF (fun _ => false) 3 FPos.R = 0 := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  have hR2 : nullValueD2t CexF (fun _ => false) 2 FPos.R = 0 := by
    rw [nullValueD2t_of_fold CexF (fun _ => false) 1 FPos.R (by decide)
      (by decide) (by decide) (by decide)]
    have hma : movesAbove CexF (val_lower 2) FPos.R = [FPos.M] := by decide
    have hT : nullTermD2t CexF (fun _ => false) 1 FPos.R = LOSS := by
      simp only [nullTermD2t]
      rw [if_neg (fun h => Bool.noConfusion h.1)]
    rw [hma, hT]
    simp only [foldMax]
    rw [cexF_t_M_depth1]
    omega
  refine ⟨hR2, ?_⟩
  have hM2 : nullValueD2t CexF (fun _ => false) 2 FPos.M = 0 := by
    rw [nullValueD2t_of_fold CexF (fun _ => false) 1 FPos.M (by decide)
      (by decide) (by decide) (by decide)]
    have hma : movesAbove CexF (val_lower 2) FPos.M = [FPos.I, FPos.E] := by
      decide
    have hT : nullTermD2t CexF (fun _ => false) 1 FPos.M = LOSS := by
      simp only [nullTermD2t]
      rw [if_neg (fun h => Bool.noConfusion h.1)]
    have hI : nullValueD2t CexF (fun _ => false) 1 FPos.I = MATE_UPPER :=
      nullValueD2t_of_capture CexF (fun _ => false) 1 FPos.I (by decide)
        (by decide)
    have hE : nullValueD2t CexF (fun _ => false) 1 FPos.E = 0 := by
      rw [nullValueD2t_of_allIllegal CexF (fun _ => false) 0 FPos.E (by decide)
        (by decide) (by decide)]
      simp only [terminalValue]
      rw [if_neg (by decide)]
    rw [hma, hT]
    simp only [foldMax]
    rw [hI, hE]
    omega
  rw [nullValueD2t_of_fold CexF (fun _ => false) 2 FPos.R (by decide)
    (by decide) (by decide) (by decide)]
  have hma : movesAbove CexF (val_lower 3) FPos.R = [FPos.M] := by decide
  have hT : nullTermD2t CexF (fun _ => false) 2 FPos.R = LOSS := by
    simp only [nullTermD2t]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
  rw [hma, hT]
  simp only [foldMax]
  rw [hM2]
  omega

end Sunfish
