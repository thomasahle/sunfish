/-
Repetition at the unstored root boundary.

The recursive search deliberately has no game-history branch.  Every interior
table entry therefore describes the same history-free value and remains valid
when the played game advances.  At the driver root, which never reads or writes
its own score-table entry, a move back into the played history receives the
exact child value 0.  The ordinary shallow move cap is then applied in exactly
the same way as for a searched child.

This file proves the three local obligations introduced by that organization:

* changing game history cannot retarget an interior table entry;
* replacing a repeated child's report by exact 0 preserves the window-report
  contract, including the ordinary move cap;
* when that cap is nonnegative, an available repeated move prevents the root
  fold from claiming a loss.
-/

import Sunfish.CanNull
import Sunfish.CappedMove

namespace Sunfish

/-- Production interior values instantiate the generic null search with an
empty history predicate. -/
abbrev interiorValue (G : NullGame) : Nat → G.Pos → Int :=
  nullValue G (fun _ => false)

/-- The production table invariant has no game-history parameter. -/
def PersistentTableOK (G : NullGame) (t : Table G.toGame) : Prop :=
  CTableOK G (fun _ => false) t

/-- A root history may change without retargeting any stored interval.  The
history arguments document the two driver calls; neither enters the invariant. -/
theorem persistentTable_history_independent (G : NullGame) (t : Table G.toGame)
    (_before _after : G.Pos → Bool) (h : PersistentTableOK G t) :
    PersistentTableOK G t :=
  h

/-- Value/report transformer for one ordinary root move.  Repeated children
are exact draws; all other child values are transferred by negamax; finally the
fixed shallow cap is applied to either case. -/
def rootMoveValue {α : Type} (hist : α → Bool) (cap : α → Int)
    (childValue : α → Int) (child : α) : Int :=
  min (cap child) (if hist child = true then 0 else -childValue child)

/-- The report emitted by the Python driver has the same transformer. -/
def rootMoveReport {α : Type} (hist : α → Bool) (cap : α → Int)
    (childReport : α → Int) (child : α) : Int :=
  min (cap child) (if hist child = true then 0 else -childReport child)

/-- An exact value is a valid report on either side of every integer window. -/
theorem WindowReport.exact (gamma value : Int) :
    WindowReport gamma value value := by
  unfold WindowReport
  omega

/-- Root repetition is report-correct.  In the repeated arm no recursive child
report is needed; otherwise this is complementary-window negation followed by
the already-proved monotone cap transform. -/
theorem rootMove_report {α : Type} (hist : α → Bool) (cap : α → Int)
    (childReport childValue : α → Int) (child : α) (gamma : Int)
    (h : WindowReport (1 - gamma) (childReport child) (childValue child)) :
    WindowReport gamma
      (rootMoveReport hist cap childReport child)
      (rootMoveValue hist cap childValue child) := by
  by_cases hrep : hist child = true
  · simp only [rootMoveReport, rootMoveValue, if_pos hrep]
    exact (WindowReport.exact gamma 0).cap (cap child) gamma 0 0
  · simp only [rootMoveReport, rootMoveValue, if_neg hrep]
    exact h.negate.cap (cap child) gamma (-childReport child) (-childValue child)

/-- The root can classify a repeated child without evaluating its subtree. -/
theorem rootMoveValue_of_repetition {α : Type} (hist : α → Bool)
    (cap : α → Int) (childValue : α → Int) (child : α)
    (hrep : hist child = true) :
    rootMoveValue hist cap childValue child = min (cap child) 0 := by
  simp [rootMoveValue, hrep]

/-- The fixed root fold corresponding to the Python real-move loop. -/
def rootHistoryFold {α : Type} (hist : α → Bool) (cap : α → Int)
    (childValue : α → Int) (moves : List α) : Int :=
  foldMax (rootMoveValue hist cap childValue) moves LOSS

/-- The actual lazy root loop, including its window cutoff. -/
def rootHistorySearch {α : Type} (hist : α → Bool) (cap : α → Int)
    (childReport : α → Int) (moves : List α) (gamma : Int) : Int :=
  searchMoves gamma (rootMoveReport hist cap childReport) moves LOSS

/-- Folding the root reports lazily preserves the report contract for the
history-shaped root value. -/
theorem rootHistorySearch_report {α : Type} (hist : α → Bool) (cap : α → Int)
    (childReport childValue : α → Int) (moves : List α) (gamma : Int)
    (hchild : ∀ child,
      WindowReport (1 - gamma) (childReport child) (childValue child)) :
    WindowReport gamma
      (rootHistorySearch hist cap childReport moves gamma)
      (rootHistoryFold hist cap childValue moves) := by
  have hedge : ∀ child,
      (gamma ≤ rootMoveReport hist cap childReport child →
        rootMoveReport hist cap childReport child ≤
          rootMoveValue hist cap childValue child) ∧
      (rootMoveReport hist cap childReport child < gamma →
        rootMoveValue hist cap childValue child ≤
          rootMoveReport hist cap childReport child) := by
    intro child
    have h := rootMove_report hist cap childReport childValue child gamma (hchild child)
    unfold WindowReport at h
    omega
  have hloop := searchMoves_spec gamma
    (rootMoveReport hist cap childReport)
    (rootMoveValue hist cap childValue) hedge moves LOSS LOSS
    (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
  unfold rootHistorySearch rootHistoryFold WindowReport
  by_cases h : searchMoves gamma (rootMoveReport hist cap childReport) moves LOSS < gamma
  · exact Or.inl ⟨h, hloop.2 h⟩
  · exact Or.inr ⟨by omega, hloop.1 (by omega)⟩

/-- A repeated move whose cap is nonnegative contributes exactly 0, so the
root cannot report a negative value.  Above the shallow-cap horizon the cap is
`MATE_UPPER`, making the premise immediate. -/
theorem rootHistoryFold_not_lost {α : Type} (hist : α → Bool)
    (cap : α → Int) (childValue : α → Int) (moves : List α) (child : α)
    (hmem : child ∈ moves) (hrep : hist child = true)
    (hcap : 0 ≤ cap child) :
    0 ≤ rootHistoryFold hist cap childValue moves := by
  have hedge : rootMoveValue hist cap childValue child = 0 := by
    rw [rootMoveValue_of_repetition hist cap childValue child hrep]
    simp only [Int.min_def]
    split <;> omega
  have hfold := foldMax_le_of_mem
    (rootMoveValue hist cap childValue) moves LOSS child hmem
  rw [hedge] at hfold
  exact hfold

end Sunfish
