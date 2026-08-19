/-
The monotone null-move clamp shipped in sunfish.py.

The recursive pass probe is made at the complementary zero window and
negated, exactly as in Searcher.bound.  The production null contribution is

    max (1 - MATE_LOWER) (min (eval + EVAL_ROUGHNESS) passReport)

When the cap is already below the window, it is a valid fail-low report for
the clamped value and the child need not be searched. Otherwise negation
transfers the child report to the parent window; fixed `min` and `max`
preserve it. Consequently at most one child probe suffices.

The move fold, king-capture substitution, sticky `live` certificate, and
post-fold mate/stalemate override are separate from this local transformer.
The model-code audit records this file as the proof of the null report emitted
by the current source.
-/

import Sunfish.Driver
import Sunfish.EvalBounds

set_option maxRecDepth 4096

namespace Sunfish

/-- The global model uses an explicit upper mate-band clamp.  It is redundant
for reachable Python evaluations, but keeps the abstract recurrence honest
without importing the mailbox evaluator into every theorem. -/
def nullClamp (eval passValue : Int) : Int :=
  max (1 - MATE_LOWER) (min (MATE_LOWER - 1) (min (eval + EVAL_ROUGHNESS) passValue))

/-- The zero-window contract stated by `Searcher.bound`: a fail-low report is
an upper bound, while a fail-high report is a lower bound. -/
def WindowReport (gamma report value : Int) : Prop :=
  (report < gamma ∧ value ≤ report) ∨
  (gamma ≤ report ∧ report ≤ value)

/-- Negating a child report at `1 - gamma` produces a valid parent report at
`gamma`.  The offset by one is the integer zero-window convention. -/
theorem WindowReport.negate {gamma report value : Int}
    (h : WindowReport (1 - gamma) report value) :
    WindowReport gamma (-report) (-value) := by
  rcases h with h | h
  · right
    constructor <;> omega
  · left
    constructor <;> omega

/-- `min cap` is monotone, so it transports any valid zero-window report to
the capped value.  If the raw report fails high but the cap lies below the
window, both capped quantities equal the cap and correctly fail low. -/
theorem WindowReport.cap (cap gamma report value : Int)
    (h : WindowReport gamma report value) :
    WindowReport gamma (min cap report) (min cap value) := by
  rcases h with h | h <;>
    simp only [WindowReport, Int.min_def] <;>
    split <;> split <;> omega

/-- `max floor` is monotone, so it transports any valid zero-window report
to the floored value. -/
theorem WindowReport.floor (floor gamma report value : Int)
    (h : WindowReport gamma report value) :
    WindowReport gamma (max floor report) (max floor value) := by
  rcases h with h | h <;>
    simp only [WindowReport, Int.max_def] <;>
    split <;> split <;> omega

/-- If a fixed cap is already below the window, the cap itself is a complete
fail-low report for the capped value. No report about `value` is needed. -/
theorem WindowReport.cap_failLow (cap gamma value : Int) (h : cap < gamma) :
    WindowReport gamma cap (min cap value) := by
  left
  constructor
  · exact h
  · simp only [Int.min_def]
    split <;> omega

/-- The upper-cap component used by the production clamp. -/
theorem cappedNull_report (cap gamma childReport childValue : Int)
    (h : WindowReport (1 - gamma) childReport childValue) :
    WindowReport gamma
      (min cap (-childReport))
      (min cap (-childValue)) :=
  WindowReport.cap cap gamma (-childReport) (-childValue) h.negate

/-- The two-sided production clamp transports the same child report and
cannot manufacture a report on the other side of the caller's window. -/
theorem clampedNull_report (floor cap gamma childReport childValue : Int)
    (h : WindowReport (1 - gamma) childReport childValue) :
    WindowReport gamma
      (max floor (min cap (-childReport)))
      (max floor (min cap (-childValue))) :=
  WindowReport.floor floor gamma _ _
    (WindowReport.cap cap gamma _ _ h.negate)

/-- If the fixed cap is below the positive mate band, so is the capped null
value.  A negative mate pass value may remain negative; only a fabricated
positive mate claim is excluded. -/
theorem cappedNull_below_positiveMate (cap passValue : Int)
    (h : cap < MATE_LOWER) :
    min cap passValue < MATE_LOWER := by
  simp only [Int.min_def]
  split <;> omega

/-- The concrete evaluation bound keeps the fixed static cap
`eval + EVAL_ROUGHNESS` strictly inside the ordinary score band.  The next
theorem states the one-sided consequence for the capped null value. -/
theorem staticCap_in_scoreBand (eval : Int)
    (h : -EvalBounds.evalBound ≤ eval ∧ eval ≤ EvalBounds.evalBound) :
    -MATE_LOWER < eval + EVAL_ROUGHNESS ∧
      eval + EVAL_ROUGHNESS < MATE_LOWER := by
  have hlo : -MATE_LOWER < -EvalBounds.evalBound + EVAL_ROUGHNESS := by decide
  have hhi : EvalBounds.evalBound + EVAL_ROUGHNESS < MATE_LOWER := by decide
  omega

/-- The exact positive-mate exclusion used by the Python null move. -/
theorem staticCappedNull_below_positiveMate (eval passValue : Int)
    (h : -EvalBounds.evalBound ≤ eval ∧ eval ≤ EvalBounds.evalBound) :
    min (eval + EVAL_ROUGHNESS) passValue < MATE_LOWER :=
  cappedNull_below_positiveMate _ _ (staticCap_in_scoreBand eval h).2

/-- The production floor and static cap keep the complete null contribution
strictly between the two mate bands, independently of the pass value. -/
theorem staticClampedNull_in_scoreBand (eval passValue : Int)
    (h : -EvalBounds.evalBound ≤ eval ∧ eval ≤ EvalBounds.evalBound) :
    -MATE_LOWER < max (1 - MATE_LOWER)
        (min (eval + EVAL_ROUGHNESS) passValue) ∧
      max (1 - MATE_LOWER) (min (eval + EVAL_ROUGHNESS) passValue) < MATE_LOWER := by
  have hcap := (staticCap_in_scoreBand eval h).2
  have hML : MATE_LOWER = 47923 := rfl
  simp only [Int.max_def, Int.min_def]
  split <;> split <;> omega

/-- The abstract clamp is always an ordinary score. -/
theorem nullClamp_in_scoreBand (eval passValue : Int) :
    -MATE_LOWER < nullClamp eval passValue ∧ nullClamp eval passValue < MATE_LOWER := by
  have hML : MATE_LOWER = 47923 := rfl
  simp only [nullClamp, Int.max_def, Int.min_def]
  split <;> split <;> split <;> omega

/-- On reachable evaluations the abstract upper clamp is redundant, so the
model is exactly the Python expression. -/
theorem nullClamp_eq_production (eval passValue : Int)
    (h : -EvalBounds.evalBound ≤ eval ∧ eval ≤ EvalBounds.evalBound) :
    nullClamp eval passValue =
      max (1 - MATE_LOWER) (min (eval + EVAL_ROUGHNESS) passValue) := by
  have hcap := (staticCap_in_scoreBand eval h).2
  simp only [nullClamp, Int.max_def, Int.min_def]
  split <;> split <;> split <;> omega

/-- If the reachable static cap is already below the window, returning the
cap without a child probe or an explicit floor is a valid report for the same
clamped value used on the searched branch. -/
theorem clampedNull_cap_failLow (eval passValue gamma : Int)
    (heval : -EvalBounds.evalBound ≤ eval ∧ eval ≤ EvalBounds.evalBound)
    (hgamma : eval + EVAL_ROUGHNESS < gamma) :
    WindowReport gamma (eval + EVAL_ROUGHNESS) (nullClamp eval passValue) := by
  rw [nullClamp_eq_production eval passValue heval]
  have hlo := (staticCap_in_scoreBand eval heval).1
  left
  constructor
  · exact hgamma
  · simp only [Int.max_def, Int.min_def]
    split <;> split <;> omega

end Sunfish
