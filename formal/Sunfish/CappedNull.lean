/-
The monotone null-move cap shipped in sunfish.py.

The recursive pass probe is made at the complementary zero window and
negated, exactly as in Searcher.bound.  The production null contribution is

    min (eval + EVAL_ROUGHNESS) passReport

When the cap is already below the window, it is a valid fail-low report for
the capped value and the child need not be searched. Otherwise negation
transfers the child report to the parent window, and `min` with a fixed cap
preserves it. Consequently at most one child probe suffices.

The move fold, king-capture substitution, sticky `live` certificate, and
post-fold mate/stalemate override are separate from this local transformer.
The model-code audit records this file as the proof of the null report emitted
by the current source.
-/

import Sunfish.Driver
import Sunfish.EvalBounds

set_option maxRecDepth 4096

namespace Sunfish

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

/-- If a fixed cap is already below the window, the cap itself is a complete
fail-low report for the capped value. No report about `value` is needed. -/
theorem WindowReport.cap_failLow (cap gamma value : Int) (h : cap < gamma) :
    WindowReport gamma cap (min cap value) := by
  left
  constructor
  · exact h
  · simp only [Int.min_def]
    split <;> omega

/-- The exact local proof obligation of the Python expression
`min(cap, -bound(pass, 1 - gamma, depth - 3))`. -/
theorem cappedNull_report (cap gamma childReport childValue : Int)
    (h : WindowReport (1 - gamma) childReport childValue) :
    WindowReport gamma
      (min cap (-childReport))
      (min cap (-childValue)) :=
  WindowReport.cap cap gamma (-childReport) (-childValue) h.negate

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
    -MATE_UPPER < eval + EVAL_ROUGHNESS ∧
      eval + EVAL_ROUGHNESS < MATE_LOWER := by
  have hlo : -MATE_UPPER < -EvalBounds.evalBound + EVAL_ROUGHNESS := by decide
  have hhi : EvalBounds.evalBound + EVAL_ROUGHNESS < MATE_LOWER := by decide
  omega

/-- The exact positive-mate exclusion used by the Python null move. -/
theorem staticCappedNull_below_positiveMate (eval passValue : Int)
    (h : -EvalBounds.evalBound ≤ eval ∧ eval ≤ EvalBounds.evalBound) :
    min (eval + EVAL_ROUGHNESS) passValue < MATE_LOWER :=
  cappedNull_below_positiveMate _ _ (staticCap_in_scoreBand eval h).2

end Sunfish
