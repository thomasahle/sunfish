/-
The monotone null-move cap shipped in sunfish.py.

The recursive pass probe is made at the complementary zero window and
negated, exactly as in Searcher.bound.  The production null contribution is

    min (eval + EVAL_ROUGHNESS) passReport

Negation transfers a child report to the parent window, and `min` with a fixed
cap preserves a valid fail-soft report. Consequently one child probe suffices.

The move fold, king-capture substitution, sticky `live` certificate, and
post-fold mate/stalemate override are separate from this local transformer.
The model-code audit records this file as the proof of the null report emitted
by the current source.
-/

import Sunfish.Driver

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

/-- The concrete balance guard keeps the fixed static cap
`eval + EVAL_ROUGHNESS` strictly inside the ordinary score band.  The next
theorem states the one-sided consequence for the capped null value. -/
theorem guardedStaticCap_in_scoreBand (eval : Int)
    (h : -500 < eval ∧ eval < 500) :
    -MATE_UPPER < eval + EVAL_ROUGHNESS ∧
      eval + EVAL_ROUGHNESS < MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hE : EVAL_ROUGHNESS = 15 := rfl
  omega

/-- The exact positive-mate exclusion used by the guarded Python null move. -/
theorem guardedCappedNull_below_positiveMate (eval passValue : Int)
    (h : -500 < eval ∧ eval < 500) :
    min (eval + EVAL_ROUGHNESS) passValue < MATE_LOWER :=
  cappedNull_below_positiveMate _ _ (guardedStaticCap_in_scoreBand eval h).2

end Sunfish
