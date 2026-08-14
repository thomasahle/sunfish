/-
The two local obligations introduced by intrinsic LMR.

Both null probes use fixed targets, so valid fail-soft reports for the same
child value determine the same hot/safe bits under every caller window and
table state. The resulting move edge spends one ply normally, one more at a
hot node, and one more for an intrinsically low move at a safe node. Thus the
child depth is a function of the position, nominal depth, and move alone, and
every real edge spends between one and three plies.
-/

import Sunfish.EventuallyWide

namespace Sunfish

/-- Two reports of each fixed null target produce the same `hot or safe`
classification. The production code short-circuits the safe probe when hot;
that merely omits the disjunct whose value cannot affect the result. -/
theorem threat_safe_bit_stable
    {hotTarget safeTarget hotReport₁ hotReport₂ safeReport₁ safeReport₂ passValue : Int}
    (hhot₁ : WindowReport (1 - hotTarget) hotReport₁ passValue)
    (hhot₂ : WindowReport (1 - hotTarget) hotReport₂ passValue)
    (hsafe₁ : WindowReport (1 - safeTarget) safeReport₁ passValue)
    (hsafe₂ : WindowReport (1 - safeTarget) safeReport₂ passValue) :
    (hotTarget ≤ -hotReport₁ ∨ safeTarget ≤ -safeReport₁) ↔
      (hotTarget ≤ -hotReport₂ ∨ safeTarget ≤ -safeReport₂) := by
  have hhot := hot_bit_stable hhot₁ hhot₂
  have hsafe := hot_bit_stable hsafe₁ hsafe₂
  constructor
  · rintro (h | h)
    · exact Or.inl (hhot.mp h)
    · exact Or.inr (hsafe.mp h)
  · rintro (h | h)
    · exact Or.inl (hhot.mpr h)
    · exact Or.inr (hsafe.mpr h)

/-- The high target implying the low target justifies the production
short-circuit `safe = hot or lower_probe`. -/
theorem hot_implies_safe
    {hotTarget safeTarget report passValue : Int}
    (htarget : safeTarget ≤ hotTarget)
    (hreport : WindowReport (1 - hotTarget) report passValue)
    (hhot : hotTarget ≤ -report) :
    safeTarget ≤ -passValue := by
  have := (hot_bit_determined hreport).mp hhot
  omega

def fuelBit (b : Bool) : Nat := if b then 1 else 0

/-- Extra fuel spent by the code: the node-wide hot bit plus the
move-specific low-value bit, enabled only at a threat-safe node. -/
def intrinsicSpend (hot safe low : Bool) : Nat :=
  fuelBit hot + fuelBit (safe && low)

theorem intrinsicSpend_le_two (hot safe low : Bool) :
    intrinsicSpend hot safe low ≤ 2 := by
  cases hot <;> cases safe <;> cases low <;> decide

/-- Exact correspondence with
`d -= hot; move_depth = d - 1 - (safe and low)` at the armed depths. -/
theorem intrinsic_child_depth (depth : Nat) (hdepth : 3 ≤ depth)
    (hot safe low : Bool) :
    depth - fuelBit hot - 1 - fuelBit (safe && low) =
      depth - 1 - min (3 - 1) (intrinsicSpend hot safe low) := by
  cases hot <;> cases safe <;> cases low <;> simp [fuelBit, intrinsicSpend] <;> omega

/-- Every intrinsic-LMR edge spends between one and three plies. -/
theorem intrinsic_edge_cost (depth : Nat) (hdepth : 3 ≤ depth)
    (hot safe low : Bool) :
    depth - 3 ≤ depth - fuelBit hot - 1 - fuelBit (safe && low) ∧
      depth - fuelBit hot - 1 - fuelBit (safe && low) ≤ depth - 1 := by
  cases hot <;> cases safe <;> cases low <;> simp [fuelBit] <;> omega

end Sunfish
