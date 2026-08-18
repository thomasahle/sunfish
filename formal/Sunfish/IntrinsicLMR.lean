/-
The local obligations introduced by intrinsic LMR.

The null probe uses a fixed target and the null child's QSearch value, so valid
fail-soft reports determine the same position-only hot bit under every caller
window and table state. At an interior node, the `eligible` bit is the static
null-move guard conjoined with `not inCheckB`; the unstored driver root supplies
`false`. A real edge
spends one ply normally, one more at a hot node, and one more for an
intrinsically low move at an eligible node. Thus child depth is a function of the
position, nominal depth, and move alone, and every real edge spends between
one and three plies.
-/

import Sunfish.EventuallyFinite

namespace Sunfish

def fuelBit (b : Bool) : Nat := if b then 1 else 0

/-- Extra fuel spent by the code: the node-wide hot bit plus the
move-specific low-value bit, enabled only by the static eligibility guard. -/
def intrinsicSpend (hot eligible low : Bool) : Nat :=
  fuelBit hot + fuelBit (eligible && low)

/-- Production's exact check-evasion exemption. Both inputs are position- and
depth-derived, so their conjunction remains a fixed property of the TT key. -/
def evasionEligible (G : QSGame) (static : G.Pos → Nat → Bool) : G.Pos → Nat → Bool :=
  fun p d => static p d && !inCheckB G.toNullGame p

theorem evasionEligible_of_check (G : QSGame) (static : G.Pos → Nat → Bool)
    (p : G.Pos) (d : Nat) (h : inCheckB G.toNullGame p = true) :
    evasionEligible G static p d = false := by
  simp [evasionEligible, h]

/-- In check, the move-specific LMR bit contributes no fuel debt. -/
theorem intrinsicSpend_evasion_of_check (G : QSGame) (static : G.Pos → Nat → Bool)
    (p : G.Pos) (d : Nat) (hot low : Bool) (h : inCheckB G.toNullGame p = true) :
    intrinsicSpend hot (evasionEligible G static p d) low = fuelBit hot := by
  simp [intrinsicSpend, evasionEligible, fuelBit, h]

theorem intrinsicSpend_le_two (hot eligible low : Bool) :
    intrinsicSpend hot eligible low ≤ 2 := by
  cases hot <;> cases eligible <;> cases low <;> decide

/-- Exact correspondence with
`d -= hot; move_depth = d - 1 - (eligible and low)` at the armed depths,
where the Python expression includes `not root` in `eligible`. -/
theorem intrinsic_child_depth (depth : Nat) (hdepth : 3 ≤ depth)
    (hot eligible low : Bool) :
    depth - fuelBit hot - 1 - fuelBit (eligible && low) =
      depth - 1 - min (3 - 1) (intrinsicSpend hot eligible low) := by
  cases hot <;> cases eligible <;> cases low <;> simp [fuelBit, intrinsicSpend] <;> omega

/-- Every intrinsic-LMR edge spends between one and three plies. -/
theorem intrinsic_edge_cost (depth : Nat) (hdepth : 3 ≤ depth)
    (hot eligible low : Bool) :
    depth - 3 ≤ depth - fuelBit hot - 1 - fuelBit (eligible && low) ∧
      depth - fuelBit hot - 1 - fuelBit (eligible && low) ≤ depth - 1 := by
  cases hot <;> cases eligible <;> cases low <;> simp [fuelBit] <;> omega

/-- The production policy as an edge selector. `hot` is a fixed-target
property of the node, `eligible` is its static, check-exempt guard, and `low` is
an intrinsic property of the move. Neither caller window nor killer ordering enters. -/
def intrinsicEdgeSpend (G : QSGame) (hot eligible : G.Pos → Nat → Bool)
    (low : G.Pos → Nat → G.Pos → Bool) : G.Pos → Nat → G.Pos → Nat :=
  fun p d m => intrinsicSpend (hot p d) (eligible p d) (low p d m)

theorem intrinsicEdgeSpend_le_two (G : QSGame) (hot eligible : G.Pos → Nat → Bool)
    (low : G.Pos → Nat → G.Pos → Bool) (p m : G.Pos) (d : Nat) :
    intrinsicEdgeSpend G hot eligible low p d m ≤ 2 :=
  intrinsicSpend_le_two _ _ _

/-- Intrinsic LMR inherits eventual mate completeness directly from the
edge-generic fuel theorem. Every real edge costs at most three plies. -/
theorem forcedMate_intrinsicValue (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 3 * k + 4 ≤ D →
      MATE_LOWER ≤ fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) D p :=
  forcedMate_fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) (by omega) hF hFM

/-- On a game tree ending within `N` plies, intrinsic LMR preserves the
full eventual W/D/L classification from depth `3*N + 9` onward. -/
theorem eventual_classification_intrinsic_finite (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {N : Nat} (p : G.Pos) (hE : EndsWithin G N p)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    ∀ D : Nat, 3 * N + 9 ≤ D →
      ((MATE_LOWER ≤ fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) D p) ↔
        (∃ k, ForcedMate G k p)) ∧
      ((fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) D p ≤ -MATE_LOWER) ↔
        (∃ k, ForcedlyMated G k p)) ∧
      ((¬ (∃ k, ForcedMate G k p)) → (¬ (∃ k, ForcedlyMated G k p)) →
        -MATE_LOWER < fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) D p ∧
          fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) D p < MATE_LOWER) := by
  intro D hD
  exact eventual_classification_fuel_finite G guard 3 (intrinsicEdgeSpend G hot eligible low)
    (by omega) hF p hE hcapf hkg D (by omega)

end Sunfish
