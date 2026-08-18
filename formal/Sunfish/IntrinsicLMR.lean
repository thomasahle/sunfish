/-
The local obligations introduced by intrinsic LMR.

At an interior node, `eligible` is the static LMR guard and `low` says the
move's intrinsic value is below the fixed LMR threshold; the unstored driver
root supplies `false`. A real edge spends one ply normally and one more when
both bits hold. Thus child depth is a function of the position, nominal depth,
and move alone, and every real edge spends either one or two plies.
-/

import Sunfish.EventuallyFinite

namespace Sunfish

def fuelBit (b : Bool) : Nat := if b then 1 else 0

/-- Extra depth spent by the code on an eligible intrinsically low move. -/
def intrinsicSpend (eligible low : Bool) : Nat := fuelBit (eligible && low)

theorem intrinsicSpend_le_one (eligible low : Bool) :
    intrinsicSpend eligible low ≤ 1 := by
  cases eligible <;> cases low <;> decide

/-- Exact correspondence with
`move_depth = depth - 1 - (eligible and low)` at the armed depths,
where the Python expression includes `not root` in `eligible`. -/
theorem intrinsic_child_depth (depth : Nat) (hdepth : 2 ≤ depth)
    (eligible low : Bool) :
    depth - 1 - fuelBit (eligible && low) =
      depth - 1 - min (2 - 1) (intrinsicSpend eligible low) := by
  cases eligible <;> cases low <;> simp [fuelBit, intrinsicSpend] <;> omega

/-- Every intrinsic-LMR edge spends between one and two plies. -/
theorem intrinsic_edge_cost (depth : Nat) (hdepth : 2 ≤ depth)
    (eligible low : Bool) :
    depth - 2 ≤ depth - 1 - fuelBit (eligible && low) ∧
      depth - 1 - fuelBit (eligible && low) ≤ depth - 1 := by
  cases eligible <;> cases low <;> simp [fuelBit] <;> omega

/-- The production policy as an edge selector. `eligible` is the node's static
guard and `low` is an intrinsic property of the move. Neither the caller's
window nor killer ordering enters. -/
def intrinsicEdgeSpend (G : QSGame) (eligible : G.Pos → Nat → Bool)
    (low : G.Pos → Nat → G.Pos → Bool) : G.Pos → Nat → G.Pos → Nat :=
  fun p d m => intrinsicSpend (eligible p d) (low p d m)

theorem intrinsicEdgeSpend_le_one (G : QSGame) (eligible : G.Pos → Nat → Bool)
    (low : G.Pos → Nat → G.Pos → Bool) (p m : G.Pos) (d : Nat) :
    intrinsicEdgeSpend G eligible low p d m ≤ 1 :=
  intrinsicSpend_le_one _ _

/-- Intrinsic LMR inherits eventual mate completeness directly from the
edge-generic depth theorem. Every real edge costs at most two plies. -/
theorem forcedMate_intrinsicValue (G : QSGame) (guard : G.Pos → Bool)
    (eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 2 * k + 4 ≤ D →
      MATE_LOWER ≤ fuelValueD2 G guard 2 (intrinsicEdgeSpend G eligible low) D p :=
  forcedMate_fuelValueD2 G guard 2 (intrinsicEdgeSpend G eligible low) (by omega) hF hFM

/-- On a game tree ending within `N` plies, intrinsic LMR preserves the
full eventual W/D/L classification from depth `2*N + 8` onward. -/
theorem eventual_classification_intrinsic_finite (G : QSGame) (guard : G.Pos → Bool)
    (eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {N : Nat} (p : G.Pos) (hE : EndsWithin G N p)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    ∀ D : Nat, 2 * N + 8 ≤ D →
      ((MATE_LOWER ≤ fuelValueD2 G guard 2 (intrinsicEdgeSpend G eligible low) D p) ↔
        (∃ k, ForcedMate G k p)) ∧
      ((fuelValueD2 G guard 2 (intrinsicEdgeSpend G eligible low) D p ≤ -MATE_LOWER) ↔
        (∃ k, ForcedlyMated G k p)) ∧
      ((¬ (∃ k, ForcedMate G k p)) → (¬ (∃ k, ForcedlyMated G k p)) →
        -MATE_LOWER < fuelValueD2 G guard 2 (intrinsicEdgeSpend G eligible low) D p ∧
          fuelValueD2 G guard 2 (intrinsicEdgeSpend G eligible low) D p < MATE_LOWER) := by
  intro D hD
  exact eventual_classification_fuel_finite G guard 2 (intrinsicEdgeSpend G eligible low)
    (by omega) hF p hE hcapf hkg D (by omega)

end Sunfish
