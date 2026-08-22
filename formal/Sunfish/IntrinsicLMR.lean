/-
The local obligations introduced by intrinsic LMR.

The null probe uses a fixed target, so valid fail-soft reports for the same
child value determine the same hot bit under every caller window and table
state. At an interior node, the `eligible` bit is the static null-move guard.
A real edge spends one ply normally, one more at a hot node, and one more for
an intrinsically low move at an eligible node. The unstored driver root has a
separate fixed selector: its hot bit is charged only to intrinsically low
moves. Thus child depth is always a function of position, nominal depth, and
move alone. Interior edges spend one to three plies; driver edges spend one
or two.
-/

import Sunfish.EventuallyFinite

namespace Sunfish

def fuelBit (b : Bool) : Nat := if b then 1 else 0

/-- Extra fuel spent by the code: the node-wide hot bit plus the
move-specific low-value bit, enabled only by the static eligibility guard. -/
def intrinsicSpend (hot eligible low : Bool) : Nat :=
  fuelBit hot + fuelBit (eligible && low)

theorem intrinsicSpend_le_two (hot eligible low : Bool) :
    intrinsicSpend hot eligible low ≤ 2 := by
  cases hot <;> cases eligible <;> cases low <;> decide

/-- The unstored driver's selector: a hot root reduces only the intrinsic
tail. This is separate from `intrinsicSpend`, which models keyed interior
nodes and is unchanged. -/
def rootIntrinsicSpend (hot low : Bool) : Nat := fuelBit (hot && low)

theorem rootIntrinsicSpend_le_one (hot low : Bool) :
    rootIntrinsicSpend hot low ≤ 1 := by
  cases hot <;> cases low <;> decide

/-- Exact correspondence with the driver's
`depth - 1 - (nmr and val < LMR)` recursion. -/
theorem rootIntrinsic_child_depth (depth : Nat) (hot low : Bool) :
    depth - 1 - fuelBit (hot && low) = depth - 1 - rootIntrinsicSpend hot low := by
  rfl

/-- Every driver-root edge spends between one and two plies. -/
theorem rootIntrinsic_edge_cost (depth : Nat) (hdepth : 2 ≤ depth)
    (hot low : Bool) :
    depth - 2 ≤ depth - 1 - fuelBit (hot && low) ∧
      depth - 1 - fuelBit (hot && low) ≤ depth - 1 := by
  cases hot <;> cases low <;> simp [fuelBit] <;> omega

/-- The fixed driver-root policy over positions and moves. Unlike the keyed
interior selector below, it has no `eligible` input: only hot, low-valued
moves pay the extra unit. -/
def rootIntrinsicEdgeSpend (G : QSGame) (hot : G.Pos → Nat → Bool)
    (low : G.Pos → Nat → G.Pos → Bool) : G.Pos → Nat → G.Pos → Nat :=
  fun p d m => rootIntrinsicSpend (hot p d) (low p d m)

theorem rootIntrinsicEdgeSpend_le_one (G : QSGame) (hot : G.Pos → Nat → Bool)
    (low : G.Pos → Nat → G.Pos → Bool) (p m : G.Pos) (d : Nat) :
    rootIntrinsicEdgeSpend G hot low p d m ≤ 1 :=
  rootIntrinsicSpend_le_one _ _

/-- Exact correspondence with the interior specialization of the current
Python expression, where `not root` is true. -/
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

/-- The production interior policy as an edge selector. `hot` is a
fixed-target property of the node, `eligible` is its static guard, and `low`
is an intrinsic property of the move. Neither the caller's window nor killer
ordering enters. The unstored driver uses `rootIntrinsicSpend` instead. -/
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
