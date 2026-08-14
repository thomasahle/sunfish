/-
The concrete finite-horizon policy used by sunfish's pawn-protected move debt.

Piece ranks are P=1, N=2, B=3, R=4, Q=5.  Kings are deliberately outside
the policy.  A quiet B/R/Q move into an enemy pawn's protection consumes one
extra depth unit exactly when `depth < 2 * rank`.  This is the arithmetic form
of Python's `piece in "PNBRQ"[depth // 2:]`.
-/

import Sunfish.IntrinsicReduction

namespace Sunfish

structure PawnDebtMove where
  rank : Nat
  quiet : Bool
  pawnProtected : Bool

def PawnDebtEligible (root : Bool) (depth : Nat) (m : PawnDebtMove) : Prop :=
  root = false ∧ 3 < depth ∧ m.quiet = true ∧ m.pawnProtected = true ∧
    3 ≤ m.rank ∧ m.rank ≤ 5 ∧ depth < 2 * m.rank

instance pawnDebtEligibleDecidable (root : Bool) (depth : Nat)
    (m : PawnDebtMove) : Decidable (PawnDebtEligible root depth m) := by
  unfold PawnDebtEligible
  infer_instance

def pawnDebtReduce (root : Bool) (depth : Nat) (m : PawnDebtMove) : Bool :=
  decide (PawnDebtEligible root depth m)

/-- The Python slice and the rank inequality select the same B/R/Q suffix. -/
theorem rank_mem_risky_iff (depth rank : Nat) :
    rank > depth / 2 ↔ depth < 2 * rank := by
  omega

/-- Pawn debt is a finite-horizon policy: it is identically off from depth 10.
The real search is therefore restored exactly, not merely approached through
unbounded residual depth. -/
theorem pawnDebtReduce_off (root : Bool) (depth : Nat) (m : PawnDebtMove)
    (hd : 10 ≤ depth) : pawnDebtReduce root depth m = false := by
  simp only [pawnDebtReduce, decide_eq_false_iff_not]
  intro h
  rcases h with ⟨_, _, _, _, _, hrank, hdepth⟩
  omega

/-- The generic fixed-edge proof applies directly to any position-derived
metadata function.  This is the interior-node contract; root moves are
unreduced and call this recurrence in their children. -/
theorem boundPawnDebt_spec (G : Game)
    (meta : G.Pos → G.Pos → PawnDebtMove) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      BoundIntrinsicSpec G
        (fun depth pos move => pawnDebtReduce false depth (meta pos move))
        d p gamma
        (boundIntrinsic G
          (fun depth pos move => pawnDebtReduce false depth (meta pos move))
          d p gamma) :=
  boundIntrinsic_spec G _

end Sunfish
