/-
Intrinsic late-move reduction as a fixed edge cost.

The Python policy reduces a narrow, position-derived class of quiet moves by
one extra depth unit.  Crucially, that class does not depend on `gamma`, the
transposition table, the killer move, or iteration order.  Those mechanisms
may decide when a move is searched, but not the depth that defines its value.

This file proves the resulting recurrence satisfies the same fail-soft
contract as `Searcher.bound`.  The policy is abstract because the important
correctness premise is its immutability, not the particular chess predicate.
-/

import Sunfish.Bound

namespace Sunfish

/-- Child depth for a real edge at a node with nominal depth `d + 1`.
An ordinary edge spends one unit; an intrinsically reduced edge spends two.
At the shallow boundary, truncated subtraction still leaves depth zero. -/
def intrinsicChildDepth {α : Type _}
    (reduce : Nat → α → α → Bool) (d : Nat) (p m : α) : Nat :=
  if reduce (d + 1) p m then d - 1 else d

theorem intrinsicChildDepth_lt {α : Type _}
    (reduce : Nat → α → α → Bool) (d : Nat) (p m : α) :
    intrinsicChildDepth reduce d p m < d + 1 := by
  simp only [intrinsicChildDepth]
  split <;> omega

/-- The fixed selective value.  `reduce depth p m` is part of the value
definition, so every window at `(p, depth)` targets this same recurrence. -/
def negamaxIntrinsic (G : Game) (reduce : Nat → G.Pos → G.Pos → Bool) :
    Nat → G.Pos → Int
  | 0, p => G.eval p
  | d + 1, p =>
    foldMax
      (fun m => -(negamaxIntrinsic G reduce
        (intrinsicChildDepth reduce d p m) m))
      (G.moves p) LOSS
termination_by d _ => d
decreasing_by exact intrinsicChildDepth_lt reduce _ _ _

/-- Null-window search for the same fixed-edge recurrence. -/
def boundIntrinsic (G : Game) (reduce : Nat → G.Pos → G.Pos → Bool) :
    Nat → G.Pos → Int → Int
  | 0, p, _gamma => G.eval p
  | d + 1, p, gamma =>
    searchMoves gamma
      (fun m => -(boundIntrinsic G reduce
        (intrinsicChildDepth reduce d p m) m (1 - gamma)))
      (G.moves p) LOSS
termination_by d _ _ => d
decreasing_by exact intrinsicChildDepth_lt reduce _ _ _

def BoundIntrinsicSpec (G : Game)
    (reduce : Nat → G.Pos → G.Pos → Bool)
    (d : Nat) (p : G.Pos) (gamma r : Int) : Prop :=
  (gamma ≤ r → r ≤ negamaxIntrinsic G reduce d p) ∧
    (r < gamma → negamaxIntrinsic G reduce d p ≤ r)

/-- Fixed intrinsic reductions preserve the fail-soft contract.  Strong
induction is the only difference from the ordinary one-ply proof: a child can
be at either `d` or `d - 1`, and both are strictly below `d + 1`. -/
theorem boundIntrinsic_spec (G : Game)
    (reduce : Nat → G.Pos → G.Pos → Bool) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      BoundIntrinsicSpec G reduce d p gamma
        (boundIntrinsic G reduce d p gamma) := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    cases d with
    | zero =>
      intro p gamma
      refine ⟨fun _ => ?_, fun _ => ?_⟩ <;>
        (simp only [boundIntrinsic, negamaxIntrinsic]; omega)
    | succ d =>
      intro p gamma
      have hchild : ∀ m : G.Pos,
          (gamma ≤ -(boundIntrinsic G reduce
              (intrinsicChildDepth reduce d p m) m (1 - gamma)) →
            -(boundIntrinsic G reduce
              (intrinsicChildDepth reduce d p m) m (1 - gamma)) ≤
              -(negamaxIntrinsic G reduce
                (intrinsicChildDepth reduce d p m) m)) ∧
          (-(boundIntrinsic G reduce
              (intrinsicChildDepth reduce d p m) m (1 - gamma)) < gamma →
            -(negamaxIntrinsic G reduce
              (intrinsicChildDepth reduce d p m) m) ≤
              -(boundIntrinsic G reduce
                (intrinsicChildDepth reduce d p m) m (1 - gamma))) := by
        intro m
        have hs := ih (intrinsicChildDepth reduce d p m)
          (intrinsicChildDepth_lt reduce d p m) m (1 - gamma)
        have h1 := hs.1
        have h2 := hs.2
        constructor
        · intro hge
          have hlt : boundIntrinsic G reduce
              (intrinsicChildDepth reduce d p m) m (1 - gamma) <
              1 - gamma := by omega
          have := h2 hlt
          omega
        · intro hlt
          have hge : 1 - gamma ≤ boundIntrinsic G reduce
              (intrinsicChildDepth reduce d p m) m (1 - gamma) := by omega
          have := h1 hge
          omega
      have h := searchMoves_spec gamma
        (fun m => -(boundIntrinsic G reduce
          (intrinsicChildDepth reduce d p m) m (1 - gamma)))
        (fun m => -(negamaxIntrinsic G reduce
          (intrinsicChildDepth reduce d p m) m))
        hchild (G.moves p) LOSS LOSS
        (fun _ => Int.le_refl LOSS)
        (fun _ => Int.le_refl LOSS)
      simpa only [BoundIntrinsicSpec, boundIntrinsic, negamaxIntrinsic] using h

/-- An intrinsically reduced edge never consumes more than two units of
nominal depth.  This is the local fuel fact behind eventual widening. -/
theorem intrinsicChildDepth_ge_sub_two {α : Type _}
    (reduce : Nat → α → α → Bool) (d : Nat) (p m : α) :
    d + 1 - 2 ≤ intrinsicChildDepth reduce d p m := by
  simp only [intrinsicChildDepth]
  split <;> omega

end Sunfish
