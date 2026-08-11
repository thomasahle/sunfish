/-
The core deliverable: sunfish's fail-soft null-window search, without the
transposition table, and a sorry-free proof of its docstring.

sunfish.py, `Searcher.bound` (lines 300-304):

    def bound(self, pos, gamma, depth, root=False):
        """ Let s* be the "true" score of the sub-tree we are searching.
            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound) """

Here `s*` is `negamax G depth p` and the two clauses are `BoundSpec` below.

The proven model keeps exactly the skeleton of sunfish's move loop
(sunfish.py lines 378-388) and the null-window recursion on children
(line 376: `-self.bound(pos.move(move), 1 - gamma, depth - 1)`).
Everything else -- the transposition table (the `tp_score` lookup and
the table-part-2 store), the repetition check, the null move, the QS
stand-pat and futility pruning, killer/IID, and the stalemate
correction -- is deliberately absent here; the interesting ones are
stated (with the hypotheses they need) in `Sunfish/Tricks.lean` and
`Sunfish/Stalemate.lean`.  (Anchors, not line numbers: this file's
model is the bare move loop, and the citations above drifted twice
before being replaced with names.)
-/

import Sunfish.GameTree

namespace Sunfish

/-- The move loop of `bound`, sunfish.py lines 378-388:

        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)          -- `max best (score m)` below
            if best >= gamma:                -- the early (beta) cutoff
                ...
                break

`score m` is the (already negated) null-window score of child `m`.
Fail-soft: on cutoff we return `best` itself, not `gamma`. -/
def searchMoves {α : Type _} (gamma : Int) (score : α → Int) :
    List α → Int → Int
  | [], best => best
  | m :: ms, best =>
    if gamma ≤ max best (score m)       -- if best >= gamma: break  (line 382)
    then max best (score m)
    else searchMoves gamma score ms (max best (score m))

/-- Pure recursive `bound`, no transposition table.

* depth 0: return the static eval (stand pat, the `depth == 0` case of
  sunfish.py line 335-336, with QS collapsed into `eval`).
* depth d+1: run the move loop; every child is searched with the flipped
  null window `1 - gamma` one ply shallower, exactly sunfish.py line 376. -/
def bound (G : Game) : Nat → G.Pos → Int → Int
  | 0, p, _gamma => G.eval p
  | d + 1, p, gamma =>
    searchMoves gamma (fun m => -(bound G d m (1 - gamma))) (G.moves p) LOSS

/-- The docstring of `Searcher.bound` (sunfish.py lines 287-290), verbatim:
with `s* = negamax G d p`,

* if `r ≥ gamma` then `r` is a valid *lower* bound: `gamma ≤ r ≤ s*`
  (the first clause also gives `gamma ≤ r` for free);
* if `r < gamma` then `r` is a valid *upper* bound: `s* ≤ r < gamma`. -/
def BoundSpec (G : Game) (d : Nat) (p : G.Pos) (gamma r : Int) : Prop :=
  (gamma ≤ r → r ≤ negamax G d p) ∧ (r < gamma → negamax G d p ≤ r)

/-- The heart of the proof: the move loop preserves fail-soft correctness.

`f m` is what the search reports for child move `m`, `w m` is the true
(negated-child-negamax) value of move `m`.  `hchild` says every child
report is itself fail-soft correct -- this is what the induction hypothesis
on depth provides.  The invariant carried through the loop is that the
running `best` is a fail-soft-correct report for the true running maximum
`acc`; the fold on the right is exactly the fold defining `negamax`. -/
theorem searchMoves_spec {α : Type _} (gamma : Int) (f w : α → Int)
    (hchild : ∀ m, (gamma ≤ f m → f m ≤ w m) ∧ (f m < gamma → w m ≤ f m)) :
    ∀ (ms : List α) (best acc : Int),
      (gamma ≤ best → best ≤ acc) →
      (best < gamma → acc ≤ best) →
      (gamma ≤ searchMoves gamma f ms best →
        searchMoves gamma f ms best ≤ foldMax w ms acc) ∧
      (searchMoves gamma f ms best < gamma →
        foldMax w ms acc ≤ searchMoves gamma f ms best) := by
  intro ms
  induction ms with
  | nil =>
    -- Loop over: `best` reports for `acc`, by assumption.
    intro best acc h1 h2
    simp only [searchMoves, foldMax]
    exact ⟨h1, h2⟩
  | cons m ms ih =>
    intro best acc h1 h2
    have hm1 := (hchild m).1
    have hm2 := (hchild m).2
    simp only [searchMoves, foldMax]
    by_cases hcut : gamma ≤ max best (f m)
    · -- Early cutoff (sunfish.py line 382): we fail high with best' ≥ gamma.
      -- Soundness: best' ≤ max acc (w m) ≤ foldMax over the rest.
      rw [if_pos hcut]
      have hrest := foldMax_ge_init w ms (max acc (w m))
      constructor
      · intro _
        -- Either f m ≥ gamma (then f m ≤ w m by hchild) or best ≥ gamma
        -- (then best ≤ acc by the invariant); both give best' ≤ max acc (w m).
        by_cases hf : gamma ≤ f m
        · have := hm1 hf
          by_cases hb : gamma ≤ best
          · have := h1 hb; omega
          · omega
        · have hb : gamma ≤ best := by omega
          have := h1 hb
          omega
      · intro hlt; omega
    · -- No cutoff: best' < gamma, so this child failed low (f m < gamma) and
      -- so did the previous best.  The invariant is re-established for
      -- best' = max best (f m) against acc' = max acc (w m).
      rw [if_neg hcut]
      have hf : f m < gamma := by omega
      have hb : best < gamma := by omega
      have hwm := hm2 hf
      have hacc := h2 hb
      exact ih (max best (f m)) (max acc (w m))
        (fun hge => absurd hge hcut)
        (fun _ => by omega)

/-- **Main theorem.** `bound` satisfies its docstring at every depth, every
position and every test value `gamma`.  Sorry-free. -/
theorem bound_spec (G : Game) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      BoundSpec G d p gamma (bound G d p gamma) := by
  intro d
  induction d with
  | zero =>
    -- depth 0: bound returns eval p = negamax 0 p exactly, so it is both
    -- a valid lower and a valid upper bound.
    intro p gamma
    refine ⟨fun _ => ?_, fun _ => ?_⟩ <;> (simp only [bound, negamax]; omega)
  | succ d ih =>
    intro p gamma
    -- Each child report -(bound d m (1-gamma)) is fail-soft correct for the
    -- negated child value -(negamax d m).  This is where the null-window
    -- integer trick lives: over Int,  b < 1 - gamma  ↔  gamma ≤ -b,
    -- so a child that fails low against 1-gamma fails high for us and
    -- vice versa (sunfish.py line 376: `1 - gamma`).
    have hchild : ∀ m : G.Pos,
        (gamma ≤ -(bound G d m (1 - gamma)) →
          -(bound G d m (1 - gamma)) ≤ -(negamax G d m)) ∧
        (-(bound G d m (1 - gamma)) < gamma →
          -(negamax G d m) ≤ -(bound G d m (1 - gamma))) := by
      intro m
      have h1 := (ih m (1 - gamma)).1
      have h2 := (ih m (1 - gamma)).2
      constructor
      · intro hge
        have hlt : bound G d m (1 - gamma) < 1 - gamma := by omega
        have := h2 hlt
        omega
      · intro hlt
        have hge : 1 - gamma ≤ bound G d m (1 - gamma) := by omega
        have := h1 hge
        omega
    -- The initial accumulators agree: best = acc = LOSS (sunfish.py line 379),
    -- which trivially reports for itself.
    have h := searchMoves_spec gamma
      (fun m => -(bound G d m (1 - gamma)))
      (fun m => -(negamax G d m))
      hchild (G.moves p) LOSS LOSS
      (fun _ => Int.le_refl LOSS)
      (fun _ => Int.le_refl LOSS)
    simpa only [BoundSpec, bound, negamax] using h

end Sunfish
