/-
Abstract finite game trees for the sunfish formalization.

This file is deliberately generic: no chess rules, no board representation.
A `Game` is just a type of positions, a legal-move function (an empty move
list means the position is terminal, i.e. the side to move has lost -- in
sunfish terms, the king was captured or there are no pseudo-legal moves),
and a static evaluation.

`negamax` is the depth-limited negamax value: this is the "true score s*"
that the docstring of `Searcher.bound` in sunfish.py (lines 287-290) talks
about.
-/

namespace Sunfish

/-- Scores are bounded by `MATE_UPPER` in sunfish (sunfish.py line 123:
`MATE_UPPER = piece["K"] + 10 * piece["Q"] = 69290`). We only need it as
a named constant; no proof depends on its exact value. -/
def MATE_UPPER : Int := 69290

/-- The score of a position in which the side to move has no moves.
Corresponds to `best = -MATE_UPPER` at sunfish.py line 379: if no move
(not even the QS stand-pat) raises `best`, the position counts as lost. -/
def LOSS : Int := -MATE_UPPER

/-- An abstract two-player zero-sum game.

* `Pos`   -- positions, always seen from the side to move (sunfish rotates
            the board on every move, sunfish.py `Position.rotate`).
* `moves` -- the list of successor positions; `[]` means terminal.
* `eval`  -- static evaluation from the point of view of the side to move
            (`pos.score` in sunfish.py). -/
structure Game where
  Pos : Type
  moves : Pos → List Pos
  eval : Pos → Int

/-- `foldMax w ms acc` = the maximum of `acc` and `w m` for all `m` in `ms`.
A first-order fold so that `negamax` and the lemmas about the search loop
talk about literally the same function. -/
def foldMax {α : Type _} (w : α → Int) : List α → Int → Int
  | [], acc => acc
  | m :: ms, acc => foldMax w ms (max acc (w m))

/-- `acc` never decreases along the fold. -/
theorem foldMax_ge_init {α : Type _} (w : α → Int) :
    ∀ (ms : List α) (acc : Int), acc ≤ foldMax w ms acc := by
  intro ms
  induction ms with
  | nil => intro acc; exact Int.le_refl acc
  | cons m ms ih =>
    intro acc
    have h1 : acc ≤ max acc (w m) := by omega
    have h2 := ih (max acc (w m))
    exact Int.le_trans h1 h2

/-- Depth-limited negamax.

* At depth 0 we return the static evaluation (in sunfish this is where
  quiescence search starts; in the proven model QS is collapsed into `eval`,
  see `formal/README.md`).
* At depth `d+1` the value is the maximum over all moves of the negated
  value of the child, starting from `LOSS` so that a terminal position
  (no moves) is lost for the side to move. -/
def negamax (G : Game) : Nat → G.Pos → Int
  | 0, p => G.eval p
  | d + 1, p => foldMax (fun m => -(negamax G d m)) (G.moves p) LOSS

end Sunfish
