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

/-- The mate threshold (sunfish.py line 122:
`MATE_LOWER = piece["K"] - 13 * piece["Q"] = 47923`).  A position whose
static score is `≤ -MATE_LOWER` has already lost its king (sunfish.py
lines 298-303); scores of magnitude above `MATE_LOWER` are mate scores. -/
def MATE_LOWER : Int := 47923

/-- The score of a position in which the side to move has no moves.
Corresponds to `best = -MATE_UPPER` at sunfish.py line 379: if no move
(not even the QS stand-pat) raises `best`, the position counts as lost. -/
def LOSS : Int := -MATE_UPPER

/-- `EVAL_ROUGHNESS` (sunfish.py's search constants).  It is the width
the MTD-bi bracket stops at -- and, since `terminalValue` scales mate
distance by it, also the score one ply of mate distance is worth. -/
def EVAL_ROUGHNESS : Int := 15

/-! The exact value the terminal correction assigns to CHECKMATE is
`1 - MATE_UPPER`:

```python
best = 1 - MATE_UPPER if pos.rotate(nullmove=True).king_capture() else 0
```

Mated HERE is the deepest LEGAL loss there is -- one better than a king
already off the board (`-MATE_UPPER`, which the fold reads as "illegal
move", so the two must stay distinct) -- and `up` walks it back toward
zero one ply at a time.  It is spelled out rather than named so that
`omega` sees the arithmetic at every one of its consumers. -/

/-- **One ply of mate distance** (`up` in sunfish.py): a child's report
`r` as the parent sees it.  Negate it, and step a mate score one place
toward zero, so a faster mate outscores a slower one.

    def up(r): return -r + (MATE_LOWER <= abs(r) < MATE_UPPER) * ((r > 0) - (r < 0))

The distance is measured FROM THE NODE, not from the root: nothing but
`(pos, depth)` enters, so the transposition table needs no store/probe
adjustment and keeps one value per key.  The band EDGES are fixed
points of the step: `up MATE_UPPER = -MATE_UPPER` keeps the
illegal-move sentinel exact (the fold's `score > -MATE_UPPER` legality
test reads it literally) and `up (-MATE_UPPER) = MATE_UPPER` keeps the
king-capture promise exact.  Below the band `up` is plain negation. -/
def up (r : Int) : Int :=
  if MATE_LOWER ≤ r ∧ r < MATE_UPPER then 1 - r
  else if -MATE_UPPER < r ∧ r ≤ -MATE_LOWER then -1 - r
  else -r

/-- `up` is antitone: a better child report is a worse parent yield.
This is all the fail-soft transport ever needs of it. -/
theorem up_anti {a b : Int} (h : a ≤ b) : up b ≤ up a := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold up
  split <;> split <;> omega

/-- Outside the mate band `up` is plain negation -- the shape every
non-mate arithmetic step in the development relies on. -/
theorem up_of_quiet {r : Int} (h1 : -MATE_LOWER < r) (h2 : r < MATE_LOWER) :
    up r = -r := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold up
  rw [if_neg (by omega), if_neg (by omega)]

/-- The illegal-move sentinel is a FIXED POINT of the step: a child that
reports `MATE_UPPER` (its king can be taken) still hands the parent
exactly `-MATE_UPPER`, so `live |= score > -MATE_UPPER` is unchanged. -/
theorem up_MATE_UPPER : up MATE_UPPER = -MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold up
  rw [if_neg (by omega), if_neg (by omega)]

/-- Dually, capturing the king NOW is distance zero: a kingless child
hands the parent the full `MATE_UPPER`. -/
theorem up_kingGone : up (-MATE_UPPER) = MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold up
  rw [if_neg (by omega), if_neg (by omega)]
  omega

/-- Delivering mate is `MATE_UPPER - 2`: the checkmated child reports
`1 - MATE_UPPER`, and one step back is the winner's mate-in-one.
Iterating, a mate `k` moves away scores `MATE_UPPER - 2k`, which is the
convention sunfish.py's constant block has claimed since 2014. -/
theorem up_of_mated : up (1 - MATE_UPPER) = MATE_UPPER - 2 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold up
  rw [if_neg (by omega), if_pos (by omega)]
  omega

/-- `up` never leaves the score band. -/
theorem up_bounded {r : Int} (h1 : -MATE_UPPER ≤ r) (h2 : r ≤ MATE_UPPER) :
    -MATE_UPPER ≤ up r ∧ up r ≤ MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold up
  split <;> (try split) <;> omega

/-- The step never overshoots plain negation by more than one, in either
direction -- the sandwich that carries every `omega` the old proofs did
with `-r` alone. -/
theorem up_sandwich (r : Int) : -r - 1 ≤ up r ∧ up r ≤ -r + 1 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold up
  split <;> (try split) <;> omega

/-- A parent yield is a mate-band win exactly when the child was a
mate-band loss -- and the step costs one point of margin, which is what
makes the band edge `MATE_LOWER` a horizon rather than a fixed point. -/
theorem up_mate_iff (r : Int) :
    (MATE_LOWER ≤ up r ↔ r ≤ -MATE_LOWER - 1) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold up
  by_cases h1 : MATE_LOWER ≤ r ∧ r < MATE_UPPER
  · rw [if_pos h1]; constructor <;> intro <;> omega
  · rw [if_neg h1]
    by_cases h2 : -MATE_UPPER < r ∧ r ≤ -MATE_LOWER
    · rw [if_pos h2]; constructor <;> intro <;> omega
    · rw [if_neg h2]; constructor <;> intro <;> omega

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

/-- The fold never exceeds an upper bound respected by the initial
accumulator and by every element. -/
theorem foldMax_le {α : Type _} (w : α → Int) {B : Int} :
    ∀ (ms : List α) (acc : Int),
      (∀ m ∈ ms, w m ≤ B) → acc ≤ B → foldMax w ms acc ≤ B := by
  intro ms
  induction ms with
  | nil => intro acc _ hacc; exact hacc
  | cons m ms ih =>
    intro acc hall hacc
    have hm : w m ≤ B := hall m (by simp)
    exact ih (max acc (w m)) (fun x hx => hall x (by simp [hx])) (by omega)

/-- Every member's value is below the fold. -/
theorem foldMax_le_of_mem {α : Type _} (w : α → Int) :
    ∀ (ms : List α) (acc : Int) (m : α), m ∈ ms → w m ≤ foldMax w ms acc := by
  intro ms
  induction ms with
  | nil => intro acc m hm; cases hm
  | cons a ms ih =>
    intro acc m hm
    cases List.mem_cons.mp hm with
    | inl heq =>
      subst heq
      have h1 : w m ≤ max acc (w m) := by omega
      have h2 := foldMax_ge_init w ms (max acc (w m))
      simpa [foldMax] using Int.le_trans h1 h2
    | inr htail =>
      simpa [foldMax] using ih (max acc (w a)) m htail

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

/-- A `Game` together with a pass ("null") move:
`pass p` = `pos.rotate(nullmove=True)` (sunfish.py line 331).  Same board,
other side to move.  Used both by null-move pruning and by the in-check
probe of the stalemate correction (line 409). -/
structure NullGame extends Game where
  pass : Pos → Pos

/-- All static evaluations live in sunfish's score band
`[-MATE_UPPER, MATE_UPPER]` -- true by construction of sunfish's evaluation,
an explicit hypothesis in the abstract model. -/
def Bounded (G : Game) : Prop :=
  ∀ p, -MATE_UPPER ≤ G.eval p ∧ G.eval p ≤ MATE_UPPER

end Sunfish
