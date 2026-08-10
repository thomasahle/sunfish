/-
pst-swap soundness (milestone 2, part D): why `search` may retarget the
evaluation between searches -- `pst["K"] = K_MID if "Q" in pos.board and
"q" in pos.board else K_END`, assigned in BOTH directions on every
search -- while keeping `tp_move` and clearing `tp_score`.

Two tables, two fates:

* **`tp_score` bounds are EVAL-RELATIVE.**  Every entry brackets the
  declared value of the game IN FORCE when it was stored -- the keyed
  invariant `CTableOK G ...` (`Sunfish/CanNull.lean`) is G-indexed, and
  the evaluation is part of `G`.  `tableEntries_eval_relative` below
  machine-checks the dependence: an EXACT entry for one evaluation
  violates the invariant for the other.  So the swap without the clear
  would be unsound -- and the clear (`self.tp_score.clear()`, every
  search) restores the invariant for the NEW evaluation
  unconditionally, because the empty table satisfies `CTableOK` for ANY
  game (`ctableOK_empty`, cited, not reproven).

* **`tp_move` survives the swap.**  `KillerInv`
  (`Sunfish/Stalemate.lean`, the `tp_move` lifecycle) is
  position-intrinsic up to the ONE eval-shaped notion it mentions: the
  king-gone CLASSIFICATION `eval ≤ -MATE_LOWER`.  K_MID/K_END move king
  PLACEMENT values only; whether a king is on the board is the material
  term (`piece["K"]`, shared by both tables, with the mate band ±13
  queens clear of any placement delta).  Under that classification
  agreement (`SameKingClass`) the whole invariant transfers
  (`killerInv_withEval`), so `KillerLegal` -- the premise every layered
  spec consumes -- keeps holding across the swap with no re-derivation
  (`killerLegal_withEval`, and end-to-end
  `killerLegal_lifecycle_pstSwap`, the cross-search mirror of
  `killerLegal_lifecycle`).

The review-edit rationale (Thomas, `search`): the assignment runs in
both directions every search precisely so that the evaluation in force
is a function of the current position alone, never of module history --
reused processes start new games with this module state, and `tp_move`
entries stored under one K-table are then consumed under the other.
That consumption is sound exactly because the lifecycle invariant is
eval-independent modulo `SameKingClass`; the eval-RELATIVE table is the
one that gets cleared.
-/

import Sunfish.Stalemate
import Sunfish.CanNull

namespace Sunfish

/-- The same game under a new evaluation: `Pos`, `moves` and `pass`
(the board and the rules) unchanged; `eval` (`pos.score`) and `val`
(`pos.value(move)`) retargeted -- exactly what the `pst["K"]` swap does
between searches. -/
def withEval (G : QSGame) (eval' : G.Pos → Int) (val' : G.Pos → G.Pos → Int) :
    QSGame where
  Pos := G.Pos
  moves := G.moves
  eval := eval'
  pass := G.pass
  val := val'

/-- The two evaluations agree on the king-gone CLASSIFICATION:
`eval ≤ -MATE_LOWER` means "this side's king is off the board", a
material fact the K_MID/K_END placement swap never moves. -/
def SameKingClass (G : QSGame) (eval' : G.Pos → Int) : Prop :=
  ∀ p, G.eval p ≤ -MATE_LOWER ↔ eval' p ≤ -MATE_LOWER

/-- King-capturability -- the legality notion `KillerInv` speaks of --
is invariant under the swap, because it only consults the king-gone
classification of the children. -/
theorem hasKingCapture_withEval (G : QSGame) (eval' : G.Pos → Int)
    (val' : G.Pos → G.Pos → Int) (hcls : SameKingClass G eval') (p : G.Pos) :
    hasKingCapture (withEval G eval' val').toNullGame.toGame p
      = hasKingCapture G.toNullGame.toGame p := by
  cases h : hasKingCapture G.toNullGame.toGame p with
  | true =>
    obtain ⟨m, hm, he⟩ := (hasKingCapture_iff G.toNullGame.toGame p).mp h
    exact (hasKingCapture_iff (withEval G eval' val').toNullGame.toGame p).mpr
      ⟨m, hm, (hcls m).mp he⟩
  | false =>
    cases h' : hasKingCapture (withEval G eval' val').toNullGame.toGame p with
    | false => rfl
    | true =>
      exfalso
      obtain ⟨m, hm, he⟩ :=
        (hasKingCapture_iff (withEval G eval' val').toNullGame.toGame p).mp h'
      have hG : hasKingCapture G.toNullGame.toGame p = true :=
        (hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, (hcls m).mpr he⟩
      rw [h] at hG
      exact Bool.noConfusion hG

/-- **The `tp_move` lifecycle invariant transfers across the pst
swap**: a stored move is still a generated move (the rules did not
change) that either wins the king (a classification both evaluations
agree on) or is legal (`hasKingCapture_withEval`). -/
theorem killerInv_withEval (G : QSGame) (eval' : G.Pos → Int)
    (val' : G.Pos → G.Pos → Int) (hcls : SameKingClass G eval')
    {t : KillTable G} (ht : KillerInv G t) :
    KillerInv (withEval G eval' val') t := by
  intro p m hpm
  obtain ⟨hm, hor⟩ := ht p m hpm
  refine ⟨hm, ?_⟩
  cases hor with
  | inl hcapv => exact Or.inl ((hcls m).mp hcapv)
  | inr hleg =>
    refine Or.inr ?_
    rw [hasKingCapture_withEval G eval' val' hcls m]
    exact hleg

/-- The consumer form: `KillerLegal` -- the premise of every layered
spec -- holds for the swapped game over the surviving table. -/
theorem killerLegal_withEval (G : QSGame) (eval' : G.Pos → Int)
    (val' : G.Pos → G.Pos → Int) (hcls : SameKingClass G eval')
    {t : KillTable G} (ht : KillerInv G t) :
    KillerLegal (withEval G eval' val') (fun p => (t p).isSome) :=
  killerLegal_of_inv (withEval G eval' val') (killerInv_withEval G eval' val' hcls ht)

/-- **End-to-end**: a `tp_move` table built by ANY store trace under
the old evaluation serves the new one -- `killerLegal_lifecycle`'s
conclusion survives the swap.  (Stores made after the swap are
`KillStore (withEval G ...)` events and continue the same lifecycle
through `killerInv_step`; only the initial-table argument changes.) -/
theorem killerLegal_lifecycle_pstSwap (G : QSGame) [DecidableEq G.Pos]
    (eval' : G.Pos → Int) (val' : G.Pos → G.Pos → Int)
    (hcls : SameKingClass G eval') (es : List (KillStore G)) :
    KillerLegal (withEval G eval' val')
      (fun p => ((es.foldl applyStore (fun _ => none)) p).isSome) :=
  killerLegal_withEval G eval' val' hcls
    (killerInv_trace G (killerInv_empty G) es)

/-! ### The other table: `tp_score` bounds are eval-relative -/

/-- A one-position game under evaluation 5... -/
def PstA : NullGame where
  Pos := Unit
  moves := fun _ => []
  eval := fun _ => 5
  pass := fun _ => ()

/-- ...and the SAME position under evaluation 7 -- the pst-swap shape:
same board, same rules, different score. -/
def PstB : NullGame where
  Pos := Unit
  moves := fun _ => []
  eval := fun _ => 7
  pass := fun _ => ()

/-- One EXACT stored entry, keyed at depth 0, typed for `PstA`. -/
def pstEntry : Table PstA.toGame :=
  ⟨fun d _ => if d = 0 then some (5, 5) else none⟩

/-- The SAME entries typed for `PstB` -- `Table` is game-indexed, so
the "table that outlives the swap" is this literal re-key; the `find`
functions are syntactically identical. -/
def pstEntryB : Table PstB.toGame :=
  ⟨fun d _ => if d = 0 then some (5, 5) else none⟩

/-- **The eval-dependence of the keyed table invariant,
machine-checked**: the exact entry satisfies `CTableOK` for the
evaluation it was proven against and VIOLATES it for the swapped one.
This is why `search` must clear `tp_score` when the pst assignment can
change the evaluation -- and `ctableOK_empty` (any game, any history)
is why clearing suffices. -/
theorem tableEntries_eval_relative :
    CTableOK PstA (fun _ => false) pstEntry ∧
      ¬ CTableOK PstB (fun _ => false) pstEntryB := by
  constructor
  · intro d p lo hi h
    cases d with
    | zero =>
      have h5 : pstEntry.find 0 p = some (5, 5) := rfl
      rw [h5] at h
      injection h with hpair
      injection hpair with h1 h2
      have hv : nullValue PstA (fun _ => false) 0 p = 5 := rfl
      rw [hv]
      omega
    | succ d =>
      have hn : pstEntry.find (d + 1) p = none := by
        show (if d + 1 = 0 then some ((5 : Int), (5 : Int)) else none) = none
        rw [if_neg (by omega)]
      rw [hn] at h
      exact Option.noConfusion h
  · intro h
    have hb := h 0 () 5 5 rfl
    have hv : nullValue PstB (fun _ => false) 0 () = 7 := rfl
    rw [hv] at hb
    omega

end Sunfish
