/-
The flagless null-move/repetition search and its (pos, depth)-keyed
table, modeled exactly as the code stands since commit eda66ee ("Remove
can_null: driver probes are unstored, the TT key is (pos, depth)");
audit of sunfish.py lines 285-518 at that commit.

`can_null` is gone.  What replaced it:

* INTERIOR calls (`root=False`, the default -- every child search, the
  null pass, the stalemate probe): semantics are uniform.  The null
  yield is gated by position-determined tests alone
  (`depth > 2 and abs(pos.score) < 500`, lines 364-365; the pass is
  searched at `depth - 3` as an interior call, so sunfish still permits
  CONSECUTIVE null moves -- reproduced exactly); the repetition gate
  (`depth > 0 and pos in self.history`, line 341) is always on; lookups
  (334-336) and stores (481-485) go through the table under the key
  `(pos, depth)` -- no flag.
* DRIVER probes (`root=True`: the search root, line 512, and IID, line
  381): skip the table in BOTH directions, skip the repetition-0 and
  the null option, and store nothing.  Every entry in the table
  therefore describes ONE value function -- `nullValue` below,
  determined by (pos, depth) and the per-search `history` -- and the
  doctrine's invariant ("every stored bound describes one value
  function determined by the transposition key") holds by construction
  instead of by key partition.

THE COLLAPSED STORY (was: a two-function layering keyed by
`(pos, depth, can_null)`; git has the last cn-keyed model at eda66ee):

* `boundNullTT_spec` (PROVEN, unconditional): the interior search
  brackets `nullValue` with a POINT spec, and the `(pos, depth)`-keyed
  table stays consistent (`CTableOK`).  No zugzwang hypothesis
  anywhere: self-consistency of search + table is unconditional.
* `rootProbe_spec`, the driver lemma (PROVEN, unconditional): the
  driver probe brackets its own `rootValue` -- the max over real moves
  of the children's interior `nullValue` -- and preserves `CTableOK`
  (its only table effect is through its interior children).
  `rootValue` differs from `nullValue` exactly where a gate would have
  fired (`rootValue_eq_nullValue` below); the divergence is harmless
  BECAUSE the driver never stores: no table entry ever describes
  `rootValue`.  (At the actual search root the difference is not even
  optional: the root position sits in `history`, so interior semantics
  would answer the root probe with the repetition 0 -- "it is in
  history, but not a draw".)
* IID (line 381) is the same rootProbe shape at `depth - 3`, and stores
  nothing under its own key; its purpose -- killer arrival via
  `tp_move` fail-highs inside the probe -- is `Sunfish/Killer.lean`'s
  territory, unchanged here.  The source guard is now
  `not killer and depth > 3`, since a depth-3 probe would enter
  quiescence and cannot store a killer.  This model collapses depth-zero
  quiescence to `eval`, so the corresponding transform in its uniform
  recurrence is definitionally the identity on the table.  At greater
  depths the probe's table effect is its children's interior stores,
  modeled as the table component of the root recursion.
* `nullValue_plain` (PROVEN under `NullBetOK`): relating `nullValue` to
  the null-free `plainValue` is where the null-move BET lives.
  Zugzwang only ever threatens this bridge, never self-consistency: a
  zugzwang position makes the engine compute the wrong VALUE, but never
  makes its table contradict its search.

Exactness notes from the audit (those that survive the collapse):

* Order of the prelude is exact: king-gone check (321-322) BEFORE table
  lookup (332-336) BEFORE repetition (341-342); the early returns
  (king-gone, TT hit, repetition) store nothing, all interior loop
  exits store through Table part 2 (481-485, the plain stores =
  `tablePart2` of `Sunfish/Tricks.lean`, reused here); driver probes
  return without storing (the store sits under `if not root`).
* The generator's LAZINESS is semantically load-bearing (a surprise of
  the original audit, still true): the null yield is pulled first, and
  if it cuts off, the IID recursion never runs -- so the table state
  differs depending on the cutoff.  `cNodeTail` therefore applies the
  IID table-effect only on the no-cutoff path.  A model that ran all
  yields eagerly would mis-model `tp_score`.
* `history` is a FIXED per-search parameter here (`hist`);
  `ctableOK_empty` below is the invariant fact that justifies sunfish
  clearing `tp_score` whenever `history` changes -- the table invariant
  is history-relative, and the empty table satisfies it for any
  history.
* The deadline `Stop` (305-310) raises at node ENTRY, before any store:
  an abort can leave the search unfinished but never a table entry
  unjustified -- aborts cannot corrupt `CTableOK`.  Not modeled, by
  that argument.
* `depth = max(depth, 0)` (line 315) corresponds to this model's use of
  `Nat` depths with saturating subtraction -- verified aligned.
* The stalemate probe (line 468) calls `bound(flipped, MATE_UPPER, 0)`
  as an ordinary interior call, so its table key is `(flipped, 0)`; the
  repetition (depth > 0) and null (depth > 2) gates are dead at depth
  0, so the probe is exactly a depth-0 interior search.
* Not modeled in THIS file (each layered elsewhere): the killer yield
  and `tp_move` (`Sunfish/Killer.lean`), futility
  (`Sunfish/Tricks.lean`), the stalemate block
  (`Sunfish/Stalemate.lean`), QS interior (collapsed to `eval`, see
  README).
-/

import Sunfish.Tricks

namespace Sunfish




/-! ### The gamma-free null-move guard and the value functions -/

/-- `abs(pos.score) < 500` (line 364): the zugzwang heuristic.  Crucially
gamma-free: whether the pass option EXISTS does not depend on the window,
only its search does. -/
def nullGuard (G : Game) (p : G.Pos) : Prop :=
  -500 < G.eval p ∧ G.eval p < 500

instance (G : Game) (p : G.Pos) : Decidable (nullGuard G p) := by
  unfold nullGuard; infer_instance

/-- `nullValue G hist d p`: THE value function of the search -- the one
every stored entry describes, determined by `(pos, depth)` and the
per-search `hist` alone.  King-gone normalization first (321-322); the
repetition gate `depth > 0 ∧ hist p` returns 0 (341-342; `depth > 0` is
carried by the patterns); the pass option, at `depth - 3`, is the fold's
initial accumulator when `depth > 2 ∧ nullGuard` (364-365); children are
interior searches at `depth - 1`.  No flag argument: interior semantics
are uniform. -/
def nullValue (G : NullGame) (hist : G.Pos → Bool) : Nat → G.Pos → Int
  | 0, p => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hist p = true then 0
    else foldMax (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS
  | 2, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hist p = true then 0
    else foldMax (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS
  | d + 3, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hist p = true then 0
    else foldMax (fun m => -(nullValue G hist (d + 2) m)) (G.moves p)
      (if nullGuard G.toGame p then
        max LOSS (-(nullValue G hist d (G.pass p)))
      else LOSS)

/-- `rootValue G hist d p`: what a driver probe (`root=True`) computes at
its own node -- the plain move fold over the children's INTERIOR
`nullValue`, with no repetition-0 and no pass option at the top node
(children are ordinary interior searches, so gates fire below as usual).
This function is never stored: the driver probes skip the table in both
directions, which is exactly why it may differ from `nullValue`
harmlessly (`rootValue_eq_nullValue` pins down where they agree). -/
def rootValue (G : NullGame) (hist : G.Pos → Bool) : Nat → G.Pos → Int
  | 0, p => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else foldMax (fun m => -(nullValue G hist d m)) (G.moves p) LOSS

/-! ### The keyed table: `Table` of Tricks.lean, invariant re-targeted -/

/-- **The keyed-table invariant**: every entry under `(depth, pos)`
brackets `nullValue` at exactly that key.  History-relative.  The table
type is `Table` from `Sunfish/Tricks.lean` -- the key needs no flag,
because the only deviant-semantics calls (the driver probes) never touch
the table; the invariant differs from `TableOK` only in its target value
function. -/
def CTableOK (G : NullGame) (hist : G.Pos → Bool) (t : Table G.toGame) : Prop :=
  ∀ (d : Nat) (p : G.Pos) (lo hi : Int),
    t.find d p = some (lo, hi) →
      lo ≤ nullValue G hist d p ∧ nullValue G hist d p ≤ hi

/-- The repetition gate's bookkeeping fact: the EMPTY table satisfies the
invariant for ANY history -- which is exactly why sunfish may (and must)
clear `tp_score` when `history` changes: entries proven against one
history mean nothing under another, and clearing restores the invariant
trivially. -/
theorem ctableOK_empty (G : NullGame) (hist : G.Pos → Bool) :
    CTableOK G hist ⟨fun _ _ => none⟩ :=
  fun _ _ _ _ h => Option.noConfusion h

/-! ### Helper lemmas -/

theorem nullValue_kingGone (G : NullGame) (hist : G.Pos → Bool) (p : G.Pos)
    (h : G.eval p ≤ -MATE_LOWER) :
    ∀ (d : Nat), nullValue G hist d p = -MATE_UPPER := by
  intro d
  match d with
  | 0 => simp only [nullValue]; rw [if_pos h]
  | 1 => simp only [nullValue]; rw [if_pos h]
  | 2 => simp only [nullValue]; rw [if_pos h]
  | d + 3 => simp only [nullValue]; rw [if_pos h]

theorem nullValue_rep (G : NullGame) (hist : G.Pos → Bool) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) (hr : hist p = true) :
    ∀ (d : Nat), nullValue G hist (d + 1) p = 0 := by
  intro d
  match d with
  | 0 => simp only [nullValue]; rw [if_neg hkg, if_pos hr]
  | 1 => simp only [nullValue]; rw [if_neg hkg, if_pos hr]
  | d + 2 => simp only [nullValue]; rw [if_neg hkg, if_pos hr]

/-- Under `Bounded`, `nullValue` stays in the score band (what validates
the fresh `Entry(-MATE_UPPER, MATE_UPPER)` default of line 334). -/
theorem nullValue_bounded (G : NullGame) (hist : G.Pos → Bool)
    (hb : Bounded G.toGame) :
    ∀ (d : Nat) (p : G.Pos),
      -MATE_UPPER ≤ nullValue G hist d p ∧
      nullValue G hist d p ≤ MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  suffices H : ∀ (n d : Nat), d ≤ n → ∀ (p : G.Pos),
      -MATE_UPPER ≤ nullValue G hist d p ∧
      nullValue G hist d p ≤ MATE_UPPER by
    exact fun d => H d d (Nat.le_refl d)
  intro n
  induction n with
  | zero =>
    intro d hd p
    have hd0 : d = 0 := by omega
    subst hd0
    have := hb p
    simp only [nullValue]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]; omega
    · rw [if_neg hkg]; omega
  | succ n ihn =>
    intro d hd p
    match d, hd with
    | 0, _ =>
      have := hb p
      simp only [nullValue]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]; omega
    | 1, _ =>
      have hunf : nullValue G hist 1 p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if hist p = true then 0
            else foldMax (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS) := rfl
      rw [hunf]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]
        by_cases hr : hist p = true
        · rw [if_pos hr]; omega
        · rw [if_neg hr]
          have hfl := foldMax_ge_init (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS
          have hfu : foldMax (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS
              ≤ MATE_UPPER := by
            refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
            show -(nullValue G hist 0 m) ≤ MATE_UPPER
            have := ihn 0 (by omega) m
            omega
          omega
    | 2, hd =>
      have hunf : nullValue G hist 2 p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if hist p = true then 0
            else foldMax (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS) := rfl
      rw [hunf]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]
        by_cases hr : hist p = true
        · rw [if_pos hr]; omega
        · rw [if_neg hr]
          have hfl := foldMax_ge_init (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS
          have hfu : foldMax (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS
              ≤ MATE_UPPER := by
            refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
            show -(nullValue G hist 1 m) ≤ MATE_UPPER
            have := ihn 1 (by omega) m
            omega
          omega
    | (d + 3), hd =>
      simp only [nullValue]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]
        by_cases hr : hist p = true
        · rw [if_pos hr]; omega
        · rw [if_neg hr]
          have hpass := ihn d (by omega) (G.pass p)
          have hinit : -MATE_UPPER ≤
              (if nullGuard G.toGame p then
                max LOSS (-(nullValue G hist d (G.pass p)))
              else LOSS) ∧
              (if nullGuard G.toGame p then
                max LOSS (-(nullValue G hist d (G.pass p)))
              else LOSS) ≤ MATE_UPPER := by
            by_cases hg : nullGuard G.toGame p
            · rw [if_pos hg]; omega
            · rw [if_neg hg]; omega
          have hfl := foldMax_ge_init (fun m => -(nullValue G hist (d + 2) m))
            (G.moves p)
            (if nullGuard G.toGame p then
              max LOSS (-(nullValue G hist d (G.pass p)))
            else LOSS)
          have hfu : foldMax (fun m => -(nullValue G hist (d + 2) m)) (G.moves p)
              (if nullGuard G.toGame p then
                max LOSS (-(nullValue G hist d (G.pass p)))
              else LOSS) ≤ MATE_UPPER := by
            refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
            show -(nullValue G hist (d + 2) m) ≤ MATE_UPPER
            have := ihn (d + 2) (by omega) m
            omega
          omega

/-- Storing a valid bracket preserves the keyed invariant (the
`nullValue` twin of `tableOK_store`). -/
theorem cTableOK_store {G : NullGame} {hist : G.Pos → Bool} [DecidableEq G.Pos]
    {t : Table G.toGame} {D : Nat} {p : G.Pos} {e' : Int × Int}
    (ht : CTableOK G hist t)
    (h1 : e'.1 ≤ nullValue G hist D p)
    (h2 : nullValue G hist D p ≤ e'.2) :
    CTableOK G hist (Table.store t D p e') := by
  intro d' p' lo hi hfind
  simp only [Table.store] at hfind
  by_cases hk : d' = D ∧ p' = p
  · rw [if_pos hk] at hfind
    injection hfind with hh
    have hl : e'.1 = lo := by rw [hh]
    have hr : e'.2 = hi := by rw [hh]
    rw [hk.1, hk.2]
    constructor
    · rw [← hl]; exact h1
    · rw [← hr]; exact h2
  · rw [if_neg hk] at hfind
    exact ht d' p' lo hi hfind

/-- The current entry (stored, or the fresh default of line 334) is a
valid bracket. -/
theorem cEntry_valid (G : NullGame) (hist : G.Pos → Bool) (hb : Bounded G.toGame)
    {t : Table G.toGame} (ht : CTableOK G hist t) (D : Nat) (p : G.Pos) :
    ((t.find D p).getD (LOSS, MATE_UPPER)).1 ≤ nullValue G hist D p ∧
    nullValue G hist D p ≤ ((t.find D p).getD (LOSS, MATE_UPPER)).2 := by
  cases hfind : t.find D p with
  | none =>
    have hband := nullValue_bounded G hist hb D p
    have hLOSS : LOSS = -MATE_UPPER := rfl
    refine ⟨?_, ?_⟩
    · show LOSS ≤ nullValue G hist D p
      omega
    · show nullValue G hist D p ≤ MATE_UPPER
      omega
  | some e =>
    exact ht D p e.1 e.2 (by rw [hfind])

/-! ### The search -/

/-- The generic state-passing fail-soft loop (generalized over the state
type; `searchMovesTT` in Tricks.lean is the same shape specialized to
`Table`). -/
def searchMovesSt {σ α : Type _} (gamma : Int) (f : α → σ → Int × σ) :
    List α → Int → σ → Int × σ
  | [], best, s => (best, s)
  | m :: ms, best, s =>
    if gamma ≤ max best (f m s).1 then (max best (f m s).1, (f m s).2)
    else searchMovesSt gamma f ms (max best (f m s).1) (f m s).2

theorem searchMovesSt_spec {σ α : Type _} (gamma : Int)
    (f : α → σ → Int × σ) (w : α → Int) (Inv : σ → Prop)
    (hf : ∀ (m : α) (s : σ), Inv s → Inv (f m s).2 ∧
      (gamma ≤ (f m s).1 → (f m s).1 ≤ w m) ∧
      ((f m s).1 < gamma → w m ≤ (f m s).1)) :
    ∀ (ms : List α) (best acc : Int) (s : σ), Inv s →
      (gamma ≤ best → best ≤ acc) →
      (best < gamma → acc ≤ best) →
      Inv (searchMovesSt gamma f ms best s).2 ∧
      (gamma ≤ (searchMovesSt gamma f ms best s).1 →
        (searchMovesSt gamma f ms best s).1 ≤ foldMax w ms acc) ∧
      ((searchMovesSt gamma f ms best s).1 < gamma →
        foldMax w ms acc ≤ (searchMovesSt gamma f ms best s).1) := by
  intro ms
  induction ms with
  | nil =>
    intro best acc s hs h1 h2
    simp only [searchMovesSt, foldMax]
    exact ⟨hs, h1, h2⟩
  | cons m ms ih =>
    intro best acc s hs h1 h2
    have hfm := hf m s hs
    have hm1 := hfm.2.1
    have hm2 := hfm.2.2
    simp only [searchMovesSt, foldMax]
    by_cases hcut : gamma ≤ max best (f m s).1
    · rw [if_pos hcut]
      have hrest := foldMax_ge_init w ms (max acc (w m))
      refine ⟨hfm.1, fun _ => ?_, fun hlt => ?_⟩
      · show max best (f m s).1 ≤ foldMax w ms (max acc (w m))
        by_cases hfge : gamma ≤ (f m s).1
        · have := hm1 hfge
          by_cases hb : gamma ≤ best
          · have := h1 hb; omega
          · omega
        · have hb : gamma ≤ best := by omega
          have := h1 hb
          omega
      · omega
    · rw [if_neg hcut]
      have hfl : (f m s).1 < gamma := by omega
      have hb : best < gamma := by omega
      have hwm := hm2 hfl
      have hacc := h2 hb
      exact ih (max best (f m s).1) (max acc (w m)) (f m s).2 hfm.1
        (fun hge => absurd hge hcut)
        (fun _ => by omega)

/-- The store step on the keyed table: `tablePart2` of Tricks.lean (the
PLAIN stores of master, `Entry(best, entry.upper)` / `Entry(entry.lower,
best)`, lines 481-485) re-proved against `nullValue`.  Interior exits
only: driver probes never reach a store. -/
theorem cTablePart2_ok (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos]
    (D : Nat) (p : G.Pos) (gamma : Int) (e : Int × Int)
    (r : Int × Table G.toGame)
    (htok : CTableOK G hist r.2)
    (he1 : e.1 ≤ nullValue G hist D p)
    (he2 : nullValue G hist D p ≤ e.2)
    (hr1 : gamma ≤ r.1 → r.1 ≤ nullValue G hist D p)
    (hr2 : r.1 < gamma → nullValue G hist D p ≤ r.1) :
    (tablePart2 G.toGame D p gamma e r).1 = r.1 ∧
      CTableOK G hist (tablePart2 G.toGame D p gamma e r).2 := by
  unfold tablePart2
  by_cases hcut : gamma ≤ r.1
  · rw [if_pos hcut]
    refine ⟨rfl, cTableOK_store htok ?_ ?_⟩
    · show r.1 ≤ nullValue G hist D p
      exact hr1 hcut
    · show nullValue G hist D p ≤ e.2
      exact he2
  · rw [if_neg hcut]
    refine ⟨rfl, cTableOK_store htok ?_ ?_⟩
    · show e.1 ≤ nullValue G hist D p
      exact he1
    · show nullValue G hist D p ≤ r.1
      exact hr2 (by omega)

/-- The tail of a searched INTERIOR node: optional null yield first
(whose cutoff skips everything else -- the generator's laziness), then
the IID table effect, then the move loop; plain keyed store on every
exit. -/
def cNodeTail (G : NullGame) [DecidableEq G.Pos] (gamma : Int) (D : Nat)
    (p : G.Pos) (e : Int × Int)
    (iid : Table G.toGame → Table G.toGame)
    (f : G.Pos → Table G.toGame → Int × Table G.toGame)
    (pass? : Option (Table G.toGame → Int × Table G.toGame))
    (t : Table G.toGame) : Int × Table G.toGame :=
  match pass? with
  | some pf =>
    if gamma ≤ max LOSS (-(pf t).1) then
      -- Null cutoff: the loop breaks before pulling another yield, so
      -- the IID recursion never runs (laziness, see module comment).
      tablePart2 G.toGame D p gamma e (max LOSS (-(pf t).1), (pf t).2)
    else
      tablePart2 G.toGame D p gamma e
        (searchMovesSt gamma f (G.moves p) (max LOSS (-(pf t).1)) (iid (pf t).2))
  | none =>
    tablePart2 G.toGame D p gamma e
      (searchMovesSt gamma f (G.moves p) LOSS (iid t))

/-- The node-tail lemma: given a valid old entry, the value-shape
equation for this node, spec/preservation of children and IID, and the
null-yield clauses when present, the tail satisfies the point spec
against `nullValue` and preserves the invariant. -/
theorem cNodeTail_spec (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos]
    (gamma : Int) (D : Nat) (p : G.Pos) (e : Int × Int)
    (iid : Table G.toGame → Table G.toGame)
    (f : G.Pos → Table G.toGame → Int × Table G.toGame)
    (pass? : Option (Table G.toGame → Int × Table G.toGame))
    (t : Table G.toGame) (w : G.Pos → Int) (acc0 : Int)
    (ht : CTableOK G hist t)
    (hiid : ∀ s, CTableOK G hist s → CTableOK G hist (iid s))
    (hf : ∀ (m : G.Pos) (s : Table G.toGame), CTableOK G hist s →
      CTableOK G hist (f m s).2 ∧
      (gamma ≤ (f m s).1 → (f m s).1 ≤ w m) ∧
      ((f m s).1 < gamma → w m ≤ (f m s).1))
    (he1 : e.1 ≤ nullValue G hist D p)
    (he2 : nullValue G hist D p ≤ e.2)
    (hV : nullValue G hist D p = foldMax w (G.moves p) acc0)
    (hpass : ∀ pf, pass? = some pf →
      CTableOK G hist (pf t).2 ∧
      (gamma ≤ max LOSS (-(pf t).1) → max LOSS (-(pf t).1) ≤ acc0) ∧
      (max LOSS (-(pf t).1) < gamma → acc0 ≤ max LOSS (-(pf t).1)))
    (hnone : pass? = none → acc0 = LOSS) :
    ((gamma ≤ (cNodeTail G gamma D p e iid f pass? t).1 →
      (cNodeTail G gamma D p e iid f pass? t).1 ≤ nullValue G hist D p) ∧
     ((cNodeTail G gamma D p e iid f pass? t).1 < gamma →
      nullValue G hist D p ≤ (cNodeTail G gamma D p e iid f pass? t).1)) ∧
    CTableOK G hist (cNodeTail G gamma D p e iid f pass? t).2 := by
  cases pass? with
  | none =>
    have hacc := hnone rfl
    subst hacc
    have hloop := searchMovesSt_spec gamma f w (CTableOK G hist) hf
      (G.moves p) LOSS LOSS (iid t) (hiid t ht)
      (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
    simp only [cNodeTail]
    have htp := cTablePart2_ok G hist D p gamma e
      (searchMovesSt gamma f (G.moves p) LOSS (iid t))
      hloop.1 he1 he2
      (fun hge => by rw [hV]; exact hloop.2.1 hge)
      (fun hlt => by rw [hV]; exact hloop.2.2 hlt)
    refine ⟨⟨fun hge => ?_, fun hlt => ?_⟩, htp.2⟩
    · rw [htp.1] at hge ⊢
      rw [hV]
      exact hloop.2.1 hge
    · rw [htp.1] at hlt ⊢
      rw [hV]
      exact hloop.2.2 hlt
  | some pf =>
    have hp := hpass pf rfl
    simp only [cNodeTail]
    by_cases hcut : gamma ≤ max LOSS (-(pf t).1)
    · rw [if_pos hcut]
      have hfold := foldMax_ge_init w (G.moves p) acc0
      have htp := cTablePart2_ok G hist D p gamma e
        (max LOSS (-(pf t).1), (pf t).2)
        hp.1 he1 he2
        (fun _ => by
          show max LOSS (-(pf t).1) ≤ nullValue G hist D p
          have := hp.2.1 hcut
          rw [hV]
          omega)
        (fun hlt => by omega)
      refine ⟨⟨fun hge => ?_, fun hlt => ?_⟩, htp.2⟩
      · rw [htp.1] at hge ⊢
        have := hp.2.1 hcut
        rw [hV]
        omega
      · rw [htp.1] at hlt
        omega
    · rw [if_neg hcut]
      have hloop := searchMovesSt_spec gamma f w (CTableOK G hist) hf
        (G.moves p) (max LOSS (-(pf t).1)) acc0 (iid (pf t).2)
        (hiid _ hp.1) hp.2.1 hp.2.2
      have htp := cTablePart2_ok G hist D p gamma e
        (searchMovesSt gamma f (G.moves p) (max LOSS (-(pf t).1)) (iid (pf t).2))
        hloop.1 he1 he2
        (fun hge => by rw [hV]; exact hloop.2.1 hge)
        (fun hlt => by rw [hV]; exact hloop.2.2 hlt)
      refine ⟨⟨fun hge => ?_, fun hlt => ?_⟩, htp.2⟩
      · rw [htp.1] at hge ⊢
        rw [hV]
        exact hloop.2.1 hge
      · rw [htp.1] at hlt ⊢
        rw [hV]
        exact hloop.2.2 hlt

/-- `bound` with the full flagless mechanics.  Interior calls
(`root = false`): king-gone (321), keyed lookup (334-336), repetition
(341), null move (364-365), IID as an unstored driver probe (381,
result discarded, table kept), move loop, plain keyed store (481-485).
Driver probes (`root = true`, the search root and IID): king-gone, then
straight to the move loop over ordinary interior children -- no lookup,
no repetition-0, no null yield, no store. Above depth 3 the probe still
runs its own nested IID (table effect only); at depth 3 the model's
depth-zero transform is the identity noted above. -/
def boundNullTT (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos] :
    Nat → Bool → G.Pos → Int → Table G.toGame → Int × Table G.toGame
  | 0, _, p, _gamma, t =>
    (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p, t)
  | 1, false, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else if gamma ≤ ((t.find 1 p).getD (LOSS, MATE_UPPER)).1 then
      (((t.find 1 p).getD (LOSS, MATE_UPPER)).1, t)
    else if ((t.find 1 p).getD (LOSS, MATE_UPPER)).2 < gamma then
      (((t.find 1 p).getD (LOSS, MATE_UPPER)).2, t)
    else if hist p = true then (0, t)
    else
      cNodeTail G gamma 1 p ((t.find 1 p).getD (LOSS, MATE_UPPER))
        (fun s => s)
        (fun m t' => (-(boundNullTT G hist 0 false m (1 - gamma) t').1,
          (boundNullTT G hist 0 false m (1 - gamma) t').2))
        none t
  | 1, true, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else
      searchMovesSt gamma
        (fun m t' => (-(boundNullTT G hist 0 false m (1 - gamma) t').1,
          (boundNullTT G hist 0 false m (1 - gamma) t').2))
        (G.moves p) LOSS t
  | 2, false, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else if gamma ≤ ((t.find 2 p).getD (LOSS, MATE_UPPER)).1 then
      (((t.find 2 p).getD (LOSS, MATE_UPPER)).1, t)
    else if ((t.find 2 p).getD (LOSS, MATE_UPPER)).2 < gamma then
      (((t.find 2 p).getD (LOSS, MATE_UPPER)).2, t)
    else if hist p = true then (0, t)
    else
      cNodeTail G gamma 2 p ((t.find 2 p).getD (LOSS, MATE_UPPER))
        (fun s => s)
        (fun m t' => (-(boundNullTT G hist 1 false m (1 - gamma) t').1,
          (boundNullTT G hist 1 false m (1 - gamma) t').2))
        none t
  | 2, true, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else
      searchMovesSt gamma
        (fun m t' => (-(boundNullTT G hist 1 false m (1 - gamma) t').1,
          (boundNullTT G hist 1 false m (1 - gamma) t').2))
        (G.moves p) LOSS t
  | d + 3, false, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else if gamma ≤ ((t.find (d + 3) p).getD (LOSS, MATE_UPPER)).1 then
      (((t.find (d + 3) p).getD (LOSS, MATE_UPPER)).1, t)
    else if ((t.find (d + 3) p).getD (LOSS, MATE_UPPER)).2 < gamma then
      (((t.find (d + 3) p).getD (LOSS, MATE_UPPER)).2, t)
    else if hist p = true then (0, t)
    else
      cNodeTail G gamma (d + 3) p ((t.find (d + 3) p).getD (LOSS, MATE_UPPER))
        (fun t' => (boundNullTT G hist d true p gamma t').2)
        (fun m t' => (-(boundNullTT G hist (d + 2) false m (1 - gamma) t').1,
          (boundNullTT G hist (d + 2) false m (1 - gamma) t').2))
        (if nullGuard G.toGame p then
          some (fun t' => boundNullTT G hist d false (G.pass p) (1 - gamma) t')
        else none)
        t
  | d + 3, true, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else
      searchMovesSt gamma
        (fun m t' => (-(boundNullTT G hist (d + 2) false m (1 - gamma) t').1,
          (boundNullTT G hist (d + 2) false m (1 - gamma) t').2))
        (G.moves p) LOSS ((boundNullTT G hist d true p gamma t).2)

/-- The master induction: interior calls bracket `nullValue`, driver
probes bracket `rootValue`, and BOTH preserve the `(pos, depth)`-keyed
invariant.  One induction because the two paths interleave: a driver
probe's children are interior searches, and an interior node's IID is a
driver probe. -/
theorem boundNullTT_spec_all (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos]
    (hb : Bounded G.toGame) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int) (t : Table G.toGame),
      CTableOK G hist t →
      (((gamma ≤ (boundNullTT G hist d false p gamma t).1 →
          (boundNullTT G hist d false p gamma t).1 ≤ nullValue G hist d p) ∧
        ((boundNullTT G hist d false p gamma t).1 < gamma →
          nullValue G hist d p ≤ (boundNullTT G hist d false p gamma t).1)) ∧
       CTableOK G hist (boundNullTT G hist d false p gamma t).2) ∧
      (((gamma ≤ (boundNullTT G hist d true p gamma t).1 →
          (boundNullTT G hist d true p gamma t).1 ≤ rootValue G hist d p) ∧
        ((boundNullTT G hist d true p gamma t).1 < gamma →
          rootValue G hist d p ≤ (boundNullTT G hist d true p gamma t).1)) ∧
       CTableOK G hist (boundNullTT G hist d true p gamma t).2) := by
  suffices H : ∀ (n d : Nat), d ≤ n → ∀ (p : G.Pos) (gamma : Int)
      (t : Table G.toGame), CTableOK G hist t →
      (((gamma ≤ (boundNullTT G hist d false p gamma t).1 →
          (boundNullTT G hist d false p gamma t).1 ≤ nullValue G hist d p) ∧
        ((boundNullTT G hist d false p gamma t).1 < gamma →
          nullValue G hist d p ≤ (boundNullTT G hist d false p gamma t).1)) ∧
       CTableOK G hist (boundNullTT G hist d false p gamma t).2) ∧
      (((gamma ≤ (boundNullTT G hist d true p gamma t).1 →
          (boundNullTT G hist d true p gamma t).1 ≤ rootValue G hist d p) ∧
        ((boundNullTT G hist d true p gamma t).1 < gamma →
          rootValue G hist d p ≤ (boundNullTT G hist d true p gamma t).1)) ∧
       CTableOK G hist (boundNullTT G hist d true p gamma t).2) by
    exact fun d => H d d (Nat.le_refl d)
  intro n
  induction n with
  | zero =>
    intro d hd p gamma t ht
    have hd0 : d = 0 := by omega
    subst hd0
    have hzI : (boundNullTT G hist 0 false p gamma t).1 = nullValue G hist 0 p := rfl
    have hzR : (boundNullTT G hist 0 true p gamma t).1 = rootValue G hist 0 p := rfl
    exact ⟨⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩,
           ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩⟩
  | succ n ihn =>
    intro d hd p gamma t ht
    match d, hd with
    | 0, _ =>
      have hzI : (boundNullTT G hist 0 false p gamma t).1 = nullValue G hist 0 p := rfl
      have hzR : (boundNullTT G hist 0 true p gamma t).1 = rootValue G hist 0 p := rfl
      exact ⟨⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩,
             ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩⟩
    | 1, _ =>
      have hchild : ∀ (m : G.Pos) (s : Table G.toGame), CTableOK G hist s →
          CTableOK G hist (boundNullTT G hist 0 false m (1 - gamma) s).2 ∧
          (gamma ≤ -(boundNullTT G hist 0 false m (1 - gamma) s).1 →
            -(boundNullTT G hist 0 false m (1 - gamma) s).1
              ≤ -(nullValue G hist 0 m)) ∧
          (-(boundNullTT G hist 0 false m (1 - gamma) s).1 < gamma →
            -(nullValue G hist 0 m)
              ≤ -(boundNullTT G hist 0 false m (1 - gamma) s).1) := by
        intro m s hs
        have hih := (ihn 0 (by omega) m (1 - gamma) s hs).1
        have h1 := hih.1.1
        have h2 := hih.1.2
        refine ⟨hih.2, fun hge => ?_, fun hlt => ?_⟩
        · have := h2 (by omega); omega
        · have := h1 (by omega); omega
      constructor
      · -- Interior (root = false).
        have hunfB : boundNullTT G hist 1 false p gamma t
            = (if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
              else if gamma ≤ ((t.find 1 p).getD (LOSS, MATE_UPPER)).1 then
                (((t.find 1 p).getD (LOSS, MATE_UPPER)).1, t)
              else if ((t.find 1 p).getD (LOSS, MATE_UPPER)).2 < gamma then
                (((t.find 1 p).getD (LOSS, MATE_UPPER)).2, t)
              else if hist p = true then (0, t)
              else
                cNodeTail G gamma 1 p ((t.find 1 p).getD (LOSS, MATE_UPPER))
                  (fun s => s)
                  (fun m t' => (-(boundNullTT G hist 0 false m (1 - gamma) t').1,
                    (boundNullTT G hist 0 false m (1 - gamma) t').2))
                  none t) := rfl
        rw [hunfB]
        by_cases hkg : G.eval p ≤ -MATE_LOWER
        · rw [if_pos hkg]
          have hv := nullValue_kingGone G hist p hkg 1
          exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
        · rw [if_neg hkg]
          have hE := cEntry_valid G hist hb ht 1 p
          by_cases hlo : gamma ≤ ((t.find 1 p).getD (LOSS, MATE_UPPER)).1
          · rw [if_pos hlo]
            exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
          · rw [if_neg hlo]
            by_cases hhi : ((t.find 1 p).getD (LOSS, MATE_UPPER)).2 < gamma
            · rw [if_pos hhi]
              exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
            · rw [if_neg hhi]
              by_cases hrep : hist p = true
              · rw [if_pos hrep]
                have hv : nullValue G hist 1 p = 0 :=
                  nullValue_rep G hist p hkg hrep 0
                exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
              · rw [if_neg hrep]
                have hV : nullValue G hist 1 p
                    = foldMax (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS := by
                  simp only [nullValue]
                  rw [if_neg hkg, if_neg hrep]
                exact cNodeTail_spec G hist gamma 1 p
                  ((t.find 1 p).getD (LOSS, MATE_UPPER))
                  (fun s => s)
                  (fun m t' => (-(boundNullTT G hist 0 false m (1 - gamma) t').1,
                    (boundNullTT G hist 0 false m (1 - gamma) t').2))
                  none t
                  (fun m => -(nullValue G hist 0 m)) LOSS
                  ht (fun s hs => hs) hchild hE.1 hE.2 hV
                  (fun pf hpf => Option.noConfusion hpf)
                  (fun _ => rfl)
      · -- Driver probe (root = true).
        have hunfR : boundNullTT G hist 1 true p gamma t
            = (if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
              else
                searchMovesSt gamma
                  (fun m t' => (-(boundNullTT G hist 0 false m (1 - gamma) t').1,
                    (boundNullTT G hist 0 false m (1 - gamma) t').2))
                  (G.moves p) LOSS t) := rfl
        rw [hunfR]
        by_cases hkg : G.eval p ≤ -MATE_LOWER
        · rw [if_pos hkg]
          have hv : rootValue G hist 1 p = -MATE_UPPER := by
            have hunfV : rootValue G hist 1 p
                = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
                  else foldMax (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS) := rfl
            rw [hunfV, if_pos hkg]
          exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
        · rw [if_neg hkg]
          have hV : rootValue G hist 1 p
              = foldMax (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS := by
            have hunfV : rootValue G hist 1 p
                = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
                  else foldMax (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS) := rfl
            rw [hunfV, if_neg hkg]
          have hloop := searchMovesSt_spec gamma
            (fun m t' => (-(boundNullTT G hist 0 false m (1 - gamma) t').1,
              (boundNullTT G hist 0 false m (1 - gamma) t').2))
            (fun m => -(nullValue G hist 0 m)) (CTableOK G hist) hchild
            (G.moves p) LOSS LOSS t ht
            (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
          refine ⟨⟨fun hge => ?_, fun hlt => ?_⟩, hloop.1⟩
          · rw [hV]; exact hloop.2.1 hge
          · rw [hV]; exact hloop.2.2 hlt
    | 2, hd =>
      have hchild : ∀ (m : G.Pos) (s : Table G.toGame), CTableOK G hist s →
          CTableOK G hist (boundNullTT G hist 1 false m (1 - gamma) s).2 ∧
          (gamma ≤ -(boundNullTT G hist 1 false m (1 - gamma) s).1 →
            -(boundNullTT G hist 1 false m (1 - gamma) s).1
              ≤ -(nullValue G hist 1 m)) ∧
          (-(boundNullTT G hist 1 false m (1 - gamma) s).1 < gamma →
            -(nullValue G hist 1 m)
              ≤ -(boundNullTT G hist 1 false m (1 - gamma) s).1) := by
        intro m s hs
        have hih := (ihn 1 (by omega) m (1 - gamma) s hs).1
        have h1 := hih.1.1
        have h2 := hih.1.2
        refine ⟨hih.2, fun hge => ?_, fun hlt => ?_⟩
        · have := h2 (by omega); omega
        · have := h1 (by omega); omega
      constructor
      · -- Interior (root = false).
        have hunfB : boundNullTT G hist 2 false p gamma t
            = (if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
              else if gamma ≤ ((t.find 2 p).getD (LOSS, MATE_UPPER)).1 then
                (((t.find 2 p).getD (LOSS, MATE_UPPER)).1, t)
              else if ((t.find 2 p).getD (LOSS, MATE_UPPER)).2 < gamma then
                (((t.find 2 p).getD (LOSS, MATE_UPPER)).2, t)
              else if hist p = true then (0, t)
              else
                cNodeTail G gamma 2 p ((t.find 2 p).getD (LOSS, MATE_UPPER))
                  (fun s => s)
                  (fun m t' => (-(boundNullTT G hist 1 false m (1 - gamma) t').1,
                    (boundNullTT G hist 1 false m (1 - gamma) t').2))
                  none t) := rfl
        rw [hunfB]
        by_cases hkg : G.eval p ≤ -MATE_LOWER
        · rw [if_pos hkg]
          have hv := nullValue_kingGone G hist p hkg 2
          exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
        · rw [if_neg hkg]
          have hE := cEntry_valid G hist hb ht 2 p
          by_cases hlo : gamma ≤ ((t.find 2 p).getD (LOSS, MATE_UPPER)).1
          · rw [if_pos hlo]
            exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
          · rw [if_neg hlo]
            by_cases hhi : ((t.find 2 p).getD (LOSS, MATE_UPPER)).2 < gamma
            · rw [if_pos hhi]
              exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
            · rw [if_neg hhi]
              by_cases hrep : hist p = true
              · rw [if_pos hrep]
                have hv : nullValue G hist 2 p = 0 :=
                  nullValue_rep G hist p hkg hrep 1
                exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
              · rw [if_neg hrep]
                have hV : nullValue G hist 2 p
                    = foldMax (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS := by
                  simp only [nullValue]
                  rw [if_neg hkg, if_neg hrep]
                exact cNodeTail_spec G hist gamma 2 p
                  ((t.find 2 p).getD (LOSS, MATE_UPPER))
                  (fun s => s)
                  (fun m t' => (-(boundNullTT G hist 1 false m (1 - gamma) t').1,
                    (boundNullTT G hist 1 false m (1 - gamma) t').2))
                  none t
                  (fun m => -(nullValue G hist 1 m)) LOSS
                  ht (fun s hs => hs) hchild hE.1 hE.2 hV
                  (fun pf hpf => Option.noConfusion hpf)
                  (fun _ => rfl)
      · -- Driver probe (root = true).
        have hunfR : boundNullTT G hist 2 true p gamma t
            = (if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
              else
                searchMovesSt gamma
                  (fun m t' => (-(boundNullTT G hist 1 false m (1 - gamma) t').1,
                    (boundNullTT G hist 1 false m (1 - gamma) t').2))
                  (G.moves p) LOSS t) := rfl
        rw [hunfR]
        by_cases hkg : G.eval p ≤ -MATE_LOWER
        · rw [if_pos hkg]
          have hv : rootValue G hist 2 p = -MATE_UPPER := by
            have hunfV : rootValue G hist 2 p
                = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
                  else foldMax (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS) := rfl
            rw [hunfV, if_pos hkg]
          exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
        · rw [if_neg hkg]
          have hV : rootValue G hist 2 p
              = foldMax (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS := by
            have hunfV : rootValue G hist 2 p
                = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
                  else foldMax (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS) := rfl
            rw [hunfV, if_neg hkg]
          have hloop := searchMovesSt_spec gamma
            (fun m t' => (-(boundNullTT G hist 1 false m (1 - gamma) t').1,
              (boundNullTT G hist 1 false m (1 - gamma) t').2))
            (fun m => -(nullValue G hist 1 m)) (CTableOK G hist) hchild
            (G.moves p) LOSS LOSS t ht
            (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
          refine ⟨⟨fun hge => ?_, fun hlt => ?_⟩, hloop.1⟩
          · rw [hV]; exact hloop.2.1 hge
          · rw [hV]; exact hloop.2.2 hlt
    | (d + 3), hd =>
      have hchild : ∀ (m : G.Pos) (s : Table G.toGame), CTableOK G hist s →
          CTableOK G hist (boundNullTT G hist (d + 2) false m (1 - gamma) s).2 ∧
          (gamma ≤ -(boundNullTT G hist (d + 2) false m (1 - gamma) s).1 →
            -(boundNullTT G hist (d + 2) false m (1 - gamma) s).1
              ≤ -(nullValue G hist (d + 2) m)) ∧
          (-(boundNullTT G hist (d + 2) false m (1 - gamma) s).1 < gamma →
            -(nullValue G hist (d + 2) m)
              ≤ -(boundNullTT G hist (d + 2) false m (1 - gamma) s).1) := by
        intro m s hs
        have hih := (ihn (d + 2) (by omega) m (1 - gamma) s hs).1
        have h1 := hih.1.1
        have h2 := hih.1.2
        refine ⟨hih.2, fun hge => ?_, fun hlt => ?_⟩
        · have := h2 (by omega); omega
        · have := h1 (by omega); omega
      have hiid : ∀ s, CTableOK G hist s →
          CTableOK G hist ((boundNullTT G hist d true p gamma s).2) :=
        fun s hs => ((ihn d (by omega) p gamma s hs).2).2
      constructor
      · -- Interior (root = false).
        simp only [boundNullTT]
        by_cases hkg : G.eval p ≤ -MATE_LOWER
        · rw [if_pos hkg]
          have hv := nullValue_kingGone G hist p hkg (d + 3)
          exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
        · rw [if_neg hkg]
          have hE := cEntry_valid G hist hb ht (d + 3) p
          by_cases hlo : gamma ≤ ((t.find (d + 3) p).getD (LOSS, MATE_UPPER)).1
          · rw [if_pos hlo]
            exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
          · rw [if_neg hlo]
            by_cases hhi : ((t.find (d + 3) p).getD (LOSS, MATE_UPPER)).2 < gamma
            · rw [if_pos hhi]
              exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
            · rw [if_neg hhi]
              by_cases hrep : hist p = true
              · rw [if_pos hrep]
                have hv : nullValue G hist (d + 3) p = 0 :=
                  nullValue_rep G hist p hkg hrep (d + 2)
                exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
              · rw [if_neg hrep]
                by_cases hg : nullGuard G.toGame p
                · rw [if_pos hg]
                  have hV : nullValue G hist (d + 3) p
                      = foldMax (fun m => -(nullValue G hist (d + 2) m)) (G.moves p)
                          (max LOSS (-(nullValue G hist d (G.pass p)))) := by
                    simp only [nullValue]
                    rw [if_neg hkg, if_neg hrep, if_pos hg]
                  refine cNodeTail_spec G hist gamma (d + 3) p
                    ((t.find (d + 3) p).getD (LOSS, MATE_UPPER))
                    (fun t' => (boundNullTT G hist d true p gamma t').2)
                    (fun m t' => (-(boundNullTT G hist (d + 2) false m (1 - gamma) t').1,
                      (boundNullTT G hist (d + 2) false m (1 - gamma) t').2))
                    (some (fun t' => boundNullTT G hist d false (G.pass p) (1 - gamma) t'))
                    t
                    (fun m => -(nullValue G hist (d + 2) m))
                    (max LOSS (-(nullValue G hist d (G.pass p))))
                    ht hiid hchild hE.1 hE.2 hV ?_ ?_
                  · intro pf hpf
                    injection hpf with hpf
                    subst hpf
                    simp only []
                    have hih := (ihn d (by omega) (G.pass p) (1 - gamma) t ht).1
                    have h1 := hih.1.1
                    have h2 := hih.1.2
                    refine ⟨hih.2, fun hge => ?_, fun hlt => ?_⟩
                    · by_cases hpv : gamma ≤
                          -(boundNullTT G hist d false (G.pass p) (1 - gamma) t).1
                      · have := h2 (by omega)
                        omega
                      · omega
                    · have : 1 - gamma ≤
                          (boundNullTT G hist d false (G.pass p) (1 - gamma) t).1 := by
                        omega
                      have := h1 this
                      omega
                  · intro hcontra
                    exact Option.noConfusion hcontra
                · rw [if_neg hg]
                  have hV : nullValue G hist (d + 3) p
                      = foldMax (fun m => -(nullValue G hist (d + 2) m)) (G.moves p)
                          LOSS := by
                    simp only [nullValue]
                    rw [if_neg hkg, if_neg hrep, if_neg hg]
                  exact cNodeTail_spec G hist gamma (d + 3) p
                    ((t.find (d + 3) p).getD (LOSS, MATE_UPPER))
                    (fun t' => (boundNullTT G hist d true p gamma t').2)
                    (fun m t' => (-(boundNullTT G hist (d + 2) false m (1 - gamma) t').1,
                      (boundNullTT G hist (d + 2) false m (1 - gamma) t').2))
                    none t
                    (fun m => -(nullValue G hist (d + 2) m)) LOSS
                    ht hiid hchild hE.1 hE.2 hV
                    (fun pf hpf => Option.noConfusion hpf)
                    (fun _ => rfl)
      · -- Driver probe (root = true): no lookup, no repetition, no null,
        -- no store; the nested IID's interior children are the only
        -- table effect.
        simp only [boundNullTT]
        by_cases hkg : G.eval p ≤ -MATE_LOWER
        · rw [if_pos hkg]
          have hv : rootValue G hist (d + 3) p = -MATE_UPPER := by
            have hunfV : rootValue G hist (d + 3) p
                = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
                  else foldMax (fun m => -(nullValue G hist (d + 2) m))
                    (G.moves p) LOSS) := rfl
            rw [hunfV, if_pos hkg]
          exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
        · rw [if_neg hkg]
          have hV : rootValue G hist (d + 3) p
              = foldMax (fun m => -(nullValue G hist (d + 2) m)) (G.moves p) LOSS := by
            have hunfV : rootValue G hist (d + 3) p
                = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
                  else foldMax (fun m => -(nullValue G hist (d + 2) m))
                    (G.moves p) LOSS) := rfl
            rw [hunfV, if_neg hkg]
          have hloop := searchMovesSt_spec gamma
            (fun m t' => (-(boundNullTT G hist (d + 2) false m (1 - gamma) t').1,
              (boundNullTT G hist (d + 2) false m (1 - gamma) t').2))
            (fun m => -(nullValue G hist (d + 2) m)) (CTableOK G hist) hchild
            (G.moves p) LOSS LOSS ((boundNullTT G hist d true p gamma t).2)
            (hiid t ht)
            (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
          refine ⟨⟨fun hge => ?_, fun hlt => ?_⟩, hloop.1⟩
          · rw [hV]; exact hloop.2.1 hge
          · rw [hV]; exact hloop.2.2 hlt

/-- **The interior point spec, proven and unconditional**: the flagless
search brackets `nullValue` -- the ONE value function every stored entry
describes -- with a point spec at every `(pos, depth)`, and preserves
the keyed-table invariant.  No zugzwang hypothesis: the search and its
table are self-consistent whatever the position. -/
theorem boundNullTT_spec (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos]
    (hb : Bounded G.toGame) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int) (t : Table G.toGame),
      CTableOK G hist t →
      ((gamma ≤ (boundNullTT G hist d false p gamma t).1 →
        (boundNullTT G hist d false p gamma t).1 ≤ nullValue G hist d p) ∧
       ((boundNullTT G hist d false p gamma t).1 < gamma →
        nullValue G hist d p ≤ (boundNullTT G hist d false p gamma t).1)) ∧
      CTableOK G hist (boundNullTT G hist d false p gamma t).2 :=
  fun d p gamma t ht => (boundNullTT_spec_all G hist hb d p gamma t ht).1

/-! ### The driver probe -/

/-- The driver probe -- `bound(pos, gamma, depth, root=True)`, used by
the MTD-bi driver (line 512) and by IID (line 381, at `depth - 3`): the
same move-loop recursion with no table access, no repetition-0 and no
null yield; its children are ordinary interior searches. -/
def rootProbe (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos]
    (gamma : Int) (d : Nat) (p : G.Pos) (t : Table G.toGame) :
    Int × Table G.toGame :=
  boundNullTT G hist d true p gamma t

/-- **The driver lemma, proven and unconditional**: the driver probe
returns fail-soft bounds on `rootValue` -- the max over real moves of
the interior `nullValue` of the children -- and preserves the
`(pos, depth)`-keyed invariant (its only table effect is through its
interior children; it stores nothing itself).  This is the honest spec
of what `search`'s binary loop consumes, and of what an IID probe
computes at `depth - 3` before its result is discarded. -/
theorem rootProbe_spec (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos]
    (hb : Bounded G.toGame) :
    ∀ (gamma : Int) (d : Nat) (p : G.Pos) (t : Table G.toGame),
      CTableOK G hist t →
      ((gamma ≤ (rootProbe G hist gamma d p t).1 →
        (rootProbe G hist gamma d p t).1 ≤ rootValue G hist d p) ∧
       ((rootProbe G hist gamma d p t).1 < gamma →
        rootValue G hist d p ≤ (rootProbe G hist gamma d p t).1)) ∧
      CTableOK G hist (rootProbe G hist gamma d p t).2 :=
  fun gamma d p t ht => (boundNullTT_spec_all G hist hb d p gamma t ht).2

/-- Where the two value functions agree: away from the gates.  If the
position is not in `history` and the null option is closed (low depth or
guard failure), `rootValue = nullValue`.  So the driver's deviation is
confined to exactly the nodes where a gate would have fired -- and it is
harmless there BECAUSE the driver never stores: no table entry ever
describes `rootValue` (see `boundNullTT`: the store sits under
`if not root`). -/
theorem rootValue_eq_nullValue (G : NullGame) (hist : G.Pos → Bool) (p : G.Pos)
    (hrep : ¬ hist p = true) :
    ∀ (d : Nat), d ≤ 2 ∨ ¬ nullGuard G.toGame p →
      rootValue G hist d p = nullValue G hist d p := by
  intro d hq
  match d with
  | 0 => rfl
  | 1 =>
    have hL : rootValue G hist 1 p
        = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
          else foldMax (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS) := rfl
    have hR : nullValue G hist 1 p
        = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
          else if hist p = true then 0
          else foldMax (fun m => -(nullValue G hist 0 m)) (G.moves p) LOSS) := rfl
    rw [hL, hR]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg, if_pos hkg]
    · rw [if_neg hkg, if_neg hkg, if_neg hrep]
  | 2 =>
    have hL : rootValue G hist 2 p
        = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
          else foldMax (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS) := rfl
    have hR : nullValue G hist 2 p
        = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
          else if hist p = true then 0
          else foldMax (fun m => -(nullValue G hist 1 m)) (G.moves p) LOSS) := rfl
    rw [hL, hR]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg, if_pos hkg]
    · rw [if_neg hkg, if_neg hkg, if_neg hrep]
  | d + 3 =>
    have hg : ¬ nullGuard G.toGame p := by
      cases hq with
      | inl h => omega
      | inr h => exact h
    have hL : rootValue G hist (d + 3) p
        = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
          else foldMax (fun m => -(nullValue G hist (d + 2) m)) (G.moves p) LOSS) := rfl
    have hR : nullValue G hist (d + 3) p
        = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
          else if hist p = true then 0
          else foldMax (fun m => -(nullValue G hist (d + 2) m)) (G.moves p)
            (if nullGuard G.toGame p then
              max LOSS (-(nullValue G hist d (G.pass p)))
            else LOSS)) := rfl
    rw [hL, hR]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg, if_pos hkg]
    · rw [if_neg hkg, if_neg hkg, if_neg hrep, if_neg hg]

/-! ### The bridge to the null-free value: where the bet lives -/

/-- The null-free value: king-capture-normalized negamax, i.e. exactly
`nullValue` with the pass option and the repetition gate deleted.  This
is the honest "s*" of a king-capture engine (the same normalization
`Sunfish/Stalemate.lean` established as the sentinel semantics); it
differs from the raw `negamax` of `GameTree.lean` only at king-gone
positions, which score the exact `-MATE_UPPER` sentinel. -/
def plainValue (G : NullGame) : Nat → G.Pos → Int
  | 0, p => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else foldMax (fun m => -(plainValue G d m)) (G.moves p) LOSS

/-- Pointwise-equal weights give equal folds. -/
theorem foldMax_congr {α : Type _} (w w' : α → Int) :
    ∀ (ms : List α) (acc : Int), (∀ m ∈ ms, w m = w' m) →
      foldMax w ms acc = foldMax w' ms acc := by
  intro ms
  induction ms with
  | nil => intro acc _; rfl
  | cons m ms ih =>
    intro acc h
    simp only [foldMax]
    rw [h m (by simp)]
    exact ih _ (fun x hx => h x (by simp [hx]))

/-- The fold commutes with a `max` in its initial accumulator. -/
theorem foldMax_max {α : Type _} (w : α → Int) :
    ∀ (ms : List α) (a x : Int),
      foldMax w ms (max a x) = max (foldMax w ms a) x := by
  intro ms
  induction ms with
  | nil => intro a x; rfl
  | cons m ms ih =>
    intro a x
    simp only [foldMax]
    rw [show max (max a x) (w m) = max (max a (w m)) x from by omega]
    exact ih (max a (w m)) x

/-- **NullBetOK** -- the bridge hypothesis, exactly as the code places
the bet: at every guard-passing position, some real move at the
CHILDREN's depth (`depth - 1`, i.e. `d + 2` when the pass sits at `d`)
matches the pass searched at its REDUCED depth (`depth - 3`).  This
folds the old `NullOK` (zugzwang: passing beats every real move)
together with the depth-reduction stability that `depth - 3`
additionally needs, and -- since it demands a WITNESS move -- also
excludes guard-passing TERMINAL positions (where the pass option has no
move to back it: the stalemate corner of the same bet).  Its negation
is zugzwang, a pass-favoring depth artifact, or a guard-passing
stalemate. -/
def NullBetOK (G : NullGame) : Prop :=
  ∀ (d : Nat) (p : G.Pos), nullGuard G.toGame p →
    ∃ m ∈ G.moves p, plainValue G (d + 2) m ≤ plainValue G d (G.pass p)

/-- **The bridge, proven under the named hypothesis**: with `NullBetOK`
and an empty history (so the repetition gate never fires), `nullValue`
coincides with the null-free `plainValue` at every depth -- the pass
option never raises the fold, so composing with `boundNullTT_spec`'s
point spec recovers the original docstring against `plainValue`.
Zugzwang (¬`NullBetOK`) breaks exactly this bridge and nothing in the
self-consistency story; non-empty histories additionally shift the
target toward the draw-aware semantics of `Sunfish/Stalemate.lean`. -/
theorem nullValue_plain (G : NullGame) (hbet : NullBetOK G) :
    ∀ (d : Nat) (p : G.Pos),
      nullValue G (fun _ => false) d p = plainValue G d p := by
  suffices H : ∀ (n d : Nat), d ≤ n → ∀ (p : G.Pos),
      nullValue G (fun _ => false) d p = plainValue G d p by
    exact fun d => H d d (Nat.le_refl d)
  intro n
  induction n with
  | zero =>
    intro d hd p
    have hd0 : d = 0 := by omega
    subst hd0
    rfl
  | succ n ihn =>
    intro d hd p
    have hrep : ¬ ((fun _ : G.Pos => false) p = true) :=
      fun h => Bool.noConfusion h
    match d, hd with
    | 0, _ => rfl
    | 1, _ =>
      have hL : nullValue G (fun _ => false) 1 p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if (fun _ : G.Pos => false) p = true then 0
            else foldMax (fun m => -(nullValue G (fun _ => false) 0 m))
              (G.moves p) LOSS) := rfl
      have hR : plainValue G 1 p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else foldMax (fun m => -(plainValue G 0 m)) (G.moves p) LOSS) := rfl
      rw [hL, hR]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg, if_pos hkg]
      · rw [if_neg hkg, if_neg hkg, if_neg hrep]
        exact foldMax_congr _ _ (G.moves p) LOSS (fun m _ => by
          show -(nullValue G (fun _ => false) 0 m) = -(plainValue G 0 m)
          rw [ihn 0 (by omega) m])
    | 2, hd =>
      have hL : nullValue G (fun _ => false) 2 p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if (fun _ : G.Pos => false) p = true then 0
            else foldMax (fun m => -(nullValue G (fun _ => false) 1 m))
              (G.moves p) LOSS) := rfl
      have hR : plainValue G 2 p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else foldMax (fun m => -(plainValue G 1 m)) (G.moves p) LOSS) := rfl
      rw [hL, hR]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg, if_pos hkg]
      · rw [if_neg hkg, if_neg hkg, if_neg hrep]
        exact foldMax_congr _ _ (G.moves p) LOSS (fun m _ => by
          show -(nullValue G (fun _ => false) 1 m) = -(plainValue G 1 m)
          rw [ihn 1 (by omega) m])
    | (d + 3), hd =>
      have hL : nullValue G (fun _ => false) (d + 3) p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if (fun _ : G.Pos => false) p = true then 0
            else foldMax (fun m => -(nullValue G (fun _ => false) (d + 2) m))
              (G.moves p)
              (if nullGuard G.toGame p then
                max LOSS (-(nullValue G (fun _ => false) d (G.pass p)))
              else LOSS)) := rfl
      have hR : plainValue G (d + 3) p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else foldMax (fun m => -(plainValue G (d + 2) m)) (G.moves p) LOSS) := rfl
      rw [hL, hR]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg, if_pos hkg]
      · rw [if_neg hkg, if_neg hkg, if_neg hrep]
        have hcong : foldMax (fun m => -(nullValue G (fun _ => false) (d + 2) m))
              (G.moves p) LOSS
            = foldMax (fun m => -(plainValue G (d + 2) m)) (G.moves p) LOSS :=
          foldMax_congr _ _ (G.moves p) LOSS (fun m _ => by
            show -(nullValue G (fun _ => false) (d + 2) m) = -(plainValue G (d + 2) m)
            rw [ihn (d + 2) (by omega) m])
        by_cases hg : nullGuard G.toGame p
        · rw [if_pos hg]
          rw [show (-(nullValue G (fun _ => false) d (G.pass p)))
              = (-(plainValue G d (G.pass p))) from by
            rw [ihn d (by omega) (G.pass p)]]
          rw [foldMax_max, hcong]
          -- The bet: the pass option never exceeds the real fold.
          cases hbet d p hg with
          | intro m hm =>
            have hmem : -(plainValue G (d + 2) m)
                ≤ foldMax (fun x => -(plainValue G (d + 2) x)) (G.moves p) LOSS :=
              foldMax_le_of_mem _ _ _ m hm.1
            omega
        · rw [if_neg hg]
          exact hcong

end Sunfish
