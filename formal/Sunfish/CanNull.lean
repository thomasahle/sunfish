/-
The can_null layering, modeled exactly as master uses it (audit of
sunfish.py lines 286-448 at commit 9b1a7b4, 2026-08-05).

`can_null` plays four roles, all modeled here:

(a) it gates the null move (line 340:
    `depth > 2 and can_null and abs(pos.score) < 500`), whose pass
    position is searched at `depth - 3` and -- note -- with
    `can_null=True` (the Python default: sunfish permits CONSECUTIVE
    null moves; `can_null` marks root/IID calls, it is not a
    no-two-passes flag.  The model reproduces this exactly);
(b) it gates the repetition check (line 325:
    `can_null and depth > 0 and pos in self.history`).  `history` is a
    FIXED per-search parameter here (`hist`); `ctableOK_empty` below is
    the invariant fact that justifies sunfish clearing `tp_score`
    whenever `history` changes -- the table invariant is
    history-relative, and the empty table satisfies it for any history;
(c) it is part of the transposition key (line 318:
    `(pos, depth, can_null)`) -- `CTable` below keys on all three;
(d) IID recurses with `can_null=False` (line 355:
    `self.bound(pos, gamma, depth - 3, can_null=False)`).
    NOTE: sunfish.py's own comment at lines 352-353 says "We set
    can_null=True", but the CODE passes False, and the code is right:
    with True, a null cutoff inside IID would yield `None` and find no
    killer, and the repetition check could truncate the probe.  The
    comment bug is being fixed separately in PR #135.  We model the
    code, not the comment.

THE LAYERED NULL-MOVE STORY.

Layer 1 (UNCONDITIONAL, proven: `boundNullTT_spec`): define
`nullValue pos depth can_null` -- with `hist` fixed -- as the value the
null-and-repetition-augmented search actually computes: the pass option
(at depth-3, when the GAMMA-FREE guard passes) is one more option in the
move fold, and the repetition gate returns 0.  Then the search brackets
`nullValue` with a POINT spec, and a `(pos, depth, can_null)`-keyed table
brackets `nullValue` consistently.  No zugzwang hypothesis appears
anywhere: self-consistency of the search + table is unconditional.

Layer 2 (CONDITIONAL, stated: `nullValue_negamax`): relating `nullValue`
to plain `negamax` is where the null-move BET lives -- `NullBetOK` below,
the (now smaller) named hypothesis.  Zugzwang only ever threatens layer
2, never layer-1 self-consistency: a zugzwang position makes the engine
compute the wrong VALUE, but never makes its table contradict its
search.

Exactness notes from the audit:

* Order of the prelude is exact: king-gone check (line 312) BEFORE table
  lookup (line 318) BEFORE repetition (line 325); the early returns
  (king-gone, TT hit, repetition) store nothing, all loop exits store
  through Table part 2 (lines 442-445, `cTablePart2` =
  the plain stores of master).
* The generator's LAZINESS is semantically load-bearing (a surprise of
  this audit): the null yield is pulled first, and if it cuts off, the
  IID recursion never runs -- so the table state differs depending on
  the cutoff.  `cNodeTail` therefore applies the IID table-effect only
  on the no-cutoff path.  A model that ran all yields eagerly would
  mis-model `tp_score`.
* The deadline `Stop` (lines 297-301) raises at node ENTRY, before any
  store: an abort can leave the search unfinished but never a table
  entry unjustified -- aborts cannot corrupt `CTableOK`.  Not modeled,
  by that argument.
* `depth = max(depth, 0)` (line 306) corresponds to this model's use of
  `Nat` depths with saturating subtraction -- verified aligned.
* The stalemate probe (line 434) calls `bound(flipped, MATE_UPPER, 0)`
  with `can_null=True` by default, so its table key is
  `(flipped, 0, True)`; harmless, since both `can_null` gates also
  require `depth > 0` (repetition) or `depth > 2` (null), which fail at
  depth 0 -- the probe behaves identically under either flag, but the
  KEY is `True` and the model says so.
* Not modeled in THIS file (each layered elsewhere): the killer yield
  and `tp_move` (`Sunfish/Killer.lean`), futility (`Sunfish/Tricks.lean`),
  the stalemate block
  (`Sunfish/Stalemate.lean`), QS interior (collapsed to `eval`, see
  README).  With no killer, the IID guard `not killer and depth > 2`
  reduces to `depth > 2`, which is what the model runs.
-/

import Sunfish.Tricks

namespace Sunfish




/-! ### The gamma-free null-move guard and the keyed table -/

/-- `abs(pos.score) < 500` (line 340): the zugzwang heuristic.  Crucially
gamma-free: whether the pass option EXISTS does not depend on the window,
only its search does. -/
def nullGuard (G : Game) (p : G.Pos) : Prop :=
  -500 < G.eval p ∧ G.eval p < 500

instance (G : Game) (p : G.Pos) : Decidable (nullGuard G p) := by
  unfold nullGuard; infer_instance

/-- `tp_score`, keyed by `(depth, can_null, pos)` -- role (c). -/
structure CTable (G : Game) where
  find : Nat → Bool → G.Pos → Option (Int × Int)

def CTable.store {G : Game} [DecidableEq G.Pos] (t : CTable G) (D : Nat)
    (cn : Bool) (p : G.Pos) (e : Int × Int) : CTable G :=
  ⟨fun d' cn' p' => if d' = D ∧ cn' = cn ∧ p' = p then some e else t.find d' cn' p'⟩

/-! ### Layer 1: the value the search actually computes -/

/-- `nullValue G hist d cn p`: the value function of the
null-and-repetition-augmented search.  King-gone normalization first
(line 312); the repetition gate `can_null ∧ depth > 0 ∧ hist p` returns 0
(line 325; `depth > 0` is carried by the patterns); the pass option, at
`depth - 3` and with `can_null := true`, is the fold's initial
accumulator when `depth > 2 ∧ can_null ∧ nullGuard` (lines 340-341);
children are searched with `can_null := true` (the Python default). -/
def nullValue (G : NullGame) (hist : G.Pos → Bool) : Nat → Bool → G.Pos → Int
  | 0, _, p => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | 1, cn, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if cn = true ∧ hist p = true then 0
    else foldMax (fun m => -(nullValue G hist 0 true m)) (G.moves p) LOSS
  | 2, cn, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if cn = true ∧ hist p = true then 0
    else foldMax (fun m => -(nullValue G hist 1 true m)) (G.moves p) LOSS
  | d + 3, cn, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if cn = true ∧ hist p = true then 0
    else foldMax (fun m => -(nullValue G hist (d + 2) true m)) (G.moves p)
      (if cn = true ∧ nullGuard G.toGame p then
        max LOSS (-(nullValue G hist d true (G.pass p)))
      else LOSS)

/-- **The keyed-table invariant**: every entry under `(depth, can_null,
pos)` brackets `nullValue` at exactly that key.  History-relative. -/
def CTableOK (G : NullGame) (hist : G.Pos → Bool) (t : CTable G.toGame) : Prop :=
  ∀ (d : Nat) (cn : Bool) (p : G.Pos) (lo hi : Int),
    t.find d cn p = some (lo, hi) →
      lo ≤ nullValue G hist d cn p ∧ nullValue G hist d cn p ≤ hi

/-- Role (b)'s bookkeeping fact: the EMPTY table satisfies the invariant
for ANY history -- which is exactly why sunfish may (and must) clear
`tp_score` when `history` changes: entries proven against one history
mean nothing under another, and clearing restores the invariant
trivially. -/
theorem ctableOK_empty (G : NullGame) (hist : G.Pos → Bool) :
    CTableOK G hist ⟨fun _ _ _ => none⟩ :=
  fun _ _ _ _ _ h => Option.noConfusion h

/-! ### Helper lemmas -/

theorem nullValue_kingGone (G : NullGame) (hist : G.Pos → Bool) (p : G.Pos)
    (h : G.eval p ≤ -MATE_LOWER) :
    ∀ (d : Nat) (cn : Bool), nullValue G hist d cn p = -MATE_UPPER := by
  intro d cn
  match d with
  | 0 => simp only [nullValue]; rw [if_pos h]
  | 1 => simp only [nullValue]; rw [if_pos h]
  | 2 => simp only [nullValue]; rw [if_pos h]
  | d + 3 => simp only [nullValue]; rw [if_pos h]

theorem nullValue_rep (G : NullGame) (hist : G.Pos → Bool) (p : G.Pos) (cn : Bool)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) (hr : cn = true ∧ hist p = true) :
    ∀ (d : Nat), nullValue G hist (d + 1) cn p = 0 := by
  intro d
  match d with
  | 0 => simp only [nullValue]; rw [if_neg hkg, if_pos hr]
  | 1 => simp only [nullValue]; rw [if_neg hkg, if_pos hr]
  | d + 2 => simp only [nullValue]; rw [if_neg hkg, if_pos hr]

/-- Under `Bounded`, `nullValue` stays in the score band (what validates
the fresh `Entry(-MATE_UPPER, MATE_UPPER)` default of line 318). -/
theorem nullValue_bounded (G : NullGame) (hist : G.Pos → Bool)
    (hb : Bounded G.toGame) :
    ∀ (d : Nat) (cn : Bool) (p : G.Pos),
      -MATE_UPPER ≤ nullValue G hist d cn p ∧
      nullValue G hist d cn p ≤ MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 50710 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  suffices H : ∀ (n d : Nat), d ≤ n → ∀ (cn : Bool) (p : G.Pos),
      -MATE_UPPER ≤ nullValue G hist d cn p ∧
      nullValue G hist d cn p ≤ MATE_UPPER by
    exact fun d => H d d (Nat.le_refl d)
  intro n
  induction n with
  | zero =>
    intro d hd cn p
    have hd0 : d = 0 := by omega
    subst hd0
    have := hb p
    simp only [nullValue]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]; omega
    · rw [if_neg hkg]; omega
  | succ n ihn =>
    intro d hd cn p
    match d, hd with
    | 0, _ =>
      have := hb p
      simp only [nullValue]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]; omega
    | 1, _ =>
      have hunf : nullValue G hist 1 cn p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if cn = true ∧ hist p = true then 0
            else foldMax (fun m => -(nullValue G hist 0 true m)) (G.moves p) LOSS) := rfl
      rw [hunf]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]
        by_cases hr : cn = true ∧ hist p = true
        · rw [if_pos hr]; omega
        · rw [if_neg hr]
          have hfl := foldMax_ge_init (fun m => -(nullValue G hist 0 true m)) (G.moves p) LOSS
          have hfu : foldMax (fun m => -(nullValue G hist 0 true m)) (G.moves p) LOSS
              ≤ MATE_UPPER := by
            refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
            show -(nullValue G hist 0 true m) ≤ MATE_UPPER
            have := ihn 0 (by omega) true m
            omega
          omega
    | 2, hd =>
      have hunf : nullValue G hist 2 cn p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if cn = true ∧ hist p = true then 0
            else foldMax (fun m => -(nullValue G hist 1 true m)) (G.moves p) LOSS) := rfl
      rw [hunf]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]
        by_cases hr : cn = true ∧ hist p = true
        · rw [if_pos hr]; omega
        · rw [if_neg hr]
          have hfl := foldMax_ge_init (fun m => -(nullValue G hist 1 true m)) (G.moves p) LOSS
          have hfu : foldMax (fun m => -(nullValue G hist 1 true m)) (G.moves p) LOSS
              ≤ MATE_UPPER := by
            refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
            show -(nullValue G hist 1 true m) ≤ MATE_UPPER
            have := ihn 1 (by omega) true m
            omega
          omega
    | (d + 3), hd =>
      simp only [nullValue]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]
        by_cases hr : cn = true ∧ hist p = true
        · rw [if_pos hr]; omega
        · rw [if_neg hr]
          have hpass := ihn d (by omega) true (G.pass p)
          have hinit : -MATE_UPPER ≤
              (if cn = true ∧ nullGuard G.toGame p then
                max LOSS (-(nullValue G hist d true (G.pass p)))
              else LOSS) ∧
              (if cn = true ∧ nullGuard G.toGame p then
                max LOSS (-(nullValue G hist d true (G.pass p)))
              else LOSS) ≤ MATE_UPPER := by
            by_cases hg : cn = true ∧ nullGuard G.toGame p
            · rw [if_pos hg]; omega
            · rw [if_neg hg]; omega
          have hfl := foldMax_ge_init (fun m => -(nullValue G hist (d + 2) true m))
            (G.moves p)
            (if cn = true ∧ nullGuard G.toGame p then
              max LOSS (-(nullValue G hist d true (G.pass p)))
            else LOSS)
          have hfu : foldMax (fun m => -(nullValue G hist (d + 2) true m)) (G.moves p)
              (if cn = true ∧ nullGuard G.toGame p then
                max LOSS (-(nullValue G hist d true (G.pass p)))
              else LOSS) ≤ MATE_UPPER := by
            refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
            show -(nullValue G hist (d + 2) true m) ≤ MATE_UPPER
            have := ihn (d + 2) (by omega) true m
            omega
          omega

/-- Storing a valid bracket preserves the keyed invariant. -/
theorem cTableOK_store {G : NullGame} {hist : G.Pos → Bool} [DecidableEq G.Pos]
    {t : CTable G.toGame} {D : Nat} {cn : Bool} {p : G.Pos} {e' : Int × Int}
    (ht : CTableOK G hist t)
    (h1 : e'.1 ≤ nullValue G hist D cn p)
    (h2 : nullValue G hist D cn p ≤ e'.2) :
    CTableOK G hist (CTable.store t D cn p e') := by
  intro d' cn' p' lo hi hfind
  simp only [CTable.store] at hfind
  by_cases hk : d' = D ∧ cn' = cn ∧ p' = p
  · rw [if_pos hk] at hfind
    injection hfind with hh
    have hl : e'.1 = lo := by rw [hh]
    have hr : e'.2 = hi := by rw [hh]
    rw [hk.1, hk.2.1, hk.2.2]
    constructor
    · rw [← hl]; exact h1
    · rw [← hr]; exact h2
  · rw [if_neg hk] at hfind
    exact ht d' cn' p' lo hi hfind

/-- The current entry (stored, or the fresh default of line 318) is a
valid bracket. -/
theorem cEntry_valid (G : NullGame) (hist : G.Pos → Bool) (hb : Bounded G.toGame)
    {t : CTable G.toGame} (ht : CTableOK G hist t) (D : Nat) (cn : Bool) (p : G.Pos) :
    ((t.find D cn p).getD (LOSS, MATE_UPPER)).1 ≤ nullValue G hist D cn p ∧
    nullValue G hist D cn p ≤ ((t.find D cn p).getD (LOSS, MATE_UPPER)).2 := by
  cases hfind : t.find D cn p with
  | none =>
    have hband := nullValue_bounded G hist hb D cn p
    have hLOSS : LOSS = -MATE_UPPER := rfl
    refine ⟨?_, ?_⟩
    · show LOSS ≤ nullValue G hist D cn p
      omega
    · show nullValue G hist D cn p ≤ MATE_UPPER
      omega
  | some e =>
    exact ht D cn p e.1 e.2 (by rw [hfind])

/-! ### The search -/

/-- The generic state-passing fail-soft loop (finally generalized over
the state type; `searchMovesTT` is the `Table` instance of the same
shape). -/
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

/-- Table part 2 (lines 439-442) on the keyed table: the PLAIN stores
of master (`Entry(best, entry.upper)` / `Entry(entry.lower, best)`) --
the 2c95ab0 clamp was removed from the code at 7f9f164 and from this
model with it (git has the history). -/
def cTablePart2 (G : Game) [DecidableEq G.Pos] (D : Nat) (cn : Bool) (p : G.Pos)
    (gamma : Int) (e : Int × Int) (r : Int × CTable G) : Int × CTable G :=
  if gamma ≤ r.1 then (r.1, CTable.store r.2 D cn p (r.1, e.2))
  else (r.1, CTable.store r.2 D cn p (e.1, r.1))

theorem cTablePart2_ok (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos]
    (D : Nat) (cn : Bool) (p : G.Pos) (gamma : Int) (e : Int × Int)
    (r : Int × CTable G.toGame)
    (htok : CTableOK G hist r.2)
    (he1 : e.1 ≤ nullValue G hist D cn p)
    (he2 : nullValue G hist D cn p ≤ e.2)
    (hr1 : gamma ≤ r.1 → r.1 ≤ nullValue G hist D cn p)
    (hr2 : r.1 < gamma → nullValue G hist D cn p ≤ r.1) :
    (cTablePart2 G.toGame D cn p gamma e r).1 = r.1 ∧
      CTableOK G hist (cTablePart2 G.toGame D cn p gamma e r).2 := by
  unfold cTablePart2
  by_cases hcut : gamma ≤ r.1
  · rw [if_pos hcut]
    refine ⟨rfl, cTableOK_store htok ?_ ?_⟩
    · show r.1 ≤ nullValue G hist D cn p
      exact hr1 hcut
    · show nullValue G hist D cn p ≤ e.2
      exact he2
  · rw [if_neg hcut]
    refine ⟨rfl, cTableOK_store htok ?_ ?_⟩
    · show e.1 ≤ nullValue G hist D cn p
      exact he1
    · show nullValue G hist D cn p ≤ r.1
      exact hr2 (by omega)

/-- The tail of a searched node: optional null yield first (whose cutoff
skips everything else -- the generator's laziness), then the IID table
effect, then the move loop; plain keyed store on every exit. -/
def cNodeTail (G : NullGame) [DecidableEq G.Pos] (gamma : Int) (D : Nat)
    (cn : Bool) (p : G.Pos) (e : Int × Int)
    (iid : CTable G.toGame → CTable G.toGame)
    (f : G.Pos → CTable G.toGame → Int × CTable G.toGame)
    (pass? : Option (CTable G.toGame → Int × CTable G.toGame))
    (t : CTable G.toGame) : Int × CTable G.toGame :=
  match pass? with
  | some pf =>
    if gamma ≤ max LOSS (-(pf t).1) then
      -- Null cutoff: the loop breaks before pulling another yield, so
      -- the IID recursion never runs (laziness, see module comment).
      cTablePart2 G.toGame D cn p gamma e (max LOSS (-(pf t).1), (pf t).2)
    else
      cTablePart2 G.toGame D cn p gamma e
        (searchMovesSt gamma f (G.moves p) (max LOSS (-(pf t).1)) (iid (pf t).2))
  | none =>
    cTablePart2 G.toGame D cn p gamma e
      (searchMovesSt gamma f (G.moves p) LOSS (iid t))

/-- `bound` with the full can_null mechanics: king-gone (312), keyed
lookup (318-320), repetition (325), null move (340-341), IID (355,
`can_null=False`, result discarded, table kept), move loop, plain
keyed store. -/
def boundNullTT (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos] :
    Nat → Bool → G.Pos → Int → CTable G.toGame → Int × CTable G.toGame
  | 0, _, p, _gamma, t =>
    (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p, t)
  | 1, cn, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else if gamma ≤ ((t.find 1 cn p).getD (LOSS, MATE_UPPER)).1 then
      (((t.find 1 cn p).getD (LOSS, MATE_UPPER)).1, t)
    else if ((t.find 1 cn p).getD (LOSS, MATE_UPPER)).2 < gamma then
      (((t.find 1 cn p).getD (LOSS, MATE_UPPER)).2, t)
    else if cn = true ∧ hist p = true then (0, t)
    else
      cNodeTail G gamma 1 cn p ((t.find 1 cn p).getD (LOSS, MATE_UPPER))
        (fun s => s)
        (fun m t' => (-(boundNullTT G hist 0 true m (1 - gamma) t').1,
          (boundNullTT G hist 0 true m (1 - gamma) t').2))
        none t
  | 2, cn, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else if gamma ≤ ((t.find 2 cn p).getD (LOSS, MATE_UPPER)).1 then
      (((t.find 2 cn p).getD (LOSS, MATE_UPPER)).1, t)
    else if ((t.find 2 cn p).getD (LOSS, MATE_UPPER)).2 < gamma then
      (((t.find 2 cn p).getD (LOSS, MATE_UPPER)).2, t)
    else if cn = true ∧ hist p = true then (0, t)
    else
      cNodeTail G gamma 2 cn p ((t.find 2 cn p).getD (LOSS, MATE_UPPER))
        (fun s => s)
        (fun m t' => (-(boundNullTT G hist 1 true m (1 - gamma) t').1,
          (boundNullTT G hist 1 true m (1 - gamma) t').2))
        none t
  | d + 3, cn, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else if gamma ≤ ((t.find (d + 3) cn p).getD (LOSS, MATE_UPPER)).1 then
      (((t.find (d + 3) cn p).getD (LOSS, MATE_UPPER)).1, t)
    else if ((t.find (d + 3) cn p).getD (LOSS, MATE_UPPER)).2 < gamma then
      (((t.find (d + 3) cn p).getD (LOSS, MATE_UPPER)).2, t)
    else if cn = true ∧ hist p = true then (0, t)
    else
      cNodeTail G gamma (d + 3) cn p ((t.find (d + 3) cn p).getD (LOSS, MATE_UPPER))
        (fun t' => (boundNullTT G hist d false p gamma t').2)
        (fun m t' => (-(boundNullTT G hist (d + 2) true m (1 - gamma) t').1,
          (boundNullTT G hist (d + 2) true m (1 - gamma) t').2))
        (if cn = true ∧ nullGuard G.toGame p then
          some (fun t' => boundNullTT G hist d true (G.pass p) (1 - gamma) t')
        else none)
        t

/-- The node-tail lemma: given a valid old entry, the value-shape
equation for this node, spec/preservation of children and IID, and the
null-yield clauses when present, the tail satisfies the point spec
against `nullValue` and preserves the invariant. -/
theorem cNodeTail_spec (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos]
    (gamma : Int) (D : Nat) (cn : Bool) (p : G.Pos) (e : Int × Int)
    (iid : CTable G.toGame → CTable G.toGame)
    (f : G.Pos → CTable G.toGame → Int × CTable G.toGame)
    (pass? : Option (CTable G.toGame → Int × CTable G.toGame))
    (t : CTable G.toGame) (w : G.Pos → Int) (acc0 : Int)
    (ht : CTableOK G hist t)
    (hiid : ∀ s, CTableOK G hist s → CTableOK G hist (iid s))
    (hf : ∀ (m : G.Pos) (s : CTable G.toGame), CTableOK G hist s →
      CTableOK G hist (f m s).2 ∧
      (gamma ≤ (f m s).1 → (f m s).1 ≤ w m) ∧
      ((f m s).1 < gamma → w m ≤ (f m s).1))
    (he1 : e.1 ≤ nullValue G hist D cn p)
    (he2 : nullValue G hist D cn p ≤ e.2)
    (hV : nullValue G hist D cn p = foldMax w (G.moves p) acc0)
    (hpass : ∀ pf, pass? = some pf →
      CTableOK G hist (pf t).2 ∧
      (gamma ≤ max LOSS (-(pf t).1) → max LOSS (-(pf t).1) ≤ acc0) ∧
      (max LOSS (-(pf t).1) < gamma → acc0 ≤ max LOSS (-(pf t).1)))
    (hnone : pass? = none → acc0 = LOSS) :
    ((gamma ≤ (cNodeTail G gamma D cn p e iid f pass? t).1 →
      (cNodeTail G gamma D cn p e iid f pass? t).1 ≤ nullValue G hist D cn p) ∧
     ((cNodeTail G gamma D cn p e iid f pass? t).1 < gamma →
      nullValue G hist D cn p ≤ (cNodeTail G gamma D cn p e iid f pass? t).1)) ∧
    CTableOK G hist (cNodeTail G gamma D cn p e iid f pass? t).2 := by
  cases pass? with
  | none =>
    have hacc := hnone rfl
    subst hacc
    have hloop := searchMovesSt_spec gamma f w (CTableOK G hist) hf
      (G.moves p) LOSS LOSS (iid t) (hiid t ht)
      (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
    simp only [cNodeTail]
    have htp := cTablePart2_ok G hist D cn p gamma e
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
      have htp := cTablePart2_ok G hist D cn p gamma e
        (max LOSS (-(pf t).1), (pf t).2)
        hp.1 he1 he2
        (fun _ => by
          show max LOSS (-(pf t).1) ≤ nullValue G hist D cn p
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
      have htp := cTablePart2_ok G hist D cn p gamma e
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

/-- **Layer 1, proven and unconditional**: the can_null-aware search
brackets `nullValue` with a POINT spec at every `(depth, can_null)`, and
preserves the keyed-table invariant.  No zugzwang hypothesis: the search
and its table are self-consistent whatever the position. -/
theorem boundNullTT_spec (G : NullGame) (hist : G.Pos → Bool) [DecidableEq G.Pos]
    (hb : Bounded G.toGame) :
    ∀ (d : Nat) (cn : Bool) (p : G.Pos) (gamma : Int) (t : CTable G.toGame),
      CTableOK G hist t →
      ((gamma ≤ (boundNullTT G hist d cn p gamma t).1 →
        (boundNullTT G hist d cn p gamma t).1 ≤ nullValue G hist d cn p) ∧
       ((boundNullTT G hist d cn p gamma t).1 < gamma →
        nullValue G hist d cn p ≤ (boundNullTT G hist d cn p gamma t).1)) ∧
      CTableOK G hist (boundNullTT G hist d cn p gamma t).2 := by
  suffices H : ∀ (n d : Nat), d ≤ n → ∀ (cn : Bool) (p : G.Pos) (gamma : Int)
      (t : CTable G.toGame), CTableOK G hist t →
      ((gamma ≤ (boundNullTT G hist d cn p gamma t).1 →
        (boundNullTT G hist d cn p gamma t).1 ≤ nullValue G hist d cn p) ∧
       ((boundNullTT G hist d cn p gamma t).1 < gamma →
        nullValue G hist d cn p ≤ (boundNullTT G hist d cn p gamma t).1)) ∧
      CTableOK G hist (boundNullTT G hist d cn p gamma t).2 by
    exact fun d => H d d (Nat.le_refl d)
  intro n
  induction n with
  | zero =>
    intro d hd cn p gamma t ht
    have hd0 : d = 0 := by omega
    subst hd0
    have hz : (boundNullTT G hist 0 cn p gamma t).1 = nullValue G hist 0 cn p := rfl
    exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
  | succ n ihn =>
    intro d hd cn p gamma t ht
    match d, hd with
    | 0, _ =>
      have hz : (boundNullTT G hist 0 cn p gamma t).1 = nullValue G hist 0 cn p := rfl
      exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
    | 1, _ =>
      have hchild : ∀ (m : G.Pos) (s : CTable G.toGame), CTableOK G hist s →
          CTableOK G hist (boundNullTT G hist 0 true m (1 - gamma) s).2 ∧
          (gamma ≤ -(boundNullTT G hist 0 true m (1 - gamma) s).1 →
            -(boundNullTT G hist 0 true m (1 - gamma) s).1
              ≤ -(nullValue G hist 0 true m)) ∧
          (-(boundNullTT G hist 0 true m (1 - gamma) s).1 < gamma →
            -(nullValue G hist 0 true m)
              ≤ -(boundNullTT G hist 0 true m (1 - gamma) s).1) := by
        intro m s hs
        have hih := ihn 0 (by omega) true m (1 - gamma) s hs
        have h1 := hih.1.1
        have h2 := hih.1.2
        refine ⟨hih.2, fun hge => ?_, fun hlt => ?_⟩
        · have := h2 (by omega); omega
        · have := h1 (by omega); omega
      have hunfB : boundNullTT G hist 1 cn p gamma t
          = (if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
            else if gamma ≤ ((t.find 1 cn p).getD (LOSS, MATE_UPPER)).1 then
              (((t.find 1 cn p).getD (LOSS, MATE_UPPER)).1, t)
            else if ((t.find 1 cn p).getD (LOSS, MATE_UPPER)).2 < gamma then
              (((t.find 1 cn p).getD (LOSS, MATE_UPPER)).2, t)
            else if cn = true ∧ hist p = true then (0, t)
            else
              cNodeTail G gamma 1 cn p ((t.find 1 cn p).getD (LOSS, MATE_UPPER))
                (fun s => s)
                (fun m t' => (-(boundNullTT G hist 0 true m (1 - gamma) t').1,
                  (boundNullTT G hist 0 true m (1 - gamma) t').2))
                none t) := rfl
      rw [hunfB]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]
        have hv := nullValue_kingGone G hist p hkg 1 cn
        exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
      · rw [if_neg hkg]
        have hE := cEntry_valid G hist hb ht 1 cn p
        by_cases hlo : gamma ≤ ((t.find 1 cn p).getD (LOSS, MATE_UPPER)).1
        · rw [if_pos hlo]
          exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
        · rw [if_neg hlo]
          by_cases hhi : ((t.find 1 cn p).getD (LOSS, MATE_UPPER)).2 < gamma
          · rw [if_pos hhi]
            exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
          · rw [if_neg hhi]
            by_cases hrep : cn = true ∧ hist p = true
            · rw [if_pos hrep]
              have hv : nullValue G hist 1 cn p = 0 :=
                nullValue_rep G hist p cn hkg hrep 0
              exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
            · rw [if_neg hrep]
              have hV : nullValue G hist 1 cn p
                  = foldMax (fun m => -(nullValue G hist 0 true m)) (G.moves p) LOSS := by
                simp only [nullValue]
                rw [if_neg hkg, if_neg hrep]
              exact cNodeTail_spec G hist gamma 1 cn p
                ((t.find 1 cn p).getD (LOSS, MATE_UPPER))
                (fun s => s)
                (fun m t' => (-(boundNullTT G hist 0 true m (1 - gamma) t').1,
                  (boundNullTT G hist 0 true m (1 - gamma) t').2))
                none t
                (fun m => -(nullValue G hist 0 true m)) LOSS
                ht (fun s hs => hs) hchild hE.1 hE.2 hV
                (fun pf hpf => Option.noConfusion hpf)
                (fun _ => rfl)
    | 2, hd =>
      have hchild : ∀ (m : G.Pos) (s : CTable G.toGame), CTableOK G hist s →
          CTableOK G hist (boundNullTT G hist 1 true m (1 - gamma) s).2 ∧
          (gamma ≤ -(boundNullTT G hist 1 true m (1 - gamma) s).1 →
            -(boundNullTT G hist 1 true m (1 - gamma) s).1
              ≤ -(nullValue G hist 1 true m)) ∧
          (-(boundNullTT G hist 1 true m (1 - gamma) s).1 < gamma →
            -(nullValue G hist 1 true m)
              ≤ -(boundNullTT G hist 1 true m (1 - gamma) s).1) := by
        intro m s hs
        have hih := ihn 1 (by omega) true m (1 - gamma) s hs
        have h1 := hih.1.1
        have h2 := hih.1.2
        refine ⟨hih.2, fun hge => ?_, fun hlt => ?_⟩
        · have := h2 (by omega); omega
        · have := h1 (by omega); omega
      have hunfB : boundNullTT G hist 2 cn p gamma t
          = (if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
            else if gamma ≤ ((t.find 2 cn p).getD (LOSS, MATE_UPPER)).1 then
              (((t.find 2 cn p).getD (LOSS, MATE_UPPER)).1, t)
            else if ((t.find 2 cn p).getD (LOSS, MATE_UPPER)).2 < gamma then
              (((t.find 2 cn p).getD (LOSS, MATE_UPPER)).2, t)
            else if cn = true ∧ hist p = true then (0, t)
            else
              cNodeTail G gamma 2 cn p ((t.find 2 cn p).getD (LOSS, MATE_UPPER))
                (fun s => s)
                (fun m t' => (-(boundNullTT G hist 1 true m (1 - gamma) t').1,
                  (boundNullTT G hist 1 true m (1 - gamma) t').2))
                none t) := rfl
      rw [hunfB]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]
        have hv := nullValue_kingGone G hist p hkg 2 cn
        exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
      · rw [if_neg hkg]
        have hE := cEntry_valid G hist hb ht 2 cn p
        by_cases hlo : gamma ≤ ((t.find 2 cn p).getD (LOSS, MATE_UPPER)).1
        · rw [if_pos hlo]
          exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
        · rw [if_neg hlo]
          by_cases hhi : ((t.find 2 cn p).getD (LOSS, MATE_UPPER)).2 < gamma
          · rw [if_pos hhi]
            exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
          · rw [if_neg hhi]
            by_cases hrep : cn = true ∧ hist p = true
            · rw [if_pos hrep]
              have hv : nullValue G hist 2 cn p = 0 :=
                nullValue_rep G hist p cn hkg hrep 1
              exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
            · rw [if_neg hrep]
              have hV : nullValue G hist 2 cn p
                  = foldMax (fun m => -(nullValue G hist 1 true m)) (G.moves p) LOSS := by
                simp only [nullValue]
                rw [if_neg hkg, if_neg hrep]
              exact cNodeTail_spec G hist gamma 2 cn p
                ((t.find 2 cn p).getD (LOSS, MATE_UPPER))
                (fun s => s)
                (fun m t' => (-(boundNullTT G hist 1 true m (1 - gamma) t').1,
                  (boundNullTT G hist 1 true m (1 - gamma) t').2))
                none t
                (fun m => -(nullValue G hist 1 true m)) LOSS
                ht (fun s hs => hs) hchild hE.1 hE.2 hV
                (fun pf hpf => Option.noConfusion hpf)
                (fun _ => rfl)
    | (d + 3), hd =>
      have hchild : ∀ (m : G.Pos) (s : CTable G.toGame), CTableOK G hist s →
          CTableOK G hist (boundNullTT G hist (d + 2) true m (1 - gamma) s).2 ∧
          (gamma ≤ -(boundNullTT G hist (d + 2) true m (1 - gamma) s).1 →
            -(boundNullTT G hist (d + 2) true m (1 - gamma) s).1
              ≤ -(nullValue G hist (d + 2) true m)) ∧
          (-(boundNullTT G hist (d + 2) true m (1 - gamma) s).1 < gamma →
            -(nullValue G hist (d + 2) true m)
              ≤ -(boundNullTT G hist (d + 2) true m (1 - gamma) s).1) := by
        intro m s hs
        have hih := ihn (d + 2) (by omega) true m (1 - gamma) s hs
        have h1 := hih.1.1
        have h2 := hih.1.2
        refine ⟨hih.2, fun hge => ?_, fun hlt => ?_⟩
        · have := h2 (by omega); omega
        · have := h1 (by omega); omega
      have hiid : ∀ s, CTableOK G hist s →
          CTableOK G hist ((boundNullTT G hist d false p gamma s).2) :=
        fun s hs => (ihn d (by omega) false p gamma s hs).2
      simp only [boundNullTT]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]
        have hv := nullValue_kingGone G hist p hkg (d + 3) cn
        exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
      · rw [if_neg hkg]
        have hE := cEntry_valid G hist hb ht (d + 3) cn p
        by_cases hlo : gamma ≤ ((t.find (d + 3) cn p).getD (LOSS, MATE_UPPER)).1
        · rw [if_pos hlo]
          exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
        · rw [if_neg hlo]
          by_cases hhi : ((t.find (d + 3) cn p).getD (LOSS, MATE_UPPER)).2 < gamma
          · rw [if_pos hhi]
            exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
          · rw [if_neg hhi]
            by_cases hrep : cn = true ∧ hist p = true
            · rw [if_pos hrep]
              have hv : nullValue G hist (d + 3) cn p = 0 :=
                nullValue_rep G hist p cn hkg hrep (d + 2)
              exact ⟨⟨fun _ => by omega, fun _ => by omega⟩, ht⟩
            · rw [if_neg hrep]
              by_cases hg : cn = true ∧ nullGuard G.toGame p
              · rw [if_pos hg]
                have hV : nullValue G hist (d + 3) cn p
                    = foldMax (fun m => -(nullValue G hist (d + 2) true m)) (G.moves p)
                        (max LOSS (-(nullValue G hist d true (G.pass p)))) := by
                  simp only [nullValue]
                  rw [if_neg hkg, if_neg hrep, if_pos hg]
                refine cNodeTail_spec G hist gamma (d + 3) cn p
                  ((t.find (d + 3) cn p).getD (LOSS, MATE_UPPER))
                  (fun t' => (boundNullTT G hist d false p gamma t').2)
                  (fun m t' => (-(boundNullTT G hist (d + 2) true m (1 - gamma) t').1,
                    (boundNullTT G hist (d + 2) true m (1 - gamma) t').2))
                  (some (fun t' => boundNullTT G hist d true (G.pass p) (1 - gamma) t'))
                  t
                  (fun m => -(nullValue G hist (d + 2) true m))
                  (max LOSS (-(nullValue G hist d true (G.pass p))))
                  ht hiid hchild hE.1 hE.2 hV ?_ ?_
                · intro pf hpf
                  injection hpf with hpf
                  subst hpf
                  simp only []
                  have hih := ihn d (by omega) true (G.pass p) (1 - gamma) t ht
                  have h1 := hih.1.1
                  have h2 := hih.1.2
                  refine ⟨hih.2, fun hge => ?_, fun hlt => ?_⟩
                  · by_cases hpv : gamma ≤
                        -(boundNullTT G hist d true (G.pass p) (1 - gamma) t).1
                    · have := h2 (by omega)
                      omega
                    · omega
                  · have : 1 - gamma ≤
                        (boundNullTT G hist d true (G.pass p) (1 - gamma) t).1 := by
                      omega
                    have := h1 this
                    omega
                · intro hcontra
                  exact Option.noConfusion hcontra
              · rw [if_neg hg]
                have hV : nullValue G hist (d + 3) cn p
                    = foldMax (fun m => -(nullValue G hist (d + 2) true m)) (G.moves p)
                        LOSS := by
                  simp only [nullValue]
                  rw [if_neg hkg, if_neg hrep, if_neg hg]
                exact cNodeTail_spec G hist gamma (d + 3) cn p
                  ((t.find (d + 3) cn p).getD (LOSS, MATE_UPPER))
                  (fun t' => (boundNullTT G hist d false p gamma t').2)
                  (fun m t' => (-(boundNullTT G hist (d + 2) true m (1 - gamma) t').1,
                    (boundNullTT G hist (d + 2) true m (1 - gamma) t').2))
                  none t
                  (fun m => -(nullValue G hist (d + 2) true m)) LOSS
                  ht hiid hchild hE.1 hE.2 hV
                  (fun pf hpf => Option.noConfusion hpf)
                  (fun _ => rfl)

/-! ### Layer 2: where the bet lives -/

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

/-- **NullBetOK** -- the layer-2 hypothesis, exactly as the code places
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

/-- **Layer 2, proven under the named hypothesis**: with `NullBetOK` and
an empty history (so the repetition gate never fires), the layer-1 value
coincides with the null-free `plainValue` at every `(depth, can_null)` --
the pass option never raises the fold, so composing with layer 1's point
spec recovers the original docstring against `plainValue`.  Zugzwang
(¬`NullBetOK`) breaks exactly this bridge and nothing in layer 1;
non-empty histories additionally shift the target toward the draw-aware
semantics of `Sunfish/Stalemate.lean`. -/
theorem nullValue_plain (G : NullGame) (hbet : NullBetOK G) :
    ∀ (d : Nat) (cn : Bool) (p : G.Pos),
      nullValue G (fun _ => false) d cn p = plainValue G d p := by
  suffices H : ∀ (n d : Nat), d ≤ n → ∀ (cn : Bool) (p : G.Pos),
      nullValue G (fun _ => false) d cn p = plainValue G d p by
    exact fun d => H d d (Nat.le_refl d)
  intro n
  induction n with
  | zero =>
    intro d hd cn p
    have hd0 : d = 0 := by omega
    subst hd0
    rfl
  | succ n ihn =>
    intro d hd cn p
    have hrep : ¬ (cn = true ∧ (fun _ : G.Pos => false) p = true) :=
      fun h => Bool.noConfusion h.2
    match d, hd with
    | 0, _ => rfl
    | 1, _ =>
      have hL : nullValue G (fun _ => false) 1 cn p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if cn = true ∧ (fun _ : G.Pos => false) p = true then 0
            else foldMax (fun m => -(nullValue G (fun _ => false) 0 true m))
              (G.moves p) LOSS) := rfl
      have hR : plainValue G 1 p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else foldMax (fun m => -(plainValue G 0 m)) (G.moves p) LOSS) := rfl
      rw [hL, hR]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg, if_pos hkg]
      · rw [if_neg hkg, if_neg hkg, if_neg hrep]
        exact foldMax_congr _ _ (G.moves p) LOSS (fun m _ => by
          show -(nullValue G (fun _ => false) 0 true m) = -(plainValue G 0 m)
          rw [ihn 0 (by omega) true m])
    | 2, hd =>
      have hL : nullValue G (fun _ => false) 2 cn p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if cn = true ∧ (fun _ : G.Pos => false) p = true then 0
            else foldMax (fun m => -(nullValue G (fun _ => false) 1 true m))
              (G.moves p) LOSS) := rfl
      have hR : plainValue G 2 p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else foldMax (fun m => -(plainValue G 1 m)) (G.moves p) LOSS) := rfl
      rw [hL, hR]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg, if_pos hkg]
      · rw [if_neg hkg, if_neg hkg, if_neg hrep]
        exact foldMax_congr _ _ (G.moves p) LOSS (fun m _ => by
          show -(nullValue G (fun _ => false) 1 true m) = -(plainValue G 1 m)
          rw [ihn 1 (by omega) true m])
    | (d + 3), hd =>
      have hL : nullValue G (fun _ => false) (d + 3) cn p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else if cn = true ∧ (fun _ : G.Pos => false) p = true then 0
            else foldMax (fun m => -(nullValue G (fun _ => false) (d + 2) true m))
              (G.moves p)
              (if cn = true ∧ nullGuard G.toGame p then
                max LOSS (-(nullValue G (fun _ => false) d true (G.pass p)))
              else LOSS)) := rfl
      have hR : plainValue G (d + 3) p
          = (if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
            else foldMax (fun m => -(plainValue G (d + 2) m)) (G.moves p) LOSS) := rfl
      rw [hL, hR]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg, if_pos hkg]
      · rw [if_neg hkg, if_neg hkg, if_neg hrep]
        have hcong : foldMax (fun m => -(nullValue G (fun _ => false) (d + 2) true m))
              (G.moves p) LOSS
            = foldMax (fun m => -(plainValue G (d + 2) m)) (G.moves p) LOSS :=
          foldMax_congr _ _ (G.moves p) LOSS (fun m _ => by
            show -(nullValue G (fun _ => false) (d + 2) true m) = -(plainValue G (d + 2) m)
            rw [ihn (d + 2) (by omega) true m])
        by_cases hg : cn = true ∧ nullGuard G.toGame p
        · rw [if_pos hg]
          rw [show (-(nullValue G (fun _ => false) d true (G.pass p)))
              = (-(plainValue G d (G.pass p))) from by
            rw [ihn d (by omega) true (G.pass p)]]
          rw [foldMax_max, hcong]
          -- The bet: the pass option never exceeds the real fold.
          cases hbet d p hg.2 with
          | intro m hm =>
            have hmem : -(plainValue G (d + 2) m)
                ≤ foldMax (fun x => -(plainValue G (d + 2) x)) (G.moves p) LOSS :=
              foldMax_le_of_mem _ _ _ m hm.1
            omega
        · rw [if_neg hg]
          exact hcong

end Sunfish
