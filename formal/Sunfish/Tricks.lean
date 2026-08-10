/-
Precise statements of the search "tricks" that the proven model in
`Sunfish/Bound.lean` deliberately leaves out.  Everything here is now
PROVEN -- unconditionally where possible, otherwise under explicit named
hypotheses in the statement (the honest form): the point is that each
trick has a *name* and a *hypothesis*, so that a search-changing PR can
say which lemma it preserves, weakens, or newly depends on.
-/

import Sunfish.Bound

namespace Sunfish

/-! ### (a) Mate-score softening  (PROVEN)

Mate scores live strictly beyond the band `[-ML, ML]` (in sunfish:
`MATE_LOWER = 47923`, sunfish.py line 122).  When such a score is passed one
ply up the tree it must be pulled one step toward zero -- "mate in k" becomes
"mate in k+1" -- otherwise the engine cannot distinguish faster from slower
mates.  `soften` is that adjustment.

The dangerous question is whether softening breaks the null-window test:
the parent compares the softened, negated child score against its own
`gamma`.  The lemma says that for any window strictly above the mate band
(`gamma > ML + 1`), the softened test `soften (-r) ≥ gamma` is *exactly* the
integer test `r ≤ -gamma - 1` on the raw child score -- so the parent's
fail-high decision is unchanged by softening, off by exactly the one mate
ply.  Note `0 ≤ ML` is required: for a negative "mate band" both `if`s can
fire at once and the equivalence is false (e.g. ML = -5, gamma = -3, r = 3).
-/

/-- Pull scores strictly beyond the mate band `[-ML, ML]` one step toward
zero: `s ↦ s - 1` above the band, `s ↦ s + 1` below it, identity inside. -/
def soften (ML s : Int) : Int :=
  s - (if ML < s then 1 else 0) + (if s < -ML then 1 else 0)

/-- The mate-score softening window lemma (sorry-free). -/
theorem soften_null_window (ML gamma r : Int) (hML : 0 ≤ ML)
    (hgamma : ML + 1 < gamma) :
    gamma ≤ soften ML (-r) ↔ r ≤ -gamma - 1 := by
  unfold soften
  split <;> split <;> omega

/-! ### (b) The null-move hypothesis  (STATED)

sunfish.py lines 364-365 (since `eda66ee` the gate is position-determined
plus the driver flag `root`, which never reaches the table):

    if depth > 2 and not root and abs(pos.score) < 500:
        score = -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)

Null-move pruning bets that *doing nothing is never your best option*: if
even passing the turn fails high, some real move surely does too.  That bet
is exactly `NullOK` below.  Its negation is zugzwang: a non-terminal
position in which every legal move is strictly worse than passing.  So the
correctness statement `boundNull_spec` is *conditional* -- this is why
sunfish guards the trick with `abs(pos.score) < 500` (line 322-330: the
FIXME about zugzwang and king-capture) rather than using it everywhere.

Modeling choice: sunfish also reduces the null search by 3 plies
(`depth - 3`); the model searches the pass at full depth, because depth
reduction needs an additional depth-stability hypothesis that is orthogonal
to the zugzwang question being named here.
-/

/-- **The null-move hypothesis**: in every non-terminal position, some legal
move scores at least as well (for us) as passing, i.e. the opponent's value
after some real move is at most their value after the pass.  Zugzwang is
precisely the failure of this Prop. -/
def NullOK (G : NullGame) : Prop :=
  ∀ (d : Nat) (p : G.Pos), G.moves p ≠ [] →
    ∃ m, m ∈ G.moves p ∧
      negamax G.toGame d m ≤ negamax G.toGame d (G.pass p)

/- RESTRUCTURED (audit; re-collapsed after `eda66ee` removed `can_null`):
the former `boundNull`/`boundNull_spec` here are superseded by
`Sunfish/CanNull.lean`, now FLAGLESS like the code:

* `boundNullTT_spec` (PROVEN, unconditional): the null-and-repetition-
  augmented interior search brackets its own value function `nullValue`
  -- ONE function, no flag -- with a point spec, and the
  `(pos, depth)`-keyed table stays consistent.  Zugzwang cannot break
  self-consistency.
* `rootProbe_spec` (PROVEN, unconditional): the driver probes (the
  search root and IID, `root=True`) skip the table in both directions
  and store nothing; they bracket their own `rootValue` and cannot
  disturb the table invariant.
* `nullValue_plain` (PROVEN under `NullBetOK`): relating `nullValue` to
  the null-free `plainValue` is where the zugzwang bet lives -- `NullOK`
  above is its same-depth core, `NullBetOK` its code-exact form with the
  `depth - 3` reduction folded in. -/

/-! ### (c) The transposition-table invariant  (STATED)

sunfish.py lines 288-289, 324-336, 481-485.  An `Entry(lower, upper)` stored
under key `(pos, depth)` promises `lower ≤ s* ≤ upper` for the
negamax value `s*` of that position at that depth; the comment at line 288
(`# lower <= s(pos) <= upper`) *is* `TableOK`.  Lookups may then answer a
query without searching (lines 309-310), and every exit of `bound` tightens
one side of the entry (lines 415-418).

`Bounded` is needed because a fresh entry starts as
`Entry(-MATE_UPPER, MATE_UPPER)` (line 308): that is only a valid bracket if
all scores actually live in `[-MATE_UPPER, MATE_UPPER]`, which sunfish
guarantees by construction of the evaluation.
-/

/-- A transposition table, abstracted as a partial map from (depth,
position) to an `Entry (lower, upper)` pair.  (Sunfish clamps QS depths
to 0, line 315 -- a refinement of the same invariant.  Since `eda66ee`
the key is exactly `(pos, depth)`: the only deviant-semantics calls, the
driver probes, never touch the table -- see `Sunfish/CanNull.lean`,
which reuses this very `Table` for its keyed invariant `CTableOK`.) -/
structure Table (G : Game) where
  find : Nat → G.Pos → Option (Int × Int)

/-- **The table invariant**: every stored entry brackets the true negamax
value at its depth.  This is sunfish.py line 275: `# lower <= s(pos) <= upper`. -/
def TableOK (G : Game) (t : Table G) : Prop :=
  ∀ (d : Nat) (p : G.Pos) (lo hi : Int),
    t.find d p = some (lo, hi) → lo ≤ negamax G d p ∧ negamax G d p ≤ hi

/-- Store an entry (functional update; sunfish's `TABLE_SIZE` eviction,
lines 419-420, only ever *forgets* entries, which trivially preserves
`TableOK`, so it is not modeled). -/
def Table.store {G : Game} [DecidableEq G.Pos] (t : Table G) (d : Nat)
    (p : G.Pos) (e : Int × Int) : Table G :=
  ⟨fun d' p' => if d' = d ∧ p' = p then some e else t.find d' p'⟩

/-- Storing a valid bracket preserves the invariant. -/
theorem tableOK_store {G : Game} [DecidableEq G.Pos] {t : Table G} {d : Nat}
    {p : G.Pos} {lo hi : Int} (ht : TableOK G t)
    (h : lo ≤ negamax G d p ∧ negamax G d p ≤ hi) :
    TableOK G (Table.store t d p (lo, hi)) := by
  intro d' p' lo' hi' hfind
  simp only [Table.store] at hfind
  by_cases hdp : d' = d ∧ p' = p
  · rw [if_pos hdp] at hfind
    injection hfind with h1
    rw [Prod.mk.injEq] at h1
    rw [hdp.1, hdp.2, ← h1.1, ← h1.2]
    exact h
  · rw [if_neg hdp] at hfind
    exact ht d' p' lo' hi' hfind

/-- Under `Bounded`, negamax values live in the band -- what makes the
fresh entry `Entry(-MATE_UPPER, MATE_UPPER)` of line 308 a valid bracket. -/
theorem negamax_bounded (G : Game) (hb : Bounded G) :
    ∀ (d : Nat) (p : G.Pos),
      -MATE_UPPER ≤ negamax G d p ∧ negamax G d p ≤ MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d with
  | zero =>
    intro p
    have := hb p
    simp only [negamax]
    omega
  | succ d ih =>
    intro p
    simp only [negamax]
    constructor
    · have := foldMax_ge_init (fun m => -(negamax G d m)) (G.moves p) LOSS
      omega
    · refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
      show -(negamax G d m) ≤ MATE_UPPER
      have := ih m
      omega

/-- The move loop of `boundTT`, threading the table through the child
searches (state-passing version of `searchMoves`). -/
def searchMovesTT {G : Game} (gamma : Int)
    (f : G.Pos → Table G → Int × Table G) :
    List G.Pos → Int → Table G → Int × Table G
  | [], best, t => (best, t)
  | m :: ms, best, t =>
    if gamma ≤ max best (f m t).1 then (max best (f m t).1, (f m t).2)
    else searchMovesTT gamma f ms (max best (f m t).1) (f m t).2

/-- Table part 2, PRE-2c95ab0 point version (the historical lines
415-418): tighten the reported side, keep the other.  The clamped
clamped-store era (2c95ab0..7f9f164) lives in git history. -/
def tablePart2 (G : Game) [DecidableEq G.Pos] (d : Nat) (p : G.Pos)
    (gamma : Int) (e : Int × Int) (r : Int × Table G) : Int × Table G :=
  if gamma ≤ r.1 then (r.1, Table.store r.2 d p (r.1, e.2))
  else (r.1, Table.store r.2 d p (e.1, r.1))

/-- `bound` with a transposition table: lookup before searching
(sunfish.py lines 308-310), store the tightened bound on exit. -/
def boundTT (G : Game) [DecidableEq G.Pos] :
    Nat → G.Pos → Int → Table G → Int × Table G
  | 0, p, _gamma, t => (G.eval p, t)
  | d + 1, p, gamma, t =>
    -- entry = tp_score.get(..., Entry(-MATE_UPPER, MATE_UPPER))  (line 308)
    if gamma ≤ ((t.find (d + 1) p).getD (LOSS, MATE_UPPER)).1 then
      (((t.find (d + 1) p).getD (LOSS, MATE_UPPER)).1, t)   -- line 309
    else if ((t.find (d + 1) p).getD (LOSS, MATE_UPPER)).2 < gamma then
      (((t.find (d + 1) p).getD (LOSS, MATE_UPPER)).2, t)   -- line 310
    else
      tablePart2 G (d + 1) p gamma ((t.find (d + 1) p).getD (LOSS, MATE_UPPER))
        (searchMovesTT gamma
          (fun m t' => (-(boundTT G d m (1 - gamma) t').1,
            (boundTT G d m (1 - gamma) t').2))
          (G.moves p) LOSS t)

/-- The state-passing loop is fail-soft correct AND preserves the table
invariant, provided every child call is (`searchMoves_spec` fused with
invariant threading). -/
theorem searchMovesTT_spec (G : Game) (gamma : Int)
    (f : G.Pos → Table G → Int × Table G) (w : G.Pos → Int)
    (hf : ∀ (m : G.Pos) (t : Table G), TableOK G t →
      TableOK G (f m t).2 ∧
      (gamma ≤ (f m t).1 → (f m t).1 ≤ w m) ∧
      ((f m t).1 < gamma → w m ≤ (f m t).1)) :
    ∀ (ms : List G.Pos) (best acc : Int) (t : Table G), TableOK G t →
      (gamma ≤ best → best ≤ acc) →
      (best < gamma → acc ≤ best) →
      TableOK G (searchMovesTT gamma f ms best t).2 ∧
      (gamma ≤ (searchMovesTT gamma f ms best t).1 →
        (searchMovesTT gamma f ms best t).1 ≤ foldMax w ms acc) ∧
      ((searchMovesTT gamma f ms best t).1 < gamma →
        foldMax w ms acc ≤ (searchMovesTT gamma f ms best t).1) := by
  intro ms
  induction ms with
  | nil =>
    intro best acc t ht h1 h2
    simp only [searchMovesTT, foldMax]
    exact ⟨ht, h1, h2⟩
  | cons m ms ih =>
    intro best acc t ht h1 h2
    have hfm := hf m t ht
    have hm1 := hfm.2.1
    have hm2 := hfm.2.2
    simp only [searchMovesTT, foldMax]
    by_cases hcut : gamma ≤ max best (f m t).1
    · rw [if_pos hcut]
      have hrest := foldMax_ge_init w ms (max acc (w m))
      refine ⟨hfm.1, fun _ => ?_, fun hlt => ?_⟩
      · show max best (f m t).1 ≤ foldMax w ms (max acc (w m))
        by_cases hfge : gamma ≤ (f m t).1
        · have := hm1 hfge
          by_cases hb : gamma ≤ best
          · have := h1 hb; omega
          · omega
        · have hb : gamma ≤ best := by omega
          have := h1 hb
          omega
      · omega
    · rw [if_neg hcut]
      have hfl : (f m t).1 < gamma := by omega
      have hb : best < gamma := by omega
      have hwm := hm2 hfl
      have hacc := h2 hb
      exact ih (max best (f m t).1) (max acc (w m)) (f m t).2 hfm.1
        (fun hge => absurd hge hcut)
        (fun _ => by omega)

/-- The point-version store step: given a valid old entry and a fail-soft
correct report, the tightened entry is a valid bracket. -/
theorem tablePart2_ok (G : Game) [DecidableEq G.Pos] (d : Nat) (p : G.Pos)
    (gamma : Int) (e : Int × Int) (r : Int × Table G)
    (htok : TableOK G r.2)
    (he1 : e.1 ≤ negamax G d p) (he2 : negamax G d p ≤ e.2)
    (hr1 : gamma ≤ r.1 → r.1 ≤ negamax G d p)
    (hr2 : r.1 < gamma → negamax G d p ≤ r.1) :
    (tablePart2 G d p gamma e r).1 = r.1 ∧
      TableOK G (tablePart2 G d p gamma e r).2 := by
  unfold tablePart2
  by_cases hcut : gamma ≤ r.1
  · rw [if_pos hcut]
    exact ⟨rfl, tableOK_store htok ⟨hr1 hcut, he2⟩⟩
  · rw [if_neg hcut]
    exact ⟨rfl, tableOK_store htok ⟨he1, hr2 (by omega)⟩⟩

/-- **A table-using `bound` both answers correctly and preserves
`TableOK` -- proven.**  The preservation half is the real content: it is
what makes it sound for later queries (with different `gamma`!) to trust
lines 309-310.  This is the POINT-spec version -- `bound_spec` plus
invariant-threading through the state-passing loop -- and is sound for
the pre-LMR search model (`Sunfish/Bound.lean`); under LMR the point
invariant was temporarily unachievable (the machine-checked TT crossing,
deleted with the mechanism -- git history has both). -/
theorem boundTT_spec (G : Game) [DecidableEq G.Pos] (hb : Bounded G) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int) (t : Table G), TableOK G t →
      BoundSpec G d p gamma (boundTT G d p gamma t).1 ∧
      TableOK G (boundTT G d p gamma t).2 := by
  intro d
  induction d with
  | zero =>
    intro p gamma t ht
    refine ⟨⟨fun _ => ?_, fun _ => ?_⟩, ?_⟩
    · simp only [boundTT, negamax]; omega
    · simp only [boundTT, negamax]; omega
    · simp only [boundTT]; exact ht
  | succ d ih =>
    intro p gamma t ht
    -- The current entry is a valid bracket (stored, or the band default).
    have hE : ((t.find (d + 1) p).getD (LOSS, MATE_UPPER)).1 ≤ negamax G (d + 1) p ∧
        negamax G (d + 1) p ≤ ((t.find (d + 1) p).getD (LOSS, MATE_UPPER)).2 := by
      cases hfind : t.find (d + 1) p with
      | none =>
        have hband := negamax_bounded G hb (d + 1) p
        have hLOSS : LOSS = -MATE_UPPER := rfl
        refine ⟨?_, ?_⟩
        · show LOSS ≤ negamax G (d + 1) p
          omega
        · show negamax G (d + 1) p ≤ MATE_UPPER
          omega
      | some e =>
        exact ht (d + 1) p e.1 e.2 (by rw [hfind])
    simp only [boundTT]
    by_cases hlo : gamma ≤ ((t.find (d + 1) p).getD (LOSS, MATE_UPPER)).1
    · -- Entry lower already answers (line 309): a valid lower bound.
      rw [if_pos hlo]
      refine ⟨⟨fun _ => ?_, fun hlt => ?_⟩, ht⟩
      · exact hE.1
      · omega
    · rw [if_neg hlo]
      by_cases hhi : ((t.find (d + 1) p).getD (LOSS, MATE_UPPER)).2 < gamma
      · -- Entry upper already answers (line 310): a valid upper bound.
        rw [if_pos hhi]
        refine ⟨⟨fun hge => ?_, fun _ => ?_⟩, ht⟩
        · omega
        · exact hE.2
      · -- The searched branch: loop spec + store step.
        rw [if_neg hhi]
        have hf : ∀ (m : G.Pos) (t' : Table G), TableOK G t' →
            TableOK G (boundTT G d m (1 - gamma) t').2 ∧
            (gamma ≤ -(boundTT G d m (1 - gamma) t').1 →
              -(boundTT G d m (1 - gamma) t').1 ≤ -(negamax G d m)) ∧
            (-(boundTT G d m (1 - gamma) t').1 < gamma →
              -(negamax G d m) ≤ -(boundTT G d m (1 - gamma) t').1) := by
          intro m t' ht'
          have hih := ih m (1 - gamma) t' ht'
          have h1 := hih.1.1
          have h2 := hih.1.2
          refine ⟨hih.2, fun hge => ?_, fun hlt => ?_⟩
          · have := h2 (by omega); omega
          · have := h1 (by omega); omega
        have hloop := searchMovesTT_spec G gamma
          (fun m t' => (-(boundTT G d m (1 - gamma) t').1,
            (boundTT G d m (1 - gamma) t').2))
          (fun m => -(negamax G d m)) hf
          (G.moves p) LOSS LOSS t ht
          (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
        have hneq : negamax G (d + 1) p
            = foldMax (fun m => -(negamax G d m)) (G.moves p) LOSS := rfl
        have htp := tablePart2_ok G (d + 1) p gamma
          ((t.find (d + 1) p).getD (LOSS, MATE_UPPER))
          (searchMovesTT gamma
            (fun m t' => (-(boundTT G d m (1 - gamma) t').1,
              (boundTT G d m (1 - gamma) t').2))
            (G.moves p) LOSS t)
          hloop.1 hE.1 hE.2
          (fun hge => by rw [hneq]; exact hloop.2.1 hge)
          (fun hlt => by rw [hneq]; exact hloop.2.2 hlt)
        rw [htp.1]
        refine ⟨⟨fun hge => ?_, fun hlt => ?_⟩, htp.2⟩
        · rw [hneq]; exact hloop.2.1 hge
        · rw [hneq]; exact hloop.2.2 hlt

/-! ### (d) The recapture-extension key problem  (STATED + counterexample)

Suppose the search extends (searches deeper) when a *recapture* is
available -- a move landing on the square of the previous capture.  Then the
value computed for a position depends not only on `(pos, depth)` but also on
*how we got there* (the last-capture square).  A transposition table keyed
by `(pos, depth)` alone would conflate a node reached by a capture with the
same node reached quietly, and return a value computed under a different
extension regime.  Hence the TT key must include the last-capture square
(or whatever state drives the extension).

sunfish sidesteps this by having no recapture extension: its QS (depth 0
region, lines 369-406) re-derives capture information from `pos` itself via
`pos.value(move)`, so its value IS a function of `pos` alone, and
`(pos, depth)` is an honest key.  `can_null` was the same phenomenon in
miniature: while root/IID calls could store, their no-null,
no-repetition semantics changed the value, so the flag had to be part of
the key; at `eda66ee` those calls became table-invisible (unstored in
both directions) and the flag left the key with them
(`Sunfish/CanNull.lean`).

We model "how we got there" as an explicit `State` threaded through the
search, and state the (false!) key-independence claim as a Prop; the
counterexample below refutes it, sorry-free.
-/

/-- A game with search-relevant history: `State` is e.g. the last-capture
square, `step` updates it across a move, `extend` decides a (bounded, one
extra ply at the horizon) extension. -/
structure ExtGame extends Game where
  State : Type
  step : State → Pos → Pos → State
  extend : State → Pos → Bool

/-- Negamax with a horizon extension driven by history: at depth 0, if the
history warrants it (e.g. a recapture is available), look one ply further
instead of standing pat. -/
def negamaxExt (G : ExtGame) : Nat → G.State → G.Pos → Int
  | 0, s, p =>
    if G.extend s p then
      foldMax (fun m => -(G.eval m)) (G.moves p) LOSS
    else G.eval p
  | d + 1, s, p =>
    foldMax (fun m => -(negamaxExt G d (G.step s p m) m)) (G.moves p) LOSS

/-- The claim a `(pos, depth)`-keyed table silently makes about an extended
search: the value does not depend on the history component.  This Prop is
FALSIFIABLE -- see `extended_value_not_key_independent`. -/
def ExtKeyIndependent (G : ExtGame) : Prop :=
  ∀ (d : Nat) (s₁ s₂ : G.State) (p : G.Pos),
    negamaxExt G d s₁ p = negamaxExt G d s₂ p

/-- Minimal counterexample: a single terminal position, with the history
bit deciding whether the horizon is extended.  Extended, the terminal node
scores `LOSS`; unextended, it stands pat at `0`. -/
def recaptureCounterexample : ExtGame where
  Pos := Unit
  moves := fun _ => []
  eval := fun _ => 0
  State := Bool
  step := fun _ _ _ => false
  extend := fun s _ => s

/-- Extended-search value is NOT a function of `(pos, depth)` alone
(sorry-free).  This is the honest statement of why a TT key must include
the last-capture square once recapture extensions exist. -/
theorem extended_value_not_key_independent :
    ¬ ExtKeyIndependent recaptureCounterexample := by
  intro h
  have h0 := h 0 true false ()
  simp [negamaxExt, recaptureCounterexample, foldMax, LOSS, MATE_UPPER] at h0

/-! ### (e) The futility yield  (DISCHARGED -- see `boundFut_spec` below)

sunfish.py lines 360-374: at `depth ≤ 1`, moves come in decreasing order
of `pos.value(move)`, and once `pos.score + val < gamma` the child search
is replaced by the STATIC ESTIMATE `pos.score + val` (and the loop breaks;
all later moves have smaller `val`).

The estimate is below `gamma`, so it is only ever a *fail-low* report, and
`BoundSpec` demands of a fail-low report exactly one thing: that it be an
upper bound on the move's true value,

    -(negamax d child)  ≤  pos.score + val.

That inequality is `FutilityOK`.  It is sunfish's own justification read
off the comment at lines 365-367 ("the opponent will for sure just stand
pat"): after our move the opponent's value is at least their stand-pat
score `-(pos.score + val)`, hence ours is at most `pos.score + val`.  The
hypothesis can fail exactly where stand-pat reasoning fails -- e.g. when
the opponent is in check after the move (they cannot "stand pat" out of a
king capture), which is why futility is confined to the QS-adjacent depths
where `eval` is trusted to be a stand-pat bound.

The special case at line 371 (`pos.score + val if val < MATE_LOWER else
MATE_UPPER`) routes king captures around the estimate: they must report
the exact sentinel `MATE_UPPER` (the requirement of sunfish.py lines
398-401; see `Sunfish/Stalemate.lean`).  It is also the one futility yield
that can fail HIGH, so its soundness is a separate named hypothesis,
`FutilityMateOK`. -/

/-- A `Game` together with sunfish's move valuation `pos.value(move)`
(indexed here by the child position the move reaches). -/
structure FutGame extends Game where
  val : Pos → Pos → Int

/-- **FutilityOK**: the static estimate dominates the true child value at
the futility depths, so a pruned move only ever under-promises.
HISTORICAL: kept for reference -- this is no longer a hypothesis.  Its
only instance the search consumes (`d = 0`) is discharged as an
*equality* by the score identity (`futilityOK_discharged` below), and
`boundFut_spec` is now proven without it. -/
def FutilityOK (G : FutGame) : Prop :=
  ∀ (d : Nat) (p : G.Pos), ∀ m ∈ G.moves p,
    -(negamax G.toGame d m) ≤ G.eval p + G.val p m

/-- **FutilityMateOK**: a move valued as a king capture
(`val ≥ MATE_LOWER`) really wins everything -- the fail-high side of the
`else MATE_UPPER` yield of line 371. -/
def FutilityMateOK (G : FutGame) : Prop :=
  ∀ (d : Nat) (p : G.Pos), ∀ m ∈ G.moves p,
    MATE_LOWER ≤ G.val p m → MATE_UPPER ≤ -(negamax G.toGame d m)

/-- `bound` with the futility yield at depth 1 (the `depth ≤ 1` zone of
line 368; our depth-0 bound is already a bare eval).  Modeling choice: we
yield the estimate for every futile move instead of breaking at the first
one; under the sorted order of line 360 the break skips only moves with
even smaller estimates, so the two variants report the same fail-low
facts, and we do not model the ordering. -/
def boundFut (G : FutGame) : Nat → G.Pos → Int → Int
  | 0, p, _gamma => G.eval p
  | 1, p, gamma =>
    searchMoves gamma
      (fun m =>
        if G.eval p + G.val p m < gamma then
          -- futility: yield the static estimate (sunfish.py line 371) ...
          if G.val p m < MATE_LOWER then G.eval p + G.val p m
          -- ... unless the move is a king capture, which must report the
          -- exact sentinel (line 371, `else MATE_UPPER`)
          else MATE_UPPER
        else -(boundFut G 0 m (1 - gamma)))
      (G.moves p) LOSS
  | d + 2, p, gamma =>
    searchMoves gamma (fun m => -(boundFut G (d + 1) m (1 - gamma))) (G.moves p) LOSS

/-! #### FutilityOK, DISCHARGED

The maintainer's observation: `FutilityOK` is not a hypothesis at all --
it is a theorem, once the model records how sunfish actually builds child
positions.  `pos.move(move)` computes the child's score *incrementally*:

    child.score = -(pos.score + pos.value(move))        (score identity)

This is literal in sunfish (`Position.move` ends with
`Position(...).rotate()`, negating the accumulated `score + value`), and
the comment at lines 365-367 -- `pos.score + val < gamma  ===
-(pos.score + val) >= 1 - gamma` -- is exactly this identity applied to
the futility test.  `ValGame` adds the identity as a structural property:
model-faithful, not an assumption.

With it, the futility yield is *not an estimate*:

1. `pos.score + val < gamma` is integer-equivalent to
   `child.score ≥ 1 - gamma` -- the child's stand-pat meets its window.
2. At `depth ≤ 1` the child is searched at depth 0, where stand-pat is
   yielded FIRST; by (1) it fails high immediately, and fail-soft returns
   exactly `child.score = -(pos.score + val)`.  So the parent's yield
   `pos.score + val` equals *exactly* what the search would have
   returned (`futilityOK_discharged` below is an equality, not a `≤`).
3. The child's true QS value is ≥ its stand-pat (stand-pat is always
   among the options), so the yield upper-bounds the move's true value --
   precisely the direction a fail-low report needs.  (A child TT hit may
   return a different number, but with the same fail direction, bounding
   the same function.)
4. The `break` after the futility yield is covered by sort order: the
   remaining moves have smaller `val`, hence smaller estimates, all
   upper-bounded by the yielded one (our model yields them all, which is
   equivalent for the fail-low facts).

Note the fine print made explicit by the model: the *quantified*
`FutilityOK` (∀ depths) is not dischargeable -- plain `negamax` at
`d ≥ 1` has no stand-pat option, so "true value ≥ stand-pat" is a QS
property.  But the search only ever consumes the `d = 0` instance (the
futility zone searches children at depth 0), and *that* instance is the
score identity on the nose.  The old statement over-required.

Contrast with the retired LMR (git history): futility's shortcut provably
one-side-bounds the SAME value function the full search targets -- hence
a single-function `BoundSpec`; LMR's reduced value is
incomparable to the full value -- hence the crossing and the
TT crossing.  `FutilityMateOK` (the `val ≥ MATE_LOWER` king-capture
bypass of line 371) remains a hypothesis as before: it asserts a fact
about king captures, not about the score arithmetic. -/

/-- A `FutGame` whose evaluation is incremental, as sunfish's really is:
the child of `p` by a move valued `val p m` evaluates to
`-(eval p + val p m)` (the board is rotated, hence the negation). -/
structure ValGame extends FutGame where
  score_identity : ∀ (p m : Pos), m ∈ moves p → eval m = -(eval p + val p m)

/-- **The old `FutilityOK` hypothesis, discharged as an equality** at the
depth where futility actually fires: the static estimate IS the depth-0
child search result, exactly. -/
theorem futilityOK_discharged (G : ValGame) :
    ∀ (p : G.Pos), ∀ m ∈ G.toFutGame.toGame.moves p,
      -(negamax G.toFutGame.toGame 0 m)
        = G.toFutGame.toGame.eval p + G.toFutGame.val p m := by
  intro p m hm
  have hid := G.score_identity p m hm
  have hn : negamax G.toFutGame.toGame 0 m = G.toFutGame.toGame.eval m := rfl
  omega

/-- `searchMoves_spec` with the per-child hypothesis restricted to actual
members of the move list (the score identity only speaks about legal
moves). -/
theorem searchMoves_spec_mem {α : Type _} (gamma : Int) (f w : α → Int) :
    ∀ (ms : List α),
      (∀ m ∈ ms, (gamma ≤ f m → f m ≤ w m) ∧ (f m < gamma → w m ≤ f m)) →
      ∀ (best acc : Int),
      (gamma ≤ best → best ≤ acc) →
      (best < gamma → acc ≤ best) →
      (gamma ≤ searchMoves gamma f ms best →
        searchMoves gamma f ms best ≤ foldMax w ms acc) ∧
      (searchMoves gamma f ms best < gamma →
        foldMax w ms acc ≤ searchMoves gamma f ms best) := by
  intro ms
  induction ms with
  | nil =>
    intro _ best acc h1 h2
    simp only [searchMoves, foldMax]
    exact ⟨h1, h2⟩
  | cons m ms ih =>
    intro hchild best acc h1 h2
    have hm1 := (hchild m (by simp)).1
    have hm2 := (hchild m (by simp)).2
    simp only [searchMoves, foldMax]
    by_cases hcut : gamma ≤ max best (f m)
    · rw [if_pos hcut]
      have hrest := foldMax_ge_init w ms (max acc (w m))
      constructor
      · intro _
        by_cases hf : gamma ≤ f m
        · have := hm1 hf
          by_cases hb : gamma ≤ best
          · have := h1 hb; omega
          · omega
        · have hb : gamma ≤ best := by omega
          have := h1 hb
          omega
      · intro hlt; omega
    · rw [if_neg hcut]
      have hf : f m < gamma := by omega
      have hb : best < gamma := by omega
      have hwm := hm2 hf
      have hacc := h2 hb
      exact ih (fun x hx => hchild x (by simp [hx])) (max best (f m)) (max acc (w m))
        (fun hge => absurd hge hcut)
        (fun _ => by omega)

/-- **Futility-pruned search satisfies `BoundSpec` -- proven, with no
`FutilityOK` hypothesis.**  Single value function: the
futility yield provably one-side-bounds the same `negamax` the full
search targets.  Only `FutilityMateOK` (line 371's king-capture bypass)
and the in-band window (which makes the `MATE_UPPER` yield always a
fail-high; cf. `Sunfish/Stalemate.lean`) remain as hypotheses. -/
theorem boundFut_spec (G : ValGame) (hFM : FutilityMateOK G.toFutGame) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      BoundSpec G.toFutGame.toGame d p gamma (boundFut G.toFutGame d p gamma) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro d
  induction d with
  | zero =>
    intro p gamma _ _
    refine ⟨fun _ => ?_, fun _ => ?_⟩ <;> (simp only [boundFut, negamax]; omega)
  | succ d ih =>
    intro p gamma hg1 hg2
    have hw1 : -MATE_UPPER < 1 - gamma := by omega
    have hw2 : 1 - gamma ≤ MATE_UPPER := by omega
    cases d with
    | zero =>
      -- Depth 1: the futility zone.  Children are depth-0 (= stand-pat)
      -- searches, so the score identity makes every futile yield exact.
      have hchild : ∀ m ∈ G.toFutGame.toGame.moves p,
          (gamma ≤ (if G.toFutGame.toGame.eval p + G.toFutGame.val p m < gamma then
              (if G.toFutGame.val p m < MATE_LOWER then
                G.toFutGame.toGame.eval p + G.toFutGame.val p m
              else MATE_UPPER)
            else -(G.toFutGame.toGame.eval m)) →
            (if G.toFutGame.toGame.eval p + G.toFutGame.val p m < gamma then
              (if G.toFutGame.val p m < MATE_LOWER then
                G.toFutGame.toGame.eval p + G.toFutGame.val p m
              else MATE_UPPER)
            else -(G.toFutGame.toGame.eval m)) ≤ -(G.toFutGame.toGame.eval m)) ∧
          ((if G.toFutGame.toGame.eval p + G.toFutGame.val p m < gamma then
              (if G.toFutGame.val p m < MATE_LOWER then
                G.toFutGame.toGame.eval p + G.toFutGame.val p m
              else MATE_UPPER)
            else -(G.toFutGame.toGame.eval m)) < gamma →
            -(G.toFutGame.toGame.eval m) ≤
            (if G.toFutGame.toGame.eval p + G.toFutGame.val p m < gamma then
              (if G.toFutGame.val p m < MATE_LOWER then
                G.toFutGame.toGame.eval p + G.toFutGame.val p m
              else MATE_UPPER)
            else -(G.toFutGame.toGame.eval m))) := by
        intro m hm
        have hid := G.score_identity p m hm
        by_cases hfut : G.toFutGame.toGame.eval p + G.toFutGame.val p m < gamma
        · rw [if_pos hfut]
          by_cases hq : G.toFutGame.val p m < MATE_LOWER
          · -- Futile quiet move: the yield is exactly -(eval child).
            rw [if_pos hq]
            constructor
            · intro hge; omega
            · intro _; omega
          · -- Futile king capture: the exact-sentinel bypass (line 371).
            rw [if_neg hq]
            have hmate := hFM 0 p m hm (by omega)
            have hn : negamax G.toFutGame.toGame 0 m = G.toFutGame.toGame.eval m := rfl
            constructor
            · intro _; omega
            · intro hlt; omega
        · -- Non-futile: an exact depth-0 child.
          rw [if_neg hfut]
          exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
      have h := searchMoves_spec_mem gamma
        (fun m =>
          if G.toFutGame.toGame.eval p + G.toFutGame.val p m < gamma then
            (if G.toFutGame.val p m < MATE_LOWER then
              G.toFutGame.toGame.eval p + G.toFutGame.val p m
            else MATE_UPPER)
          else -(G.toFutGame.toGame.eval m))
        (fun m => -(G.toFutGame.toGame.eval m))
        (G.toFutGame.toGame.moves p) hchild LOSS LOSS
        (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
      simp only [BoundSpec, boundFut, negamax]
      exact h
    | succ d' =>
      -- Depth d'+2: the ordinary loop, exactly as in `bound_spec`.
      have hchild : ∀ m : G.Pos,
          (gamma ≤ -(boundFut G.toFutGame (d' + 1) m (1 - gamma)) →
            -(boundFut G.toFutGame (d' + 1) m (1 - gamma))
              ≤ -(negamax G.toFutGame.toGame (d' + 1) m)) ∧
          (-(boundFut G.toFutGame (d' + 1) m (1 - gamma)) < gamma →
            -(negamax G.toFutGame.toGame (d' + 1) m)
              ≤ -(boundFut G.toFutGame (d' + 1) m (1 - gamma))) := by
        intro m
        have h1 := (ih m (1 - gamma) hw1 hw2).1
        have h2 := (ih m (1 - gamma) hw1 hw2).2
        constructor
        · intro hge
          have := h2 (by omega)
          omega
        · intro hlt
          have := h1 (by omega)
          omega
      have h := searchMoves_spec gamma
        (fun m => -(boundFut G.toFutGame (d' + 1) m (1 - gamma)))
        (fun m => -(negamax G.toFutGame.toGame (d' + 1) m))
        hchild (G.toFutGame.toGame.moves p) LOSS LOSS
        (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
      simp only [BoundSpec, boundFut, negamax]
      simpa only [negamax] using h

/-! ### (f) Depth-independent mate entries  (STATED + one proven lemma)

An experimental sunfish variant stores mate results under a depth
sentinel (e.g. depth = 1000) and serves them at ANY queried depth,
including shallower ones.  What would justify that?

* Serving at DEEPER depths needs exactly `MateDepthMonotone`: a mate-band
  lower bound survives one more ply.  `mateEntry_deep_service` (proven,
  sorry-free) lifts it to arbitrary deeper depths.

* Serving at SHALLOWER depths is NOT justified by monotonicity: the
  depth-indexed `BoundSpec` is violated outright -- the stored number is a
  fact about `negamax d p`, the query is answered about `negamax d' p`
  with `d' < d`, and no lemma connects them downward.  The violation is
  not hypothetical: `negamaxDraw_depth_inconsistent`
  (`Sunfish/Stalemate.lean`) exhibits a position whose value flips between
  `LOSS` and `0` purely with depth.

* Shallow service can still be *chess-harmless* if a mate-band value
  reflects a permanent fact -- in a king-capture engine, "the king is
  captured regardless of horizon".  `KingGoneStable` is the eval-level
  seed (a shown king capture never evaporates with more search, on either
  side of the sign alternation), and together with the mates-are-real
  hypothesis (the `Game`-level analogue of `MateValuesAreKingCaptures`)
  it yields `MateDepthStable`, mate-band membership at every depth ≥ 1.
  Depth 0 must stay excluded: a bare `eval` cannot see a hanging king.

Even under `MateDepthStable`, such a variant should document itself as
WEAKENING `BoundSpec`: what survives is only which side of the mate band
the score is on, not the depth-indexed numeric bracket that `TableOK` and
the MTD-bi driver are stated against. -/

/-- A mate-band lower bound survives one extra ply of search. -/
def MateDepthMonotone (G : Game) (ML : Int) : Prop :=
  ∀ (d : Nat) (p : G.Pos), ML ≤ negamax G d p → ML ≤ negamax G (d + 1) p

/-- PROVEN: `MateDepthMonotone` is exactly what serving a mate entry at
any *deeper* depth requires. -/
theorem mateEntry_deep_service (G : Game) (ML : Int)
    (h : MateDepthMonotone G ML) :
    ∀ (d d' : Nat) (p : G.Pos), d ≤ d' →
      ML ≤ negamax G d p → ML ≤ negamax G d' p := by
  intro d d' p
  induction d' with
  | zero =>
    intro hdd hm
    have hz : d = 0 := by omega
    subst hz
    exact hm
  | succ d' ih =>
    intro hdd hm
    by_cases hlt : d ≤ d'
    · exact h d' p (ih hlt hm)
    · have he : d = d' + 1 := by omega
      subst he
      exact hm

/-- Mate-band membership is depth-independent (away from depth 0, where a
bare `eval` cannot see a hanging king).  This -- not `MateDepthMonotone` --
is what SHALLOW service of sentinel mate entries assumes. -/
def MateDepthStable (G : Game) (ML : Int) : Prop :=
  ∀ (d d' : Nat) (p : G.Pos), 1 ≤ d → 1 ≤ d' →
    ML ≤ negamax G d p → ML ≤ negamax G d' p

/-- **KingGoneStable**: a captured king is permanent, on both sides of the
sign alternation.  If the static score already shows the opponent king
gone, every horizon confirms the win; if it shows our king gone, every
horizon confirms the loss. -/
def KingGoneStable (G : Game) (ML : Int) : Prop :=
  (∀ p, ML ≤ G.eval p → ∀ (d : Nat), ML ≤ negamax G d p) ∧
  (∀ p, G.eval p ≤ -ML → ∀ (d : Nat), negamax G d p ≤ -ML)

/-- **Proven under its named hypotheses**: permanence of king capture
plus "mate scores only come from real king captures" (the `Game`-level
analogue of `MateValuesAreKingCaptures`) give depth-stability of the
mate band.  The proof runs exactly on the `foldMax` member lemma: a
depth-`d+1` mate finds a capture move `m` by `hOnly`, and `hK.2` pins
`m` below `-ML` at EVERY depth, so the parent stays above `ML` at every
depth ≥ 1. -/
theorem mateDepthStable_of_kingGoneStable (G : Game) (ML : Int)
    (hK : KingGoneStable G ML)
    (hOnly : ∀ (d : Nat) (p : G.Pos), ML ≤ negamax G (d + 1) p →
      ∃ m ∈ G.moves p, G.eval m ≤ -ML) :
    MateDepthStable G ML := by
  intro d d' p hd hd' hm
  cases d with
  | zero => exact absurd hd (by omega)
  | succ d0 =>
    cases hOnly d0 p hm with
    | intro m hmm =>
      cases d' with
      | zero => exact absurd hd' (by omega)
      | succ d1 =>
        have hv := hK.2 m hmm.2 d1
        have hfold : -(negamax G d1 m)
            ≤ foldMax (fun x => -(negamax G d1 x)) (G.moves p) LOSS :=
          foldMax_le_of_mem _ _ _ m hmm.1
        have hneq : negamax G (d1 + 1) p
            = foldMax (fun x => -(negamax G d1 x)) (G.moves p) LOSS := rfl
        rw [hneq]
        omega

end Sunfish
