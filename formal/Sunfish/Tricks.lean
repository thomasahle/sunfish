/-
Precise statements of the search "tricks" that the proven model in
`Sunfish/Bound.lean` deliberately leaves out.  Some are proven (the easy
ones), the rest are `sorry`d on purpose: the point is that each trick has a
*name* and a *hypothesis*, so that a search-changing PR can say which lemma
it preserves, weakens, or newly depends on.
-/

import Sunfish.Bound

namespace Sunfish

/-! ### (a) Mate-score softening  (PROVEN)

Mate scores live strictly beyond the band `[-ML, ML]` (in sunfish:
`MATE_LOWER = 50710`, sunfish.py line 122).  When such a score is passed one
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

sunfish.py lines 330-331:

    if depth > 2 and can_null and abs(pos.score) < 500:
        yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)

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

/-- A `Game` together with a pass ("null") move:
`pass p` = `pos.rotate(nullmove=True)` (sunfish.py line 331). -/
structure NullGame extends Game where
  pass : Pos → Pos

/-- **The null-move hypothesis**: in every non-terminal position, some legal
move scores at least as well (for us) as passing, i.e. the opponent's value
after some real move is at most their value after the pass.  Zugzwang is
precisely the failure of this Prop. -/
def NullOK (G : NullGame) : Prop :=
  ∀ (d : Nat) (p : G.Pos), G.moves p ≠ [] →
    ∃ m, m ∈ G.moves p ∧
      negamax G.toGame d m ≤ negamax G.toGame d (G.pass p)

/-- `bound` with null-move pruning: before the move loop, try passing; if
the pass already fails high (and the position is not terminal -- a terminal
position has no move to back the claim), cut off with the pass's value. -/
def boundNull (G : NullGame) : Nat → G.Pos → Int → Int
  | 0, p, _gamma => G.eval p
  | d + 1, p, gamma =>
    let nullVal := -(boundNull G d (G.pass p) (1 - gamma))
    if (G.moves p).isEmpty = false ∧ gamma ≤ nullVal then
      nullVal
    else
      searchMoves gamma (fun m => -(boundNull G d m (1 - gamma))) (G.moves p) LOSS

/-- Null-move-pruned search is fail-soft correct ONLY under `NullOK`.
Proof sketch: as `bound_spec`, plus: when the null cutoff fires, the spec of
the recursive call gives `gamma ≤ nullVal ≤ -(negamax d (pass p))`, and
`NullOK` supplies a real move `m` with
`-(negamax d (pass p)) ≤ -(negamax d m) ≤ negamax (d+1) p`. -/
theorem boundNull_spec (G : NullGame) (hnull : NullOK G) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      BoundSpec G.toGame d p gamma (boundNull G d p gamma) := by
  sorry

/-! ### (c) The transposition-table invariant  (STATED)

sunfish.py lines 275-276, 305-310, 414-420.  An `Entry(lower, upper)` stored
under key `(pos, depth, can_null)` promises `lower ≤ s* ≤ upper` for the
negamax value `s*` of that position at that depth; the comment at line 275
(`# lower <= s(pos) <= upper`) *is* `TableOK`.  Lookups may then answer a
query without searching (lines 309-310), and every exit of `bound` tightens
one side of the entry (lines 415-418).

`Bounded` is needed because a fresh entry starts as
`Entry(-MATE_UPPER, MATE_UPPER)` (line 308): that is only a valid bracket if
all scores actually live in `[-MATE_UPPER, MATE_UPPER]`, which sunfish
guarantees by construction of the evaluation.
-/

/-- A transposition table, abstracted as a partial map from (depth,
position) to an `Entry (lower, upper)` pair.  (Sunfish additionally keys on
`can_null` and clamps QS depths to 0, line 296 -- refinements of the same
invariant.) -/
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

/-- All static evaluations live in sunfish's score band. -/
def Bounded (G : Game) : Prop :=
  ∀ p, -MATE_UPPER ≤ G.eval p ∧ G.eval p ≤ MATE_UPPER

/-- The move loop of `boundTT`, threading the table through the child
searches (state-passing version of `searchMoves`). -/
def searchMovesTT {G : Game} (gamma : Int)
    (f : G.Pos → Table G → Int × Table G) :
    List G.Pos → Int → Table G → Int × Table G
  | [], best, t => (best, t)
  | m :: ms, best, t =>
    let r := f m t
    if gamma ≤ max best r.1 then (max best r.1, r.2)
    else searchMovesTT gamma f ms (max best r.1) r.2

/-- `bound` with a transposition table: lookup before searching
(sunfish.py lines 308-310), store the tightened bound on exit (415-418). -/
def boundTT (G : Game) [DecidableEq G.Pos] :
    Nat → G.Pos → Int → Table G → Int × Table G
  | 0, p, _gamma, t => (G.eval p, t)
  | d + 1, p, gamma, t =>
    -- entry = tp_score.get(..., Entry(-MATE_UPPER, MATE_UPPER))  (line 308)
    let e := (t.find (d + 1) p).getD (LOSS, MATE_UPPER)
    if gamma ≤ e.1 then (e.1, t)          -- if entry.lower >= gamma  (line 309)
    else if e.2 < gamma then (e.2, t)     -- if entry.upper < gamma   (line 310)
    else
      let r := searchMovesTT gamma
        (fun m t' =>
          let s := boundTT G d m (1 - gamma) t'
          (-s.1, s.2))
        (G.moves p) LOSS t
      -- Table part 2 (lines 414-418): tighten one side, keep the other.
      let t' := if gamma ≤ r.1 then Table.store r.2 (d + 1) p (r.1, e.2)
                else Table.store r.2 (d + 1) p (e.1, r.1)
      (r.1, t')

/-- A table-using `bound` both answers correctly and *preserves* `TableOK`.
The preservation half is the real content: it is what makes it sound for
later queries (with different `gamma`!) to trust lines 309-310. -/
theorem boundTT_spec (G : Game) [DecidableEq G.Pos] (hb : Bounded G)
    (d : Nat) (p : G.Pos) (gamma : Int) (t : Table G) (ht : TableOK G t) :
    BoundSpec G d p gamma (boundTT G d p gamma t).1 ∧
      TableOK G (boundTT G d p gamma t).2 := by
  sorry

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
region, lines 348-374) re-derives capture information from `pos` itself via
`pos.value(move)`, so its value IS a function of `pos` alone, and
`(pos, depth, can_null)` is an honest key.  `can_null` is the same
phenomenon in miniature: the null-move and repetition checks change the
value, so the flag must be part of the key (line 308).

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

end Sunfish
