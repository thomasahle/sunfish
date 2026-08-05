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

/-! ### (e) The futility yield  (STATED)

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
the futility depths, so a pruned move only ever under-promises. -/
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

/-- Futility-pruned search is fail-soft correct ONLY under `FutilityOK`
(fail-low estimates) and `FutilityMateOK` (the fail-high `MATE_UPPER`
yield); the in-band window makes the `MATE_UPPER` yield always a fail-high
(cf. the window discussion in `Sunfish/Stalemate.lean`).

Proof sketch (`sorry`d): induction on depth as in `bound_spec`.  At depth
1 the per-move clause required by the loop invariant splits three ways:
a futile quiet move reports `eval p + val p m < gamma` and `FutilityOK`
gives `w m ≤ f m`, exactly the fail-low clause; a futile king capture
reports `MATE_UPPER ≥ gamma` and `FutilityMateOK` gives `f m ≤ w m`, the
fail-high clause; a non-futile move is an ordinary (exact) depth-0 child.
The only new machinery needed is a variant of `searchMoves_spec` whose
per-child hypothesis is restricted to members of the move list, since
`FutilityOK` speaks only about actual moves. -/
theorem boundFut_spec (G : FutGame) (hF : FutilityOK G) (hFM : FutilityMateOK G) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      BoundSpec G.toGame d p gamma (boundFut G d p gamma) := by
  sorry

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

/-- STATED: permanence of king capture plus "mate scores only come from
real king captures" (the `Game`-level analogue of
`MateValuesAreKingCaptures`) give depth-stability of the mate band.
`sorry`d; a proof runs on the `foldMax` member lemmas: a depth-`d+1` mate
finds a capture move `m` by `hOnly`, and `hK.2` pins `m` below `-ML` at
every depth, so the parent stays above `ML` at every depth ≥ 1. -/
theorem mateDepthStable_of_kingGoneStable (G : Game) (ML : Int)
    (hK : KingGoneStable G ML)
    (hOnly : ∀ (d : Nat) (p : G.Pos), ML ≤ negamax G (d + 1) p →
      ∃ m ∈ G.moves p, G.eval m ≤ -ML) :
    MateDepthStable G ML := by
  sorry

end Sunfish
