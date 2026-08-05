/-
Late Move Reductions (commit 58883ea, sunfish.py lines 370, 386-397):

    for i_m, (val, move) in enumerate(sorted(...)):
        ...
        if depth >= 3 and i_m >= 5 and val < QS:
            score = -self.bound(pos.move(move), 1 - gamma, depth - 2)
            if score < gamma:
                yield move, score
                continue
        yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1)

A reducible (late, quiet, deep-enough) move is probed one ply shallower;
only a reduced FAIL LOW is yielded, a reduced fail high falls through to
the full depth-1 re-search and only that result is yielded.

`bound`'s docstring is now unprovable as stated: the reduced fail-low
yields a depth-(d-2) fact as a depth-(d-1) claim, so no single value
function can back both sides of `BoundSpec`.  The honest replacement is an
INTERVAL spec, and modeling the merged code produced two surprises that
shape it:

SURPRISE 1 (the naive fail-low target is wrong).  "Value reducible moves
one ply shallower" (`negamaxShallow` below) is NOT a sound upper-bound
target for fail lows, because of the re-search fall-through: when the
reduced probe fails high but the full-depth re-search comes back low, the
yielded value bounds the DEEP child value, while the shallow child value
-- which is what `negamaxShallow` scores, and which just failed high --
can exceed everything yielded.  The sound per-move fail-low entry for a
reducible move is the MINIMUM of its shallow and deep values (the
opponent gets the better of the two readings, whichever branch actually
yielded).

SURPRISE 2 (fail highs are not sound against full negamax either).  The
re-search guard does make the *immediately* reduced move safe -- a reduced
fail high is never yielded, which is why the fail-high side needs no
`min`.  But a fail high at the parent is fed by a fail LOW of the child,
and the child's fail low is only sound against the child's *reduced*
value; the contamination is recursive and cannot be scrubbed at one
level.  So the two sides need a mutually recursive pair of value
functions,

    Vhi d p = max over moves of -(Vlo (d-1) m)          (all full depth)
    Vlo d p = max over moves of
                if reducible then min (-(Vhi (d-2) m)) (-(Vhi (d-1) m))
                else -(Vhi (d-1) m)

with `Vlo ≤ negamax ≤ Vhi` pointwise (`lmrVal_sandwich`): LMR weakens the
point spec "both sides bracket `negamax d p`" to the interval spec
"fail highs are ≤ `Vhi`, fail lows are ≥ `Vlo`, and `negamax` lies in
between".  Neither side individually brackets `negamax` any more --
`lmr_tt_crossing` below machine-checks a position where a fail-high
report strictly exceeds a fail-low report at the same `(pos, depth)`,
which `bound_no_crossing` shows is impossible for the unreduced search.

Consequences for the transposition table: `tp_score` entries now mix
lower bounds on `Vhi` with upper bounds on `Vlo`, so crossing REPORTS
(`fail-high value > fail-low value` at the same key) are possible exactly
when `Vlo < Vhi` there.  sunfish's MTD-bi consistency reasoning
(sunfish.py lines 449-461: `lower <= score <= upper`) becomes conditional
on the `Vhi - Vlo` gap staying within what `EVAL_ROUGHNESS` and the
re-probing loop absorb: the binary search still terminates (each probe
still tightens the side it reported), but the bracket it maintains is
around the interval, not around a single true value.  Since commit
2c95ab0 the store CLAMPS (widening the stale side), so a crossing can no
longer be *stored* -- `Sunfish/TableClamp.lean` proves the clamp keeps
every entry an honest interval claim (`IntervalTableOK`) and non-crossing
by construction (`clamp_no_crossing`).

KillerIsKingCapture and the stalemate correction are UNAFFECTED:

* King captures have `pos.value ≥ MATE_LOWER > QS = 40`, so `val < QS`
  never holds for them: they are never reducible (`RedRespectsCaptures`
  names this, true by construction in sunfish).  The killer is yielded
  before the sorted loop (line 367) and thus structurally never reduced.
  A reduced fail low is `< gamma`, so it can never trigger the fail-high
  cutoff that stores into `tp_move`; a reduced fail high stores only
  after (and with the move of) the full-depth re-search.  Every leg of
  `boundKill_spec`'s induction is therefore untouched.
* The stalemate correction rests on the king-capture normalization
  returning the exact `-MATE_UPPER` sentinel, which is depth-INDEPENDENT
  (`boundKill_kingGone`, `negamaxDraw_kingGone` hold at every depth), so
  a reduced search still answers the sentinel for a king-loss child and
  `best == -MATE_UPPER` detection still works.
-/

import Sunfish.Bound

namespace Sunfish

/-! ### Indexed folds and the indexed move loop -/

/-- `foldMax` with the move index threaded through, so that value
functions can depend on the sort position (`i_m` of sunfish.py line 370). -/
def foldMaxIdx {α : Type _} (w : Nat → α → Int) : Nat → List α → Int → Int
  | _, [], acc => acc
  | i, m :: ms, acc => foldMaxIdx w (i + 1) ms (max acc (w i m))

theorem foldMaxIdx_ge_init {α : Type _} (w : Nat → α → Int) :
    ∀ (ms : List α) (i : Nat) (acc : Int), acc ≤ foldMaxIdx w i ms acc := by
  intro ms
  induction ms with
  | nil => intro _ acc; exact Int.le_refl acc
  | cons m ms ih =>
    intro i acc
    have h1 : acc ≤ max acc (w i m) := by omega
    exact Int.le_trans h1 (ih (i + 1) (max acc (w i m)))

theorem foldMaxIdx_mono {α : Type _} (w w' : Nat → α → Int)
    (h : ∀ (i : Nat) (m : α), w i m ≤ w' i m) :
    ∀ (ms : List α) (i : Nat) (acc acc' : Int), acc ≤ acc' →
      foldMaxIdx w i ms acc ≤ foldMaxIdx w' i ms acc' := by
  intro ms
  induction ms with
  | nil => intro _ acc acc' hacc; exact hacc
  | cons m _ ih =>
    intro i acc acc' hacc
    refine ih (i + 1) _ _ ?_
    have := h i m
    omega

theorem foldMax_eq_foldMaxIdx {α : Type _} (w : α → Int) :
    ∀ (ms : List α) (i : Nat) (acc : Int),
      foldMax w ms acc = foldMaxIdx (fun _ m => w m) i ms acc := by
  intro ms
  induction ms with
  | nil => intro _ acc; rfl
  | cons m ms ih => intro i acc; simp only [foldMax, foldMaxIdx]; exact ih (i + 1) _

/-- The fail-soft cutoff loop with the move index (enumerate of line 370). -/
def searchMovesIdx {α : Type _} (gamma : Int) (score : Nat → α → Int) :
    Nat → List α → Int → Int
  | _, [], best => best
  | i, m :: ms, best =>
    if gamma ≤ max best (score i m) then max best (score i m)
    else searchMovesIdx gamma score (i + 1) ms (max best (score i m))

/-- The two-target loop invariant: each move's report is bounded above by
`whi` when it fails high and below by `wlo` when it fails low; then so is
the loop's fail-soft result, against the respective indexed folds.  This
generalizes `searchMoves_spec`, whose two targets coincide. -/
theorem searchMovesIdx_spec {α : Type _} (gamma : Int) (f whi wlo : Nat → α → Int)
    (hchild : ∀ (i : Nat) (m : α),
      (gamma ≤ f i m → f i m ≤ whi i m) ∧ (f i m < gamma → wlo i m ≤ f i m)) :
    ∀ (ms : List α) (i : Nat) (best accHi accLo : Int),
      (gamma ≤ best → best ≤ accHi) →
      (best < gamma → accLo ≤ best) →
      (gamma ≤ searchMovesIdx gamma f i ms best →
        searchMovesIdx gamma f i ms best ≤ foldMaxIdx whi i ms accHi) ∧
      (searchMovesIdx gamma f i ms best < gamma →
        foldMaxIdx wlo i ms accLo ≤ searchMovesIdx gamma f i ms best) := by
  intro ms
  induction ms with
  | nil =>
    intro i best accHi accLo h1 h2
    simp only [searchMovesIdx, foldMaxIdx]
    exact ⟨h1, h2⟩
  | cons m ms ih =>
    intro i best accHi accLo h1 h2
    have hm1 := (hchild i m).1
    have hm2 := (hchild i m).2
    simp only [searchMovesIdx, foldMaxIdx]
    by_cases hcut : gamma ≤ max best (f i m)
    · rw [if_pos hcut]
      have hrest := foldMaxIdx_ge_init whi ms (i + 1) (max accHi (whi i m))
      constructor
      · intro _
        by_cases hf : gamma ≤ f i m
        · have := hm1 hf
          by_cases hb : gamma ≤ best
          · have := h1 hb; omega
          · omega
        · have hb : gamma ≤ best := by omega
          have := h1 hb
          omega
      · intro hlt; omega
    · rw [if_neg hcut]
      have hf : f i m < gamma := by omega
      have hb : best < gamma := by omega
      have hwm := hm2 hf
      have hacc := h2 hb
      exact ih (i + 1) (max best (f i m)) (max accHi (whi i m)) (max accLo (wlo i m))
        (fun hge => absurd hge hcut)
        (fun _ => by omega)

/-! ### The LMR search and the interval value pair -/

/-- `red d i m`: is the `i`-th sorted move `m` reducible at remaining
depth `d`?  Abstracts `depth >= 3 and i_m >= 5 and val < QS`
(sunfish.py line 392); the model permits reduction from depth 2 up (a
reduced child needs a legal depth `d - 2 ≥ 0`).  Note king captures have
`val ≥ MATE_LOWER > QS`, so sunfish's instance satisfies
`RedRespectsCaptures` below by construction. -/
def boundLmr (G : Game) (red : Nat → Nat → G.Pos → Bool) :
    Nat → G.Pos → Int → Int
  | 0, p, _gamma => G.eval p
  | 1, p, gamma => searchMoves gamma (fun m => -(G.eval m)) (G.moves p) LOSS
  | d + 2, p, gamma =>
    searchMovesIdx gamma
      (fun i m =>
        -- lines 392-396: probe reducible moves at depth-2; yield only a
        -- reduced fail low ...
        if red (d + 2) i m = true ∧ -(boundLmr G red d m (1 - gamma)) < gamma then
          -(boundLmr G red d m (1 - gamma))
        -- ... line 397: otherwise (or on a reduced fail high) yield the
        -- full depth-1 search.
        else -(boundLmr G red (d + 1) m (1 - gamma)))
      0 (G.moves p) LOSS

/-- The interval value pair.  `lmrVal G red d true p` (= `Vhi`) is what
fail-high reports are sound against; `lmrVal G red d false p` (= `Vlo`)
is what fail-low reports are sound against.  Mutually recursive with side
alternation: a fail high is certified by a child fail low and vice versa.
The `min` in the `Vlo` entry for a reducible move is Surprise 1; the
`Vlo` (not `negamax`) inside `Vhi`'s children is Surprise 2. -/
def lmrVal (G : Game) (red : Nat → Nat → G.Pos → Bool) :
    Nat → Bool → G.Pos → Int
  | 0, _, p => G.eval p
  | 1, _, p => foldMax (fun m => -(G.eval m)) (G.moves p) LOSS
  | d + 2, true, p =>
    foldMaxIdx (fun _ m => -(lmrVal G red (d + 1) false m)) 0 (G.moves p) LOSS
  | d + 2, false, p =>
    foldMaxIdx (fun i m =>
      if red (d + 2) i m = true then
        min (-(lmrVal G red d true m)) (-(lmrVal G red (d + 1) true m))
      else -(lmrVal G red (d + 1) true m)) 0 (G.moves p) LOSS

/-- The naive fail-low target from the LMR folklore -- reducible moves
simply valued one ply shallower.  Kept for the record: it is NOT what
fail lows are sound against (Surprise 1; see the module comment).  The
`min`-entry `Vlo` above is the honest version. -/
def negamaxShallow (G : Game) (red : Nat → Nat → G.Pos → Bool) :
    Nat → G.Pos → Int
  | 0, p => G.eval p
  | 1, p => foldMax (fun m => -(G.eval m)) (G.moves p) LOSS
  | d + 2, p =>
    foldMaxIdx (fun i m =>
      if red (d + 2) i m = true then -(negamax G d m)
      else -(negamax G (d + 1) m)) 0 (G.moves p) LOSS

/-- **The LMR interval spec, proven.**  For every depth, position and
window: a fail-high report is a sound lower-bound claim against `Vhi`,
and a fail-low report is a sound upper-bound claim against `Vlo`.
Together with `lmrVal_sandwich` (`Vlo ≤ negamax ≤ Vhi`) this is the
precise sense in which LMR weakens `BoundSpec` from a point to an
interval. -/
theorem boundLmr_spec (G : Game) (red : Nat → Nat → G.Pos → Bool) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      (gamma ≤ boundLmr G red d p gamma →
        boundLmr G red d p gamma ≤ lmrVal G red d true p) ∧
      (boundLmr G red d p gamma < gamma →
        lmrVal G red d false p ≤ boundLmr G red d p gamma) := by
  suffices H : ∀ (n d : Nat), d ≤ n → ∀ (p : G.Pos) (gamma : Int),
      (gamma ≤ boundLmr G red d p gamma →
        boundLmr G red d p gamma ≤ lmrVal G red d true p) ∧
      (boundLmr G red d p gamma < gamma →
        lmrVal G red d false p ≤ boundLmr G red d p gamma) by
    exact fun d => H d d (Nat.le_refl d)
  intro n
  induction n with
  | zero =>
    intro d hd p gamma
    have hd0 : d = 0 := by omega
    subst hd0
    refine ⟨fun _ => ?_, fun _ => ?_⟩ <;> (simp only [boundLmr, lmrVal]; omega)
  | succ n ihn =>
    intro d hd p gamma
    cases d with
    | zero =>
      refine ⟨fun _ => ?_, fun _ => ?_⟩ <;> (simp only [boundLmr, lmrVal]; omega)
    | succ d' =>
      cases d' with
      | zero =>
        -- Depth 1: children are exact evals, both targets coincide.
        have h := searchMoves_spec gamma (fun m => -(G.eval m)) (fun m => -(G.eval m))
          (fun m => ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩)
          (G.moves p) LOSS LOSS
          (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
        simp only [boundLmr, lmrVal]
        exact h
      | succ d'' =>
        -- Depth d''+2: the per-move clause splits on the LMR branch taken.
        have hchild : ∀ (i : Nat) (m : G.Pos),
            (gamma ≤ (if red (d'' + 2) i m = true ∧
                  -(boundLmr G red d'' m (1 - gamma)) < gamma then
                -(boundLmr G red d'' m (1 - gamma))
              else -(boundLmr G red (d'' + 1) m (1 - gamma))) →
              (if red (d'' + 2) i m = true ∧
                  -(boundLmr G red d'' m (1 - gamma)) < gamma then
                -(boundLmr G red d'' m (1 - gamma))
              else -(boundLmr G red (d'' + 1) m (1 - gamma)))
                ≤ -(lmrVal G red (d'' + 1) false m)) ∧
            ((if red (d'' + 2) i m = true ∧
                  -(boundLmr G red d'' m (1 - gamma)) < gamma then
                -(boundLmr G red d'' m (1 - gamma))
              else -(boundLmr G red (d'' + 1) m (1 - gamma))) < gamma →
              (if red (d'' + 2) i m = true then
                min (-(lmrVal G red d'' true m)) (-(lmrVal G red (d'' + 1) true m))
              else -(lmrVal G red (d'' + 1) true m))
                ≤ (if red (d'' + 2) i m = true ∧
                    -(boundLmr G red d'' m (1 - gamma)) < gamma then
                  -(boundLmr G red d'' m (1 - gamma))
                else -(boundLmr G red (d'' + 1) m (1 - gamma)))) := by
          intro i m
          have ih1a := (ihn (d'' + 1) (by omega) m (1 - gamma)).1
          have ih1b := (ihn (d'' + 1) (by omega) m (1 - gamma)).2
          have ih0a := (ihn d'' (by omega) m (1 - gamma)).1
          by_cases hred : red (d'' + 2) i m = true ∧
              -(boundLmr G red d'' m (1 - gamma)) < gamma
          · -- Reduced fail low was yielded.
            rw [if_pos hred, if_pos hred.1]
            constructor
            · intro hge; omega
            · intro _
              -- The reduced probe failed high FOR THE CHILD, so the
              -- child's Vhi at the reduced depth bounds it from above,
              -- and the min-entry is below that.
              have hfh : 1 - gamma ≤ boundLmr G red d'' m (1 - gamma) := by omega
              have := ih0a hfh
              omega
          · -- Full-depth yield (non-reducible, or reduced fail high fell
            -- through to the re-search: the guard in action).
            rw [if_neg hred]
            by_cases hr : red (d'' + 2) i m = true
            · rw [if_pos hr]
              constructor
              · intro hge
                have hfl : boundLmr G red (d'' + 1) m (1 - gamma) < 1 - gamma := by omega
                have := ih1b hfl
                omega
              · intro hlt
                have hfh : 1 - gamma ≤ boundLmr G red (d'' + 1) m (1 - gamma) := by omega
                have := ih1a hfh
                omega
            · rw [if_neg hr]
              constructor
              · intro hge
                have hfl : boundLmr G red (d'' + 1) m (1 - gamma) < 1 - gamma := by omega
                have := ih1b hfl
                omega
              · intro hlt
                have hfh : 1 - gamma ≤ boundLmr G red (d'' + 1) m (1 - gamma) := by omega
                have := ih1a hfh
                omega
        have h := searchMovesIdx_spec gamma
          (fun i m =>
            if red (d'' + 2) i m = true ∧ -(boundLmr G red d'' m (1 - gamma)) < gamma then
              -(boundLmr G red d'' m (1 - gamma))
            else -(boundLmr G red (d'' + 1) m (1 - gamma)))
          (fun _ m => -(lmrVal G red (d'' + 1) false m))
          (fun i m =>
            if red (d'' + 2) i m = true then
              min (-(lmrVal G red d'' true m)) (-(lmrVal G red (d'' + 1) true m))
            else -(lmrVal G red (d'' + 1) true m))
          hchild (G.moves p) 0 LOSS LOSS LOSS
          (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
        simp only [boundLmr, lmrVal]
        exact h

/-- **The sandwich, proven**: the interval really is an interval around
the full-depth value, `Vlo ≤ negamax ≤ Vhi` pointwise.  So LMR reports
still confine `negamax d p` -- but only from the side that happened to be
reported, and against the widened pair, not the point. -/
theorem lmrVal_sandwich (G : Game) (red : Nat → Nat → G.Pos → Bool) :
    ∀ (d : Nat) (p : G.Pos),
      lmrVal G red d false p ≤ negamax G d p ∧
      negamax G d p ≤ lmrVal G red d true p := by
  suffices H : ∀ (n d : Nat), d ≤ n → ∀ (p : G.Pos),
      lmrVal G red d false p ≤ negamax G d p ∧
      negamax G d p ≤ lmrVal G red d true p by
    exact fun d => H d d (Nat.le_refl d)
  intro n
  induction n with
  | zero =>
    intro d hd p
    have hd0 : d = 0 := by omega
    subst hd0
    simp only [lmrVal, negamax]
    omega
  | succ n ihn =>
    intro d hd p
    cases d with
    | zero =>
      simp only [lmrVal, negamax]
      omega
    | succ d' =>
      cases d' with
      | zero =>
        simp only [lmrVal, negamax]
        exact ⟨Int.le_refl _, Int.le_refl _⟩
      | succ d'' =>
        have hnege : negamax G (d'' + 1 + 1) p
            = foldMaxIdx (fun _ m => -(negamax G (d'' + 1) m)) 0 (G.moves p) LOSS := by
          rw [← foldMax_eq_foldMaxIdx (fun m => -(negamax G (d'' + 1) m)) (G.moves p) 0 LOSS]
          rfl
        constructor
        · -- Vlo ≤ negamax: every Vlo entry is ≤ the full-depth entry.
          simp only [lmrVal]
          rw [hnege]
          refine foldMaxIdx_mono _ _ (fun i m => ?_) (G.moves p) 0 LOSS LOSS (Int.le_refl _)
          have hhi := (ihn (d'' + 1) (by omega) m).2
          by_cases hr : red (d'' + 1 + 1) i m = true
          · rw [if_pos hr]; omega
          · rw [if_neg hr]; omega
        · -- negamax ≤ Vhi: every full-depth entry is ≤ the Vhi entry.
          simp only [lmrVal]
          rw [hnege]
          refine foldMaxIdx_mono _ _ (fun _ m => ?_) (G.moves p) 0 LOSS LOSS (Int.le_refl _)
          have hlo := (ihn (d'' + 1) (by omega) m).1
          omega

/-! ### The TT-crossing phenomenon (deliverable 3) -/

/-- For the UNREDUCED search, a fail-high report can never exceed a
fail-low report at the same `(pos, depth)`: both bracket `negamax d p`,
so a `tp_score` entry always satisfies `lower ≤ upper`.  This is the
consistency the MTD-bi comments (sunfish.py lines 449-461) rely on. -/
theorem bound_no_crossing (G : Game) (d : Nat) (p : G.Pos)
    {g1 g2 r1 r2 : Int}
    (h1 : bound G d p g1 = r1) (hh : g1 ≤ r1)
    (h2 : bound G d p g2 = r2) (hl : r2 < g2) :
    r1 ≤ r2 := by
  have s1 := (bound_spec G d p g1).1
  have s2 := (bound_spec G d p g2).2
  rw [h1] at s1
  rw [h2] at s2
  exact Int.le_trans (s1 hh) (s2 hl)

/-- A three-position game exhibiting a TT crossing under LMR.  The single
root move `m` looks bad shallow (`eval m = 50`, so the depth-0 probe
scores it `-50` for us) but is good deep (its only reply `g` has
`eval g = 10`, so at depth 1 the child is worth `-10` and the move scores
`+10`).  Probing `gamma = 0` takes the reduced branch and fails low at
`-50`; probing `gamma = -60` bypasses it (the probe fails high),
re-searches at full depth and fails high at `+10`.  Stored in the same
`tp_score` slot, that is `Entry(lower = 10, upper = -50)`:
`lower > upper`, impossible without LMR (`bound_no_crossing`). -/
inductive LPos where
  | X | m | g

def LG : Game where
  Pos := LPos
  moves := fun p => match p with
    | .X => [.m]
    | .m => [.g]
    | .g => []
  eval := fun p => match p with
    | .X => 0
    | .m => 50
    | .g => 10

/-- Everything is reducible: a legitimate instance of the abstract `red`
(in sunfish only late quiet moves qualify; the phenomenon needs just one). -/
def lred : Nat → Nat → LPos → Bool := fun _ _ _ => true

theorem lmr_fail_high : boundLmr LG lred 2 LPos.X (-60) = 10 := by decide

theorem lmr_fail_low : boundLmr LG lred 2 LPos.X 0 = -50 := by decide

/-- **TT crossing, machine-checked**: the same position and depth yields
a fail-high report (a would-be stored `lower`) strictly ABOVE a fail-low
report (a would-be stored `upper`).  Point-`TableOK` reasoning about
`tp_score` is therefore unachievable under LMR; MTD-bi's bracket
`lower ≤ score ≤ upper` becomes conditional on the `Vhi - Vlo` gap
staying within what `EVAL_ROUGHNESS` and re-probing absorb.  (Since
commit 2c95ab0 the store clamp prevents the crossing from being STORED --
see `Sunfish/TableClamp.lean` -- but the crossing reports themselves, as
exhibited here, remain.) -/
theorem lmr_tt_crossing :
    ∃ (glo ghi : Int),
      ghi ≤ boundLmr LG lred 2 LPos.X ghi ∧
      boundLmr LG lred 2 LPos.X glo < glo ∧
      boundLmr LG lred 2 LPos.X glo < boundLmr LG lred 2 LPos.X ghi := by
  refine ⟨0, -60, ?_, ?_, ?_⟩ <;> simp only [lmr_fail_high, lmr_fail_low] <;> omega

/-! ### Killer and stalemate lemmas are preserved (deliverable 4) -/

/-- Reductions never touch king captures.  True by construction in
sunfish: a king capture has `pos.value ≥ MATE_LOWER = 50710`, far above
`QS = 40`, so `val < QS` fails.  Under this condition every leg of
`boundKill_spec` survives LMR verbatim.  The precise killer claim (audit
refinement): the PRE-LOOP killer yield (sunfish.py line 367) is
structurally never reduced -- LMR lives inside the sorted loop -- but the
killer's DUPLICATE in the sorted loop (it is among `gen_moves`) follows
normal LMR rules and CAN be reduced when late-sorted and quiet.  That is
still sound: a reduced fail high falls through to the full-depth search
before any store, a reduced fail low is `< gamma` and never reaches the
`tp_move` store, and the duplicate itself contributes the same move to
the loop's `max` twice, which is harmless (`foldMax_dup` below).
Likewise the stalemate correction: the `-MATE_UPPER` king-loss
normalization is depth-independent (`boundKill_kingGone` /
`negamaxDraw_kingGone` hold at every depth), so reduced searches still
return the sentinel for king-loss children and the `best == -MATE_UPPER`
detection is unchanged. -/
def RedRespectsCaptures (G : Game) (red : Nat → Nat → G.Pos → Bool) : Prop :=
  ∀ (d i : Nat) (m : G.Pos), G.eval m ≤ -MATE_LOWER → red d i m = false

/-- A duplicated move contributes to the fold's max twice, which changes
nothing: the killer's re-appearance in the sorted loop is value-neutral. -/
theorem foldMax_dup {α : Type _} (w : α → Int) (m : α) (ms : List α) (acc : Int) :
    foldMax w (m :: m :: ms) acc = foldMax w (m :: ms) acc := by
  simp only [foldMax]
  congr 1
  omega

/-- **FutilityLmrDisjoint**: futility fires only at `depth ≤ 1`
(sunfish.py line 375), LMR only at `depth ≥ 3` (line 392) -- they never
co-occur at a node.  This is the unstated reason `Sunfish/Tricks.lean`'s
futility model and this file's LMR model may be studied separately and
composed freely: no node runs both.  A future PR overlapping the depth
ranges would be composing something unstudied and should say so. -/
theorem futilityLmrDisjoint (d : Nat) : ¬ (d ≤ 1 ∧ 3 ≤ d) := by omega

end Sunfish
