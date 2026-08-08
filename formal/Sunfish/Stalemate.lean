/-
The stalemate correction, sunfish.py lines 388-412.

sunfish is a king-capture engine: it does not generate "legal" moves, it
generates pseudo-legal moves and scores an actual king capture as an
immediate `-MATE_UPPER` for the side that just lost the king (lines
298-303).  Consequently "no legal moves" (mate OR stalemate) surfaces in
the search as `best == -MATE_UPPER` after the move loop, and the long
comment at lines 388-407 explains the fix: since a stalemate is a draw,
not a loss, a `best == -MATE_UPPER` at `depth > 2` is post-processed by an
in-check probe

    in_check = self.bound(pos.rotate(nullmove=True), MATE_UPPER, 0) == MATE_UPPER
    best = -MATE_LOWER if in_check else 0            -- lines 408-412

This file models that block and proves its contract.  Modeling it honestly
turned up three things that are easy to miss when reading the Python:

1.  **The correction is only sound for in-band windows.**  `bound`'s
    docstring quantifies over every `gamma`, but the corrected search
    satisfies it against the draw-aware value only for
    `-MATE_UPPER < gamma ≤ MATE_UPPER` (an interval that is closed under
    the null-window flip `gamma ↦ 1 - gamma`).  The reason: the argument
    "the loop ended with `best = -MATE_UPPER`, hence there really is no
    king-safe move" requires the loop to have run to completion, i.e. to
    have failed LOW; a `gamma ≤ -MATE_UPPER` would let the loop cut off
    early at `best = -MATE_UPPER` with unsearched moves remaining.
    sunfish respects this implicitly: MTD-bi only probes
    `gamma ∈ (-MATE_LOWER, MATE_LOWER]` (sunfish.py lines 439-447).

2.  **The correction leans on the sentinel invariant** stated at lines
    398-401: `bound` must return *exactly* `MATE_UPPER` whenever the
    opponent's king is capturable, so that a parent in a mated/stalemated
    position sees *exactly* `-MATE_UPPER` from every move.  In sunfish
    this invariant is enforced by move ordering (king captures have the
    highest `pos.value`, are searched first, and score exactly
    `MATE_UPPER` via the child's line 302) plus the explicit
    `else MATE_UPPER` in the futility yield (line 371).  Our model
    enforces it by construction with an explicit king-capture pre-check.
    NOTE: the killer move is yielded *before* the sorted moves (lines
    356-357), so a non-capture killer failing high would break the
    requirement.  `Sunfish/Killer.lean` proves this cannot happen
    (`boundKill_spec`): the position-keyed `tp_move` maintains the
    invariant that its entry at a king-capturable position is itself a
    king capture, so the killer path also reports the exact sentinel.
    The residual exception is the null-move yield, harmless only because
    of its `abs(pos.score) < 500` guard
    (`NullGuardBlocksAtCaptures` in `Sunfish/Killer.lean`).

3.  **Even with the sentinel invariant and in-band windows the faithful,
    depth-gated correction does not satisfy the docstring against the
    draw-aware value unconditionally.**  The gate `depth > 2` makes the
    draw-aware value function itself depth-inconsistent (a stalemate
    scores `-MATE_UPPER` at remaining depth ≤ 2 but `0` at depth ≥ 3), and
    a deep node can inherit a "mate score" that is really such a shallow
    stalemate artifact, with no king capture anywhere to back it.
    `boundStale_not_unconditional` below exhibits a machine-checked
    counterexample; `MateValuesAreKingCaptures` is the named hypothesis
    that outlaws the artifacts, and `boundStale_spec` proves the contract
    under it (sorry-free).  This is exactly sunfish's own caveat at lines
    403-405 ("sunfish may report 'mate', but then after more search
    realize it's not a mate after all"), turned into a hypothesis.

Why `-MATE_LOWER` and not `-MATE_UPPER` for mate (line 412)?  Because
`-MATE_UPPER` is the reserved sentinel meaning "the king is *actually*
capturable/gone".  Both the parent-level detection `best == -MATE_UPPER`
and our `MateValuesAreKingCaptures` hypothesis depend on mate-by-
correction being distinguishable from king-capture; scoring checkmate as
`-MATE_UPPER` would re-inject the sentinel at positions whose king is not
capturable and destroy the detection one ply up (it also orders "mated
next move" above "king already lost", which is what makes sunfish prefer
legal defenses).

THE SECOND HALF of this file (from "The QS val-filter and the exhaustion
gate") upgrades this model with the engine's quiescence filtering
(`val_lower`, `movesAbove`) and the #136 gate: the loop-exhaustion fact
that this first half gets by construction (it searches ALL moves) is
proven there from the gate itself, and the A1 `best_real` fix for the
null-yield/sentinel interaction is modeled ahead of its code.

THE THIRD PART (from "The refuted-assumptions ledger") records the
real-chess refutations that retired the gate-and-sentinel design
entirely, and models its replacement: the verify-on-suspicion search
(`boundD2`), whose correction and null cutoff are certified by a
dedicated legality probe (`legalityProbeCorrect`) instead of any
score-shaped sentinel, with the point spec and table non-crossing
(`boundD2_spec`, `d2_no_crossing`) holding WITHOUT the pseudo-option
hypotheses the old design needed.
-/

import Sunfish.Bound
import Sunfish.EvalBounds

namespace Sunfish

/-! ### The in-check probe -/

/-- One-ply king-capture test: some move of `p` reaches a position whose
static score says the side to move there has lost its king (the test of
sunfish.py line 302, seen from the parent). -/
def hasKingCapture (G : Game) (p : G.Pos) : Bool :=
  (G.moves p).any (fun m => decide (G.eval m ≤ -MATE_LOWER))

theorem hasKingCapture_iff (G : Game) (p : G.Pos) :
    hasKingCapture G p = true ↔ ∃ m ∈ G.moves p, G.eval m ≤ -MATE_LOWER := by
  simp [hasKingCapture, List.any_eq_true]

/-- The in-check probe of sunfish.py line 409-411, as computed by the real
engine: pass the turn (`pos.rotate(nullmove=True)`) and ask whether the
opponent can now capture our king.  In the engine the question is asked as
`bound(flipped, MATE_UPPER, 0) == MATE_UPPER`, a depth-0 QS probe; it
answers "yes" exactly when QS finds a king capture, and QS does search
every king capture (their `pos.value` is ≥ MATE_LOWER, far above the QS
threshold `val_lower`, line 350).  Our model's depth-0 bound is a bare
`eval`, so we model the probe by its *content*: a one-ply king-capture
search over the passed position. -/
def inCheckB (G : NullGame) (p : G.Pos) : Bool :=
  hasKingCapture G.toGame (G.pass p)

/-- Propositional form of the probe: after passing, the opponent has a
move that captures our king. -/
def InCheck (G : NullGame) (p : G.Pos) : Prop :=
  ∃ m ∈ G.moves (G.pass p), G.eval m ≤ -MATE_LOWER

theorem inCheckB_iff (G : NullGame) (p : G.Pos) :
    inCheckB G p = true ↔ InCheck G p :=
  hasKingCapture_iff G.toGame (G.pass p)

/-- **CheckProbeOK**: the search's probe agrees with the one-ply
king-capture notion of check.  For the real engine this is the assumption
that the depth-0 QS null probe of line 411 decides check -- true because
QS never prunes king captures. -/
def CheckProbeOK (G : NullGame) (probe : G.Pos → Bool) : Prop :=
  ∀ p, probe p = inCheckB G p

theorem CheckProbeOK_inCheck {G : NullGame} {probe : G.Pos → Bool}
    (h : CheckProbeOK G probe) (p : G.Pos) : probe p = true ↔ InCheck G p := by
  rw [h p]; exact inCheckB_iff G p

/-! ### The draw-aware value and the corrected search -/

/-- The correction applied to the *true* value: exactly sunfish.py lines
408-412 with the probe replaced by real check.  Note the faithful
`2 < d` gate ("This is too expensive to test at depth == 0", line 407):
because of it the value function is depth-inconsistent near stalemates --
see `negamaxDraw_depth_inconsistent`. -/
def drawFix (G : NullGame) (d : Nat) (best : Int) (p : G.Pos) : Int :=
  if 2 < d ∧ best = LOSS then (if inCheckB G p then -MATE_LOWER else 0) else best

/-- Draw-aware, king-capture-normalized negamax: the value that the
corrected search is trying to bracket.

* `eval p ≤ -MATE_LOWER` (our king is already gone): score `-MATE_UPPER`
  exactly -- sunfish.py lines 298-303.  This normalization is what makes
  `-MATE_UPPER` a reliable sentinel.
* otherwise fold the children as in `negamax`, then apply the stalemate
  correction (depth-gated, as in the engine). -/
def negamaxDraw (G : NullGame) : Nat → G.Pos → Int
  | 0, p => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else drawFix G (d + 1) (foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS) p

/-- The correction as applied to the *search's* `best`, with the engine's
probe: sunfish.py lines 408-412 verbatim (`best = -MATE_LOWER if in_check
else 0`). -/
def staleFix (G : NullGame) (probe : G.Pos → Bool) (d : Nat) (best : Int) (p : G.Pos) : Int :=
  if 2 < d ∧ best = LOSS then (if probe p then -MATE_LOWER else 0) else best

/-- `bound` with the king-capture termination and the stalemate correction.

* line 302-303: our king is gone -> `-MATE_UPPER`.
* the sentinel invariant of lines 398-401, by construction: if we can
  capture the opponent king, return *exactly* `MATE_UPPER` (in sunfish
  this emerges from move ordering; see the module comment, point 2).
* otherwise the fail-soft null-window loop of `bound` (lines 378-388),
  post-processed by the correction (lines 408-412). -/
def boundStale (G : NullGame) (probe : G.Pos → Bool) : Nat → G.Pos → Int → Int
  | 0, p, _gamma => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | d + 1, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toGame p = true then MATE_UPPER
    else staleFix G probe (d + 1)
      (searchMoves gamma (fun m => -(boundStale G probe d m (1 - gamma))) (G.moves p) LOSS) p

/-- `BoundSpec` against the draw-aware value. -/
def BoundSpecD (G : NullGame) (d : Nat) (p : G.Pos) (gamma r : Int) : Prop :=
  (gamma ≤ r → r ≤ negamaxDraw G d p) ∧ (r < gamma → negamaxDraw G d p ≤ r)

/-- **MateValuesAreKingCaptures**: whenever the draw-aware value of a
position (at remaining depth ≥ 1) is the full sentinel `MATE_UPPER`, there
really is a king capture available -- the score is not a shallow-horizon
stalemate artifact.  This is the honest form of sunfish's requirement at
lines 398-405; its failure is exactly the "sunfish may report mate, then
realize it's not a mate after all" caveat, and `boundStale_not_
unconditional` shows the spec genuinely needs it. -/
def MateValuesAreKingCaptures (G : NullGame) : Prop :=
  ∀ (d : Nat) (p : G.Pos), 1 ≤ d → negamaxDraw G d p = MATE_UPPER →
    ∃ m ∈ G.moves p, G.eval m ≤ -MATE_LOWER

/-! ### Helper lemmas -/

/-- A position whose king is gone scores `-MATE_UPPER` at every depth. -/
theorem negamaxDraw_kingGone (G : NullGame) (d : Nat) (p : G.Pos)
    (h : G.eval p ≤ -MATE_LOWER) : negamaxDraw G d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [negamaxDraw]; rw [if_pos h]
  | succ d => simp only [negamaxDraw]; rw [if_pos h]

/-- With a band-bounded evaluation, the draw-aware value stays in the band
(the fact that makes `Entry(-MATE_UPPER, MATE_UPPER)` a valid fresh table
entry, cf. `TableOK`). -/
theorem negamaxDraw_bounded (G : NullGame) (hB : Bounded G.toGame) :
    ∀ (d : Nat) (p : G.Pos),
      -MATE_UPPER ≤ negamaxDraw G d p ∧ negamaxDraw G d p ≤ MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d with
  | zero =>
    intro p
    have hband := hB p
    simp only [negamaxDraw]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]; omega
    · rw [if_neg hkg]; omega
  | succ d ih =>
    intro p
    simp only [negamaxDraw]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]; omega
    · rw [if_neg hkg]
      have hfl := foldMax_ge_init (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS
      have hfu : foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS ≤ MATE_UPPER := by
        refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
        show -(negamaxDraw G d m) ≤ MATE_UPPER
        have := ih m
        omega
      unfold drawFix
      by_cases hgate : 2 < d + 1 ∧
          foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS = LOSS
      · rw [if_pos hgate]
        by_cases hic : inCheckB G p = true
        · rw [if_pos hic]; omega
        · rw [if_neg hic]; omega
      · rw [if_neg hgate]; omega

/-- The loop keeps its running `best` at or above the initial value. -/
theorem searchMoves_ge_init {α : Type _} (gamma : Int) (f : α → Int) :
    ∀ (ms : List α) (b : Int), b ≤ searchMoves gamma f ms b := by
  intro ms
  induction ms with
  | nil => intro b; simp only [searchMoves]; omega
  | cons m ms ih =>
    intro b
    simp only [searchMoves]
    by_cases hcut : gamma ≤ max b (f m)
    · rw [if_pos hcut]; omega
    · rw [if_neg hcut]
      have := ih (max b (f m))
      omega

/-- If every move's report stays at or below the (below-window) initial
value, the loop returns the initial value unchanged: the "every move is a
refuted king-loss" scenario. -/
theorem searchMoves_eq_init {α : Type _} (gamma : Int) (f : α → Int) :
    ∀ (ms : List α) (b : Int), (∀ m ∈ ms, f m ≤ b) → b < gamma →
      searchMoves gamma f ms b = b := by
  intro ms
  induction ms with
  | nil => intro b _ _; simp only [searchMoves]
  | cons m ms ih =>
    intro b hall hb
    have hm : f m ≤ b := hall m (by simp)
    have hmax : max b (f m) = b := by omega
    simp only [searchMoves, hmax]
    rw [if_neg (by omega)]
    exact ih b (fun x hx => hall x (by simp [hx])) hb

/-- If the opponent king is capturable (and ours is not gone), the
corrected search reports *exactly* `MATE_UPPER` -- the sentinel invariant
of sunfish.py lines 398-401, here by construction. -/
theorem boundStale_of_capture (G : NullGame) (probe : G.Pos → Bool) (d : Nat)
    (m : G.Pos) (gamma : Int) (hd : 1 ≤ d) (hkg : ¬ (G.eval m ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toGame m = true) :
    boundStale G probe d m gamma = MATE_UPPER := by
  cases d with
  | zero => exact absurd hd (by omega)
  | succ d => simp only [boundStale, if_neg hkg, if_pos hcap]

/-- The pointwise heart of the correction: if the search's `best` (`S`)
is a fail-soft-correct report of the true fold (`F`), the probe is
correct, the window is above the loss floor, and a fold of `LOSS` can
only be seen when the search also saw `LOSS` (`hmask` -- this is where
`MateValuesAreKingCaptures` enters at the call site), then the
*corrected* `best` is a fail-soft-correct report of the *corrected*
fold. -/
theorem staleFix_spec_core (G : NullGame) (probe : G.Pos → Bool) (p : G.Pos)
    (d : Nat) (gamma S F : Int)
    (hg1 : -MATE_UPPER < gamma)
    (hP : CheckProbeOK G probe)
    (hspec1 : gamma ≤ S → S ≤ F)
    (hspec2 : S < gamma → F ≤ S)
    (hFinit : LOSS ≤ F)
    (hmask : F = LOSS → S = LOSS) :
    (gamma ≤ staleFix G probe (d + 1) S p →
      staleFix G probe (d + 1) S p ≤ drawFix G (d + 1) F p) ∧
    (staleFix G probe (d + 1) S p < gamma →
      drawFix G (d + 1) F p ≤ staleFix G probe (d + 1) S p) := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  by_cases hhi : gamma ≤ S
  · -- Fail high: `best ≥ gamma > LOSS`, so neither gate can fire.
    have hSF := hspec1 hhi
    have hnc1 : ¬ (2 < d + 1 ∧ S = LOSS) := by
      intro hand; omega
    have hnc2 : ¬ (2 < d + 1 ∧ F = LOSS) := by
      intro hand; omega
    have hsf : staleFix G probe (d + 1) S p = S := by
      simp only [staleFix, if_neg hnc1]
    have hdf : drawFix G (d + 1) F p = F := by
      simp only [drawFix, if_neg hnc2]
    rw [hsf, hdf]
    exact ⟨fun _ => hSF, fun h => by omega⟩
  · have hlow : S < gamma := by omega
    have hFS := hspec2 hlow
    by_cases hSL : S = LOSS
    · -- Fail low at exactly LOSS: the loop ran to completion and every
      -- move was refuted by a king capture (or there were none), so the
      -- true fold is LOSS too, and both sides apply the *same*
      -- correction (this is where the correct probe is used).
      have hFL : F = LOSS := by omega
      by_cases hd2 : 2 < d + 1
      · have hsf : staleFix G probe (d + 1) S p
            = (if probe p = true then -MATE_LOWER else 0) := by
          simp only [staleFix, if_pos (And.intro hd2 hSL)]
        have hdf : drawFix G (d + 1) F p
            = (if inCheckB G p = true then -MATE_LOWER else 0) := by
          simp only [drawFix, if_pos (And.intro hd2 hFL)]
        rw [hsf, hdf, hP p]
        exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
      · have hnc1 : ¬ (2 < d + 1 ∧ S = LOSS) := fun hand => hd2 hand.1
        have hnc2 : ¬ (2 < d + 1 ∧ F = LOSS) := fun hand => hd2 hand.1
        have hsf : staleFix G probe (d + 1) S p = S := by
          simp only [staleFix, if_neg hnc1]
        have hdf : drawFix G (d + 1) F p = F := by
          simp only [drawFix, if_neg hnc2]
        rw [hsf, hdf]
        exact ⟨fun h => by omega, fun _ => hFS⟩
    · -- Fail low above LOSS: the search saw a real move; `hmask`
      -- guarantees the value side is not secretly a corrected stalemate.
      have hFnL : ¬ (F = LOSS) := fun hFL => hSL (hmask hFL)
      have hncS : ¬ (2 < d + 1 ∧ S = LOSS) := fun hand => hSL hand.2
      have hncF : ¬ (2 < d + 1 ∧ F = LOSS) := fun hand => hFnL hand.2
      have hsf : staleFix G probe (d + 1) S p = S := by
        simp only [staleFix, if_neg hncS]
      have hdf : drawFix G (d + 1) F p = F := by
        simp only [drawFix, if_neg hncF]
      rw [hsf, hdf]
      exact ⟨fun h => by omega, fun _ => hFS⟩

/-! ### The main theorem -/

/-- **The stalemate-corrected search satisfies the docstring against the
draw-aware value** -- for in-band windows, a correct probe, a band-bounded
evaluation, and under `MateValuesAreKingCaptures`.  Sorry-free.  Each
hypothesis is necessary in an explicit sense: for the window see the
module comment (point 1); for `MateValuesAreKingCaptures` see
`boundStale_not_unconditional`. -/
theorem boundStale_spec (G : NullGame) (probe : G.Pos → Bool)
    (hB : Bounded G.toGame) (hM : MateValuesAreKingCaptures G)
    (hP : CheckProbeOK G probe) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      BoundSpecD G d p gamma (boundStale G probe d p gamma) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d with
  | zero =>
    -- Depth 0: search and value are literally the same expression.
    intro p gamma _ _
    simp only [BoundSpecD, boundStale, negamaxDraw]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]
      exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
    · rw [if_neg hkg]
      exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
  | succ d ih =>
    intro p gamma hg1 hg2
    simp only [BoundSpecD, boundStale, negamaxDraw]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · -- Our king is gone: both sides are exactly -MATE_UPPER.
      rw [if_pos hkg, if_pos hkg]
      exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
    · rw [if_neg hkg, if_neg hkg]
      by_cases hcap : hasKingCapture G.toGame p = true
      · -- We can capture their king: report exactly MATE_UPPER, and the
        -- capture child contributes exactly MATE_UPPER to the true fold,
        -- so the value is ≥ MATE_UPPER and its gate cannot fire.
        rw [if_pos hcap]
        cases (hasKingCapture_iff G.toGame p).mp hcap with
        | intro c hc =>
          have hcval := negamaxDraw_kingGone G d c hc.2
          have hcontrib : -(negamaxDraw G d c)
              ≤ foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS :=
            foldMax_le_of_mem _ _ _ c hc.1
          have hfold : MATE_UPPER
              ≤ foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS := by
            rw [hcval] at hcontrib; omega
          have hnc : ¬ (2 < d + 1 ∧
              foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS = LOSS) := by
            intro hand; omega
          have hdf : drawFix G (d + 1)
                (foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS) p
              = foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS := by
            simp only [drawFix, if_neg hnc]
          rw [hdf]
          exact ⟨fun _ => hfold, fun h => by omega⟩
      · -- The move loop.  The window flips into itself: gamma ∈ (-MU, MU]
        -- iff 1 - gamma ∈ (-MU, MU]  (the integer null-window trick again).
        rw [if_neg hcap]
        have hw1 : -MATE_UPPER < 1 - gamma := by omega
        have hw2 : 1 - gamma ≤ MATE_UPPER := by omega
        have hchild : ∀ m : G.Pos,
            (gamma ≤ -(boundStale G probe d m (1 - gamma)) →
              -(boundStale G probe d m (1 - gamma)) ≤ -(negamaxDraw G d m)) ∧
            (-(boundStale G probe d m (1 - gamma)) < gamma →
              -(negamaxDraw G d m) ≤ -(boundStale G probe d m (1 - gamma))) := by
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
        have hloop := searchMoves_spec gamma
          (fun m => -(boundStale G probe d m (1 - gamma)))
          (fun m => -(negamaxDraw G d m))
          hchild (G.moves p) LOSS LOSS
          (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
        -- hmask: a true fold of LOSS forces every child to be an exact
        -- king-capture mate (via the band, MateValuesAreKingCaptures and
        -- the sentinel invariant), hence every report is exactly LOSS and
        -- the search's best is LOSS too.
        have hmask : foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS = LOSS →
            searchMoves gamma (fun m => -(boundStale G probe d m (1 - gamma)))
              (G.moves p) LOSS = LOSS := by
          intro hFL
          have hall : ∀ m ∈ G.moves p,
              -(boundStale G probe d m (1 - gamma)) ≤ LOSS := by
            intro m hm
            have hwm : -(negamaxDraw G d m) ≤ LOSS := by
              have := foldMax_le_of_mem (fun x => -(negamaxDraw G d x)) (G.moves p) LOSS m hm
              rw [hFL] at this
              exact this
            have hband := negamaxDraw_bounded G hB d m
            have hveq : negamaxDraw G d m = MATE_UPPER := by omega
            have hmkg : ¬ (G.eval m ≤ -MATE_LOWER) := by
              intro hh
              exact hcap ((hasKingCapture_iff G.toGame p).mpr ⟨m, hm, hh⟩)
            have hbm : boundStale G probe d m (1 - gamma) = MATE_UPPER := by
              cases d with
              | zero =>
                -- Depth-0 reports are exact, so a MATE_UPPER value is
                -- reported as exactly MATE_UPPER.
                have h0 : negamaxDraw G 0 m
                    = (if G.eval m ≤ -MATE_LOWER then -MATE_UPPER else G.eval m) := by
                  simp only [negamaxDraw]
                rw [h0] at hveq
                simp only [boundStale]
                exact hveq
              | succ d' =>
                have hd1 : 1 ≤ d' + 1 := by omega
                exact boundStale_of_capture G probe (d' + 1) m (1 - gamma) hd1 hmkg
                  ((hasKingCapture_iff G.toGame m).mpr (hM (d' + 1) m hd1 hveq))
            rw [hbm]
            omega
          exact searchMoves_eq_init gamma
            (fun m => -(boundStale G probe d m (1 - gamma))) (G.moves p) LOSS
            hall (by omega)
        exact staleFix_spec_core G probe p d gamma
          (searchMoves gamma (fun m => -(boundStale G probe d m (1 - gamma))) (G.moves p) LOSS)
          (foldMax (fun m => -(negamaxDraw G d m)) (G.moves p) LOSS)
          hg1 hP hloop.1 hloop.2
          (foldMax_ge_init _ (G.moves p) LOSS)
          hmask

/-! ### The counterexample: the hypothesis is not optional

A seven-position game, all evaluations honest (in band, no king anywhere
near capture), probe exactly correct, window in band -- and the faithful
depth-gated correction still breaks the docstring.  The mechanism: `g` and
`h2` are terminal (stalemate) positions.  Seen at remaining depth 1 they
score `LOSS` (the gate is off), so at depth 2 the positions `m0` and `m1`
score a full `MATE_UPPER` -- mate scores fabricated out of stalemates, with
no king capture anywhere (`cex_violates_hypothesis`).  At the root (depth
3, gamma = -5) both moves therefore report `-MATE_UPPER`-flavored or
ordinary values whose maximum is `-20 < gamma`: an ordinary fail low, no
correction.  But the root's *true* fold is `LOSS`, so the draw-aware value
corrects to `0`, and the reported upper bound `-20` is simply wrong:

    boundStale = -20  <  gamma = -5,   yet   negamaxDraw = 0 > -20.
-/

/-- Positions of the counterexample game. -/
inductive CPos where
  | root | m0 | m1 | g | h1 | h2 | e

open CPos in
/-- root -> {m0, m1};  m0 -> {g};  m1 -> {h1, h2};  h1 -> {e};
g, h2, e terminal.  `eval e = 20`, every other eval 0; `pass` is the
identity (no position is in check under the one-ply probe). -/
def Cex : NullGame where
  Pos := CPos
  moves := fun p => match p with
    | root => [m0, m1]
    | m0 => [g]
    | m1 => [h1, h2]
    | h1 => [e]
    | _ => []
  eval := fun p => match p with
    | e => 20
    | _ => 0
  pass := fun p => p

theorem cex_bounded : Bounded Cex.toGame := by
  intro p
  cases p <;> decide

theorem cex_probeOK : CheckProbeOK Cex (fun p => inCheckB Cex p) :=
  fun _ => rfl

/-- `Cex` fabricates a mate score with no king capture behind it:
`negamaxDraw Cex 2 m0 = MATE_UPPER`, but `m0`'s only move goes to the
quiet position `g`. -/
theorem cex_violates_hypothesis : ¬ MateValuesAreKingCaptures Cex := by
  intro h
  cases h 2 CPos.m0 (by omega) (by decide) with
  | intro c hc =>
    have h1 : c ∈ [CPos.g] := hc.1
    rw [List.mem_singleton] at h1
    subst h1
    exact absurd hc.2 (by decide)

/-- The faithful `2 < depth` gate makes the draw-aware value itself
depth-inconsistent: the stalemate `g` is a loss at remaining depth 2 and a
draw at depth 3.  (This is the divergence between `negamaxDraw` and any
single depth-independent game value, stated on a concrete position.) -/
theorem negamaxDraw_depth_inconsistent :
    negamaxDraw Cex 2 CPos.g = LOSS ∧ negamaxDraw Cex 3 CPos.g = 0 := by
  constructor <;> decide

/-- **The stalemate correction is NOT unconditionally spec-preserving.**
Bounded evaluation, a perfectly correct probe and an in-band window are
not enough: without `MateValuesAreKingCaptures` the corrected search can
return a fail-low "upper bound" that the draw-aware value exceeds.
Machine-checked on `Cex` at the root with `gamma = -5`. -/
theorem boundStale_not_unconditional :
    ¬ (∀ (G : NullGame) (probe : G.Pos → Bool),
        Bounded G.toGame → CheckProbeOK G probe →
        ∀ (d : Nat) (p : G.Pos) (gamma : Int),
          -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
          BoundSpecD G d p gamma (boundStale G probe d p gamma)) := by
  intro h
  have hspec := h Cex (fun p => inCheckB Cex p) cex_bounded cex_probeOK
    3 CPos.root (-5) (by decide) (by decide)
  have hsearch : boundStale Cex (fun p => inCheckB Cex p) 3 CPos.root (-5) = -20 := by
    decide
  have hvalue : negamaxDraw Cex 3 CPos.root = 0 := by decide
  have h2 := hspec.2
  rw [hsearch, hvalue] at h2
  have := h2 (by omega)
  omega

/-! # The QS val-filter and the exhaustion gate

(References are to master at `bf72b43`.)  Everything above modeled the
move loop as running over ALL moves.  The engine does not: the loop is
quiescence-filtered by the move-value threshold

    val_lower = QS - depth * QS_A                      -- line 355
    for val, move in sorted(...):
        if val < val_lower: break                      -- lines 399-401

(the killer try respects the same threshold, line 395, so the killer
path cannot smuggle a below-threshold move into the loop), and the
stalemate correction is gated by the #136 fix:

    if best == -MATE_UPPER and (depth > 2 or
            all(pos.value(m) >= val_lower for m in pos.gen_moves())):
                                                       -- lines 471-472

The models above ASSUMED what this gate provides: `boundStale` searches
`G.moves p` unfiltered, so "the loop ended at `LOSS`, hence every legal
move was searched and refuted" held by construction -- the code
discharged a hypothesis the model never stated.  This section closes the
gap.  The loop runs over `movesAbove` (the filter; the `break` is
modeled as a filter exactly as `boundFut` models the futility break:
under the sort of line 399 the two are equivalent), the correction is
gated exactly as in the code, and the exhaustion argument becomes a
theorem:

* **`boundA1_exhaustion` / `correction_trustworthy`** -- if no legal
  move falls below the threshold (`allAboveB`), `best == -MATE_UPPER`
  after the FILTERED loop still certifies that every legal move loses
  the king, at ANY depth.  This is the formal content of the gate's
  `all(...)` arm.
* The `depth > 2` arm is sound because at depth ≥ 3 the threshold
  `val_lower = 40 - 140 * depth ≤ -380` sits below every move value: a
  named floor hypothesis `ValFloor` (concretely -192, machine-checked
  from the piece-square tables -- `EvalBounds.quietDropMax_eq`), under
  which `gate_implies_no_filtering` reduces this arm to the first.
  **Finding**: `val_lower 2 = -240` is ALREADY below the -192 floor, so
  with the shipped tables `allAboveB` is identically true at depth ≥ 2
  and the `depth > 2` arm is redundant -- a scan-skipping optimization,
  one ply more conservative than the tables require
  (`depth_arm_redundant`, `tables_kill_filter_at_depth2`).
* Dropping the gate is not an option: `qsUngated_not_sound` exhibits a
  machine-checked position where the filter skipped a legal quiet move,
  every SEARCHED move lost the king, and an ungated correction
  mislabels the non-stalemate as a draw.

The value function the filtered search brackets is `negamaxQS`: the
draw-aware value folded over `movesAbove` with the gated correction.  It
is determined by `(pos, depth)` alone -- the point-spec doctrine of
`formal/README.md` is preserved (QS filtering is depth-keyed and
pos-derived).  Note `negamaxQS` is still depth-inconsistent
(`negamaxQS_depth_inconsistent`) -- the filter itself moves with depth --
but the #136 gate REPAIRS the inconsistency at genuinely moveless
positions: their correction now fires at every depth ≥ 1
(`stalemate_fixed_all_depths`), where `negamaxDraw` scored them `LOSS`
at depth ≤ 2.

## The A1 exposure and its fix (modeled ahead of the code)

The loop also contains the null-move yield (lines 371-372), and in the
shipped code it feeds `best`: a fail-low null yield `rn` with
`-MATE_UPPER < rn < gamma` at a genuinely stalemated node leaves
`best = rn ≠ -MATE_UPPER`, the sentinel test never fires, and the search
returns `rn` as an "upper bound" that the draw-aware value 0 exceeds.
That is audit finding A1, and master currently has the hole:
`a1_unfixed_not_sound` is a machine-checked witness in which every other
hypothesis (including the null-move bet) is satisfied.  The fix modeled
here follows the agreed `a1-fix` design -- **at modeling time the
`a1-fix` branch carries no code beyond master** (verified: empty diff),
so this is a model of the design, to be re-audited against the code when
it lands:

* `best_real` tracks real-move yields only, and the sentinel test reads
  `best_real`, never `best` -- the null yield cannot mask a stalemate;
* the gate becomes `best < gamma and best_real == -MATE_UPPER and (...)`;
* the null yield is suppressed when `rn >= MATE_LOWER`, so the null-move
  bet (`NullBetQS`) need only be trusted BELOW the mate band -- exactly
  where "some real move is at least as good as passing" is plausible.
  Unsuppressed, a mate-band null claim with no move behind it would
  demand the bet where the `MateValuesAreKingCaptures` doctrine forbids
  trusting fabricated mate scores.

`boundA1_spec` proves the fixed search brackets `negamaxQS` under named
hypotheses; `boundQS_spec` is the null-free instance.  The loop is
modeled as `a1Fix .. gamma (max rn best_real) best_real ..`; the
equivalence with the code's single running `best` (the null yield is
just the loop's first, cutoff-checked accumulator update) is
`searchMoves_init_max`. -/

/-! ### The threshold -/

/-- sunfish.py line 149: `QS = 40`. -/
def QS : Int := 40

/-- sunfish.py line 150: `QS_A = 140`. -/
def QS_A : Int := 140

/-- The QS move-value threshold, sunfish.py line 355:
`val_lower = QS - depth * QS_A`.  (`depth` is already clamped to ≥ 0 at
line 329, matching the `Nat` here.) -/
def val_lower (d : Nat) : Int := QS - d * QS_A

theorem val_lower_le_QS (d : Nat) : val_lower d ≤ QS := by
  unfold val_lower QS QS_A
  omega

/-- The threshold never reaches the mate band: king captures
(`val ≥ MATE_LOWER`) pass the filter at every depth. -/
theorem val_lower_lt_ML (d : Nat) : val_lower d < MATE_LOWER := by
  have h := val_lower_le_QS d
  have hQ : QS = 40 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  omega

/-- At depth ≥ 3 the threshold is at most -380 (the `depth > 2` arm's
arithmetic; `val_lower 3 = 40 - 420`). -/
theorem val_lower_deep (d : Nat) (h : 3 ≤ d) : val_lower d ≤ -380 := by
  unfold val_lower QS QS_A
  omega

/-! ### The filtered move list -/

/-- A `NullGame` together with sunfish's move valuation `pos.value(move)`
(indexed by the child position the move reaches -- the same convention as
`FutGame.val` in `Sunfish/Tricks.lean`). -/
structure QSGame extends NullGame where
  val : Pos → Pos → Int

/-- The moves the val-filter keeps: `val >= thr`.  The code's sorted
`break` (lines 399-401) searches exactly this set; as with the futility
break (`boundFut`), the sort order makes break and filter equivalent, and
we do not model the ordering. -/
def movesAbove (G : QSGame) (thr : Int) (p : G.Pos) : List G.Pos :=
  (G.moves p).filter (fun m => decide (thr ≤ G.val p m))

theorem mem_movesAbove {G : QSGame} {thr : Int} {p m : G.Pos} :
    m ∈ movesAbove G thr p ↔ m ∈ G.moves p ∧ thr ≤ G.val p m := by
  simp [movesAbove, List.mem_filter]

theorem movesAbove_subset (G : QSGame) (thr : Int) (p : G.Pos) :
    ∀ m ∈ movesAbove G thr p, m ∈ G.moves p :=
  fun _ hm => (List.mem_filter.mp hm).1

/-- `all(pos.value(m) >= val_lower for m in pos.gen_moves())` -- the
computable no-skip test of the gate's second arm (line 472). -/
def allAboveB (G : QSGame) (d : Nat) (p : G.Pos) : Bool :=
  (G.moves p).all (fun m => decide (val_lower d ≤ G.val p m))

theorem filter_eq_self_of_all {α : Type _} (f : α → Bool) :
    ∀ (l : List α), (∀ a ∈ l, f a = true) → l.filter f = l := by
  intro l
  induction l with
  | nil => intro _; rfl
  | cons a l ih =>
    intro h
    rw [List.filter_cons, if_pos (h a (List.mem_cons_self a l)),
      ih (fun x hx => h x (List.mem_cons_of_mem a hx))]

/-- If the no-skip test passes, the filter kept every legal move. -/
theorem movesAbove_all (G : QSGame) (d : Nat) (p : G.Pos)
    (h : allAboveB G d p = true) :
    movesAbove G (val_lower d) p = G.moves p := by
  rw [allAboveB, List.all_eq_true] at h
  exact filter_eq_self_of_all _ (G.moves p) h

/-- The gate of lines 471-472: `depth > 2 or all(...)`. -/
def qsGateB (G : QSGame) (d : Nat) (p : G.Pos) : Bool :=
  decide (2 < d) || allAboveB G d p

/-! ### The move-value floor -/

/-- **ValFloor**: every legal move's value is at least `-B`.  For the
shipped tables `B = 192` works: `pos.value` is the mover's table delta
(≥ -192, the queen's worst case) plus nonnegative terms -- all
machine-checked in `Sunfish/EvalBounds.lean` (`quietDropMax_eq`,
`capture_terms_nonneg`, `promotion_terms_nonneg`, `castle_rook_deltas`);
the link from board strings to tables is not modeled (the same caveat as
`Bounded`'s discharge there). -/
def ValFloor (G : QSGame) (B : Int) : Prop :=
  ∀ (p : G.Pos), ∀ m ∈ G.moves p, -B ≤ G.val p m

theorem allAboveB_of_floor (G : QSGame) {B : Int} (hF : ValFloor G B)
    (d : Nat) (p : G.Pos) (h : val_lower d ≤ -B) : allAboveB G d p = true := by
  rw [allAboveB, List.all_eq_true]
  intro m hm
  rw [decide_eq_true_eq]
  have := hF p m hm
  omega

/-- **The `depth > 2` arm, justified**: whenever the gate is on -- by
either arm -- the filter provably kept every legal move, provided the
move values respect a floor of at least -380 (tables: -192).  This is
the lemma that lets the depth arm inherit the exhaustion argument. -/
theorem gate_implies_no_filtering (G : QSGame) {B : Int} (hF : ValFloor G B)
    (hB : B ≤ 380) (d : Nat) (p : G.Pos) (hg : qsGateB G d p = true) :
    movesAbove G (val_lower d) p = G.moves p := by
  rw [qsGateB, Bool.or_eq_true, decide_eq_true_eq] at hg
  cases hg with
  | inl hd =>
    refine movesAbove_all G d p (allAboveB_of_floor G hF d p ?_)
    have := val_lower_deep d hd
    omega
  | inr hall => exact movesAbove_all G d p hall

/-- **Finding**: with the shipped tables' floor (-192 ≥ -240 =
`val_lower 2`) the no-skip test is identically true at depth ≥ 2, so the
`depth > 2` arm never decides anything -- it only skips the `all(...)`
scan, and is one ply more conservative than the tables require. -/
theorem depth_arm_redundant (G : QSGame) {B : Int} (hF : ValFloor G B)
    (hB : B ≤ 240) (d : Nat) (hd : 2 ≤ d) (p : G.Pos) :
    allAboveB G d p = true := by
  refine allAboveB_of_floor G hF d p ?_
  unfold val_lower QS QS_A
  omega

set_option maxRecDepth 4096 in
/-- The table-level arithmetic behind the previous two theorems:
`val_lower 2 = -240` already clears the concrete -192 floor;
`val_lower 3 = -380` is what the depth arm actually relies on. -/
theorem tables_kill_filter_at_depth2 :
    val_lower 2 = -240 ∧ val_lower 2 ≤ -EvalBounds.quietDropMax ∧
    val_lower 3 = -380 :=
  ⟨by decide, by decide, by decide⟩

/-! ### The filtered draw-aware value -/

/-- **KingCaptureValHigh**: a move that captures the king (child eval in
the king-gone zone) is valued in the mate band, `val ≥ MATE_LOWER` --
sunfish's move ordering fact (line 399, cf. `orderedMoves` in
`Sunfish/Killer.lean`), which with `val_lower_lt_ML` puts king captures
in `movesAbove` at every depth.  Concrete backing:
`EvalBounds.kingCapture_val_above`. -/
def KingCaptureValHigh (G : QSGame) : Prop :=
  ∀ (p : G.Pos), ∀ m ∈ G.moves p, G.eval m ≤ -MATE_LOWER → MATE_LOWER ≤ G.val p m

/-- The gated correction applied to the true (filtered) fold: fire iff
the fold is the untouched sentinel AND the gate of lines 471-472 is on. -/
def qsDrawFix (G : QSGame) (d : Nat) (F : Int) (p : G.Pos) : Int :=
  if qsGateB G d p = true ∧ F = LOSS then
    (if inCheckB G.toNullGame p = true then -MATE_LOWER else 0)
  else F

/-- Draw-aware, king-capture-normalized, VAL-FILTERED negamax: the value
function the filtered search brackets.  Structure of `negamaxDraw`, with
the fold over `movesAbove` and the correction gated as in the code.
Determined by `(pos, depth)` alone -- the point-spec doctrine holds. -/
def negamaxQS (G : QSGame) : Nat → G.Pos → Int
  | 0, p => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else qsDrawFix G (d + 1)
      (foldMax (fun m => -(negamaxQS G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS) p

theorem negamaxQS_kingGone (G : QSGame) (d : Nat) (p : G.Pos)
    (h : G.eval p ≤ -MATE_LOWER) : negamaxQS G d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [negamaxQS]; rw [if_pos h]
  | succ d => simp only [negamaxQS]; rw [if_pos h]

/-- Band-boundedness of the filtered draw value (mirror of
`negamaxDraw_bounded`). -/
theorem negamaxQS_bounded (G : QSGame) (hB : Bounded G.toNullGame.toGame) :
    ∀ (d : Nat) (p : G.Pos),
      -MATE_UPPER ≤ negamaxQS G d p ∧ negamaxQS G d p ≤ MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d with
  | zero =>
    intro p
    have hband := hB p
    simp only [negamaxQS]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]; omega
    · rw [if_neg hkg]; omega
  | succ d ih =>
    intro p
    simp only [negamaxQS]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]; omega
    · rw [if_neg hkg]
      have hfl := foldMax_ge_init (fun m => -(negamaxQS G d m))
        (movesAbove G (val_lower (d + 1)) p) LOSS
      have hfu : foldMax (fun m => -(negamaxQS G d m))
          (movesAbove G (val_lower (d + 1)) p) LOSS ≤ MATE_UPPER := by
        refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
        show -(negamaxQS G d m) ≤ MATE_UPPER
        have := ih m
        omega
      unfold qsDrawFix
      by_cases hgate : qsGateB G (d + 1) p = true ∧
          foldMax (fun m => -(negamaxQS G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS
      · rw [if_pos hgate]
        by_cases hic : inCheckB G.toNullGame p = true
        · rw [if_pos hic]; omega
        · rw [if_neg hic]; omega
      · rw [if_neg hgate]; omega

/-- **The #136 repair, stated**: at a genuinely moveless position the
`all(...)` arm is vacuously true, so the correction fires at EVERY depth
≥ 1 -- where the un-`all`-gated `negamaxDraw` scored the same position
`LOSS` at depth ≤ 2 (`negamaxDraw_depth_inconsistent`).  This is what
stops a depth ≤ 2 node above a stalemate from feeding `+MATE_UPPER` to
its parent (the `Qc4??` bug of the code comment, lines 469-470). -/
theorem stalemate_fixed_all_depths (G : QSGame) (p : G.Pos)
    (hm : G.moves p = []) (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) (d : Nat) :
    negamaxQS G (d + 1) p
      = (if inCheckB G.toNullGame p = true then -MATE_LOWER else 0) := by
  simp only [negamaxQS]
  rw [if_neg hkg]
  have hma : movesAbove G (val_lower (d + 1)) p = [] := by
    rw [movesAbove, hm]
    rfl
  rw [hma]
  have hgate : qsGateB G (d + 1) p = true := by
    have hall : allAboveB G (d + 1) p = true := by
      rw [allAboveB, hm]
      rfl
    rw [qsGateB, hall, Bool.or_true]
  simp only [foldMax, qsDrawFix]
  rw [if_pos (And.intro hgate (by trivial))]

/-! ### The filtered, A1-fixed search -/

/-- The filtered form of `MateValuesAreKingCaptures`: a `MATE_UPPER`
value of `negamaxQS` (at remaining depth ≥ 1) is always backed by a real
king capture.  Same honest caveat as the unfiltered version -- its
necessity is inherited from `boundStale_not_unconditional`. -/
def MateValuesAreKingCapturesQS (G : QSGame) : Prop :=
  ∀ (d : Nat) (p : G.Pos), 1 ≤ d → negamaxQS G d p = MATE_UPPER →
    ∃ m ∈ G.moves p, G.eval m ≤ -MATE_LOWER

/-- **NullBetQS**: the null-move bet, A1-shaped.  `nully d p gamma`
abstracts the null yield `-bound(pos.rotate(nullmove=True), 1 - gamma,
depth - 3)` of lines 371-372 (the `depth - 3` reduction and the
`can_null` layering live in `Sunfish/CanNull.lean`; here the yield is an
oracle and this is the one fact the search consumes from it): when the
guard passes, at depth > 2, a fail-HIGH yield BELOW the mate band really
lower-bounds the position's value.  Fail-low yields need no hypothesis
(they only raise a fail-soft upper bound that the real loop already
justifies), and mate-band yields are suppressed by the A1 fix, so the
bet is never trusted where `MateValuesAreKingCaptures` doctrine forbids
fabricated mate scores. -/
def NullBetQS (G : QSGame) (nully : Nat → G.Pos → Int → Int)
    (guard : G.Pos → Bool) : Prop :=
  ∀ (d : Nat) (p : G.Pos) (gamma : Int),
    guard p = true → 2 < d → gamma ≤ nully d p gamma →
    nully d p gamma < MATE_LOWER → nully d p gamma ≤ negamaxQS G d p

/-- The A1 null-use test: the engine guard (`can_null` and
`abs(pos.score) < 500`, abstracted as `guard`), the `depth > 2` gate of
line 371, and the A1 suppression `rn < MATE_LOWER`. -/
def useNull (G : QSGame) (nully : Nat → G.Pos → Int → Int)
    (guard : G.Pos → Bool) (d : Nat) (p : G.Pos) (gamma : Int) : Bool :=
  guard p && decide (2 < d) && decide (nully d p gamma < MATE_LOWER)

/-- `best`: the null yield joins the fail-soft maximum (when used), but
never `best_real`. -/
def nullMax (G : QSGame) (nully : Nat → G.Pos → Int → Int)
    (guard : G.Pos → Bool) (d : Nat) (p : G.Pos) (gamma : Int) (S : Int) : Int :=
  if useNull G nully guard d p gamma = true then max (nully d p gamma) S else S

/-- The A1-fixed correction as applied to the search: fire on
`best < gamma  AND  best_real == -MATE_UPPER  AND  (depth > 2 or all(...))`,
with the engine's probe.  `best` is the null-inclusive maximum, `S` is
`best_real`. -/
def a1Fix (G : QSGame) (probe : G.Pos → Bool) (d : Nat)
    (gamma best S : Int) (p : G.Pos) : Int :=
  if best < gamma ∧ S = LOSS ∧ qsGateB G d p = true then
    (if probe p = true then -MATE_LOWER else 0)
  else best

/-- The filtered, null-aware, A1-fixed `bound`:

* line 337-338: our king is gone -> `-MATE_UPPER`;
* the sentinel invariant by construction (as in `boundStale`);
* the null yield first: cutoff if used and `gamma ≤ rn` (fail-soft, the
  loop ends before any real move) -- suppressed entirely when
  `rn ≥ MATE_LOWER` (A1);
* otherwise the fail-soft loop over the FILTERED moves accumulates
  `best_real` from `LOSS`, `best = max rn best_real` when the null was
  used (`searchMoves_init_max` proves this equals the code's single
  running `best`), and the gated correction reads `best_real`. -/
def boundA1 (G : QSGame) (probe : G.Pos → Bool)
    (nully : Nat → G.Pos → Int → Int) (guard : G.Pos → Bool) :
    Nat → G.Pos → Int → Int
  | 0, p, _gamma => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | d + 1, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if useNull G nully guard (d + 1) p gamma = true ∧ gamma ≤ nully (d + 1) p gamma then
      nully (d + 1) p gamma
    else
      a1Fix G probe (d + 1) gamma
        (nullMax G nully guard (d + 1) p gamma
          (searchMoves gamma (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS))
        (searchMoves gamma (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
          (movesAbove G (val_lower (d + 1)) p) LOSS)
        p

/-- The null-free instance: `boundStale` upgraded with the val-filter and
the #136 gate.  With `guard ≡ false` the null branch is dead code, `best
= best_real`, and (for in-band `gamma`, where `best_real = LOSS` implies
`best < gamma`) the `a1Fix` gate is exactly the code's lines 471-472. -/
def boundQS (G : QSGame) (probe : G.Pos → Bool) : Nat → G.Pos → Int → Int :=
  boundA1 G probe (fun _ _ _ => 0) (fun _ => false)

/-- `BoundSpec` against the filtered draw-aware value. -/
def BoundSpecQS (G : QSGame) (d : Nat) (p : G.Pos) (gamma r : Int) : Prop :=
  (gamma ≤ r → r ≤ negamaxQS G d p) ∧ (r < gamma → negamaxQS G d p ≤ r)

/-- The sentinel invariant, by construction (mirror of
`boundStale_of_capture`): king-capturable positions report exactly
`MATE_UPPER` at depth ≥ 1, before null or loop can interfere. -/
theorem boundA1_of_capture (G : QSGame) (probe : G.Pos → Bool)
    (nully : Nat → G.Pos → Int → Int) (guard : G.Pos → Bool) (d : Nat)
    (m : G.Pos) (gamma : Int) (hd : 1 ≤ d) (hkg : ¬ (G.eval m ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame m = true) :
    boundA1 G probe nully guard d m gamma = MATE_UPPER := by
  cases d with
  | zero => exact absurd hd (by omega)
  | succ d => simp only [boundA1]; rw [if_neg hkg, if_pos hcap]

/-! ### Loop lemmas -/

/-- Exhaustion of the loop, converse of `searchMoves_eq_init`: a
below-window loop that ends AT its initial value saw every move fail at
or below it -- "the consumption break needs `best >= gamma`", the code's
own argument at lines 462-464. -/
theorem searchMoves_eq_init_all {α : Type _} (gamma : Int) (f : α → Int) :
    ∀ (ms : List α) (b : Int), b < gamma → searchMoves gamma f ms b = b →
      ∀ m ∈ ms, f m ≤ b := by
  intro ms
  induction ms with
  | nil => intro b _ _ m hm; cases hm
  | cons a ms ih =>
    intro b hb heq m hm
    simp only [searchMoves] at heq
    by_cases hcut : gamma ≤ max b (f a)
    · rw [if_pos hcut] at heq
      omega
    · rw [if_neg hcut] at heq
      have hge := searchMoves_ge_init gamma f ms (max b (f a))
      have hmax : max b (f a) = b := by omega
      cases List.mem_cons.mp hm with
      | inl he => subst he; omega
      | inr ht =>
        rw [hmax] at heq
        exact ih b hb heq m ht

/-- Fidelity of the `max rn best_real` restructuring: seeding the
fail-soft loop with a below-window value (the code's null yield updating
the single running `best` first) is the same as folding from `LOSS` and
taking the `max` at the end -- the cutoff points coincide because a
below-window seed never triggers the break. -/
theorem searchMoves_init_max {α : Type _} (gamma : Int) (f : α → Int) :
    ∀ (ms : List α) (b : Int), LOSS ≤ b → b < gamma →
      searchMoves gamma f ms b = max b (searchMoves gamma f ms LOSS) := by
  intro ms
  induction ms with
  | nil =>
    intro b hL _hb
    simp only [searchMoves]
    omega
  | cons a ms ih =>
    intro b hL hb
    have hLg : LOSS < gamma := by omega
    simp only [searchMoves]
    by_cases hf : gamma ≤ f a
    · rw [if_pos (by omega), if_pos (by omega)]
      omega
    · rw [if_neg (by omega), if_neg (by omega)]
      rw [ih (max b (f a)) (by omega) (by omega), ih (max LOSS (f a)) (by omega) (by omega)]
      omega

/-! ### The main theorem -/

/-- The pointwise heart of the A1-fixed correction (mirror of
`staleFix_spec_core`, with the gate shared between search and value and
the null-inflated `best` riding above `best_real = S`):

* `hSb`/`hbS`: `S ≤ best`, and a fail-high `best` collapses to `S` (the
  null yield sits below the window whenever the loop ran);
* `hspec1`/`hspec2`: the loop is fail-soft correct against the filtered
  fold `F`;
* `hexh`: search exhaustion -- a loop ending at `LOSS` forces the true
  fold to `LOSS` (every filtered move's value is the exact sentinel);
* `hmask`: the converse, where `MateValuesAreKingCapturesQS` enters at
  the call site. -/
theorem a1Fix_spec_core (G : QSGame) (probe : G.Pos → Bool) (p : G.Pos)
    (d : Nat) (gamma S F best : Int)
    (hg1 : -MATE_UPPER < gamma)
    (hP : CheckProbeOK G.toNullGame probe)
    (hSb : S ≤ best)
    (hbS : gamma ≤ best → best ≤ S)
    (hspec1 : gamma ≤ S → S ≤ F)
    (hspec2 : S < gamma → F ≤ S)
    (hexh : S = LOSS → F = LOSS)
    (hmask : F = LOSS → S = LOSS) :
    (gamma ≤ a1Fix G probe d gamma best S p →
      a1Fix G probe d gamma best S p ≤ qsDrawFix G d F p) ∧
    (a1Fix G probe d gamma best S p < gamma →
      qsDrawFix G d F p ≤ a1Fix G probe d gamma best S p) := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  by_cases hgate : best < gamma ∧ S = LOSS ∧ qsGateB G d p = true
  · -- The correction fires: search exhaustion makes the value fold LOSS
    -- too, the gate is shared, and probe correctness aligns the results.
    have hFL := hexh hgate.2.1
    have hfix : a1Fix G probe d gamma best S p
        = (if probe p = true then -MATE_LOWER else 0) := by
      simp only [a1Fix, if_pos hgate]
    have hdf : qsDrawFix G d F p
        = (if inCheckB G.toNullGame p = true then -MATE_LOWER else 0) := by
      simp only [qsDrawFix, if_pos (And.intro hgate.2.2 hFL)]
    rw [hfix, hdf, hP p]
    exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
  · have hfix : a1Fix G probe d gamma best S p = best := by
      simp only [a1Fix, if_neg hgate]
    rw [hfix]
    constructor
    · -- Fail high: best = S ≥ gamma, the loop's lower bound stands, and
      -- the value gate cannot fire above LOSS.
      intro hge
      have hbs := hbS hge
      have hSF := hspec1 (by omega)
      have hdf : qsDrawFix G d F p = F := by
        simp only [qsDrawFix]
        rw [if_neg (fun hand => absurd hand.2 (by omega))]
      rw [hdf]
      omega
    · -- Fail low: if the value gate fired, hmask forces S = LOSS and the
      -- search gate would have fired too -- contradiction; otherwise the
      -- upper bound is the loop's.
      intro hlt
      have hFS := hspec2 (by omega)
      by_cases hfire : qsGateB G d p = true ∧ F = LOSS
      · exact absurd ⟨hlt, hmask hfire.2, hfire.1⟩ hgate
      · have hdf : qsDrawFix G d F p = F := by
          simp only [qsDrawFix]
          rw [if_neg (fun hand => hfire ⟨hand.1, hand.2⟩)]
        rw [hdf]
        omega

/-- **The filtered, A1-fixed search satisfies the docstring against the
filtered draw-aware value** -- for in-band windows, a correct probe, a
band-bounded evaluation, king captures valued in the mate band, the
below-band null bet, and `MateValuesAreKingCapturesQS`.  Sorry-free.
This is `boundStale_spec` with the assumed exhaustion ("the loop
searched every move") replaced by the PROVEN gate: the correction only
consumes the sentinel where the val-filter provably skipped nothing
(`hexh`/`hmask` below), and the null yield can neither trigger nor mask
it (`best_real`). -/
theorem boundA1_spec (G : QSGame) (probe : G.Pos → Bool)
    (nully : Nat → G.Pos → Int → Int) (guard : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hV : KingCaptureValHigh G)
    (hM : MateValuesAreKingCapturesQS G)
    (hP : CheckProbeOK G.toNullGame probe)
    (hN : NullBetQS G nully guard) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      BoundSpecQS G d p gamma (boundA1 G probe nully guard d p gamma) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d with
  | zero =>
    -- Depth 0: search and value are literally the same expression.
    intro p gamma _ _
    simp only [BoundSpecQS, boundA1, negamaxQS]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]
      exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
    · rw [if_neg hkg]
      exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
  | succ d ih =>
    intro p gamma hg1 hg2
    simp only [BoundSpecQS, boundA1, negamaxQS]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg, if_pos hkg]
      exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
    · rw [if_neg hkg, if_neg hkg]
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · -- King capturable: the capture passes the filter
        -- (KingCaptureValHigh + val_lower_lt_ML), so the value fold is
        -- at least MATE_UPPER and its gate cannot fire.
        rw [if_pos hcap]
        cases (hasKingCapture_iff G.toNullGame.toGame p).mp hcap with
        | intro c hc =>
          have hcval := negamaxQS_kingGone G d c hc.2
          have hcmem : c ∈ movesAbove G (val_lower (d + 1)) p := by
            rw [mem_movesAbove]
            refine ⟨hc.1, ?_⟩
            have hv := hV p c hc.1 hc.2
            have hvl := val_lower_lt_ML (d + 1)
            omega
          have hcontrib : -(negamaxQS G d c)
              ≤ foldMax (fun m => -(negamaxQS G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS :=
            foldMax_le_of_mem _ _ _ c hcmem
          have hfold : MATE_UPPER
              ≤ foldMax (fun m => -(negamaxQS G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS := by
            rw [hcval] at hcontrib
            omega
          have hdf : qsDrawFix G (d + 1)
                (foldMax (fun m => -(negamaxQS G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS) p
              = foldMax (fun m => -(negamaxQS G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS := by
            simp only [qsDrawFix]
            rw [if_neg (fun hand => absurd hand.2 (by omega))]
          rw [hdf]
          exact ⟨fun _ => hfold, fun h => absurd h (by omega)⟩
      · rw [if_neg hcap]
        -- The loop.  Window flips into itself (the null-window trick).
        have hw1 : -MATE_UPPER < 1 - gamma := by omega
        have hw2 : 1 - gamma ≤ MATE_UPPER := by omega
        have hchild : ∀ m : G.Pos,
            (gamma ≤ -(boundA1 G probe nully guard d m (1 - gamma)) →
              -(boundA1 G probe nully guard d m (1 - gamma)) ≤ -(negamaxQS G d m)) ∧
            (-(boundA1 G probe nully guard d m (1 - gamma)) < gamma →
              -(negamaxQS G d m) ≤ -(boundA1 G probe nully guard d m (1 - gamma))) := by
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
        have hloop := searchMoves_spec gamma
          (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
          (fun m => -(negamaxQS G d m))
          hchild (movesAbove G (val_lower (d + 1)) p) LOSS LOSS
          (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
        -- hexh: search exhaustion.  A loop ending at LOSS reported every
        -- filtered move at or below LOSS, so every filtered child search
        -- returned >= MATE_UPPER; fail-soft correctness plus the band pin
        -- each child value at exactly MATE_UPPER, so the true fold is
        -- LOSS as well.  (No MateValues hypothesis needed here.)
        have hexh : searchMoves gamma
              (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
              (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS →
            foldMax (fun m => -(negamaxQS G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS
              = LOSS := by
          intro hS
          have hallle := searchMoves_eq_init_all gamma
            (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS (by omega) hS
          have hup : foldMax (fun m => -(negamaxQS G d m))
              (movesAbove G (val_lower (d + 1)) p) LOSS ≤ LOSS := by
            refine foldMax_le _ _ _ (fun m hm => ?_) (Int.le_refl _)
            show -(negamaxQS G d m) ≤ LOSS
            have hf : -(boundA1 G probe nully guard d m (1 - gamma)) ≤ LOSS := hallle m hm
            have hband := negamaxQS_bounded G hB d m
            have := (ih m (1 - gamma) hw1 hw2).1 (by omega)
            omega
          have hdown := foldMax_ge_init (fun m => -(negamaxQS G d m))
            (movesAbove G (val_lower (d + 1)) p) LOSS
          omega
        -- hmask: a true fold of LOSS forces every filtered child to be an
        -- exact king-capture mate (band + MateValuesAreKingCapturesQS +
        -- the sentinel invariant), hence every report is exactly LOSS and
        -- the loop ends at LOSS too.
        have hmask : foldMax (fun m => -(negamaxQS G d m))
              (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS →
            searchMoves gamma
              (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
              (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS := by
          intro hFL
          have hall : ∀ m ∈ movesAbove G (val_lower (d + 1)) p,
              -(boundA1 G probe nully guard d m (1 - gamma)) ≤ LOSS := by
            intro m hm
            have hwm : -(negamaxQS G d m) ≤ LOSS := by
              have := foldMax_le_of_mem (fun x => -(negamaxQS G d x))
                (movesAbove G (val_lower (d + 1)) p) LOSS m hm
              rw [hFL] at this
              exact this
            have hband := negamaxQS_bounded G hB d m
            have hveq : negamaxQS G d m = MATE_UPPER := by omega
            have hmkg : ¬ (G.eval m ≤ -MATE_LOWER) := by
              intro hh
              have := negamaxQS_kingGone G d m hh
              omega
            have hbm : boundA1 G probe nully guard d m (1 - gamma) = MATE_UPPER := by
              cases d with
              | zero =>
                have h0 : negamaxQS G 0 m
                    = (if G.eval m ≤ -MATE_LOWER then -MATE_UPPER else G.eval m) := by
                  simp only [negamaxQS]
                rw [h0] at hveq
                simp only [boundA1]
                exact hveq
              | succ d' =>
                have hd1 : 1 ≤ d' + 1 := by omega
                exact boundA1_of_capture G probe nully guard (d' + 1) m (1 - gamma) hd1 hmkg
                  ((hasKingCapture_iff G.toNullGame.toGame m).mpr (hM (d' + 1) m hd1 hveq))
            rw [hbm]
            omega
          exact searchMoves_eq_init gamma
            (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS hall (by omega)
        by_cases hnc : useNull G nully guard (d + 1) p gamma = true ∧
            gamma ≤ nully (d + 1) p gamma
        · -- Null cutoff: fail high on the oracle; NullBetQS (usable only
          -- because the A1 suppression pinned rn below the mate band) is
          -- exactly the required lower bound.
          rw [if_pos hnc]
          have hun := hnc.1
          simp only [useNull, Bool.and_eq_true, decide_eq_true_eq] at hun
          have hbet := hN (d + 1) p gamma hun.1.1 hun.1.2 hnc.2 hun.2
          have hval : negamaxQS G (d + 1) p
              = qsDrawFix G (d + 1)
                  (foldMax (fun m => -(negamaxQS G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS)
                  p := by
            simp only [negamaxQS]
            rw [if_neg hkg]
          rw [hval] at hbet
          exact ⟨fun _ => hbet, fun hlt => absurd hnc.2 (by omega)⟩
        · rw [if_neg hnc]
          -- No null cutoff: rn (if used at all) is below the window, so
          -- `best = max rn best_real` collapses onto best_real whenever it
          -- matters; the core lemma does the rest.
          refine a1Fix_spec_core G probe p (d + 1) gamma
            (searchMoves gamma (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
              (movesAbove G (val_lower (d + 1)) p) LOSS)
            (foldMax (fun m => -(negamaxQS G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS)
            (nullMax G nully guard (d + 1) p gamma
              (searchMoves gamma (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
                (movesAbove G (val_lower (d + 1)) p) LOSS))
            hg1 hP ?_ ?_ hloop.1 hloop.2 hexh hmask
          · -- S ≤ best
            simp only [nullMax]
            by_cases hu : useNull G nully guard (d + 1) p gamma = true
            · rw [if_pos hu]; omega
            · rw [if_neg hu]; exact Int.le_refl _
          · -- gamma ≤ best → best ≤ S
            intro hge
            simp only [nullMax] at hge ⊢
            by_cases hu : useNull G nully guard (d + 1) p gamma = true
            · rw [if_pos hu] at hge ⊢
              have hrn : nully (d + 1) p gamma < gamma := by
                by_cases h : gamma ≤ nully (d + 1) p gamma
                · exact absurd ⟨hu, h⟩ hnc
                · omega
              omega
            · rw [if_neg hu] at hge ⊢
              exact Int.le_refl _

/-- The null-free corollary: the filtered, #136-gated search (the code
with `can_null=False`, or below the null-depth threshold) satisfies the
docstring against `negamaxQS` -- no null bet needed. -/
theorem boundQS_spec (G : QSGame) (probe : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hV : KingCaptureValHigh G)
    (hM : MateValuesAreKingCapturesQS G)
    (hP : CheckProbeOK G.toNullGame probe) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      BoundSpecQS G d p gamma (boundQS G probe d p gamma) :=
  boundA1_spec G probe (fun _ _ _ => 0) (fun _ => false) hB hV hM hP
    (fun _ _ _ hg _ _ _ => absurd hg (by simp))

/-! ### The exhaustion theorems: the sentinel is trustworthy -/

/-- **The exhaustion lemma, search level** (the formal content of the
#136 gate's `all(...)` arm): if no legal move falls below the threshold
and the FILTERED loop still ended at the untouched `LOSS` sentinel, then
EVERY legal move of `p` -- the full, unfiltered list -- has the exact
king-capture value `MATE_UPPER`: the position is genuinely mate or
stalemate, at ANY depth.  Compare the module comment's point 3 and the
code's own argument at lines 459-467, now a theorem. -/
theorem boundA1_exhaustion (G : QSGame) (probe : G.Pos → Bool)
    (nully : Nat → G.Pos → Int → Int) (guard : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hV : KingCaptureValHigh G)
    (hM : MateValuesAreKingCapturesQS G)
    (hP : CheckProbeOK G.toNullGame probe)
    (hN : NullBetQS G nully guard)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_UPPER)
    (hall : allAboveB G (d + 1) p = true)
    (hS : searchMoves gamma
        (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
        (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS) :
    ∀ m ∈ G.moves p, negamaxQS G d m = MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro m hm
  rw [movesAbove_all G (d + 1) p hall] at hS
  have hf : -(boundA1 G probe nully guard d m (1 - gamma)) ≤ LOSS :=
    searchMoves_eq_init_all gamma
      (fun m => -(boundA1 G probe nully guard d m (1 - gamma)))
      (G.moves p) LOSS (by omega) hS m hm
  have hspec := (boundA1_spec G probe nully guard hB hV hM hP hN
    d m (1 - gamma) (by omega) (by omega)).1
  have hband := negamaxQS_bounded G hB d m
  have := hspec (by omega)
  omega

/-- With `MateValuesAreKingCapturesQS` on top: every legal move is
refuted by a REAL king capture one ply down (depth ≥ 1; at depth 0 the
sentinel value is a bare static fact and no capture witness exists --
the honest limit of the claim). -/
theorem boundA1_exhaustion_captures (G : QSGame) (probe : G.Pos → Bool)
    (nully : Nat → G.Pos → Int → Int) (guard : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hV : KingCaptureValHigh G)
    (hM : MateValuesAreKingCapturesQS G)
    (hP : CheckProbeOK G.toNullGame probe)
    (hN : NullBetQS G nully guard)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_UPPER)
    (hall : allAboveB G (d + 2) p = true)
    (hS : searchMoves gamma
        (fun m => -(boundA1 G probe nully guard (d + 1) m (1 - gamma)))
        (movesAbove G (val_lower (d + 2)) p) LOSS = LOSS) :
    ∀ m ∈ G.moves p, ∃ c ∈ G.moves m, G.eval c ≤ -MATE_LOWER :=
  fun m hm => hM (d + 1) m (by omega)
    (boundA1_exhaustion G probe nully guard hB hV hM hP hN
      (d + 1) p gamma hg1 hg2 hall hS m hm)

/-- **The exhaustion lemma, value level, both gate arms**: whenever the
gate condition of lines 471-472 holds -- by EITHER arm, given the value
floor -- and the filtered fold is the untouched sentinel, every legal
move's value is the exact king-capture `MATE_UPPER`.  This is the
statement that discharges the sentinel-trust assumption for the whole
gate, `depth > 2` arm included (via `gate_implies_no_filtering`). -/
theorem correction_trustworthy (G : QSGame) (hB : Bounded G.toNullGame.toGame)
    {B : Int} (hF : ValFloor G B) (hB380 : B ≤ 380) (d : Nat) (p : G.Pos)
    (hgate : qsGateB G (d + 1) p = true)
    (hL : foldMax (fun m => -(negamaxQS G d m))
        (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS) :
    ∀ m ∈ G.moves p, negamaxQS G d m = MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro m hm
  rw [gate_implies_no_filtering G hF hB380 (d + 1) p hgate] at hL
  have hcontrib : -(negamaxQS G d m)
      ≤ foldMax (fun x => -(negamaxQS G d x)) (G.moves p) LOSS :=
    foldMax_le_of_mem (fun x => -(negamaxQS G d x)) (G.moves p) LOSS m hm
  rw [hL] at hcontrib
  have hband := negamaxQS_bounded G hB d m
  omega

/-! ### Counterexample: the low-depth correction NEEDS the `all(...)` guard

A four-position game.  At the root `r` (depth 2, `gamma = -5`) the
val-filter (threshold `val_lower 2 = -240`) keeps only the move to `a`
(val 0) and skips the legal quiet move to `q` (val -300).  The searched
move loses the king (`a`'s only reply `k` is a king capture), so the
filtered loop ends at the untouched `LOSS` sentinel -- but the sentinel
LIES about the position: `r` is not stalemated, `q` is a legal, playable
move the filter hid.  An UNGUARDED correction (fire on `best == -MATE_
UPPER` alone, no depth/`all(...)` gate) probes `r` (not in check),
mislabels it a draw and returns a fail-high `0` -- while the filtered
draw-aware value of `r` at depth 2 is `LOSS`.  Every hypothesis of
`boundA1_spec` holds for this game (including `MateValuesAreKingCaptures
QS`, proven for all depths), so the blame is pinned on the missing gate:
this is the negative result that justifies the gate's existence, and
`negamaxQS 2 r = LOSS` versus the gated search's own `LOSS` report shows
the gated search is exactly right here (`cexQ_gated_ok`). -/

/-- The unguarded correction: fire whenever the loop ends at `LOSS`,
regardless of depth or filtering.  (The naive "just correct at every
depth" alternative to the #136 gate.) -/
def staleFixUn (G : QSGame) (probe : G.Pos → Bool) (best : Int) (p : G.Pos) : Int :=
  if best = LOSS then (if probe p = true then -MATE_LOWER else 0) else best

/-- `boundQS` with the unguarded correction. -/
def boundQSUngated (G : QSGame) (probe : G.Pos → Bool) : Nat → G.Pos → Int → Int
  | 0, p, _gamma => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | d + 1, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else
      staleFixUn G probe
        (searchMoves gamma (fun m => -(boundQSUngated G probe d m (1 - gamma)))
          (movesAbove G (val_lower (d + 1)) p) LOSS) p

/-- Positions of the counterexample game. -/
inductive QPos where
  | r | a | q | k
  deriving DecidableEq

open QPos in
/-- `r -> {a, q}`; `a -> {k}`; `q`, `k` terminal.  `k` is a captured
king (`eval -60000`); `q` is an ordinary quiet position; `pass` is the
identity (nobody is in check under the one-ply probe).  The move to `q`
is valued -300: below `val_lower 2 = -240` (filtered at depth 2) but
above `val_lower 3 = -380` (searched at depth 3). -/
def CexQ : QSGame where
  Pos := QPos
  moves := fun p => match p with
    | r => [a, q]
    | a => [k]
    | _ => []
  eval := fun p => match p with
    | q => -30
    | k => -60000
    | _ => 0
  pass := fun p => p
  val := fun p m => match p, m with
    | r, q => -300
    | a, k => MATE_LOWER
    | _, _ => 0

theorem cexQ_bounded : Bounded CexQ.toNullGame.toGame := by
  intro p
  cases p <;> decide

theorem cexQ_valHigh : KingCaptureValHigh CexQ := by
  intro p m hm hev
  cases p with
  | r =>
    have hm' : m ∈ [QPos.a, QPos.q] := hm
    cases List.mem_cons.mp hm' with
    | inl h => subst h; exact absurd hev (by decide)
    | inr h =>
      have h' : m = QPos.q := List.mem_singleton.mp h
      subst h'
      exact absurd hev (by decide)
  | a =>
    have hm' : m ∈ [QPos.k] := hm
    have h' : m = QPos.k := List.mem_singleton.mp hm'
    subst h'
    decide
  | q => exact absurd (show m ∈ ([] : List QPos) from hm) (by simp)
  | k => exact absurd (show m ∈ ([] : List QPos) from hm) (by simp)

theorem cexQ_probeOK :
    CheckProbeOK CexQ.toNullGame (fun p => inCheckB CexQ.toNullGame p) :=
  fun _ => rfl

/-- `q` is a genuine stalemate (moveless, not in check): the #136 gate
scores it 0 at every depth ≥ 1. -/
theorem cexQ_q (d : Nat) : negamaxQS CexQ (d + 1) QPos.q = 0 := by
  rw [stalemate_fixed_all_depths CexQ QPos.q rfl (by decide) d]
  decide

/-- `a` is king-capturable, hence the exact sentinel at every depth ≥ 1
(the capture `k` passes the filter at every threshold). -/
theorem cexQ_a (d : Nat) : negamaxQS CexQ (d + 1) QPos.a = MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hkmem : QPos.k ∈ movesAbove CexQ (val_lower (d + 1)) QPos.a := by
    rw [mem_movesAbove]
    refine ⟨show QPos.k ∈ [QPos.k] from List.mem_singleton.mpr rfl, ?_⟩
    have hvk : CexQ.val QPos.a QPos.k = MATE_LOWER := rfl
    have hvl := val_lower_lt_ML (d + 1)
    omega
  have hkv := negamaxQS_kingGone CexQ d QPos.k (by decide)
  have hlow : -(negamaxQS CexQ d QPos.k)
      ≤ foldMax (fun m => -(negamaxQS CexQ d m))
          (movesAbove CexQ (val_lower (d + 1)) QPos.a) LOSS :=
    foldMax_le_of_mem _ _ _ QPos.k hkmem
  have hup : foldMax (fun m => -(negamaxQS CexQ d m))
      (movesAbove CexQ (val_lower (d + 1)) QPos.a) LOSS ≤ MATE_UPPER := by
    refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
    show -(negamaxQS CexQ d m) ≤ MATE_UPPER
    have := negamaxQS_bounded CexQ cexQ_bounded d m
    omega
  have hF : foldMax (fun m => -(negamaxQS CexQ d m))
      (movesAbove CexQ (val_lower (d + 1)) QPos.a) LOSS = MATE_UPPER := by
    rw [hkv] at hlow
    omega
  simp only [negamaxQS]
  rw [if_neg (by decide), hF]
  simp only [qsDrawFix]
  rw [if_neg (fun hand => absurd hand.2 (by decide))]

/-- At depth ≥ 3 the filter admits `q` and `r`'s value is the ordinary
fold: 0 (the -MATE_UPPER contribution of `a` is dominated by the
stalemate `q`'s 0). -/
theorem cexQ_r_deep (d : Nat) : negamaxQS CexQ (d + 3) QPos.r = 0 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hqv : negamaxQS CexQ (d + 2) QPos.q = 0 := cexQ_q (d + 1)
  have hav : negamaxQS CexQ (d + 2) QPos.a = MATE_UPPER := cexQ_a (d + 1)
  have hqmem : QPos.q ∈ movesAbove CexQ (val_lower (d + 2 + 1)) QPos.r := by
    rw [mem_movesAbove]
    refine ⟨show QPos.q ∈ [QPos.a, QPos.q] from
      List.mem_cons_of_mem _ (List.mem_singleton.mpr rfl), ?_⟩
    have hvq : CexQ.val QPos.r QPos.q = -300 := rfl
    have hvl := val_lower_deep (d + 2 + 1) (by omega)
    omega
  have hlow : -(negamaxQS CexQ (d + 2) QPos.q)
      ≤ foldMax (fun m => -(negamaxQS CexQ (d + 2) m))
          (movesAbove CexQ (val_lower (d + 2 + 1)) QPos.r) LOSS :=
    foldMax_le_of_mem _ _ _ QPos.q hqmem
  have hup : foldMax (fun m => -(negamaxQS CexQ (d + 2) m))
      (movesAbove CexQ (val_lower (d + 2 + 1)) QPos.r) LOSS ≤ 0 := by
    refine foldMax_le _ _ _ (fun m hm => ?_) (by omega)
    show -(negamaxQS CexQ (d + 2) m) ≤ 0
    have hmm : m ∈ [QPos.a, QPos.q] := movesAbove_subset _ _ _ m hm
    cases List.mem_cons.mp hmm with
    | inl h => subst h; omega
    | inr h =>
      have h' : m = QPos.q := List.mem_singleton.mp h
      subst h'
      omega
  have hF : foldMax (fun m => -(negamaxQS CexQ (d + 2) m))
      (movesAbove CexQ (val_lower (d + 2 + 1)) QPos.r) LOSS = 0 := by
    rw [hqv] at hlow
    omega
  show negamaxQS CexQ (d + 2 + 1) QPos.r = 0
  conv =>
    lhs
    rw [negamaxQS]
  rw [if_neg (by decide), hF]
  simp only [qsDrawFix]
  rw [if_neg (fun hand => absurd hand.2 (by decide))]

/-- `MateValuesAreKingCapturesQS` holds for the whole game, at every
depth: the only mate-band values are `a`'s, backed by the real capture
`k`. -/
theorem cexQ_mateValues : MateValuesAreKingCapturesQS CexQ := by
  intro d p hd hMU'
  cases p with
  | a =>
    exact ⟨QPos.k, show QPos.k ∈ [QPos.k] from List.mem_singleton.mpr rfl, by decide⟩
  | k =>
    rw [negamaxQS_kingGone CexQ d QPos.k (by decide)] at hMU'
    exact absurd hMU' (by decide)
  | q =>
    cases d with
    | zero => exact absurd hd (by omega)
    | succ n =>
      rw [cexQ_q n] at hMU'
      exact absurd hMU' (by decide)
  | r =>
    cases d with
    | zero => exact absurd hd (by omega)
    | succ n =>
      cases n with
      | zero =>
        rw [(by decide : negamaxQS CexQ 1 QPos.r = 0)] at hMU'
        exact absurd hMU' (by decide)
      | succ n' =>
        cases n' with
        | zero =>
          rw [(by decide : negamaxQS CexQ 2 QPos.r = LOSS)] at hMU'
          exact absurd hMU' (by decide)
        | succ n'' =>
          rw [show n'' + 1 + 1 + 1 = n'' + 3 from rfl, cexQ_r_deep n''] at hMU'
          exact absurd hMU' (by decide)

/-- The filter is depth-keyed, so the filtered draw value is
depth-inconsistent even where the old gate was not involved: `r` is 0 at
depth 1 (the capture line is invisible to the depth-0 child), `LOSS` at
depth 2 (the refutation is seen, `q` is filtered), 0 at depth 3 (`q`
enters).  The analogue of `negamaxDraw_depth_inconsistent`, now driven
by `val_lower` instead of the `depth > 2` gate. -/
theorem negamaxQS_depth_inconsistent :
    negamaxQS CexQ 1 QPos.r = 0 ∧ negamaxQS CexQ 2 QPos.r = LOSS ∧
    negamaxQS CexQ 3 QPos.r = 0 :=
  ⟨by decide, by decide, by decide⟩

/-- **The unguarded low-depth correction is UNSOUND** -- the negative
result that justifies the #136 gate.  With every `boundA1_spec`
hypothesis satisfied, the ungated search still mislabels the
filter-truncated `r` as a draw: it returns a fail-high `0` where the
filtered draw-aware value is `LOSS`.  (The full-move value is no refuge
either: the filter hid a merely bad move, not a losing one, so `0` is a
fabricated draw claim in every reading.) -/
theorem qsUngated_not_sound :
    ¬ (∀ (G : QSGame) (probe : G.Pos → Bool),
        Bounded G.toNullGame.toGame → KingCaptureValHigh G →
        MateValuesAreKingCapturesQS G → CheckProbeOK G.toNullGame probe →
        ∀ (d : Nat) (p : G.Pos) (gamma : Int),
          -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
          BoundSpecQS G d p gamma (boundQSUngated G probe d p gamma)) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro h
  have hspec := h CexQ (fun p => inCheckB CexQ.toNullGame p) cexQ_bounded cexQ_valHigh
    cexQ_mateValues cexQ_probeOK 2 QPos.r (-5) (by decide) (by decide)
  have hsearch : boundQSUngated CexQ (fun p => inCheckB CexQ.toNullGame p)
      2 QPos.r (-5) = 0 := by decide
  have hvalue : negamaxQS CexQ 2 QPos.r = LOSS := by decide
  have h1 := hspec.1
  rw [hsearch, hvalue] at h1
  have := h1 (by omega)
  omega

/-- The GATED search on the same position reports `LOSS` -- a correct
fail-low against `negamaxQS 2 r = LOSS` (and one that deeper search
retracts, exactly as the code comment promises). -/
theorem cexQ_gated_ok :
    boundQS CexQ (fun p => inCheckB CexQ.toNullGame p) 2 QPos.r (-5) = LOSS := by
  decide

/-! ### Counterexample: the null yield must not feed the sentinel test (A1)

Master's loop lets the null yield update the same `best` the sentinel
test reads.  Take a single genuinely stalemated position `s` (no moves,
not in check, quiet eval -50 -- well inside the `abs(pos.score) < 500`
null guard) and a perfectly well-behaved null oracle that returns -50 (a
sound, fail-low report; `NullBetQS` holds).  At depth 4, `gamma = -20`:
the null yield sets `best = -50 ≠ -MATE_UPPER`, the sentinel test never
fires, and the search returns -50 as an "upper bound" -- but the
draw-aware value of a stalemate is 0.  Every hypothesis holds; the
unfixed loop shape is the bug.  The A1-fixed `boundA1` on the same
inputs reads `best_real = -MATE_UPPER`, corrects, and returns the exact
0 (`a1_fix_repairs`).  NOTE: `boundA1Un` is the SHIPPED loop shape --
this hole is in master today; the existing `NullGuardBlocksAtCaptures`
analysis (Killer.lean) covers king-capturable nodes only, not
stalemates. -/

/-- Master's gate: sentinel test on the null-inclusive `best`
(`best == -MATE_UPPER and (depth > 2 or all(...))`, lines 471-472), no
`best_real`, no mate-band suppression of the null yield. -/
def a1FixUn (G : QSGame) (probe : G.Pos → Bool) (d : Nat) (best : Int) (p : G.Pos) : Int :=
  if best = LOSS ∧ qsGateB G d p = true then
    (if probe p = true then -MATE_LOWER else 0)
  else best

/-- The UNFIXED search: `boundA1` minus the A1 design -- the null yield
feeds the single running `best` that the correction gate reads. -/
def boundA1Un (G : QSGame) (probe : G.Pos → Bool)
    (nully : Nat → G.Pos → Int → Int) (guard : G.Pos → Bool) :
    Nat → G.Pos → Int → Int
  | 0, p, _gamma => if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p
  | d + 1, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if (guard p && decide (2 < d + 1)) = true ∧ gamma ≤ nully (d + 1) p gamma then
      nully (d + 1) p gamma
    else
      a1FixUn G probe (d + 1)
        (if (guard p && decide (2 < d + 1)) = true then
          max (nully (d + 1) p gamma)
            (searchMoves gamma (fun m => -(boundA1Un G probe nully guard d m (1 - gamma)))
              (movesAbove G (val_lower (d + 1)) p) LOSS)
        else
          searchMoves gamma (fun m => -(boundA1Un G probe nully guard d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS)
        p

/-- The A1 counterexample game: one stalemated position. -/
def CexN : QSGame where
  Pos := Unit
  moves := fun _ => []
  eval := fun _ => -50
  pass := fun p => p
  val := fun _ _ => 0

theorem cexN_value (d : Nat) : negamaxQS CexN (d + 1) () = 0 := by
  rw [stalemate_fixed_all_depths CexN () rfl (by decide) d]
  decide

theorem cexN_bounded : Bounded CexN.toNullGame.toGame := by
  intro p
  cases p
  decide

theorem cexN_valHigh : KingCaptureValHigh CexN := by
  intro p m hm _
  exact absurd (show m ∈ ([] : List Unit) from hm) (by simp)

theorem cexN_probeOK :
    CheckProbeOK CexN.toNullGame (fun p => inCheckB CexN.toNullGame p) :=
  fun _ => rfl

theorem cexN_mateValues : MateValuesAreKingCapturesQS CexN := by
  intro d p hd hMU'
  cases p
  cases d with
  | zero => exact absurd hd (by omega)
  | succ n =>
    rw [cexN_value n] at hMU'
    exact absurd hMU' (by decide)

/-- The constant -50 null oracle is perfectly well behaved: it never
fails high above the position's value. -/
theorem cexN_nullBet : NullBetQS CexN (fun _ _ _ => -50) (fun _ => true) := by
  intro d p gamma _ _ _ _
  show (-50 : Int) ≤ negamaxQS CexN d p
  cases p
  cases d with
  | zero => decide
  | succ n =>
    rw [cexN_value n]
    omega

/-- **Master's loop shape is UNSOUND at stalemates** -- audit finding A1,
machine-checked.  Bounded evals, mate-band king captures, real mates
only, a correct probe AND a sound null bet are not enough: the fail-low
null yield masks `best == -MATE_UPPER`, the correction never fires, and
the returned "upper bound" -50 is exceeded by the stalemate's true value
0. -/
theorem a1_unfixed_not_sound :
    ¬ (∀ (G : QSGame) (probe : G.Pos → Bool)
          (nully : Nat → G.Pos → Int → Int) (guard : G.Pos → Bool),
        Bounded G.toNullGame.toGame → KingCaptureValHigh G →
        MateValuesAreKingCapturesQS G → CheckProbeOK G.toNullGame probe →
        NullBetQS G nully guard →
        ∀ (d : Nat) (p : G.Pos) (gamma : Int),
          -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
          BoundSpecQS G d p gamma (boundA1Un G probe nully guard d p gamma)) := by
  intro h
  have hspec := h CexN (fun p => inCheckB CexN.toNullGame p) (fun _ _ _ => -50)
    (fun _ => true) cexN_bounded cexN_valHigh cexN_mateValues cexN_probeOK cexN_nullBet
    4 () (-20) (by decide) (by decide)
  have hsearch : boundA1Un CexN (fun p => inCheckB CexN.toNullGame p) (fun _ _ _ => -50)
      (fun _ => true) 4 () (-20) = -50 := by decide
  have hvalue : negamaxQS CexN 4 () = 0 := cexN_value 3
  have h2 := hspec.2
  rw [hsearch, hvalue] at h2
  have := h2 (by omega)
  omega

/-- The A1-fixed search on the same inputs: `best_real = -MATE_UPPER`
survives the null yield, the gate fires, and the stalemate is scored
exactly 0.  (Its spec is an instance of `boundA1_spec`, whose
hypotheses `cexN_*` above verify.) -/
theorem a1_fix_repairs :
    boundA1 CexN (fun p => inCheckB CexN.toNullGame p) (fun _ _ _ => -50)
      (fun _ => true) 4 () (-20) = 0 := by
  decide

/-! # The refuted-assumptions ledger, and the verified search (d2)

(References are to branch `d2-verify-pending` at `29c7887`, the
verify-on-suspicion landing; the A1 fix shipped at `0998739` and the d2
change reworks the same block.)

The A1 fix protected the correction's FAIL-LOW side: a pseudo-option
yield can no longer MASK the untouched sentinel.  But the correction is
an ASSIGNMENT of exact terminal knowledge -- an override outside the
max-fold (the fold rule, formal/README.md) -- and an override has a
FAIL-HIGH dual: a pseudo-option scoring ABOVE the terminal value at a
terminal node stores `lower > terminal` at one window while another
window stores `upper = terminal` -- a crossed table entry.  The first
formal treatment of that dual isolated it as a named hypothesis,
`TerminalPseudoSafe` below, split into a mate side (a theorem -- see
`nullAtMateD2` further down) and two open sides.  Both open sides were
then REFUTED on real boards, which is what forced the verified design:

* **`NullAtStalemateNonpositive` is FALSE in real chess.**  Witness
  `8/8/8/5k1p/3b1P1P/1p1P1P1P/pN1P1P2/K7 w` (score +175): a genuine
  stalemate -- king-defended pinned knight, promotion-blocked guard
  pawn, frozen pawn towers -- where the pass search honestly consumed
  +11.  Pre-fix master stored `lower = 11, upper = 0` on it.
  Mechanism: the "re-freezing protection" intuition (the opponent can
  pass right back, re-freezing the stalemate at value ~0) fails at
  null-depth 1, where the re-frozen stalemate is evaluated by QS as
  stand-pat -- the static +175-ish score -- not as 0.  `CexT` below is
  the abstract shape of this witness (eval +30, oracle +30), and
  `cexT_crossing` machine-checks the crossing on the A1-FIXED loop.

* **`StandPatAtTerminal` is FALSE in real chess.**  Witness
  `6Rk/6QP/8/8/8/8/8/K7 b`: a mated QS node whose every pseudo-move is
  a valuable-but-illegal defended capture, so the depth-0 stand-pat
  carries a normal score while the old depth-0 correction stored the
  exact mate -- pre-fix `lower = -1711, upper = -47923`.  The automated
  sweep (formal/scripts/standpat_terminal_search.py, seed 0: 5,000
  playouts + 1,459,694 sparse positions + 214,004 corner packings)
  found 100+ natural corner-mate hits (simplest:
  `8/8/8/7p/7P/1K5P/p7/k5R1 b`, K_MID pricing the mated king's quiet
  moves at 65-145, above `QS = 40`): the shape is common, not exotic.
  The d2 answer is not to defend the hypothesis but to remove the
  consumer: `if depth and ...` excludes depth 0 from the correction
  entirely -- QS evaluates the fold and never claims an exact terminal
  value.

* **`KingCapturableReportsExact` is FALSE under general fail-soft
  windows** (defined and machine-refuted in the probe section below):
  a king-capturable child may soundly cut off on its stand-pat or on a
  partial table lower without ever reporting the mate band -- the same
  move was machine-traced scoring -368 at one window and -69290 at
  another.  The REPLACEMENT theorem is not that every search of such a
  node reports `MATE_UPPER`; it is that the DEDICATED legality probe
  does (`legalityProbeCorrect`).  This refutation names the root cause
  of the one channel d2 leaves open -- "sentinel masking", a
  king-capturable child's partial cutoff setting the `live` bit at a
  terminal node -- and is the next arc's target.

The refuted statements are kept below, re-labeled as countermodels.
Nothing after this section consumes them: the d2 search (`boundD2`)
re-derives terminality with the legality oracle instead of assuming
any pseudo-option inequality. -/

/-! ### The terminal override and the retired obligation -/

/-- **Correction-terminal node** (at remaining depth `d`; meaningful for
`d ≥ 1`): our king is on the board, the pre-d2 gate is on, and the true
filtered fold is the untouched sentinel -- real-move exhaustion.  This
was the firing condition of the pre-d2 correction on the value side
(`negamaxQS_of_correctionTerminal`).  RETIRED as a load-bearing notion:
the d2 correction and value function key on `allIllegalB` instead --
position-intrinsic, not search-history-shaped. -/
def CorrectionTerminal (G : QSGame) (d : Nat) (p : G.Pos) : Prop :=
  ¬ (G.eval p ≤ -MATE_LOWER) ∧ qsGateB G d p = true ∧
  foldMax (fun m => -(negamaxQS G (d - 1) m)) (movesAbove G (val_lower d) p) LOSS = LOSS

/-- The exact terminal value the correction assigns (line 472 on
`29c7887`): `-MATE_LOWER` for mate (in check), `0` for stalemate. -/
def terminalValue (G : QSGame) (p : G.Pos) : Int :=
  if inCheckB G.toNullGame p = true then -MATE_LOWER else 0

/-- At a correction-terminal node the filtered draw-aware value IS the
terminal value -- the bridge between the old gate condition and the
exact knowledge the correction asserts. -/
theorem negamaxQS_of_correctionTerminal (G : QSGame) (d : Nat) (p : G.Pos)
    (h : CorrectionTerminal G (d + 1) p) :
    negamaxQS G (d + 1) p = terminalValue G p := by
  obtain ⟨hkg, hgate, hfold⟩ := h
  simp only [negamaxQS]
  rw [if_neg hkg]
  have hfold' : foldMax (fun m => -(negamaxQS G d m))
      (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS := hfold
  rw [hfold']
  simp only [qsDrawFix]
  rw [if_pos (And.intro hgate (by trivial))]
  rfl

/-- **TerminalPseudoSafe** (RETIRED -- an obligation no theorem consumes
any more): at every correction-terminal node, every ENABLED
pseudo-option score is at most the terminal value.  This was the named
hypothesis under which the PRE-d2 loop's point spec could be recovered;
its stalemate instance is `NullAtStalemateNonpositive` below, REFUTED
on a real board (module comment above).  Kept because
`terminalPseudoSafe_not_free` and `cexT_crossing` document what goes
wrong on any design that assumes rather than verifies. -/
def TerminalPseudoSafe (G : QSGame) (nully : Nat → G.Pos → Int → Int)
    (guard : G.Pos → Bool) : Prop :=
  ∀ (d : Nat) (p : G.Pos) (gamma : Int),
    CorrectionTerminal G d p → useNull G nully guard d p gamma = true →
    nully d p gamma ≤ terminalValue G p

/-- **NullAtStalemateNonpositive** (REFUTED in real chess -- the +175
witness in the module comment): at a correction-terminal node not in
check, an enabled null yield is at most the draw value 0.  Zugzwang-
shaped but sharper: at a terminal node there is NO real move to match
the pass, so the usual `NullBetQS` justification fails in principle.
The engine now VERIFIES instead of assuming this
(`positiveNullCutoffVerified`). -/
def NullAtStalemateNonpositive (G : QSGame) (nully : Nat → G.Pos → Int → Int)
    (guard : G.Pos → Bool) : Prop :=
  ∀ (d : Nat) (p : G.Pos) (gamma : Int),
    CorrectionTerminal G d p → inCheckB G.toNullGame p = false →
    useNull G nully guard d p gamma = true → nully d p gamma ≤ 0

/-! ### Countermodel: the obligation was never free

A one-position game: a genuinely stalemated position with a POSITIVE
static score -- the material-up stalemate trap, the abstract shape of
the real `8/8/8/5k1p/3b1P1P/1p1P1P1P/pN1P1P2/K7 w` witness.  `eval =
30` fits the engine's `abs(pos.score) < 500` null guard; the null
oracle answers +30 (exactly what an honest pass search reports when the
material-down opponent cannot reach 0 either); every non-bet hypothesis
of `boundA1_spec` holds.  Yet the enabled null scores ABOVE the
terminal value 0, and the crossing is machine-checked on the A1-FIXED
loop: `best_real` repaired the fail-low mask, not this fail-high
dual. -/

/-- The fail-high-dual countermodel game. -/
def CexT : QSGame where
  Pos := Unit
  moves := fun _ => []
  eval := fun _ => 30
  pass := fun p => p
  val := fun _ _ => 0

theorem cexT_value (d : Nat) : negamaxQS CexT (d + 1) () = 0 := by
  rw [stalemate_fixed_all_depths CexT () rfl (by decide) d]
  decide

theorem cexT_bounded : Bounded CexT.toNullGame.toGame := by
  intro p
  cases p
  decide

theorem cexT_valHigh : KingCaptureValHigh CexT := by
  intro p m hm _
  exact absurd (show m ∈ ([] : List Unit) from hm) (by simp)

theorem cexT_probeOK :
    CheckProbeOK CexT.toNullGame (fun p => inCheckB CexT.toNullGame p) :=
  fun _ => rfl

theorem cexT_mateValues : MateValuesAreKingCapturesQS CexT := by
  intro d p hd hMU'
  cases p
  cases d with
  | zero => exact absurd hd (by omega)
  | succ n =>
    rw [cexT_value n] at hMU'
    exact absurd hMU' (by decide)

/-- The stalemate is correction-terminal at depth 4 (any depth ≥ 1 would
do). -/
theorem cexT_correctionTerminal : CorrectionTerminal CexT 4 () :=
  ⟨by decide, by decide, by decide⟩

/-- The +30 oracle violates `TerminalPseudoSafe`: enabled by every
engine gate (guard on, depth > 2, well below the mate band), it scores
above the terminal value 0. -/
theorem cexT_violates :
    ¬ TerminalPseudoSafe CexT (fun _ _ _ => 30) (fun _ => true) := by
  intro h
  have := h 4 () 10 cexT_correctionTerminal (by decide)
  exact absurd this (by decide)

/-- **TerminalPseudoSafe was NOT implied by the pre-d2 assumptions.**
Bounded evaluation, mate-band king captures, real mates only and a
correct probe -- every non-bet hypothesis of `boundA1_spec` -- leave the
fail-high pseudo path at terminal nodes entirely unconstrained. -/
theorem terminalPseudoSafe_not_free :
    ¬ (∀ (G : QSGame) (nully : Nat → G.Pos → Int → Int) (guard : G.Pos → Bool),
        Bounded G.toNullGame.toGame → KingCaptureValHigh G →
        MateValuesAreKingCapturesQS G →
        CheckProbeOK G.toNullGame (fun p => inCheckB G.toNullGame p) →
        TerminalPseudoSafe G nully guard) := by
  intro h
  exact cexT_violates
    (h CexT (fun _ _ _ => 30) (fun _ => true)
      cexT_bounded cexT_valHigh cexT_mateValues cexT_probeOK)

/-- **The crossing, machine-checked, on the A1-FIXED (pre-d2) loop.**
Low probe `gamma = 10`: the null option cuts off before the (empty)
real-move loop and `boundA1` returns 30 -- a fail-high, stored as
`lower = 30` by table part 2.  High probe `gamma = 100` on the same
`(pos, depth)`: the null fails low, `best_real` keeps the sentinel, the
correction fires and returns the exact terminal 0 -- a fail-low, stored
as `upper = 0`.  `lower = 30 > 0 = upper`, and the point value they
disagree about is `negamaxQS = 0`: the low probe's stored bound is
simply false.  On the real +175 witness the shipped pre-d2 engine
stored `lower = 11, upper = 0`.  `d2_repairs_cexT` (below) shows the
verified loop returning the exact 0 at both windows on the same
inputs. -/
theorem cexT_crossing :
    boundA1 CexT (fun p => inCheckB CexT.toNullGame p) (fun _ _ _ => 30)
        (fun _ => true) 4 () 10 = 30 ∧
    boundA1 CexT (fun p => inCheckB CexT.toNullGame p) (fun _ _ _ => 30)
        (fun _ => true) 4 () 100 = 0 ∧
    negamaxQS CexT 4 () = 0 :=
  ⟨by decide, by decide, cexT_value 3⟩

/-! ### The depth-0 pseudo-option: the stand-pat (REFUTED, consumer removed) -/

/-- Depth-0 correction-terminality, by content (this model's depth 0 is
QS-as-eval, so the old engine's depth-0 loop is stated by what its gate
certified): our king is on the board, no pseudo-legal move falls below
the depth-0 threshold `val_lower 0 = QS = 40`, and every pseudo-legal
move loses the king to an immediate recapture. -/
def CorrectionTerminal0 (G : QSGame) (p : G.Pos) : Prop :=
  ¬ (G.eval p ≤ -MATE_LOWER) ∧ allAboveB G 0 p = true ∧
  ∀ m ∈ G.moves p, hasKingCapture G.toNullGame.toGame m = true

/-- **StandPatAtTerminal** (REFUTED in real chess -- the corner-mate
witnesses in the module comment): at a depth-0 correction-terminal
position the stand-pat score does not exceed the terminal value.  The
pre-d2 depth-0 correction silently consumed this; the sweep found 100+
natural violations.  d2 removes the consumer: `if depth and ...`
excludes depth 0 from the correction, so QS evaluates the fold and
never claims an exact terminal value. -/
def StandPatAtTerminal (G : QSGame) : Prop :=
  ∀ p, CorrectionTerminal0 G p → G.eval p ≤ terminalValue G p

/-- The mate arm of `StandPatAtTerminal`, unfolded: the hypothesis
forces every depth-0 correction-terminal node to be a stalemate, never a
mate (the stand-pat, living in the static band, always exceeds the mate
value `-MATE_LOWER`).  This is the crisp shape the counterexample sweep
probed -- and hit: checkmated depth-0 nodes whose every pseudo-move is
priced above `QS` exist in ordinary corner mates, so the hypothesis is
false exactly where it was needed. -/
theorem standPatAtTerminal_mate_arm (G : QSGame) (hS : StandPatAtTerminal G)
    (p : G.Pos) (h0 : CorrectionTerminal0 G p) :
    inCheckB G.toNullGame p = false := by
  cases hic : inCheckB G.toNullGame p with
  | false => rfl
  | true =>
    have hsp := hS p h0
    simp only [terminalValue] at hsp
    rw [if_pos hic] at hsp
    exact absurd hsp h0.1

/-! # The dedicated legality probe

The d2 engine derives legality from a DEDICATED probe (lines 383, 464
on `29c7887`):

    self.bound(pos.move(m), MATE_UPPER, 0, root=True) == MATE_UPPER

An unstored driver probe (`root=True`: no table read, no store -- the
`rootProbe` semantics of `Sunfish/CanNull.lean` -- so no table entry
can enter the definition of legality), at the one window where QS
fail-soft cutoffs cannot hide the sentinel: nothing but an exact king
capture reaches `MATE_UPPER`.  `qsProbe` below models the probed
child's depth-0 loop by its content -- the stand-pat cutoff, then the
fail-soft loop over the QS-filtered moves with each grandchild reported
by its depth-0 value (the same QS-as-eval abstraction the whole file
uses; in the engine the loop is cut short by the futility break, whose
king-capture bypass `else MATE_UPPER` at line 421 preserves exactly the
property the model takes from `KingCaptureValHigh`: the capture tops
the order and outranks every threshold).

* **`legalityProbeCorrect`** -- at window `MATE_UPPER` the probe is a
  complete decision procedure: it returns `MATE_UPPER` iff the probed
  child has a king capture, i.e. iff the move that produced the child
  left the mover's own king capturable.  The easy direction is
  `KingCaptureValHigh` (+ `val_lower_lt_ML`); the hard direction is
  closed by the depth-0 pin -- at the probe's own depth every sentinel
  has static origins, the same reason `killer_probe_sound`
  (Killer.lean) is pinned at its probe depth.  At any deeper probe
  depth the hard direction would demand the sentinel-origins
  hypothesis (`MateValuesAreKingCapturesQS`), which is exactly the
  machinery the pre-d2 corrections leaned on.
* **`kingCapturableReportsExact_refuted`** -- the same loop at a
  GENERAL window is NOT exact: a king-capturable child can soundly cut
  off on its stand-pat and report a normal score (machine-checked on
  `CexR`; machine-traced on real boards as the same move scoring -368
  and -69290 across two probes).  The replacement theorem is about the
  dedicated probe, never about search reports in general -- this is the
  root cause of the open "sentinel masking" channel and the reason the
  d2 correction trusts the oracle scan, not any score-shaped sentinel.
-/

/-! ### Loop lemmas for the probe -/

/-- The fail-soft loop never exceeds a bound respected by its seed and
by every yield (mirror of `foldMax_le`). -/
theorem searchMoves_le_max {α : Type _} (gamma : Int) (f : α → Int) :
    ∀ (ms : List α) (b U : Int), (∀ m ∈ ms, f m ≤ U) → b ≤ U →
      searchMoves gamma f ms b ≤ U := by
  intro ms
  induction ms with
  | nil => intro b U _ hb; simpa [searchMoves] using hb
  | cons a ms ih =>
    intro b U hall hb
    have ha := hall a (List.mem_cons_self a ms)
    simp only [searchMoves]
    by_cases hcut : gamma ≤ max b (f a)
    · rw [if_pos hcut]; omega
    · rw [if_neg hcut]
      exact ih (max b (f a)) U (fun m hm => hall m (List.mem_cons_of_mem a hm)) (by omega)

/-- Every member forces the loop's result up to `min (f m) gamma`: the
loop either reaches `m` (result ≥ f m) or cut off before it
(result ≥ gamma). -/
theorem searchMoves_ge_min_of_mem {α : Type _} (gamma : Int) (f : α → Int) :
    ∀ (ms : List α) (b : Int) (m : α), m ∈ ms →
      min (f m) gamma ≤ searchMoves gamma f ms b := by
  intro ms
  induction ms with
  | nil => intro b m hm; cases hm
  | cons a ms ih =>
    intro b m hm
    simp only [searchMoves]
    by_cases hcut : gamma ≤ max b (f a)
    · rw [if_pos hcut]; omega
    · rw [if_neg hcut]
      cases List.mem_cons.mp hm with
      | inl he =>
        subst he
        have := searchMoves_ge_init gamma f ms (max b (f m))
        omega
      | inr ht => exact ih (max b (f a)) m ht

/-- A fail-high from a below-window seed names a fail-high yield. -/
theorem searchMoves_failHigh_witness {α : Type _} (gamma : Int) (f : α → Int) :
    ∀ (ms : List α) (b : Int), b < gamma → gamma ≤ searchMoves gamma f ms b →
      ∃ m ∈ ms, gamma ≤ f m := by
  intro ms
  induction ms with
  | nil =>
    intro b hb hge
    simp only [searchMoves] at hge
    omega
  | cons a ms ih =>
    intro b hb hge
    simp only [searchMoves] at hge
    by_cases hcut : gamma ≤ max b (f a)
    · exact ⟨a, List.mem_cons_self a ms, by omega⟩
    · rw [if_neg hcut] at hge
      obtain ⟨m, hm, hf⟩ := ih (max b (f a)) (by omega) hge
      exact ⟨m, List.mem_cons_of_mem a hm, hf⟩

/-! ### The probe and its exactness -/

/-- **EvalQuiet**: static evaluations outside the king-gone zone stay
below the mate band -- the fact `EvalBounds.evalBound_lt_MATE_LOWER`
machine-checks from the shipped tables, stated as the named hypothesis
the probe's stand-pat branch needs (a stand-pat can never fake the
`MATE_UPPER` classification). -/
def EvalQuiet (G : Game) : Prop :=
  ∀ p, ¬ (G.eval p ≤ -MATE_LOWER) → G.eval p < MATE_LOWER

/-- The probed child's depth-0 QS loop at window `w`, by content:
king-gone normalization, the stand-pat cutoff (the depth-0
`yield None, pos.score` consumed first, lines 390-391), then the
fail-soft loop over the depth-0-filtered moves seeded with the
stand-pat, each grandchild reported by its depth-0 value.  The engine
probe is `qsProbe` at `w = MATE_UPPER`, run unstored (`root=True`). -/
def qsProbe (G : QSGame) (w : Int) (c : G.Pos) : Int :=
  if G.eval c ≤ -MATE_LOWER then -MATE_UPPER
  else if w ≤ G.eval c then G.eval c
  else searchMoves w (fun m => -(negamaxQS G 0 m)) (movesAbove G (val_lower 0) c) (G.eval c)

/-- **LegalityProbeCorrect**: at window `MATE_UPPER` the probe
classifies pseudo-moves -- it reports exactly `MATE_UPPER` iff the
probed child has a king capture (iff the move that reached `c` left the
moving side's king capturable, i.e. was illegal).  Easy direction:
king captures are valued in the mate band (`KingCaptureValHigh`), so
they pass the depth-0 threshold (`val_lower_lt_ML`) and force the loop
to the exact sentinel.  Hard direction: at depth 0 every yield has
static origins -- a non-capture grandchild evaluates inside the band and
its negation cannot reach `MATE_UPPER`, and `EvalQuiet` keeps the
stand-pat below the band. -/
theorem legalityProbeCorrect (G : QSGame)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    (hV : KingCaptureValHigh G)
    (c : G.Pos) (hkg : ¬ (G.eval c ≤ -MATE_LOWER)) :
    qsProbe G MATE_UPPER c = MATE_UPPER
      ↔ hasKingCapture G.toNullGame.toGame c = true := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hq := hQ c hkg
  have hsp : ¬ (MATE_UPPER ≤ G.eval c) := by omega
  simp only [qsProbe]
  rw [if_neg hkg, if_neg hsp]
  constructor
  · intro hres
    obtain ⟨m, hm, hf⟩ := searchMoves_failHigh_witness MATE_UPPER
      (fun m => -(negamaxQS G 0 m)) (movesAbove G (val_lower 0) c) (G.eval c)
      (by omega) (by omega)
    refine (hasKingCapture_iff G.toNullGame.toGame c).mpr
      ⟨m, movesAbove_subset G (val_lower 0) c m hm, ?_⟩
    by_cases hkgm : G.eval m ≤ -MATE_LOWER
    · exact hkgm
    · exfalso
      have hqm := hQ m hkgm
      simp only [negamaxQS] at hf
      rw [if_neg hkgm] at hf
      omega
  · intro hcap
    obtain ⟨k, hk, hkev⟩ := (hasKingCapture_iff G.toNullGame.toGame c).mp hcap
    have hkmem : k ∈ movesAbove G (val_lower 0) c := by
      rw [mem_movesAbove]
      refine ⟨hk, ?_⟩
      have := hV c k hk hkev
      have := val_lower_lt_ML 0
      omega
    have hkv := negamaxQS_kingGone G 0 k hkev
    have hlow : min (-(negamaxQS G 0 k)) MATE_UPPER
        ≤ searchMoves MATE_UPPER (fun m => -(negamaxQS G 0 m))
            (movesAbove G (val_lower 0) c) (G.eval c) :=
      searchMoves_ge_min_of_mem MATE_UPPER
        (fun m => -(negamaxQS G 0 m)) (movesAbove G (val_lower 0) c) (G.eval c) k hkmem
    have hup := searchMoves_le_max MATE_UPPER
      (fun m => -(negamaxQS G 0 m)) (movesAbove G (val_lower 0) c) (G.eval c) MATE_UPPER
      (fun m _ => by
        show -(negamaxQS G 0 m) ≤ MATE_UPPER
        have := negamaxQS_bounded G hB 0 m
        omega)
      (by omega)
    rw [hkv] at hlow
    omega

/-- The probe over a parent's pseudo-moves: for `m ∈ moves p` at a
parent that cannot capture the opponent king (the only place the
correction runs -- a capturable opponent king fails high long before
it), the probe decides legality of the move to `m`.  Legality is
position-intrinsic -- `hasKingCapture` is a function of the child alone
-- so the certificate can never go stale. -/
theorem legalityProbe_decides (G : QSGame)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    (hV : KingCaptureValHigh G)
    (p m : G.Pos) (hm : m ∈ G.moves p)
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true)) :
    qsProbe G MATE_UPPER m = MATE_UPPER
      ↔ hasKingCapture G.toNullGame.toGame m = true := by
  refine legalityProbeCorrect G hB hQ hV m (fun hkg => hcap ?_)
  exact (hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, hkg⟩

/-- A fail-LOW probe report at any window `w ≤ MATE_UPPER` certifies
legality outright: had the child a king capture, the loop would have
been forced to at least `min MATE_UPPER w = w`.  This is the depth-0
half of `storedMoveLegal`: a parent's fail-high real yield is exactly a
child fail-low, so a stored move's child was scanned past every capture
without finding one. -/
theorem qsProbe_failLow_legal (G : QSGame) (hV : KingCaptureValHigh G)
    (w : Int) (c : G.Pos) (hw : w ≤ MATE_UPPER)
    (hkg : ¬ (G.eval c ≤ -MATE_LOWER))
    (hlow : qsProbe G w c < w) :
    hasKingCapture G.toNullGame.toGame c = false := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  cases hcap : hasKingCapture G.toNullGame.toGame c with
  | false => rfl
  | true =>
    exfalso
    simp only [qsProbe] at hlow
    rw [if_neg hkg] at hlow
    by_cases hsp : w ≤ G.eval c
    · rw [if_pos hsp] at hlow
      omega
    · rw [if_neg hsp] at hlow
      obtain ⟨k, hk, hkev⟩ := (hasKingCapture_iff G.toNullGame.toGame c).mp hcap
      have hkmem : k ∈ movesAbove G (val_lower 0) c := by
        rw [mem_movesAbove]
        refine ⟨hk, ?_⟩
        have := hV c k hk hkev
        have := val_lower_lt_ML 0
        omega
      have hkv := negamaxQS_kingGone G 0 k hkev
      have hlo : min (-(negamaxQS G 0 k)) w
          ≤ searchMoves w (fun m => -(negamaxQS G 0 m))
              (movesAbove G (val_lower 0) c) (G.eval c) :=
        searchMoves_ge_min_of_mem w
          (fun m => -(negamaxQS G 0 m)) (movesAbove G (val_lower 0) c) (G.eval c) k hkmem
      rw [hkv] at hlo
      omega

/-! ### The refuted exactness assumption -/

/-- **KingCapturableReportsExact** -- refuted for the UNREPAIRED loop
(`kingCapturableReportsExact_refuted` below, on the pre-kcx interior
`qsProbe` whose stand-pat could cut below the sentinel), then RESTORED
BY CONSTRUCTION in the kcx consumer
(`kingCapturableReportsExact_restored`, final section): "every
fail-soft report of a king-capturable node is the exact `MATE_UPPER`
sentinel, at every window."  `CexR` below stays as the machine-checked
record of why the repair was needed -- the -368 vs -69290 double report,
the root cause of the sentinel-masking channel the kcx landing
closed. -/
def KingCapturableReportsExact (G : QSGame) : Prop :=
  ∀ (w : Int) (c : G.Pos), -MATE_UPPER < w → w ≤ MATE_UPPER →
    ¬ (G.eval c ≤ -MATE_LOWER) → hasKingCapture G.toNullGame.toGame c = true →
    qsProbe G w c = MATE_UPPER

/-- The countermodel: `c` (eval +30) has exactly one pseudo-move -- a
king capture.  At a low window the stand-pat cuts off first. -/
inductive RPos where
  | c | k
  deriving DecidableEq

open RPos in
def CexR : QSGame where
  Pos := RPos
  moves := fun p => match p with
    | c => [k]
    | k => []
  eval := fun p => match p with
    | c => 30
    | k => -60000
  pass := fun p => p
  val := fun _ _ => MATE_LOWER

theorem cexR_bounded : Bounded CexR.toNullGame.toGame := by
  intro p
  cases p <;> decide

theorem cexR_quiet : EvalQuiet CexR.toNullGame.toGame := by
  intro p
  cases p <;> decide

theorem cexR_valHigh : KingCaptureValHigh CexR := by
  intro p m _ _
  cases p <;> cases m <;> decide

/-- The two windows, machine-checked: the same king-capturable child
reports 30 (a sound stand-pat fail-high -- its true value is
`MATE_UPPER`, and 30 is a valid lower bound) at window 10, and the
exact `MATE_UPPER` at the dedicated probe window.  A score-shaped
sentinel test reading the first report would conclude "no king capture
here"; only the dedicated probe classifies. -/
theorem cexR_two_windows :
    qsProbe CexR 10 RPos.c = 30 ∧
    qsProbe CexR MATE_UPPER RPos.c = MATE_UPPER ∧
    hasKingCapture CexR.toNullGame.toGame RPos.c = true :=
  ⟨by decide, by decide, by decide⟩

/-- **The exactness assumption is FALSE under general fail-soft
windows**, even with bounded, quiet evaluations and mate-band king
captures.  Machine-checked on `CexR` at window 10. -/
theorem kingCapturableReportsExact_refuted :
    ¬ (∀ (G : QSGame), Bounded G.toNullGame.toGame →
        EvalQuiet G.toNullGame.toGame → KingCaptureValHigh G →
        KingCapturableReportsExact G) := by
  intro h
  have := h CexR cexR_bounded cexR_quiet cexR_valHigh
    10 RPos.c (by decide) (by decide) (by decide) (by decide)
  exact absurd this (by decide)

/-! # The reference search: verify-on-suspicion, invariant by fiat

`boundD2` below models `reference.py` of the kcx landing -- the
executable spec whose EAGER ENTRY SCAN returns the exact `MATE_UPPER`
at any king-capturable node before table, repetition or loop (that
scan IS this model's by-construction king-capture branch), and whose
verify-on-suspicion machinery is the `29c7887` loop it inherited.  The
PRODUCTION consumer (`kcx-verify` at `560799c`) computes the same
function without the eager scan -- `production_eq_reference`, final
section -- so everything proven here transfers.  The loop replaces
every assumed pseudo-option inequality with a verified one:

* The consumption fold carries a one-bit certificate: `best, live` with
  `live` true iff the current maximum was set by a real-move yield
  (lines 436-439).  Modeled here without an extra bit: with the null
  contribution `n` folded first and the real-move fold `S`, `live` is
  exactly `n < S`, so the code's `not live` is the gate `S ≤ n` in
  `d2Fix`.
* **Fail-low arm** (lines 463-472): an uncertified fail-low
  (`best < gamma and not live`) is SUSPECT -- a virtual option beat
  every legal move, or no legal move exists -- and the correction
  re-derives terminality directly: every generated move (searched or
  QS-filtered alike) is probed with the legality oracle.  The scan is
  `allIllegalB` -- a predicate of the position alone -- which is what the
  engine's `all(bound(pos.move(m), MATE_UPPER, 0, root=True) ==
  MATE_UPPER ...)` computes under `legalityProbeCorrect`
  (`legalScan_iff_allIllegal`).
* **Fail-high arm** (lines 382-385): a positive uncertified null cutoff
  (`0 < score`, `gamma <= score < MATE_LOWER`, no killer) is verified
  with the same oracle and withdrawn to the fold identity `-MATE_UPPER`
  at a verified terminal (`nullVerify`).  The killer short-circuit is
  sound because a stored move is a legal move (`storedMoveLegal`), so a
  position holding one is never terminal (`KillerLegal`).  The
  mate-band suppression stays in fold-identity form on the yield
  (line 386), modeled as option disabling per the fold rule (`useD2`).
* **Depth 0 is excluded** (`if depth and ...`): QS evaluates the fold,
  stand-pat included, and never claims an exact terminal value -- the
  `StandPatAtTerminal` refutation removed, not repaired.

The decisive structural change: the correction's firing condition and
the value function's terminal branch are the SAME position-intrinsic
predicate `allIllegalB`, instead of a search-history-shaped sentinel
(`best_real == -MATE_UPPER`) on one side and a filtered-fold condition
on the other.  Alignment between search and value becomes definitional,
and the old hypothesis inventory collapses:

* CONSUMED NO MORE by the spec: `MateValuesAreKingCapturesQS` (the
  sentinel-origins hypothesis -- nothing reads score-shaped sentinels),
  `KingCaptureValHigh` (moved into the probe theorem, where the engine
  actually spends it), the whole-tree `NullBetQS`, `TerminalPseudoSafe`
  and both its refuted instances, the `qsGateB`/`allAboveB` exhaustion
  gate and its `ValFloor` arithmetic (the scan covers filtered moves
  too, so exhaustion is irrelevant to terminality).
* CONSUMED INSTEAD: `KillerLegal` (backed by `storedMoveLegal`, the
  invariant the code comment at lines 362-365 states -- now a THEOREM,
  `killerLegal_lifecycle`, given the store trace), and
  `NullBetD2` -- the null bet needed only AWAY from verified terminals,
  where its justification ("some real move matches the pass") is at
  least possible because a legal move exists.

Depth-0 idealization, stated honestly: `boundD2`'s and `negamaxD2`'s
depth 0 report the exact king-capture-aware QS value (the `MATE_UPPER`
branch), i.e. the model's depth-0 CHILD reports are sentinel-exact.
The engine's are NOT -- that is precisely the refuted
`KingCapturableReportsExact`, and the machine-checked gap (`cexR_two_
windows`) is the open "sentinel masking" channel: a king-capturable
child soundly cutting off on a partial bound can set `live` at a
terminal node and suppress the correction.  The dedicated probe is
immune (`legalityProbeCorrect` needs no such idealization), so the
correction itself is certified; the residual exposure is confined to
the `live`/fold path and is the next arc's target.  The futility-yield
caution recorded here in earlier revisions -- a truthy `Move` on an
unsearched child letting the `live` bit claim mobility the search never
earned -- turned out to be a LIVE BUG (three bench witnesses; crossed
entry `Entry(lower=0, upper=-1054)` at a stalemated child) and is now
resolved IN CODE: the kcx landing makes sub-mate futility yields
VIRTUAL (line 417), so every truthy yield is a searched real result or
the mate-case futility yield, itself a real king capture.  The model
keeps futility out of this loop as always (`boundFut` covers it; its
sub-mate arm can never cut, `score < gamma` by construction). -/

/-! ### The oracle scan -/

/-- Every generated move loses the king to an immediate recapture: the
position-intrinsic terminality predicate the d2 correction verifies
(lines 463-465 compute it through the legality probe; the scan runs
over `pos.gen_moves()`, the FULL move list -- QS filtering is irrelevant
to terminality).  Vacuously true at genuinely moveless positions. -/
def allIllegalB (G : QSGame) (p : G.Pos) : Bool :=
  (G.moves p).all (fun m => hasKingCapture G.toNullGame.toGame m)

theorem allIllegalB_true_iff {G : QSGame} {p : G.Pos} :
    allIllegalB G p = true
      ↔ ∀ m ∈ G.moves p, hasKingCapture G.toNullGame.toGame m = true := by
  simp [allIllegalB, List.all_eq_true]

theorem allIllegalB_false_of_legal {G : QSGame} {p m : G.Pos}
    (hm : m ∈ G.moves p)
    (hleg : hasKingCapture G.toNullGame.toGame m = false) :
    allIllegalB G p = false := by
  cases h : allIllegalB G p with
  | false => rfl
  | true =>
    have := allIllegalB_true_iff.mp h m hm
    rw [hleg] at this
    exact Bool.noConfusion this

/-- The engine's scan (lines 463-465) computes `allIllegalB`: under
`legalityProbeCorrect`, probing every generated move at the dedicated
window and testing for the exact sentinel is the same Boolean as "every
move leaves our king capturable".  Stated at a parent that cannot
capture the opponent king -- the only kind of node whose correction (or
null verifier) runs the scan; a capturable opponent king fails high
before either. -/
theorem legalScan_iff_allIllegal (G : QSGame)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    (hV : KingCaptureValHigh G) (p : G.Pos)
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true)) :
    ((G.moves p).all (fun m => decide (qsProbe G MATE_UPPER m = MATE_UPPER))) = true
      ↔ allIllegalB G p = true := by
  rw [List.all_eq_true, allIllegalB_true_iff]
  constructor
  · intro h m hm
    have := h m hm
    rw [decide_eq_true_eq] at this
    exact (legalityProbe_decides G hB hQ hV p m hm hcap).mp this
  · intro h m hm
    rw [decide_eq_true_eq]
    exact (legalityProbe_decides G hB hQ hV p m hm hcap).mpr (h m hm)

/-! ### The search -/

/-- The null verifier (lines 382-385): a POSITIVE pass score that would
cut off (`gamma <= score`), below the mate band, with no killer to
certify mobility, is trusted only if the oracle scan FAILS to prove the
node terminal; at a verified terminal it is withdrawn to the fold
identity `-MATE_UPPER`.  Everything else passes through untouched. -/
def nullVerify (G : QSGame) (kill : G.Pos → Bool) (rn gamma : Int) (p : G.Pos) : Int :=
  if 0 < rn ∧ gamma ≤ rn ∧ rn < MATE_LOWER ∧ kill p = false ∧ allIllegalB G p = true
  then -MATE_UPPER else rn

/-- The null-use gate: the engine guard (`abs(pos.score) < 500` and the
piece test, abstracted as `guard`), the `depth > 2` gate of line 379,
and the A1 mate-band suppression of line 386 applied to the VERIFIED
yield -- in the code the suppression is the fold-identity form
`yield None, score if score < MATE_LOWER else -MATE_UPPER`; disabling
the option is equivalent by the fold rule (formal/README.md). -/
def useD2 (G : QSGame) (guard kill : G.Pos → Bool) (rn : Int)
    (d : Nat) (p : G.Pos) (gamma : Int) : Bool :=
  guard p && decide (2 < d) &&
    decide (nullVerify G kill rn gamma p < MATE_LOWER)

/-- The null option's contribution to the consumption fold: the
verified yield when the option is enabled, the fold identity when it is
not. -/
def nullPartD2 (G : QSGame) (guard kill : G.Pos → Bool) (rn : Int)
    (d : Nat) (p : G.Pos) (gamma : Int) : Int :=
  if useD2 G guard kill rn d p gamma = true
  then nullVerify G kill rn gamma p else LOSS

/-- The verify-on-suspicion correction (lines 463-472), applied to the
consumption fold: `best` is the full maximum, `S` the real-move fold,
`n` the null contribution, so `S ≤ n` is exactly the code's `not live`
(the maximum was not set by a real-move yield).  Fire only on a
fail-low, uncertified, ORACLE-CONFIRMED terminal; then assign the exact
terminal value -- computed since `c72cf6d` by the direct
`pos.rotate(nullmove=True).king_capture()` scan, which IS `inCheckB`
(the null rotation zeroes `kp`, so the `abs(m.j - kp) < 2` disjunct is
dead there and the scan is exactly "some rotated move lands on our
king"); no probe, no `CheckProbeOK`-style premise. -/
def d2Fix (G : QSGame)
    (gamma best S n : Int) (p : G.Pos) : Int :=
  if best < gamma ∧ S ≤ n ∧ allIllegalB G p = true then
    (if inCheckB G.toNullGame p = true then -MATE_LOWER else 0)
  else best

/-- The verified search, modeling `reference.py`.  Structure of
`boundA1` with three changes:

* depth 0 reports the king-capture-aware QS value, and the interior
  king-capture branch is the reference's EAGER ENTRY SCAN;
* the null yield IS the search's own pass probe -- the term
  `-(boundD2 ... (d - 3) (G.pass p) (1 - gamma))` is part of the
  DEFINITION (the dissolved `NullIsPassSearchD2`: fidelity now lives in
  the audit row for sunfish.py's `-self.bound(pos.rotate(nullmove=True),
  1 - gamma, depth - 3)`, not in any theorem's premises) -- passed
  through the verifier (`nullVerify`) before the cutoff test and the
  fold;
* the correction is `d2Fix`: gated by the oracle scan `allIllegalB`,
  not by any score-shaped sentinel, and `best_real` is gone -- the
  `live` bit is the comparison `S ≤ n`.

The `0/1/2/d+3` pattern split keeps the recursion structural (the pass
recurses three levels down), exactly as `nullValue` in CanNull.lean;
`boundD2_succ` restores the uniform `d + 1` equation. -/
def boundD2 (G : QSGame) (guard kill : G.Pos → Bool) :
    Nat → G.Pos → Int → Int
  | 0, p, _gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else G.eval p
  | 1, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if useD2 G guard kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) 1 p gamma = true ∧
        gamma ≤ nullVerify G kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) gamma p then
      nullVerify G kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) gamma p
    else
      d2Fix G gamma
        (max (nullPartD2 G guard kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) 1 p gamma)
          (searchMoves gamma
            (fun m => -(boundD2 G guard kill 0 m (1 - gamma)))
            (movesAbove G (val_lower 1) p) LOSS))
        (searchMoves gamma
          (fun m => -(boundD2 G guard kill 0 m (1 - gamma)))
          (movesAbove G (val_lower 1) p) LOSS)
        (nullPartD2 G guard kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) 1 p gamma)
        p
  | 2, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if useD2 G guard kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) 2 p gamma = true ∧
        gamma ≤ nullVerify G kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) gamma p then
      nullVerify G kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) gamma p
    else
      d2Fix G gamma
        (max (nullPartD2 G guard kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) 2 p gamma)
          (searchMoves gamma
            (fun m => -(boundD2 G guard kill 1 m (1 - gamma)))
            (movesAbove G (val_lower 2) p) LOSS))
        (searchMoves gamma
          (fun m => -(boundD2 G guard kill 1 m (1 - gamma)))
          (movesAbove G (val_lower 2) p) LOSS)
        (nullPartD2 G guard kill (-(boundD2 G guard kill 0 (G.pass p) (1 - gamma))) 2 p gamma)
        p
  | d + 3, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if useD2 G guard kill (-(boundD2 G guard kill d (G.pass p) (1 - gamma))) (d + 3) p gamma = true ∧
        gamma ≤ nullVerify G kill (-(boundD2 G guard kill d (G.pass p) (1 - gamma))) gamma p then
      nullVerify G kill (-(boundD2 G guard kill d (G.pass p) (1 - gamma))) gamma p
    else
      d2Fix G gamma
        (max (nullPartD2 G guard kill (-(boundD2 G guard kill d (G.pass p) (1 - gamma))) (d + 3) p gamma)
          (searchMoves gamma
            (fun m => -(boundD2 G guard kill (d + 2) m (1 - gamma)))
            (movesAbove G (val_lower (d + 3)) p) LOSS))
        (searchMoves gamma
          (fun m => -(boundD2 G guard kill (d + 2) m (1 - gamma)))
          (movesAbove G (val_lower (d + 3)) p) LOSS)
        (nullPartD2 G guard kill (-(boundD2 G guard kill d (G.pass p) (1 - gamma))) (d + 3) p gamma)
        p

/-- The uniform successor equation (`d + 1 - 3` covers the three
patterns definitionally: `1-3 = 2-3 = 0`, `(d+3)-3 = d`). -/
theorem boundD2_succ (G : QSGame) (guard kill : G.Pos → Bool) (d : Nat) (p : G.Pos) (gamma : Int) :
    boundD2 G guard kill (d + 1) p gamma
      = if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
        else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
        else if useD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true ∧
            gamma ≤ nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p then
          nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
        else
          d2Fix G gamma
            (max (nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma)
              (searchMoves gamma
                (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                (movesAbove G (val_lower (d + 1)) p) LOSS))
            (searchMoves gamma
              (fun m => -(boundD2 G guard kill d m (1 - gamma)))
              (movesAbove G (val_lower (d + 1)) p) LOSS)
            (nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma)
            p := by
  match d with
  | 0 => rfl
  | 1 => rfl
  | d + 2 => rfl

/-- The value function the verified search brackets: king-capture
normalization, the sentinel branch, then -- the d2 change -- the exact
terminal value at ORACLE-TERMINAL positions (`allIllegalB`, the same
predicate the search verifies) and the plain filtered fold elsewhere.
Determined by `(pos, depth)` alone: the point-spec doctrine holds, and
no gamma-dependent condition appears anywhere in it. -/
def negamaxD2 (G : QSGame) : Nat → G.Pos → Int
  | 0, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if allIllegalB G p = true then terminalValue G p
    else foldMax (fun m => -(negamaxD2 G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS

/-- `BoundSpec` against the d2 value. -/
def BoundSpecD2 (G : QSGame) (d : Nat) (p : G.Pos) (gamma r : Int) : Prop :=
  (gamma ≤ r → r ≤ negamaxD2 G d p) ∧ (r < gamma → negamaxD2 G d p ≤ r)

/-! ### Branch lemmas -/

theorem boundD2_kingGone (G : QSGame) (guard kill : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (gamma : Int) (h : G.eval p ≤ -MATE_LOWER) :
    boundD2 G guard kill d p gamma = -MATE_UPPER := by
  cases d with
  | zero => simp only [boundD2]; rw [if_pos h]
  | succ d => rw [boundD2_succ, if_pos h]

/-- The sentinel, by construction and AT EVERY DEPTH (depth 0 included:
the sentinel-exact idealization).  Its contrapositive is
`storedMoveLegal`. -/
theorem boundD2_of_capture (G : QSGame) (guard kill : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (gamma : Int) (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    boundD2 G guard kill d p gamma = MATE_UPPER := by
  cases d with
  | zero => simp only [boundD2]; rw [if_neg hkg, if_pos hcap]
  | succ d => rw [boundD2_succ, if_neg hkg, if_pos hcap]

theorem negamaxD2_kingGone (G : QSGame) (d : Nat) (p : G.Pos)
    (h : G.eval p ≤ -MATE_LOWER) : negamaxD2 G d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [negamaxD2]; rw [if_pos h]
  | succ d => simp only [negamaxD2]; rw [if_pos h]

theorem negamaxD2_of_capture (G : QSGame) (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    negamaxD2 G d p = MATE_UPPER := by
  cases d with
  | zero => simp only [negamaxD2]; rw [if_neg hkg, if_pos hcap]
  | succ d => simp only [negamaxD2]; rw [if_neg hkg, if_pos hcap]

/-- The value-side bridge: at an oracle-terminal node the d2 value IS
the terminal value.  Search and value share the predicate, so no
`hexh`/`hmask` alignment machinery (and no `MateValuesAreKingCapturesQS`)
is needed anywhere. -/
theorem negamaxD2_of_allIllegal (G : QSGame) (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true) :
    negamaxD2 G (d + 1) p = terminalValue G p := by
  simp only [negamaxD2]
  rw [if_neg hkg, if_neg hcap, if_pos hai]

theorem negamaxD2_of_fold (G : QSGame) (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false) :
    negamaxD2 G (d + 1) p
      = foldMax (fun m => -(negamaxD2 G d m)) (movesAbove G (val_lower (d + 1)) p) LOSS := by
  simp only [negamaxD2]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai])]

/-! ### The certificates -/

/-- **StoredMoveLegal**: `tp_move` stores only on a real fail-high
(lines 440-445), and a fail-high real yield at an in-band `gamma`
certifies its move legal -- the yield is `-(child search)` ≥ gamma >
`-MATE_UPPER`, while a king-capturable child reports the exact
`MATE_UPPER` sentinel (`boundD2_of_capture`), which would negate to
exactly `-MATE_UPPER`.  Legality (`hasKingCapture` of the child) is a
function of the child position alone, so the certificate is
position-intrinsic and can never go stale, however old the entry.  This
is the code comment at lines 362-365, and the invariant `KillerLegal`
consumes. -/
theorem storedMoveLegal (G : QSGame)
    (guard kill : G.Pos → Bool)
    (d : Nat) (m : G.Pos) (gamma : Int) (hg1 : -MATE_UPPER < gamma)
    (hkg : ¬ (G.eval m ≤ -MATE_LOWER))
    (hhi : gamma ≤ -(boundD2 G guard kill d m (1 - gamma))) :
    hasKingCapture G.toNullGame.toGame m = false := by
  cases hcap : hasKingCapture G.toNullGame.toGame m with
  | false => rfl
  | true =>
    exfalso
    rw [boundD2_of_capture G guard kill d m (1 - gamma) hkg hcap] at hhi
    omega

/-- The engine-level depth-0 instance of `storedMoveLegal`, free of the
sentinel-exact idealization: a depth-1 parent's fail-high real yield is
a fail-LOW of the child's depth-0 QS loop, and `qsProbe_failLow_legal`
certifies legality from the fail-low alone (the loop was scanned past
every king capture without cutting off). -/
theorem storedMoveLegal_qs (G : QSGame) (hV : KingCaptureValHigh G)
    (m : G.Pos) (gamma : Int) (hg1 : -MATE_UPPER < gamma)
    (hkg : ¬ (G.eval m ≤ -MATE_LOWER))
    (hhi : gamma ≤ -(qsProbe G (1 - gamma) m)) :
    hasKingCapture G.toNullGame.toGame m = false :=
  qsProbe_failLow_legal G hV (1 - gamma) m (by omega) hkg (by omega)

/-- **KillerLegal**: away from king-capturable nodes, any position the
killer table holds a move for has a legal move.  This is the invariant
`storedMoveLegal` maintains -- `tp_move` is exact-position-keyed and its
store paths are real fail-high winners at in-band windows (plus, in the
kcx production consumer, the SUBSTITUTED king capture, which only
occurs at king-capturable nodes and is why the invariant is stated
conditionally: at a king-capturable node the stored move is a king
capture, `KillerAtKingCapturable`, Killer.lean's territory -- and no
verification logic ever consults the killer there, since such nodes
fail high before any scan).  The futility `MATE_UPPER` store is also a
king capture, so it too lives outside the conditional.  Consumed by
the reference verifier's `not killer` short-circuit: a killer
certifies mobility, so the verification scan may be skipped. -/
def KillerLegal (G : QSGame) (kill : G.Pos → Bool) : Prop :=
  ∀ p, ¬ (G.eval p ≤ -MATE_LOWER) →
    hasKingCapture G.toNullGame.toGame p = false → kill p = true →
    ∃ m ∈ G.moves p, hasKingCapture G.toNullGame.toGame m = false

/-- A (non-king-capturable) position holding a killer is never
oracle-terminal: the `not killer` short-circuit skips only scans that
would have found a legal move. -/
theorem killerLegal_not_terminal (G : QSGame) (kill : G.Pos → Bool)
    (hK : KillerLegal G kill) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (h : kill p = true) :
    allIllegalB G p = false := by
  obtain ⟨m, hm, hleg⟩ := hK p hkg hcapf h
  exact allIllegalB_false_of_legal hm hleg

/-! ### The null-side hypotheses -/

/-- **NullAtMate -- a theorem, with nothing to assume at all now**: at
an in-check node a suppression-passing pass probe is exactly the fold
identity.  Passing while in check loses the king -- the passed position
is king-capturable, so the (now definitional) pass search reports the
exact sentinel through the reference's eager branch; the one other
conceivable case (the passed king already gone, negating to
`+MATE_UPPER`) is excluded by the suppression itself. -/
theorem nullAtMateD2 (G : QSGame) (guard kill : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hic : inCheckB G.toNullGame p = true)
    (hsup : -(boundD2 G guard kill d (G.pass p) gamma) < MATE_LOWER) :
    -(boundD2 G guard kill d (G.pass p) gamma) = -MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hcap : hasKingCapture G.toNullGame.toGame (G.pass p) = true := hic
  by_cases hkg : G.eval (G.pass p) ≤ -MATE_LOWER
  · exfalso
    rw [boundD2_kingGone G guard kill d (G.pass p) gamma hkg] at hsup
    omega
  · rw [boundD2_of_capture G guard kill d (G.pass p) gamma hkg hcap]

/-! ### Boundedness -/

/-- Search reports stay in the score band, for in-band windows.  (The
lower side of the null oracle needs no hypothesis: an out-of-band pass
yield is either suppressed or dominated by the fold.) -/
theorem boundD2_bounded (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int), -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      -MATE_UPPER ≤ boundD2 G guard kill d p gamma ∧
        boundD2 G guard kill d p gamma ≤ MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d with
  | zero =>
    intro p gamma _ _
    have hband := hB p
    simp only [boundD2]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]; omega
    · rw [if_neg hkg]
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [if_pos hcap]; omega
      · rw [if_neg hcap]; omega
  | succ d ih =>
    intro p gamma hg1 hg2
    rw [boundD2_succ]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]; omega
    · rw [if_neg hkg]
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [if_pos hcap]; omega
      · rw [if_neg hcap]
        by_cases hcut : useD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true ∧
            gamma ≤ nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
        · rw [if_pos hcut]
          have hu := hcut.1
          simp only [useD2, Bool.and_eq_true, decide_eq_true_eq] at hu
          omega
        · rw [if_neg hcut]
          have hSl := searchMoves_ge_init gamma
            (fun m => -(boundD2 G guard kill d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS
          have hSu := searchMoves_le_max gamma
            (fun m => -(boundD2 G guard kill d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS MATE_UPPER
            (fun m _ => by
              show -(boundD2 G guard kill d m (1 - gamma)) ≤ MATE_UPPER
              have := ih m (1 - gamma) (by omega) (by omega)
              omega)
            (by omega)
          have hn : nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma ≤ MATE_LOWER := by
            simp only [nullPartD2]
            by_cases hu : useD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true
            · rw [if_pos hu]
              simp only [useD2, Bool.and_eq_true, decide_eq_true_eq] at hu
              omega
            · rw [if_neg hu]; omega
          simp only [d2Fix]
          by_cases hfire :
              max (nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma)
                (searchMoves gamma
                  (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                  (movesAbove G (val_lower (d + 1)) p) LOSS) < gamma ∧
              searchMoves gamma
                (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                (movesAbove G (val_lower (d + 1)) p) LOSS
                ≤ nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma ∧
              allIllegalB G p = true
          · rw [if_pos hfire]
            by_cases hpr : inCheckB G.toNullGame p = true
            · rw [if_pos hpr]; omega
            · rw [if_neg hpr]; omega
          · rw [if_neg hfire]; omega

/-! ### The two verifier arms, as standalone theorems -/

/-- **PositiveNullCutoffVerified** (fail-high arm): at an
oracle-terminal node, any null cutoff that survives the verifier is
NON-POSITIVE -- so a verified-terminal node never stores a positive
lower bound.  Mechanics: a killer cannot exist there (`KillerLegal` +
the scan), so a surviving positive cutoff would have satisfied every
withdrawal conjunct and been withdrawn to the fold identity, which is
non-positive outright. -/
theorem positiveNullCutoffVerified (G : QSGame)
    (guard kill : G.Pos → Bool) (rn : Int)
    (hK : KillerLegal G kill)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hai : allIllegalB G p = true)
    (hu : useD2 G guard kill rn d p gamma = true)
    (hge : gamma ≤ nullVerify G kill rn gamma p) :
    nullVerify G kill rn gamma p ≤ 0 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hkf : kill p = false := by
    cases hk : kill p with
    | false => rfl
    | true =>
      have := killerLegal_not_terminal G kill hK p hkg hcapf hk
      rw [hai] at this
      exact Bool.noConfusion this
  simp only [useD2, Bool.and_eq_true, decide_eq_true_eq] at hu
  obtain ⟨⟨hgu, hd2⟩, hsup⟩ := hu
  by_cases hw : 0 < rn ∧ gamma ≤ rn ∧
      rn < MATE_LOWER ∧ kill p = false ∧ allIllegalB G p = true
  · have hv : nullVerify G kill rn gamma p = -MATE_UPPER := by
      simp only [nullVerify]; rw [if_pos hw]
    rw [hv]
    omega
  · have hv : nullVerify G kill rn gamma p = rn := by
      simp only [nullVerify]; rw [if_neg hw]
    rw [hv] at hge hsup ⊢
    by_cases hpos : 0 < rn
    · exact absurd ⟨hpos, hge, hsup, hkf, hai⟩ hw
    · omega

/-- **NegativeFailLowVerified** (fail-low arm), exactness half: when the
correction fires -- fail-low, uncertified (`S ≤ n`, the code's `not
live`), and ORACLE-CONFIRMED terminal -- it returns exactly the d2
value.  Under `legalityProbeCorrect` the engine's scan computes the
same confirmation, so the correction is exact, not a bet. -/
theorem negativeFailLowVerified (G : QSGame)
    (d : Nat) (p : G.Pos) (gamma best S n : Int)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true)
    (hbest : best < gamma) (hlive : S ≤ n) :
    d2Fix G gamma best S n p = negamaxD2 G (d + 1) p := by
  have hfix : d2Fix G gamma best S n p
      = (if inCheckB G.toNullGame p = true then -MATE_LOWER else 0) := by
    simp only [d2Fix]; rw [if_pos ⟨hbest, hlive, hai⟩]
  rw [hfix, negamaxD2_of_allIllegal G d p hkg hcap hai]
  simp only [terminalValue]

/-- **NegativeFailLowVerified**, guard half: without the oracle's
confirmation an uncertified fail-low is NOT converted to a terminal
value -- the fold result passes through untouched.  (The pre-d2 design
converted on a score-shaped sentinel instead; `qsUngated_not_sound` and
`a1_unfixed_not_sound` above are what that cost.) -/
theorem d2Fix_unverified_passthrough (G : QSGame)
    (gamma best S n : Int) (p : G.Pos)
    (hai : allIllegalB G p = false) :
    d2Fix G gamma best S n p = best := by
  simp only [d2Fix]
  rw [if_neg (fun h => by rw [hai] at h; exact Bool.noConfusion h.2.2)]

/-! ### The point spec -/

/-- The pointwise heart of the d2 correction.  Contrast with
`a1Fix_spec_core`: the gate is the SHARED, position-intrinsic
`allIllegalB`, so instead of the exhaustion/masking alignment
(`hexh`/`hmask`, which is where `MateValuesAreKingCapturesQS` used to
enter), the lemma needs only: the null part sits below the window
(`hnlt`, else the cutoff branch would have fired), and at a terminal
the fold is the identity, the null part is no smaller, and the value is
the terminal value (`hterm`). -/
theorem d2Fix_spec_core (G : QSGame) (p : G.Pos)
    (gamma S n V : Int)
    (hnlt : n < gamma)
    (hterm : allIllegalB G p = true → S = LOSS ∧ LOSS ≤ n ∧ V = terminalValue G p)
    (hsp1 : allIllegalB G p = false → gamma ≤ S → S ≤ V)
    (hsp2 : allIllegalB G p = false → S < gamma → V ≤ max n S) :
    (gamma ≤ d2Fix G gamma (max n S) S n p →
      d2Fix G gamma (max n S) S n p ≤ V) ∧
    (d2Fix G gamma (max n S) S n p < gamma →
      V ≤ d2Fix G gamma (max n S) S n p) := by
  cases hai : allIllegalB G p with
  | true =>
    obtain ⟨hSL, hnge, hV⟩ := hterm hai
    have hfire : max n S < gamma ∧ S ≤ n ∧ allIllegalB G p = true :=
      ⟨by omega, by omega, hai⟩
    have hfix : d2Fix G gamma (max n S) S n p
        = (if inCheckB G.toNullGame p = true then -MATE_LOWER else 0) := by
      simp only [d2Fix]; rw [if_pos hfire]
    rw [hfix, hV]
    simp only [terminalValue]
    exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
  | false =>
    have hfix : d2Fix G gamma (max n S) S n p = max n S :=
      d2Fix_unverified_passthrough G gamma (max n S) S n p hai
    rw [hfix]
    constructor
    · intro hge
      have h1 := hsp1 hai (by omega)
      omega
    · intro hlt
      have h2 := hsp2 hai (by omega)
      omega

/-- The verified loop on the countermodel that broke the old one
(`cexT_crossing`): with the SAME +30 oracle at the SAME material-up
stalemate, both windows now return the exact 0 -- the low probe's pass
is withdrawn by the verifier (no killer can exist at a moveless
position), the fold stays at the identity, and the oracle-confirmed
correction stores the draw. -/
theorem d2_repairs_cexT :
    boundD2 CexT (fun _ => true) (fun _ => false) 4 () 10 = 0 ∧
    boundD2 CexT (fun _ => true) (fun _ => false) 4 () 100 = 0 :=
  ⟨by decide, by decide⟩

/-! # The two-layer spec: the fold defines the semantics

Restructure of the retired `NullBetD2`-carrying spec (decision: Thomas).

* **Layer 1** (`bound_null_spec`): the docstring bracketing property
  against the NULL-INCLUSIVE declared value function `nullValueD2` --
  the negamax whose non-terminal nodes include the pass term (the
  option exists by DEFINITION, not assumption) and whose
  oracle-terminal nodes are the verified exact values.  No null BET
  anywhere in the premises; what remains is fidelity (`Bounded`,
  `CheckProbeQuiet` -- DISCHARGED for the shipped probe by
  `checkProbe_discharged` -- and `KillerLegal`, itself a theorem given
  the store trace), the driver window range (`Driver.lean` proves what
  the bisection guarantees), and ONE chess-position statement whose
  necessity is genuine and documented below (`NoZugzwangInMateBand`).
  The pass term needs no hypothesis at all: it is part of the
  definition.
* **Layer 2** (`nullValue_eq_realValue_of_noZugzwang`): the
  null-inclusive function equals the draw-aware REAL-MOVE value
  (`negamaxD2`) under `NoZugzwang` -- the pass term never beats the
  best real move.  Zugzwang lives here, stated once, as the validity
  region of the approximation; the chess-facing `boundD2_spec` is the
  corollary of layer 1 + layer 2.

WHY ONE BAND PREMISE CANNOT LEAVE LAYER 1 (checked against every
candidate definition, including the mate-band CAP): the suppression
tests the pass REPORT for band membership, but a gamma-free declared
function can only test the pass VALUE -- and fail-soft reports may
straddle the band across windows.  Concretely: let the pass position
be "opponent mated in 2" (value exactly `-MATE_LOWER`, an ordinary
chess shape) whose own search cuts off at one window on a sub-band
partial bound (reporting, say, -900) and sees the mate at another.
Now run the two windows against the SAME node:

* probe A (low window): the pass yield is 900, sub-band, so it
  survives suppression, cuts off, and STORES `lower = 900`;
* probe B (high window): the pass reports the mate band, is folded to
  the identity, the real moves fold to some small `S`, and the node
  STORES `upper = S`.

If the real-move fold is genuinely below 900 -- mate-band zugzwang,
"passing force-mates but no real move comes close" -- the two stores
are `lower = 900 > S = upper`: A REAL CROSSED ENTRY IN THE SHIPPED
ENGINE, before any model is chosen.  No declared-function definition
can dissolve a definition-independent crossing; a premise excluding
the position class is forced, and `NoZugzwangInMateBand` is its
weakest natural form: *if passing wins in the mate band, some real
move reaches the band too* -- the code comment's own redundancy
argument promoted from one ply (where it is the THEOREM
`nullAtMateD2` / the kcx substitution arm) to the band (where it is
chess, not logic).  For the record, the candidate definitions
apportion the SAME obstruction, they do not remove it: declining
band-valued pass terms (the definition used here) needs the premise
for probe A's cutoff; CAPPING the pass term at `MATE_LOWER - 1`
(min-with-band-edge, position-determined and gamma-free) makes probe
A's store sound by definition but turns probe B's fail-low upper into
the unsound side (the declared value would carry the capped
`MATE_LOWER - 1` term the identity-fold withheld) -- the requirement
"T ≥ every admissible sub-band cutoff" against "T ≤ every possible
fail-low upper" pins T to both ends of the band at the same position
class, which is contradictory.  It is a strict weakening of the
retired bet -- nothing sub-band, nothing about accuracy -- and it is
implied by layer 2's `NoZugzwang`
(`noZugzwangInMateBand_of_noZugzwang`), so the corollary carries a
SINGLE chess assumption.  Partial upgrade under kcx, for the record:
with reports exact at king-capturable nodes and pass-of-pass reading
back as king-capturability, the band premise is dischargeable for pass
depths ≤ 2 (band pass VALUES there force either a same-board capture
-- excluded by the branch -- or an oracle-terminal pass, which reads
back through the involution); from pass depth 3 up a "mate in 2 after
passing" value built from an in-check terminal two plies down needs no
same-board capture, and the premise is genuinely chess.

WHY THE DRIVER RANGE IS PART OF THE STATEMENT: outside
`(-MATE_LOWER, MATE_LOWER]` (sunfish.py lines 506-510: MTD-bi probes
only this range, and the null-window flip `gamma ↦ 1 - gamma` preserves
it) the suppression could discard a LEGITIMATE sub-band pass term
through the same straddle run in the other direction -- a band report
over a sub-band value fails high only at windows beyond the band. -/

/-! ### Fold lemmas -/

/-- Splitting the fold's initial accumulator off. -/
theorem foldMax_init_split {α : Type _} (w : α → Int) :
    ∀ (ms : List α) (a : Int), LOSS ≤ a →
      foldMax w ms a = max a (foldMax w ms LOSS) := by
  intro ms
  induction ms with
  | nil =>
    intro a ha
    simp only [foldMax]
    omega
  | cons m ms ih =>
    intro a ha
    simp only [foldMax]
    rw [ih (max a (w m)) (by omega), ih (max LOSS (w m)) (by omega)]
    omega

/-- Pointwise-equal weights fold identically. -/
theorem foldMax_congr {α : Type _} (w1 w2 : α → Int) :
    ∀ (ms : List α) (i : Int), (∀ m ∈ ms, w1 m = w2 m) →
      foldMax w1 ms i = foldMax w2 ms i := by
  intro ms
  induction ms with
  | nil => intro i _; rfl
  | cons m ms ih =>
    intro i h
    simp only [foldMax]
    rw [h m (List.mem_cons_self m ms)]
    exact ih _ (fun x hx => h x (List.mem_cons_of_mem m hx))

/-- Pointwise-equal yields search identically. -/
theorem searchMoves_congr {α : Type _} (gamma : Int) (f1 f2 : α → Int) :
    ∀ (ms : List α) (b : Int), (∀ m ∈ ms, f1 m = f2 m) →
      searchMoves gamma f1 ms b = searchMoves gamma f2 ms b := by
  intro ms
  induction ms with
  | nil => intro b _; rfl
  | cons m ms ih =>
    intro b h
    simp only [searchMoves]
    rw [h m (List.mem_cons_self m ms)]
    by_cases hcut : gamma ≤ max b (f2 m)
    · rw [if_pos hcut, if_pos hcut]
    · rw [if_neg hcut, if_neg hcut]
      exact ih _ (fun x hx => h x (List.mem_cons_of_mem m hx))

/-! ### The null-inclusive declared value function -/

/-- **The null-inclusive declared value function** (layer 1's subject):
king-capture normalization and the exact sentinel as always; the
verified exact terminal value at oracle-terminal nodes; and at every
other node the plain filtered fold whose INITIAL ACCUMULATOR is the
pass term -- the pass's own declared value at `depth - 3`, admitted
below the mate band and declined (fold identity) inside it, which is
the fold-rule, value-side reading of the suppression.  Determined by
`(pos, depth)` alone.

Position-determinedness despite the search's verifier consulting the
killer table: this function consults `kill` nowhere.  `KillerLegal`
makes oracle-terminal (non-king-capturable) nodes killer-free, so the
search's verifier fires deterministically exactly where this
function's terminal branch is; at non-terminal nodes the verifier is
the identity regardless of the killer. -/
def nullValueD2 (G : QSGame) (guard : G.Pos → Bool) : Nat → G.Pos → Int
  | 0, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if allIllegalB G p = true then terminalValue G p
    else
      foldMax (fun m => -(nullValueD2 G guard d m)) (movesAbove G (val_lower (d + 1)) p)
        (if guard p = true ∧ 2 < d + 1 then
          (if -(nullValueD2 G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER then
            max LOSS (-(nullValueD2 G guard (d + 1 - 3) (G.pass p)))
          else LOSS)
        else LOSS)
termination_by d _ => d
decreasing_by all_goals omega

/-- The pass term (the declared fold's initial accumulator), named. -/
def nullTermD2 (G : QSGame) (guard : G.Pos → Bool) (d : Nat) (p : G.Pos) : Int :=
  if guard p = true ∧ 2 < d + 1 then
    (if -(nullValueD2 G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER then
      max LOSS (-(nullValueD2 G guard (d + 1 - 3) (G.pass p)))
    else LOSS)
  else LOSS

theorem nullValueD2_kingGone (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (h : G.eval p ≤ -MATE_LOWER) :
    nullValueD2 G guard d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [nullValueD2]; rw [if_pos h]
  | succ d => simp only [nullValueD2]; rw [if_pos h]

theorem nullValueD2_of_capture (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    nullValueD2 G guard d p = MATE_UPPER := by
  cases d with
  | zero => simp only [nullValueD2]; rw [if_neg hkg, if_pos hcap]
  | succ d => simp only [nullValueD2]; rw [if_neg hkg, if_pos hcap]

theorem nullValueD2_of_allIllegal (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true) :
    nullValueD2 G guard (d + 1) p = terminalValue G p := by
  simp only [nullValueD2]
  rw [if_neg hkg, if_neg hcap, if_pos hai]

theorem nullValueD2_of_fold (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false) :
    nullValueD2 G guard (d + 1) p
      = foldMax (fun m => -(nullValueD2 G guard d m))
          (movesAbove G (val_lower (d + 1)) p) (nullTermD2 G guard d p) := by
  simp only [nullValueD2]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai])]
  rfl

theorem nullTermD2_ge_LOSS (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) : LOSS ≤ nullTermD2 G guard d p := by
  simp only [nullTermD2]
  by_cases h1 : guard p = true ∧ 2 < d + 1
  · rw [if_pos h1]
    by_cases h2 : -(nullValueD2 G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
    · rw [if_pos h2]; omega
    · rw [if_neg h2]; omega
  · rw [if_neg h1]; omega

/-- The declared value stays in the score band. -/
theorem nullValueD2_bounded (G : QSGame) (guard : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) :
    ∀ (d : Nat) (p : G.Pos),
      -MATE_UPPER ≤ nullValueD2 G guard d p ∧ nullValueD2 G guard d p ≤ MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p
    cases d with
    | zero =>
      have hband := hB p
      simp only [nullValueD2]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]
        by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [if_pos hcap]; omega
        · rw [if_neg hcap]; omega
    | succ d =>
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [nullValueD2_kingGone G guard (d + 1) p hkg]; omega
      · by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [nullValueD2_of_capture G guard (d + 1) p hkg hcap]; omega
        · cases hai : allIllegalB G p with
          | true =>
            rw [nullValueD2_of_allIllegal G guard d p hkg hcap hai]
            simp only [terminalValue]
            by_cases hic : inCheckB G.toNullGame p = true
            · rw [if_pos hic]; omega
            · rw [if_neg hic]; omega
          | false =>
            rw [nullValueD2_of_fold G guard d p hkg hcap hai]
            have hTl := nullTermD2_ge_LOSS G guard d p
            have hTu : nullTermD2 G guard d p ≤ MATE_UPPER := by
              simp only [nullTermD2]
              by_cases h1 : guard p = true ∧ 2 < d + 1
              · rw [if_pos h1]
                have := ih (d + 1 - 3) (by omega) (G.pass p)
                by_cases h2 : -(nullValueD2 G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
                · rw [if_pos h2]; omega
                · rw [if_neg h2]; omega
              · rw [if_neg h1]; omega
            have hfl := foldMax_ge_init (fun m => -(nullValueD2 G guard d m))
              (movesAbove G (val_lower (d + 1)) p) (nullTermD2 G guard d p)
            have hfu : foldMax (fun m => -(nullValueD2 G guard d m))
                (movesAbove G (val_lower (d + 1)) p) (nullTermD2 G guard d p)
                ≤ MATE_UPPER := by
              refine foldMax_le _ _ _ (fun m _ => ?_) hTu
              show -(nullValueD2 G guard d m) ≤ MATE_UPPER
              have := ih d (by omega) m
              omega
            omega

/-! ### The zugzwang predicates -/

/-- **NoZugzwang** (layer 2's validity region, Thomas's phrasing:
"pass-value ≤ best real move"): at every node where the pass option
exists, the RAW pass term never strictly beats the real-move fold.
Stated once, against the declared function itself; zugzwang is
precisely its failure, and threatens only the accuracy of the
null-inclusive approximation -- never the search/table consistency
(layer 1 is spec'd against the null-inclusive function directly). -/
def NoZugzwang (G : QSGame) (guard : G.Pos → Bool) : Prop :=
  ∀ (d : Nat) (p : G.Pos),
    ¬ (G.eval p ≤ -MATE_LOWER) →
    ¬ (hasKingCapture G.toNullGame.toGame p = true) →
    allIllegalB G p = false → guard p = true → 2 < d + 1 →
    -(nullValueD2 G guard (d + 1 - 3) (G.pass p))
      ≤ foldMax (fun m => -(nullValueD2 G guard d m))
          (movesAbove G (val_lower (d + 1)) p) LOSS

/-- **NoZugzwangInMateBand** (layer 1's one chess premise): if PASSING
wins in the mate band, some real move reaches the band too -- you
cannot be in zugzwang while delivering forced mate.  The module
comment above records both why layer 1 provably cannot shed it (the
suppression is report-keyed; the -900/`-MATE_LOWER` straddle) and how
far kcx discharges it (pass depths ≤ 2). -/
def NoZugzwangInMateBand (G : QSGame) (guard : G.Pos → Bool) : Prop :=
  ∀ (d : Nat) (p : G.Pos),
    ¬ (G.eval p ≤ -MATE_LOWER) →
    ¬ (hasKingCapture G.toNullGame.toGame p = true) →
    allIllegalB G p = false → guard p = true → 2 < d + 1 →
    nullValueD2 G guard (d + 1 - 3) (G.pass p) ≤ -MATE_LOWER →
    MATE_LOWER ≤ foldMax (fun m => -(nullValueD2 G guard d m))
        (movesAbove G (val_lower (d + 1)) p) LOSS

/-- The band premise is the mate-band fragment of `NoZugzwang`. -/
theorem noZugzwangInMateBand_of_noZugzwang (G : QSGame) (guard : G.Pos → Bool)
    (hZ : NoZugzwang G guard) : NoZugzwangInMateBand G guard := by
  have hML : MATE_LOWER = 47923 := rfl
  intro d p hkg hcap hai hgu hd2 hband
  have := hZ d p hkg hcap hai hgu hd2
  omega

/-! ### Layer 1: the spec against the null-inclusive function -/

/-- **bound_null_spec** (layer 1): the search brackets its OWN declared
value function -- the null-inclusive `nullValueD2` -- with no null bet
anywhere, and no pass-search hypothesis anywhere (the pass term is the
search's own recursion).  Premises: `Bounded` (fidelity),
`CheckProbeQuiet` (discharged for the shipped probe by
`checkProbe_discharged`), `KillerLegal` (a theorem given the store
trace, `killerLegal_lifecycle`), driver windows (`Driver.lean`), and
the single chess-position statement `NoZugzwangInMateBand` (see the
section comment for the machine-level scenario showing it cannot be
dropped while the suppression is report-keyed).  Sorry-free, by strong
induction (the pass recursion sits at `depth - 3`). -/
theorem bound_null_spec (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hK : KillerLegal G kill)
    (hR : NoZugzwangInMateBand G guard) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_LOWER < gamma → gamma ≤ MATE_LOWER →
      (gamma ≤ boundD2 G guard kill d p gamma →
        boundD2 G guard kill d p gamma ≤ nullValueD2 G guard d p) ∧
      (boundD2 G guard kill d p gamma < gamma →
        nullValueD2 G guard d p ≤ boundD2 G guard kill d p gamma) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p gamma hg1 hg2
    cases d with
    | zero =>
      simp only [boundD2, nullValueD2]
      exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
    | succ d =>
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [boundD2_kingGone G guard kill (d + 1) p gamma hkg,
          nullValueD2_kingGone G guard (d + 1) p hkg]
        exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
      · by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [boundD2_of_capture G guard kill (d + 1) p gamma hkg hcap,
            nullValueD2_of_capture G guard (d + 1) p hkg hcap]
          exact ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩
        · have hcapf : hasKingCapture G.toNullGame.toGame p = false := by
            cases h : hasKingCapture G.toNullGame.toGame p
            · rfl
            · exact absurd h hcap
          by_cases hcut : useD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true ∧
              gamma ≤ nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
          · -- The (verified) null cutoff.
            have hs : boundD2 G guard kill (d + 1) p gamma
                = nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p := by
              rw [boundD2_succ, if_neg hkg, if_neg hcap, if_pos hcut]
            rw [hs]
            cases hai : allIllegalB G p with
            | true =>
              -- Verified terminal: the surviving cutoff is non-positive,
              -- and the mate side cannot cut off at all.
              have h0 := positiveNullCutoffVerified G guard kill _ hK
                (d + 1) p gamma hkg hcapf hai hcut.1 hcut.2
              cases hic : inCheckB G.toNullGame p with
              | true =>
                exfalso
                by_cases hw : 0 < (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
                    gamma ≤ (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
                    (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) < MATE_LOWER ∧ kill p = false ∧
                    allIllegalB G p = true
                · have hv : nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                      = -MATE_UPPER := by
                    simp only [nullVerify]; rw [if_pos hw]
                  have h := hcut.2
                  rw [hv] at h
                  omega
                · have hv : nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                      = (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) := by
                    simp only [nullVerify]; rw [if_neg hw]
                  have hu := hcut.1
                  simp only [useD2, Bool.and_eq_true, decide_eq_true_eq] at hu
                  obtain ⟨⟨hgu, hd2⟩, hsup⟩ := hu
                  rw [hv] at hsup
                  have hmate := nullAtMateD2 G guard kill
                    (d + 1 - 3) p (1 - gamma) hic hsup
                  have h := hcut.2
                  rw [hv, hmate] at h
                  omega
              | false =>
                have hv : nullValueD2 G guard (d + 1) p = 0 := by
                  rw [nullValueD2_of_allIllegal G guard d p hkg hcap hai]
                  simp only [terminalValue]
                  rw [if_neg (by simp [hic])]
                rw [hv]
                exact ⟨fun _ => h0, fun hlt => absurd hcut.2 (by omega)⟩
            | false =>
              -- Non-terminal cutoff: the pass IH bounds the report by the
              -- pass value; below the band the declared term admits it,
              -- inside the band the redundancy premise covers it.
              have hnv : nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                  = (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) := by
                simp only [nullVerify]
                rw [if_neg (fun h => by rw [hai] at h; exact Bool.noConfusion h.2.2.2.2)]
              have hu := hcut.1
              simp only [useD2, Bool.and_eq_true, decide_eq_true_eq] at hu
              obtain ⟨⟨hgu, hd2⟩, hsup⟩ := hu
              rw [hnv] at hsup
              have hge := hcut.2
              rw [hnv] at hge
              rw [hnv]
              have hpass := ih (d + 1 - 3) (by omega) (G.pass p) (1 - gamma)
                (by omega) (by omega)
              have hplow : boundD2 G guard kill (d + 1 - 3)
                  (G.pass p) (1 - gamma) < 1 - gamma := by omega
              have hraw := hpass.2 hplow
              rw [nullValueD2_of_fold G guard d p hkg hcap hai]
              constructor
              · intro _
                by_cases hml : -(nullValueD2 G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
                · have hT : nullTermD2 G guard d p
                      = max LOSS (-(nullValueD2 G guard (d + 1 - 3) (G.pass p))) := by
                    simp only [nullTermD2]
                    rw [if_pos ⟨hgu, hd2⟩, if_pos hml]
                  rw [hT]
                  have hinit := foldMax_ge_init (fun m => -(nullValueD2 G guard d m))
                    (movesAbove G (val_lower (d + 1)) p)
                    (max LOSS (-(nullValueD2 G guard (d + 1 - 3) (G.pass p))))
                  omega
                · have hband : nullValueD2 G guard (d + 1 - 3) (G.pass p)
                      ≤ -MATE_LOWER := by omega
                  have hF := hR d p hkg hcap hai hgu hd2 hband
                  have hT : nullTermD2 G guard d p = LOSS := by
                    simp only [nullTermD2]
                    rw [if_pos ⟨hgu, hd2⟩, if_neg hml]
                  rw [hT]
                  omega
              · intro hlt
                exact absurd hge (by omega)
          · -- The loop and the correction, through the core lemma.
            have hs : boundD2 G guard kill (d + 1) p gamma
                = d2Fix G gamma
                    (max (nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma)
                      (searchMoves gamma
                        (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                        (movesAbove G (val_lower (d + 1)) p) LOSS))
                    (searchMoves gamma
                      (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                      (movesAbove G (val_lower (d + 1)) p) LOSS)
                    (nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma)
                    p := by
              rw [boundD2_succ, if_neg hkg, if_neg hcap, if_neg hcut]
            rw [hs]
            have hchild : ∀ m : G.Pos,
                (gamma ≤ -(boundD2 G guard kill d m (1 - gamma)) →
                  -(boundD2 G guard kill d m (1 - gamma))
                    ≤ -(nullValueD2 G guard d m)) ∧
                (-(boundD2 G guard kill d m (1 - gamma)) < gamma →
                  -(nullValueD2 G guard d m)
                    ≤ -(boundD2 G guard kill d m (1 - gamma))) := by
              intro m
              have h1 := (ih d (by omega) m (1 - gamma) (by omega) (by omega)).1
              have h2 := (ih d (by omega) m (1 - gamma) (by omega) (by omega)).2
              constructor
              · intro hge'
                have := h2 (by omega)
                omega
              · intro hlt'
                have := h1 (by omega)
                omega
            have hloop := searchMoves_spec gamma
              (fun m => -(boundD2 G guard kill d m (1 - gamma)))
              (fun m => -(nullValueD2 G guard d m))
              hchild (movesAbove G (val_lower (d + 1)) p) LOSS LOSS
              (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
            refine d2Fix_spec_core G p gamma _ _ _ ?_ ?_ ?_ ?_
            · -- The null part sits below the window (else the cutoff fired).
              simp only [nullPartD2]
              by_cases hu : useD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true
              · rw [if_pos hu]
                by_cases hge : gamma ≤ nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                · exact absurd ⟨hu, hge⟩ hcut
                · omega
              · rw [if_neg hu]; omega
            · -- At a verified terminal: fold at the identity, null part no
              -- smaller, value the terminal value.
              intro hai
              refine ⟨?_, ?_, nullValueD2_of_allIllegal G guard d p hkg hcap hai⟩
              · have hall : ∀ m ∈ movesAbove G (val_lower (d + 1)) p,
                    -(boundD2 G guard kill d m (1 - gamma)) ≤ LOSS := by
                  intro m hm
                  have hmm := movesAbove_subset G (val_lower (d + 1)) p m hm
                  have hcm : hasKingCapture G.toNullGame.toGame m = true :=
                    allIllegalB_true_iff.mp hai m hmm
                  have hmkg : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
                    hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hmm, hh⟩)
                  rw [boundD2_of_capture G guard kill d m (1 - gamma) hmkg hcm]
                  omega
                exact searchMoves_eq_init gamma
                  (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                  (movesAbove G (val_lower (d + 1)) p) LOSS hall (by omega)
              · simp only [nullPartD2]
                by_cases hu : useD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true
                · rw [if_pos hu]
                  simp only [useD2, Bool.and_eq_true, decide_eq_true_eq] at hu
                  obtain ⟨⟨_, _⟩, _⟩ := hu
                  have hpb := boundD2_bounded G guard kill hB
                    (d + 1 - 3) (G.pass p) (1 - gamma) (by omega) (by omega)
                  simp only [nullVerify]
                  by_cases hw : 0 < (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
                      gamma ≤ (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
                      (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) < MATE_LOWER ∧ kill p = false ∧
                      allIllegalB G p = true
                  · rw [if_pos hw]; omega
                  · rw [if_neg hw]; omega
                · rw [if_neg hu]; omega
            · -- Away from terminals, fail-high: the loop's lower bound,
              -- and the declared value only grows with its pass term.
              intro hai hgeS
              rw [nullValueD2_of_fold G guard d p hkg hcap hai]
              have hsplit := foldMax_init_split (fun m => -(nullValueD2 G guard d m))
                (movesAbove G (val_lower (d + 1)) p) (nullTermD2 G guard d p)
                (nullTermD2_ge_LOSS G guard d p)
              have := hloop.1 hgeS
              omega
            · -- Away from terminals, fail-low: the loop's upper bound plus
              -- the pass term's own admissibility.
              intro hai hltS
              rw [nullValueD2_of_fold G guard d p hkg hcap hai]
              have hsplit := foldMax_init_split (fun m => -(nullValueD2 G guard d m))
                (movesAbove G (val_lower (d + 1)) p) (nullTermD2 G guard d p)
                (nullTermD2_ge_LOSS G guard d p)
              have hF0 := hloop.2 hltS
              have hSl := searchMoves_ge_init gamma
                (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                (movesAbove G (val_lower (d + 1)) p) LOSS
              have hT : nullTermD2 G guard d p
                  ≤ max (nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma)
                      (searchMoves gamma
                        (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                        (movesAbove G (val_lower (d + 1)) p) LOSS) := by
                by_cases hen : guard p = true ∧ 2 < d + 1
                · by_cases hml : -(nullValueD2 G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
                  · have hT' : nullTermD2 G guard d p
                        = max LOSS (-(nullValueD2 G guard (d + 1 - 3) (G.pass p))) := by
                      simp only [nullTermD2]
                      rw [if_pos hen, if_pos hml]
                    by_cases hu : useD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true
                    · have hnlt' : nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                          < gamma := by
                        by_cases hge : gamma ≤ nullVerify G kill
                            ((-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))) gamma p
                        · exact absurd ⟨hu, hge⟩ hcut
                        · omega
                      have hn : nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma
                          = nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p := by
                        simp only [nullPartD2]; rw [if_pos hu]
                      by_cases hw : 0 < (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
                          gamma ≤ (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
                          (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) < MATE_LOWER ∧ kill p = false ∧
                          allIllegalB G p = true
                      · rw [hai] at hw
                        exact absurd hw.2.2.2.2 (fun h => Bool.noConfusion h)
                      · have hnv : nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                            = (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) := by
                          simp only [nullVerify]; rw [if_neg hw]
                        have hphigh : 1 - gamma ≤ boundD2 G guard kill
                            (d + 1 - 3) (G.pass p) (1 - gamma) := by omega
                        have hpass := ih (d + 1 - 3) (by omega) (G.pass p) (1 - gamma)
                          (by omega) (by omega)
                        have hup := hpass.1 hphigh
                        rw [hT', hn, hnv]
                        omega
                    · exfalso
                      have hnvge : ¬ (nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                          < MATE_LOWER) := by
                        intro hml'
                        exact hu (by
                          simp only [useD2, Bool.and_eq_true, decide_eq_true_eq]
                          exact ⟨⟨hen.1, hen.2⟩, hml'⟩)
                      by_cases hw : 0 < (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
                          gamma ≤ (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
                          (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) < MATE_LOWER ∧ kill p = false ∧
                          allIllegalB G p = true
                      · have hv : nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                            = -MATE_UPPER := by
                          simp only [nullVerify]; rw [if_pos hw]
                        rw [hv] at hnvge
                        omega
                      · have hnv : nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                            = (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) := by
                          simp only [nullVerify]; rw [if_neg hw]
                        rw [hnv] at hnvge
                        have hplow' : boundD2 G guard kill (d + 1 - 3)
                            (G.pass p) (1 - gamma) < 1 - gamma := by omega
                        have hpass := ih (d + 1 - 3) (by omega) (G.pass p) (1 - gamma)
                          (by omega) (by omega)
                        have hlow := hpass.2 hplow'
                        omega
                  · have hT' : nullTermD2 G guard d p = LOSS := by
                      simp only [nullTermD2]
                      rw [if_pos hen, if_neg hml]
                    rw [hT']
                    omega
                · have hT' : nullTermD2 G guard d p = LOSS := by
                    simp only [nullTermD2]
                    rw [if_neg hen]
                  rw [hT']
                  omega
              omega

/-! ### Layer 2: the accuracy lemma, and the chess-facing corollaries -/

/-- **nullValue_eq_realValue_of_noZugzwang** (layer 2): under
`NoZugzwang` the null-inclusive declared function collapses onto the
draw-aware REAL-MOVE value `negamaxD2` -- the pass term never decides a
fold.  This is the ONLY place zugzwang is assumed, and it is an
ACCURACY statement about the approximation's validity region, not a
correctness statement about the search. -/
theorem nullValue_eq_realValue_of_noZugzwang (G : QSGame) (guard : G.Pos → Bool)
    (hZ : NoZugzwang G guard) :
    ∀ (d : Nat) (p : G.Pos), nullValueD2 G guard d p = negamaxD2 G d p := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p
    cases d with
    | zero =>
      simp only [nullValueD2, negamaxD2]
    | succ d =>
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [nullValueD2_kingGone G guard (d + 1) p hkg,
          negamaxD2_kingGone G (d + 1) p hkg]
      · by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [nullValueD2_of_capture G guard (d + 1) p hkg hcap,
            negamaxD2_of_capture G (d + 1) p hkg hcap]
        · cases hai : allIllegalB G p with
          | true =>
            rw [nullValueD2_of_allIllegal G guard d p hkg hcap hai,
              negamaxD2_of_allIllegal G d p hkg hcap hai]
          | false =>
            rw [nullValueD2_of_fold G guard d p hkg hcap hai,
              negamaxD2_of_fold G d p hkg hcap hai]
            have hsplit := foldMax_init_split (fun m => -(nullValueD2 G guard d m))
              (movesAbove G (val_lower (d + 1)) p) (nullTermD2 G guard d p)
              (nullTermD2_ge_LOSS G guard d p)
            have hcongr := foldMax_congr (fun m => -(nullValueD2 G guard d m))
              (fun m => -(negamaxD2 G d m))
              (movesAbove G (val_lower (d + 1)) p) LOSS
              (fun m _ => by
                show -(nullValueD2 G guard d m) = -(negamaxD2 G d m)
                rw [ih d (by omega) m])
            have hT : nullTermD2 G guard d p
                ≤ foldMax (fun m => -(nullValueD2 G guard d m))
                    (movesAbove G (val_lower (d + 1)) p) LOSS := by
              have hfl := foldMax_ge_init (fun m => -(nullValueD2 G guard d m))
                (movesAbove G (val_lower (d + 1)) p) LOSS
              by_cases hen : guard p = true ∧ 2 < d + 1
              · by_cases hml : -(nullValueD2 G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
                · have hT' : nullTermD2 G guard d p
                      = max LOSS (-(nullValueD2 G guard (d + 1 - 3) (G.pass p))) := by
                    simp only [nullTermD2]
                    rw [if_pos hen, if_pos hml]
                  have hZ' := hZ d p hkg hcap hai hen.1 hen.2
                  rw [hT']
                  omega
                · have hT' : nullTermD2 G guard d p = LOSS := by
                    simp only [nullTermD2]
                    rw [if_pos hen, if_neg hml]
                  rw [hT']
                  omega
              · have hT' : nullTermD2 G guard d p = LOSS := by
                  simp only [nullTermD2]
                  rw [if_neg hen]
                rw [hT']
                omega
            omega

/-- **The chess-facing spec, recovered as a corollary** of layer 1 +
layer 2: where `NoZugzwang` holds, the search brackets the REAL-MOVE
value `negamaxD2`.  The single chess assumption is `hZ`; the band
fragment layer 1 needs is implied by it. -/
theorem boundD2_spec (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hK : KillerLegal G kill)
    (hZ : NoZugzwang G guard) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_LOWER < gamma → gamma ≤ MATE_LOWER →
      BoundSpecD2 G d p gamma (boundD2 G guard kill d p gamma) := by
  intro d p gamma h1 h2
  have h := bound_null_spec G guard kill hB hK
    (noZugzwangInMateBand_of_noZugzwang G guard hZ) d p gamma h1 h2
  have he := nullValue_eq_realValue_of_noZugzwang G guard hZ d p
  simp only [BoundSpecD2]
  rw [← he]
  exact h

/-- At a verified-terminal node the two stores bracket the exact
terminal value -- ZUGZWANG-FREE (layer 1 alone: the declared function's
terminal branch is the verified exact value). -/
theorem d2_terminal_stores (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hK : KillerLegal G kill)
    (hR : NoZugzwangInMateBand G guard)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true)
    (gamma : Int) (hg1 : -MATE_LOWER < gamma) (hg2 : gamma ≤ MATE_LOWER) :
    (gamma ≤ boundD2 G guard kill (d + 1) p gamma →
      boundD2 G guard kill (d + 1) p gamma ≤ terminalValue G p) ∧
    (boundD2 G guard kill (d + 1) p gamma < gamma →
      terminalValue G p ≤ boundD2 G guard kill (d + 1) p gamma) := by
  have h := bound_null_spec G guard kill hB hK hR
    (d + 1) p gamma hg1 hg2
  rw [nullValueD2_of_allIllegal G guard d p hkg hcap hai] at h
  exact h

/-- **D2NoCrossing, zugzwang-free**: two driver-range probes of the same
`(pos, depth)` can never store contradictory bounds, because both
bracket the SAME `(pos, depth)`-determined `nullValueD2` -- layer 1
alone.  No bet, no `NoZugzwang`: table consistency never depends on the
approximation's accuracy. -/
theorem d2_no_crossing (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hK : KillerLegal G kill)
    (hR : NoZugzwangInMateBand G guard)
    (d : Nat) (p : G.Pos) (g1 g2 : Int)
    (hg1a : -MATE_LOWER < g1) (hg1b : g1 ≤ MATE_LOWER)
    (hg2a : -MATE_LOWER < g2) (hg2b : g2 ≤ MATE_LOWER)
    (hhi : g1 ≤ boundD2 G guard kill d p g1)
    (hlo : boundD2 G guard kill d p g2 < g2) :
    boundD2 G guard kill d p g1
      ≤ boundD2 G guard kill d p g2 := by
  have h1 := (bound_null_spec G guard kill hB hK hR
    d p g1 hg1a hg1b).1 hhi
  have h2 := (bound_null_spec G guard kill hB hK hR
    d p g2 hg2a hg2b).2 hlo
  omega

/-! # The kcx production consumer, and reference ≡ production

`boundD2` above models `reference.py` -- the executable spec whose
eager entry scan (any generated move landing on the king or within one
of the king-passant square returns the exact `MATE_UPPER` before table,
repetition or loop) IS the model's by-construction king-capture branch.
The PRODUCTION consumer (`kcx-verify` at `560799c`, sunfish.py lines
451-459) restores the same invariant without the eager scan, by
validating every VIRTUAL (`None`) fail-high before it may cut:

* if a real king capture exists it is SUBSTITUTED -- the node reports
  `MATE_UPPER` through a real-move cutoff and `tp_move` stores the true
  capture (active preservation of `KillerAtKingCapturable`);
* a mate-band claim without a capture is vacuous -- fold identity ("if
  passing wins the king, capturing it is a real move too");
* a positive claim at a verified-terminal node is folded to the
  identity (depth-gated: at depth 0 QS evaluates the fold and must not
  RETURN the reserved sentinel, the 96-mismatch lesson).

Futility yields are VIRTUAL below the mate band (`(move, MATE_UPPER)
if val >= MATE_LOWER else (None, pos.score + val)`): a sub-mate
futility estimate prices an UNSEARCHED (possibly illegal) move, so its
truthy `Move` existed only to lie to the `live` bit -- the yield-species
typing caution was a live bug (three bench witnesses; crossed entry
`Entry(lower=0, upper=-1054)` at a stalemated child), now resolved in
code: every truthy yield is a searched real result (or the mate-case
futility yield, a real king capture that can cut).

**`production_eq_reference`** is the top statement: the two consumers
compute the SAME function, at every driver-range window, given
`CaptureFirst` (king captures head the sorted move list --
`KingCaptureValHigh` + the sort) and `KillerLegal` (terminal nodes are
killer-free, so the reference verifier's `not killer` short-circuit
never diverges).  Everything proven for the reference -- the layer-1/2
specs, `d2_no_crossing`, the verified-arm theorems -- transfers to
production by rewriting, and `KingCapturableReportsExact` moves from
REFUTED (for the pre-kcx loop: `CexR`, the stand-pat cut) back to a
THEOREM (`kingCapturableReportsExact_restored`).

The measured battery behind the model (build agent, 2026-08-08):
invariant test 0 violations over 9,600 probes on 200 king-capturable
positions; bound-level equivalence reference == production == optimized
over 9,600 probes (cold+warm, both probe orders); driver and ladder
crossings 0; terminal bench 148/148; all mate floors; +4.4% nodes.

Not modeled here, as always: the table (CanNull.lean -- production's
TT cutoff serves stored brackets of the one key-determined function),
the repetition-0 (which PRECEDES the consumer: a king-capturable
position inside game history would evade the invariant in production,
where the reference dodges it via the entry scan -- closed by the
input-validity hypothesis `HistoryLegal` below, fidelity-class like
`Bounded`), the killer yield (Killer.lean;
`KillerAtKingCapturable` is now ACTIVELY preserved by the substitution
arm), and the futility break (`boundFut`; its sub-mate arm can never
cut -- `score < gamma` by construction -- and is virtual, and its
mate-case arm is a real king capture). -/

/-! ### Move ordering, as the equivalence needs it -/

/-- **CaptureFirst**: at a king-capturable position, a king capture
heads the move list.  Engine backing: the sort of `moves()` orders by
`pos.value` descending and king captures are valued in the mate band
(`KingCaptureValHigh`), above every other move value (the
`EvalBounds` margins). -/
def CaptureFirst (G : QSGame) : Prop :=
  ∀ p, hasKingCapture G.toNullGame.toGame p = true →
    ∃ k rest, G.moves p = k :: rest ∧ G.eval k ≤ -MATE_LOWER

/-- A heading king capture survives every QS filter. -/
theorem movesAbove_cons_of_captureFirst (G : QSGame) (hV : KingCaptureValHigh G)
    {p k : G.Pos} {rest : List G.Pos}
    (hm : G.moves p = k :: rest) (hkev : G.eval k ≤ -MATE_LOWER) (d : Nat) :
    ∃ rest', movesAbove G (val_lower d) p = k :: rest' := by
  refine ⟨rest.filter (fun m => decide (val_lower d ≤ G.val p m)), ?_⟩
  have hk : k ∈ G.moves p := by rw [hm]; exact List.mem_cons_self k rest
  have hval := hV p k hk hkev
  have hthr := val_lower_lt_ML d
  unfold movesAbove
  rw [hm, List.filter_cons, if_pos (by rw [decide_eq_true_eq]; omega)]

/-! ### CaptureFirst, proven from the sort -/

/-- **HighValIsKingCapture** (the converse of `KingCaptureValHigh`,
same `EvalBounds` backing from the other side): a move valued in the
mate band IS a king capture -- no sum of piece values, table deltas and
promotion bonuses reaches `MATE_LOWER` without the king term
(`piece["K"]` dominates the margins, cf. `EvalBounds.margin_covers`).
A PR retuning `piece` values must re-check it. -/
def HighValIsKingCapture (G : QSGame) : Prop :=
  ∀ (p : G.Pos), ∀ m ∈ G.moves p, MATE_LOWER ≤ G.val p m →
    G.eval m ≤ -MATE_LOWER

/-- **MovesSortedByVal** (the ONE trusted primitive here: Python's
`sorted(..., reverse=True)` sorts): the move list is ordered by
descending `pos.value`.  Stability is not needed. -/
def MovesSortedByVal (G : QSGame) : Prop :=
  ∀ (p : G.Pos), List.Pairwise (fun a b => G.val p b ≤ G.val p a) (G.moves p)

/-- **CaptureFirst, DISCHARGED**: from the sort spec plus the two value
facts, a king-capturable position's move list is headed by a king
capture -- the head's value dominates the capture's mate-band value
(`KingCaptureValHigh`), and a mate-band value is itself a capture
(`HighValIsKingCapture`).  The hypothesis leaves the theorem
signatures; what remains trusted is that `sorted` sorts. -/
theorem captureFirst_of_sorted (G : QSGame)
    (hSort : MovesSortedByVal G) (hV : KingCaptureValHigh G)
    (hHi : HighValIsKingCapture G) : CaptureFirst G := by
  intro p hcap
  obtain ⟨m, hm, hmev⟩ := (hasKingCapture_iff G.toNullGame.toGame p).mp hcap
  cases hmoves : G.moves p with
  | nil =>
    rw [hmoves] at hm
    cases hm
  | cons k rest =>
    refine ⟨k, rest, rfl, ?_⟩
    have hkm : k ∈ G.moves p := by rw [hmoves]; exact List.mem_cons_self k rest
    have hmv := hV p m hm hmev
    have hkv : MATE_LOWER ≤ G.val p k := by
      rw [hmoves] at hm
      cases List.mem_cons.mp hm with
      | inl he => rw [← he]; exact hmv
      | inr ht =>
        have hpw := hSort p
        rw [hmoves] at hpw
        have := (List.pairwise_cons.mp hpw).1 m ht
        omega
    exact hHi p k hkm hkv

/-! ### The tp_move lifecycle, machine-checked -/

/-- The killer table: exact-position-keyed, one stored move per
position (`tp_move`). -/
def KillTable (G : QSGame) : Type := G.Pos → Option G.Pos

/-- The three things the engine ever does to `tp_move`, each carrying
the fact its code site establishes:

* `legal` -- the consumption loop's fail-high store of a SEARCHED real
  winner (sunfish.py line 465): legality is the store-site theorem
  `storedMoveLegal` (a real fail-high at an in-band window scores above
  `-MATE_UPPER`, which a king-capturable child cannot);
* `capture` -- a store whose move wins the king outright: the kcx
  SUBSTITUTION store (line 456, the found `king` move -- the engine's
  `board[m.j] == "k" or abs(m.j - pos.kp) < 2` test is exactly "the
  child's king is gone", the same test `gen_moves`/`value` use) and the
  mate-case FUTILITY store (line 417's `(move, MATE_UPPER)` arm: its
  `val >= MATE_LOWER` guard is a capture by `HighValIsKingCapture`);
* `evict` -- FIFO eviction (line 466-467) only forgets.

Every `tp_move` mutation in `bound()` is one of these three. -/
inductive KillStore (G : QSGame) where
  | legal (p m : G.Pos) (hm : m ∈ G.moves p)
      (hleg : hasKingCapture G.toNullGame.toGame m = false)
  | capture (p m : G.Pos) (hm : m ∈ G.moves p)
      (hcap : G.eval m ≤ -MATE_LOWER)
  | evict (p : G.Pos)

def applyStore {G : QSGame} [DecidableEq G.Pos] (t : KillTable G) :
    KillStore G → KillTable G
  | KillStore.legal p m _ _ => fun q => if q = p then some m else t q
  | KillStore.capture p m _ _ => fun q => if q = p then some m else t q
  | KillStore.evict p => fun q => if q = p then none else t q

/-- The lifecycle invariant: every stored move is a generated move that
either wins the king (possible only at king-capturable positions) or is
proven legal.  Position-intrinsic: no search state, no depth, no
window appears -- which is exactly why it PERSISTS across searches and
why exact-position keying is load-bearing. -/
def KillerInv (G : QSGame) (t : KillTable G) : Prop :=
  ∀ (p m : G.Pos), t p = some m →
    m ∈ G.moves p ∧
      (G.eval m ≤ -MATE_LOWER ∨ hasKingCapture G.toNullGame.toGame m = false)

theorem killerInv_empty (G : QSGame) : KillerInv G (fun _ => none) :=
  fun _ _ h => Option.noConfusion h

/-- One step: every store species preserves the invariant (eviction
only forgets). -/
theorem killerInv_step (G : QSGame) [DecidableEq G.Pos]
    {t : KillTable G} (ht : KillerInv G t) :
    ∀ e : KillStore G, KillerInv G (applyStore t e) := by
  intro e q x hq
  cases e with
  | legal p m hm hleg =>
    simp only [applyStore] at hq
    by_cases hqp : q = p
    · rw [if_pos hqp] at hq
      cases hq
      subst hqp
      exact ⟨hm, Or.inr hleg⟩
    · rw [if_neg hqp] at hq
      exact ht q x hq
  | capture p m hm hcap =>
    simp only [applyStore] at hq
    by_cases hqp : q = p
    · rw [if_pos hqp] at hq
      cases hq
      subst hqp
      exact ⟨hm, Or.inl hcap⟩
    · rw [if_neg hqp] at hq
      exact ht q x hq
  | evict p =>
    simp only [applyStore] at hq
    by_cases hqp : q = p
    · rw [if_pos hqp] at hq
      exact Option.noConfusion hq
    · rw [if_neg hqp] at hq
      exact ht q x hq

/-- The full lifecycle: any trace of engine stores, from the empty
table or any invariant-satisfying one, keeps the invariant -- across
searches too, since nothing search-relative appears in it (`tp_move`
is the one table sunfish does NOT clear per search, and this is the
fact that makes that safe). -/
theorem killerInv_trace (G : QSGame) [DecidableEq G.Pos]
    {t0 : KillTable G} (h0 : KillerInv G t0) :
    ∀ es : List (KillStore G), KillerInv G (es.foldl applyStore t0) := by
  intro es
  induction es generalizing t0 with
  | nil => exact h0
  | cons e _ ih => exact ih (killerInv_step G h0 e)

/-- **KillerLegal, now a THEOREM** given an empty-or-invariant initial
table and the store trace: at a non-king-capturable position, a stored
move is a legal move (the `capture` species cannot live there -- the
stored capture would itself witness king-capturability). -/
theorem killerLegal_of_inv (G : QSGame) {t : KillTable G}
    (ht : KillerInv G t) : KillerLegal G (fun p => (t p).isSome) := by
  intro p _ hcapf hk
  have hk' : (t p).isSome = true := hk
  cases hp : t p with
  | none => rw [hp] at hk'; exact Bool.noConfusion hk'
  | some m =>
    obtain ⟨hm, hor⟩ := ht p m hp
    refine ⟨m, hm, ?_⟩
    cases hor with
    | inl hcap =>
      exfalso
      have : hasKingCapture G.toNullGame.toGame p = true :=
        (hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, hcap⟩
      rw [hcapf] at this
      exact Bool.noConfusion this
    | inr hleg => exact hleg

/-- The lifecycle end-to-end, from the empty table. -/
theorem killerLegal_lifecycle (G : QSGame) [DecidableEq G.Pos]
    (es : List (KillStore G)) :
    KillerLegal G (fun p => ((es.foldl applyStore (fun _ => none)) p).isSome) :=
  killerLegal_of_inv G (killerInv_trace G (killerInv_empty G) es)

/-! ### The production consumer -/

/-- The production interception's CUT condition for the (sole modeled)
virtual yield, the null option: a virtual fail-high is allowed to end
the loop iff a king capture exists to substitute (a REAL cutoff at
`MATE_UPPER`), or the claim is sub-band and not a positive claim at a
verified terminal.  Everything else is normalized to the fold identity
and the loop continues. -/
def NCut (G : QSGame) (guard : G.Pos → Bool) (rn : Int)
    (d : Nat) (p : G.Pos) (gamma : Int) : Prop :=
  (guard p = true ∧ 2 < d) ∧ gamma ≤ rn ∧
    (hasKingCapture G.toNullGame.toGame p = true ∨
      (rn < MATE_LOWER ∧ (rn ≤ 0 ∨ allIllegalB G p = false)))

instance (G : QSGame) (guard : G.Pos → Bool) (rn : Int)
    (d : Nat) (p : G.Pos) (gamma : Int) : Decidable (NCut G guard rn d p gamma) := by
  unfold NCut; infer_instance

/-- The null option's fold contribution when it does NOT cut: a
normalized fail-high is the identity, a fail-low is RAW (production
yields the raw score; the old fail-low suppression is dead in the
driver range, where a mate-band report can never fail low). -/
def nFoldKCX (G : QSGame) (guard : G.Pos → Bool) (rn : Int)
    (d : Nat) (p : G.Pos) (gamma : Int) : Int :=
  if guard p = true ∧ 2 < d then
    (if gamma ≤ rn then -MATE_UPPER else rn)
  else -MATE_UPPER

/-- The golfed correction scan (`c72cf6d`): a move needs no probe when
the depth ≥ 2 threshold admitted it -- a fail-low loop at depth ≥ 2
searched EXACTLY the admitted moves (no futility above depth 1, the
killer is val-gated, the sorted break is the filter), and the sticky
evidence bit already certified every searched move illegal.  Only the
filtered remainder is probed; at depth 1 futility skips admitted moves,
so everything is probed (one of the golf agent's two countermodels for
the naive reduction; the other -- a raw null fail-low outscoring a legal
real move's yield -- is why `live` is evidence-shaped, not
winner-shaped). -/
def scanNewB (G : QSGame) (d : Nat) (p : G.Pos) : Bool :=
  (G.moves p).all (fun m =>
    (decide (1 < d) && decide (val_lower d ≤ G.val p m)) ||
    hasKingCapture G.toNullGame.toGame m)

/-- The golfed correction: fail-low, `not live` (the sticky evidence
bit: NO searched real yield ever exceeded the sentinel -- exactly
`S = LOSS`), and the reduced scan. -/
def golfFix (G : QSGame) (d : Nat) (gamma best S : Int) (p : G.Pos) : Int :=
  if best < gamma ∧ S = LOSS ∧ scanNewB G d p = true then
    (if inCheckB G.toNullGame p = true then -MATE_LOWER else 0)
  else best

/-- **NoMaskedMobility** (the premise the reduced scan genuinely needs
-- `reducedScan_needs_premise` below is the countermodel; the hoped-for
"chess-assumption-free" equivalence is FALSE without it): a position
whose every depth-1-admitted move is illegal has no legal move at all.
Failure shape: all high-valued moves illegal while some legal move
drops more than `QS_A - QS` (= 100) of table value -- the depth-1 node
then returns the raw `-MATE_UPPER` fold although the scan SAW the legal
move, a depth-2 parent converts that to a spurious `MATE_UPPER`, and
from depth 3 up the probe-free scan trusts the corrupted sentinel.
Chess-plausibility: no natural position with ONLY >100cp-dropping legal
moves is known; table arithmetic does not exclude it (`ValFloor` is
192 > 100). -/
def NoMaskedMobility (G : QSGame) : Prop :=
  ∀ p, (∀ m ∈ movesAbove G (val_lower 1) p, hasKingCapture G.toNullGame.toGame m = true) →
    ∀ m ∈ G.moves p, hasKingCapture G.toNullGame.toGame m = true

/-- **The production search**: NO eager king-capture branch -- the
invariant is restored by the consumer.  A virtual fail-high either
substitutes the real capture (`MATE_UPPER`), cuts validated, or is
normalized into the fold; the correction is the same `not live`
verify-on-suspicion block (`d2Fix`).  As in `boundD2`, the null yield
is the search's OWN pass probe, part of the definition. -/
def boundKCX (G : QSGame) (guard : G.Pos → Bool) :
    Nat → G.Pos → Int → Int
  | 0, p, _gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else G.eval p
  | 1, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if NCut G guard (-(boundKCX G guard 0 (G.pass p) (1 - gamma))) 1 p gamma then
      (if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
       else -(boundKCX G guard 0 (G.pass p) (1 - gamma)))
    else
      golfFix G 1 gamma
        (max (nFoldKCX G guard (-(boundKCX G guard 0 (G.pass p) (1 - gamma))) 1 p gamma)
          (searchMoves gamma
            (fun m => -(boundKCX G guard 0 m (1 - gamma)))
            (movesAbove G (val_lower 1) p) LOSS))
        (searchMoves gamma
          (fun m => -(boundKCX G guard 0 m (1 - gamma)))
          (movesAbove G (val_lower 1) p) LOSS)
        p
  | 2, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if NCut G guard (-(boundKCX G guard 0 (G.pass p) (1 - gamma))) 2 p gamma then
      (if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
       else -(boundKCX G guard 0 (G.pass p) (1 - gamma)))
    else
      golfFix G 2 gamma
        (max (nFoldKCX G guard (-(boundKCX G guard 0 (G.pass p) (1 - gamma))) 2 p gamma)
          (searchMoves gamma
            (fun m => -(boundKCX G guard 1 m (1 - gamma)))
            (movesAbove G (val_lower 2) p) LOSS))
        (searchMoves gamma
          (fun m => -(boundKCX G guard 1 m (1 - gamma)))
          (movesAbove G (val_lower 2) p) LOSS)
        p
  | d + 3, p, gamma =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if NCut G guard (-(boundKCX G guard d (G.pass p) (1 - gamma))) (d + 3) p gamma then
      (if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
       else -(boundKCX G guard d (G.pass p) (1 - gamma)))
    else
      golfFix G (d + 3) gamma
        (max (nFoldKCX G guard (-(boundKCX G guard d (G.pass p) (1 - gamma))) (d + 3) p gamma)
          (searchMoves gamma
            (fun m => -(boundKCX G guard (d + 2) m (1 - gamma)))
            (movesAbove G (val_lower (d + 3)) p) LOSS))
        (searchMoves gamma
          (fun m => -(boundKCX G guard (d + 2) m (1 - gamma)))
          (movesAbove G (val_lower (d + 3)) p) LOSS)
        p

/-- The uniform successor equation for the production search. -/
theorem boundKCX_succ (G : QSGame) (guard : G.Pos → Bool) (d : Nat) (p : G.Pos) (gamma : Int) :
    boundKCX G guard (d + 1) p gamma
      = if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
        else if NCut G guard (-(boundKCX G guard (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma then
          (if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
           else -(boundKCX G guard (d + 1 - 3) (G.pass p) (1 - gamma)))
        else
          golfFix G (d + 1) gamma
            (max (nFoldKCX G guard (-(boundKCX G guard (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma)
              (searchMoves gamma
                (fun m => -(boundKCX G guard d m (1 - gamma)))
                (movesAbove G (val_lower (d + 1)) p) LOSS))
            (searchMoves gamma
              (fun m => -(boundKCX G guard d m (1 - gamma)))
              (movesAbove G (val_lower (d + 1)) p) LOSS)
            p := by
  match d with
  | 0 => rfl
  | 1 => rfl
  | d + 2 => rfl

theorem boundKCX_kingGone (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (gamma : Int) (h : G.eval p ≤ -MATE_LOWER) :
    boundKCX G guard d p gamma = -MATE_UPPER := by
  cases d with
  | zero => simp only [boundKCX]; rw [if_pos h]
  | succ d => rw [boundKCX_succ, if_pos h]

/-- **VirtualCutoffNormalized**, the validation half: a virtual cutoff
that survives the interception at a NON-capturable, oracle-terminal
node is non-positive -- production never stores a positive lower bound
at a verified terminal, with no killer consulted at all. -/
theorem virtualCutoffValidated (G : QSGame)
    (guard : G.Pos → Bool) (rn : Int) (d : Nat) (p : G.Pos) (gamma : Int)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hai : allIllegalB G p = true)
    (hnc : NCut G guard rn d p gamma) :
    rn ≤ 0 := by
  obtain ⟨_, _, hor⟩ := hnc
  cases hor with
  | inl h => rw [hcapf] at h; exact Bool.noConfusion h
  | inr h =>
    cases h.2 with
    | inl h0 => exact h0
    | inr hA => rw [hai] at hA; exact Bool.noConfusion hA

/-! ### The null arms match -/

/-- The production interception and the reference verifier make the
SAME decisions about the null option at non-capturable nodes: same cut
condition, same cut value, same fold contribution.  The one place the
reference consults the killer -- the withdrawal's `not killer` -- is
covered by `KillerLegal`: at an oracle-terminal node no killer exists,
and away from them the killer disjunct changes nothing. -/
theorem nullArm_match (G : QSGame)
    (guard kill : G.Pos → Bool) (rn : Int) (hK : KillerLegal G kill)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_LOWER < gamma) (hg2 : gamma ≤ MATE_LOWER)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcapf : hasKingCapture G.toNullGame.toGame p = false) :
    (NCut G guard rn (d + 1) p gamma ↔
      (useD2 G guard kill rn (d + 1) p gamma = true ∧
        gamma ≤ nullVerify G kill rn gamma p)) ∧
    (NCut G guard rn (d + 1) p gamma →
      nullVerify G kill rn gamma p = rn) ∧
    (¬ NCut G guard rn (d + 1) p gamma →
      nFoldKCX G guard rn (d + 1) p gamma
        = nullPartD2 G guard kill rn (d + 1) p gamma) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by
    rw [hcapf]; exact fun h => Bool.noConfusion h
  by_cases hen : guard p = true ∧ 2 < d + 1
  · by_cases hgeraw : gamma ≤ rn
    · by_cases hml : rn < MATE_LOWER
      · by_cases hwd : 0 < rn ∧ allIllegalB G p = true
        · -- withdrawal (reference) / normalize (production): no cut, identity
          have hkf : kill p = false := by
            cases hk : kill p with
            | false => rfl
            | true =>
              have := killerLegal_not_terminal G kill hK p hkg hcapf hk
              rw [hwd.2] at this
              exact Bool.noConfusion this
          have hnc : ¬ NCut G guard rn (d + 1) p gamma := by
            intro h
            rcases h.2.2 with h' | ⟨_, h' | h'⟩
            · exact hcap h'
            · omega
            · rw [hwd.2] at h'; exact Bool.noConfusion h'
          have hv : nullVerify G kill rn gamma p
              = -MATE_UPPER := by
            simp only [nullVerify]
            rw [if_pos ⟨hwd.1, hgeraw, hml, hkf, hwd.2⟩]
          have hu : useD2 G guard kill rn (d + 1) p gamma = true := by
            simp only [useD2, Bool.and_eq_true, decide_eq_true_eq]
            exact ⟨⟨hen.1, hen.2⟩, by rw [hv]; omega⟩
          refine ⟨⟨fun h => absurd h hnc, fun h => absurd h.2 (by rw [hv]; omega)⟩,
            fun h => absurd h hnc, fun _ => ?_⟩
          simp only [nFoldKCX, nullPartD2]
          rw [if_pos hen, if_pos hgeraw, if_pos hu, hv]
        · -- surviving cutoff, both sides, value = raw
          have hor : rn ≤ 0 ∨ allIllegalB G p = false := by
            by_cases h0 : 0 < rn
            · right
              cases hA : allIllegalB G p with
              | false => rfl
              | true => exact absurd ⟨h0, hA⟩ hwd
            · left; omega
          have hnc : NCut G guard rn (d + 1) p gamma :=
            ⟨hen, hgeraw, Or.inr ⟨hml, hor⟩⟩
          have hv : nullVerify G kill rn gamma p
              = rn := by
            simp only [nullVerify]
            refine if_neg (fun h => ?_)
            exact hwd ⟨h.1, h.2.2.2.2⟩
          have hu : useD2 G guard kill rn (d + 1) p gamma = true := by
            simp only [useD2, Bool.and_eq_true, decide_eq_true_eq]
            exact ⟨⟨hen.1, hen.2⟩, by rw [hv]; exact hml⟩
          exact ⟨⟨fun _ => ⟨hu, by rw [hv]; exact hgeraw⟩, fun _ => hnc⟩,
            fun _ => hv, fun h => absurd hnc h⟩
      · -- mate-band fail-high: normalized (production) / suppressed (reference)
        have hnc : ¬ NCut G guard rn (d + 1) p gamma := by
          intro h
          rcases h.2.2 with h' | ⟨h', _⟩
          · exact hcap h'
          · exact hml h'
        have hv : nullVerify G kill rn gamma p
            = rn := by
          simp only [nullVerify]
          refine if_neg (fun h => hml h.2.2.1)
        have hu : useD2 G guard kill rn (d + 1) p gamma = false := by
          simp only [useD2]
          have : decide (nullVerify G kill rn gamma p
              < MATE_LOWER) = false := by
            rw [hv, decide_eq_false_iff_not]
            exact hml
          rw [this, Bool.and_false]
        refine ⟨⟨fun h => absurd h hnc, fun h => by rw [hu] at h; exact Bool.noConfusion h.1⟩,
          fun h => absurd h hnc, fun _ => ?_⟩
        simp only [nFoldKCX, nullPartD2]
        rw [if_pos hen, if_pos hgeraw, if_neg (by rw [hu]; exact fun h => Bool.noConfusion h)]
        rfl
    · -- fail-low: raw contribution, both sides
      have hnc : ¬ NCut G guard rn (d + 1) p gamma := fun h => hgeraw h.2.1
      have hv : nullVerify G kill rn gamma p
          = rn := by
        simp only [nullVerify]
        refine if_neg (fun h => hgeraw h.2.1)
      have hu : useD2 G guard kill rn (d + 1) p gamma = true := by
        simp only [useD2, Bool.and_eq_true, decide_eq_true_eq]
        refine ⟨⟨hen.1, hen.2⟩, ?_⟩
        rw [hv]
        omega
      refine ⟨⟨fun h => absurd h hnc, fun h => absurd h.2 (by rw [hv]; omega)⟩,
        fun h => absurd h hnc, fun _ => ?_⟩
      simp only [nFoldKCX, nullPartD2]
      rw [if_pos hen, if_neg hgeraw, if_pos hu, hv]
  · -- option disabled
    have hnc : ¬ NCut G guard rn (d + 1) p gamma := fun h => hen h.1
    have hu : useD2 G guard kill rn (d + 1) p gamma = false := by
      by_cases hgp : guard p = true
      · have hd2 : ¬ (2 < d + 1) := fun h => hen ⟨hgp, h⟩
        simp only [useD2]
        have : decide (2 < d + 1) = false := by
          rw [decide_eq_false_iff_not]; exact hd2
        rw [this, Bool.and_false, Bool.false_and]
      · have hgf : guard p = false := by
          cases h : guard p
          · rfl
          · exact absurd h hgp
        simp only [useD2]
        rw [hgf, Bool.false_and, Bool.false_and]
    refine ⟨⟨fun h => absurd h hnc, fun h => by rw [hu] at h; exact Bool.noConfusion h.1⟩,
      fun h => absurd h hnc, fun _ => ?_⟩
    simp only [nFoldKCX, nullPartD2]
    rw [if_neg hen, if_neg (by rw [hu]; exact fun h => Bool.noConfusion h)]
    rfl

/-! ### The two-way sentinel, and the reduced-scan bridge -/

/-- A result at least `Y` (above the seed) names a yield at least `Y`. -/
theorem searchMoves_exists_ge {α : Type _} (gamma : Int) (f : α → Int) :
    ∀ (ms : List α) (b Y : Int), b < Y → Y ≤ searchMoves gamma f ms b →
      ∃ m ∈ ms, Y ≤ f m := by
  intro ms
  induction ms with
  | nil =>
    intro b Y hb hge
    simp only [searchMoves] at hge
    omega
  | cons a ms ih =>
    intro b Y hb hge
    simp only [searchMoves] at hge
    by_cases hcut : gamma ≤ max b (f a)
    · rw [if_pos hcut] at hge
      exact ⟨a, List.mem_cons_self a ms, by omega⟩
    · rw [if_neg hcut] at hge
      by_cases hfa : Y ≤ f a
      · exact ⟨a, List.mem_cons_self a ms, hfa⟩
      · obtain ⟨m, hm, h⟩ := ih (max b (f a)) Y (by omega) hge
        exact ⟨m, List.mem_cons_of_mem a hm, h⟩

/-- The null contribution never sinks below the fold identity (the
pass probe is band-bounded). -/
theorem nullPartD2_ge_LOSS (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_LOWER < gamma) (hg2 : gamma ≤ MATE_LOWER) :
    LOSS ≤ nullPartD2 G guard kill
      (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))
      (d + 1) p gamma := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hpb := boundD2_bounded G guard kill hB
    (d + 1 - 3) (G.pass p) (1 - gamma) (by omega) (by omega)
  simp only [nullPartD2]
  by_cases hu : useD2 G guard kill
      (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true
  · rw [if_pos hu]
    simp only [nullVerify]
    by_cases hw : 0 < (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
        gamma ≤ (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) ∧
        (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) < MATE_LOWER ∧
        kill p = false ∧ allIllegalB G p = true
    · rw [if_pos hw]; omega
    · rw [if_neg hw]; omega
  · rw [if_neg hu]; omega

/-- **The two-way sentinel** (the golfed consumer's load-bearing claim,
proven for the reference semantics): at a quiet node the search never
returns the reserved sentinel, and a returned `MATE_UPPER` proves a
king capture.  This is what makes a searched real yield two-way
legality EVIDENCE.  Premises: the value facts, plus `NoMaskedMobility`
-- without which the claim is FALSE from depth 2 up
(`reducedScan_needs_premise`). -/
theorem sentinel_two_way_D2 (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    {B : Int} (hF : ValFloor G B) (h240 : B ≤ 240)
    (hNM : NoMaskedMobility G) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_LOWER < gamma → gamma ≤ MATE_LOWER →
      ¬ (G.eval p ≤ -MATE_LOWER) →
      (boundD2 G guard kill d p gamma = MATE_UPPER →
        hasKingCapture G.toNullGame.toGame p = true) ∧
      (hasKingCapture G.toNullGame.toGame p = false →
        -MATE_UPPER < boundD2 G guard kill d p gamma) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p gamma hg1 hg2 hkg
    have hqp := hQ p hkg
    cases d with
    | zero =>
      constructor
      · intro h
        cases hc : hasKingCapture G.toNullGame.toGame p with
        | true => rfl
        | false =>
          exfalso
          simp only [boundD2] at h
          rw [if_neg hkg, if_neg (by simp [hc])] at h
          omega
      · intro hc
        simp only [boundD2]
        rw [if_neg hkg, if_neg (by simp [hc])]
        omega
    | succ d =>
      -- The shared branch analysis, once.
      have core : ∀ r : Int, r = boundD2 G guard kill (d + 1) p gamma →
          hasKingCapture G.toNullGame.toGame p = false →
          -MATE_UPPER < r ∧ (r = MATE_UPPER → False) := by
        intro r hr hc
        have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by
          simp [hc]
        rw [boundD2_succ, if_neg hkg, if_neg hcap] at hr
        by_cases hcut : useD2 G guard kill
            (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true ∧
            gamma ≤ nullVerify G kill
              (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
        · rw [if_pos hcut] at hr
          have hu := hcut.1
          simp only [useD2, Bool.and_eq_true, decide_eq_true_eq] at hu
          constructor
          · omega
          · intro hMUr; omega
        · rw [if_neg hcut] at hr
          -- All quiet-node exits of the correction stay strictly inside
          -- (-MATE_UPPER, MATE_UPPER).
          have hSl := searchMoves_ge_init gamma
            (fun m => -(boundD2 G guard kill d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS
          -- every admitted child that reports the negated sentinel is a
          -- king capture (strong IH), so an untouched fold certifies the
          -- admitted moves illegal
          have hadm : searchMoves gamma
              (fun m => -(boundD2 G guard kill d m (1 - gamma)))
              (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS →
              ∀ m ∈ movesAbove G (val_lower (d + 1)) p,
                hasKingCapture G.toNullGame.toGame m = true := by
            intro hS m hm
            have hle : -(boundD2 G guard kill d m (1 - gamma)) ≤ LOSS :=
              searchMoves_eq_init_all gamma
                (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                (movesAbove G (val_lower (d + 1)) p) LOSS (by omega) hS m hm
            have hmkg : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
              hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr
                ⟨m, movesAbove_subset G (val_lower (d + 1)) p m hm, hh⟩)
            have hmm := (ih d (by omega) m (1 - gamma) (by omega) (by omega) hmkg).1
            have hub := boundD2_bounded G guard kill hB d m (1 - gamma)
              (by omega) (by omega)
            have := hmm (by omega)
            exact this
          have hAI : searchMoves gamma
              (fun m => -(boundD2 G guard kill d m (1 - gamma)))
              (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS →
              allIllegalB G p = true := by
            intro hS
            rw [allIllegalB_true_iff]
            by_cases hd1 : d = 0
            · subst hd1
              exact hNM p (hadm hS)
            · have hall : allAboveB G (d + 1) p = true :=
                allAboveB_of_floor G hF (d + 1) p (by
                  unfold val_lower QS QS_A
                  omega)
              intro m hm
              refine hadm hS m ?_
              rw [movesAbove_all G (d + 1) p hall]
              exact hm
          have hnge := nullPartD2_ge_LOSS G guard kill hB d p gamma hg1 hg2
          have hnub : nullPartD2 G guard kill
              (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))
              (d + 1) p gamma ≤ MATE_LOWER := by
            simp only [nullPartD2]
            by_cases hu : useD2 G guard kill
                (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true
            · rw [if_pos hu]
              simp only [useD2, Bool.and_eq_true, decide_eq_true_eq] at hu
              omega
            · rw [if_neg hu]; omega
          simp only [d2Fix] at hr
          by_cases hfire : max (nullPartD2 G guard kill
                (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma)
                (searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                  (movesAbove G (val_lower (d + 1)) p) LOSS) < gamma ∧
              searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                (movesAbove G (val_lower (d + 1)) p) LOSS
                ≤ nullPartD2 G guard kill
                    (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma ∧
              allIllegalB G p = true
          · rw [if_pos hfire] at hr
            by_cases hic : inCheckB G.toNullGame p = true
            · rw [if_pos hic] at hr
              constructor
              · omega
              · intro h; omega
            · rw [if_neg hic] at hr
              constructor
              · omega
              · intro h; omega
          · rw [if_neg hfire] at hr
            -- passthrough: if the fold sits at the identity the gate
            -- would have fired, so it does not
            by_cases hS : searchMoves gamma
                (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS
            · exfalso
              have hnlt : nullPartD2 G guard kill
                  (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))
                  (d + 1) p gamma < gamma := by
                simp only [nullPartD2]
                by_cases hu : useD2 G guard kill
                    (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (d + 1) p gamma = true
                · rw [if_pos hu]
                  by_cases hge : gamma ≤ nullVerify G kill
                      (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma p
                  · exact absurd ⟨hu, hge⟩ hcut
                  · omega
                · rw [if_neg hu]; omega
              exact hfire ⟨by omega, by omega, hAI hS⟩
            · -- a live fold: some yield above the identity, and no yield
              -- exceeds what a capture-free child can justify
              constructor
              · omega
              · intro hMUr
                rw [hMUr] at hr
                -- max n S = MATE_UPPER with n ≤ MATE_LOWER forces S = MATE_UPPER
                have hSMU : MATE_UPPER ≤ searchMoves gamma
                    (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                    (movesAbove G (val_lower (d + 1)) p) LOSS := by omega
                obtain ⟨m, hm, hf⟩ := searchMoves_exists_ge gamma
                  (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                  (movesAbove G (val_lower (d + 1)) p) LOSS MATE_UPPER (by omega) hSMU
                have hmkg : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
                  hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr
                    ⟨m, movesAbove_subset G (val_lower (d + 1)) p m hm, hh⟩)
                cases hcm : hasKingCapture G.toNullGame.toGame m with
                | true =>
                  have := boundD2_of_capture G guard kill d m (1 - gamma) hmkg hcm
                  omega
                | false =>
                  have := (ih d (by omega) m (1 - gamma) (by omega) (by omega) hmkg).2 hcm
                  omega
      constructor
      · intro h
        cases hc : hasKingCapture G.toNullGame.toGame p with
        | true => rfl
        | false => exact ((core _ rfl hc).2 h).elim
      · intro hc
        exact (core _ rfl hc).1

/-- **The reduced scan equals the oracle scan where it fires** -- the
golfed gate and the reference gate compute the same correction, given
the two-way sentinel's premises.  (⇐) an oracle-confirmed terminal
passes the reduced scan move-by-move and pins the fold to the identity;
(⇒) an identity fold certifies the admitted moves illegal through the
two-way sentinel, and `ValFloor` (depth ≥ 2) or `NoMaskedMobility`
(depth 1) extends that to every move -- so the reduced-scan gate can
only fire where the oracle gate does. -/
theorem golfFix_eq_d2Fix (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    {B : Int} (hF : ValFloor G B) (h240 : B ≤ 240) (hNM : NoMaskedMobility G)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_LOWER < gamma) (hg2 : gamma ≤ MATE_LOWER)
    (_hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (n : Int) (hnge : LOSS ≤ n) :
    golfFix G (d + 1) gamma
        (max n (searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
          (movesAbove G (val_lower (d + 1)) p) LOSS))
        (searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
          (movesAbove G (val_lower (d + 1)) p) LOSS) p
      = d2Fix G gamma
          (max n (searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS))
          (searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS) n p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hadm : searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
      (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS →
      ∀ m ∈ movesAbove G (val_lower (d + 1)) p,
        hasKingCapture G.toNullGame.toGame m = true := by
    intro hS m hm
    have hle : -(boundD2 G guard kill d m (1 - gamma)) ≤ LOSS :=
      searchMoves_eq_init_all gamma
        (fun m => -(boundD2 G guard kill d m (1 - gamma)))
        (movesAbove G (val_lower (d + 1)) p) LOSS (by omega) hS m hm
    have hmkg : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
      hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr
        ⟨m, movesAbove_subset G (val_lower (d + 1)) p m hm, hh⟩)
    have hub := boundD2_bounded G guard kill hB d m (1 - gamma)
      (by omega) (by omega)
    exact (sentinel_two_way_D2 G guard kill hB hQ hF h240 hNM
      d m (1 - gamma) (by omega) (by omega) hmkg).1 (by omega)
  have hAI : searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
      (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS →
      allIllegalB G p = true := by
    intro hS
    rw [allIllegalB_true_iff]
    by_cases hd1 : d = 0
    · subst hd1
      exact hNM p (hadm hS)
    · have hall : allAboveB G (d + 1) p = true :=
        allAboveB_of_floor G hF (d + 1) p (by
          unfold val_lower QS QS_A
          omega)
      intro m hm
      refine hadm hS m ?_
      rw [movesAbove_all G (d + 1) p hall]
      exact hm
  by_cases hai : allIllegalB G p = true
  · have hSL : searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
        (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS := by
      refine searchMoves_eq_init gamma _ _ LOSS (fun m hm => ?_) (by omega)
      show -(boundD2 G guard kill d m (1 - gamma)) ≤ LOSS
      have hmm := movesAbove_subset G (val_lower (d + 1)) p m hm
      have hcm := allIllegalB_true_iff.mp hai m hmm
      have hmkg : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
        hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hmm, hh⟩)
      rw [boundD2_of_capture G guard kill d m (1 - gamma) hmkg hcm]
      omega
    have hscan : scanNewB G (d + 1) p = true := by
      refine List.all_eq_true.mpr fun m hm => ?_
      rw [Bool.or_eq_true]
      exact Or.inr (allIllegalB_true_iff.mp hai m hm)
    simp only [golfFix, d2Fix]
    by_cases hb : max n (searchMoves gamma
        (fun m => -(boundD2 G guard kill d m (1 - gamma)))
        (movesAbove G (val_lower (d + 1)) p) LOSS) < gamma
    · have hfg : max n (searchMoves gamma
          (fun m => -(boundD2 G guard kill d m (1 - gamma)))
          (movesAbove G (val_lower (d + 1)) p) LOSS) < gamma ∧
          searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS = LOSS ∧
          scanNewB G (d + 1) p = true := ⟨hb, hSL, hscan⟩
      have hfd : max n (searchMoves gamma
          (fun m => -(boundD2 G guard kill d m (1 - gamma)))
          (movesAbove G (val_lower (d + 1)) p) LOSS) < gamma ∧
          searchMoves gamma (fun m => -(boundD2 G guard kill d m (1 - gamma)))
            (movesAbove G (val_lower (d + 1)) p) LOSS ≤ n ∧
          allIllegalB G p = true := ⟨hb, by omega, hai⟩
      rw [if_pos hfg, if_pos hfd]
    · rw [if_neg (fun h => hb h.1), if_neg (fun h => hb h.1)]
  · simp only [golfFix, d2Fix]
    rw [if_neg (fun h => hai (hAI h.2.1)),
      if_neg (fun h => hai h.2.2)]

/-! ### The top statement: reference ≡ production -/

/-- **production_eq_reference**: the production consumer computes
EXACTLY the reference function, at every position, depth and
driver-range window.  At a king-capturable node the eager scan's
`MATE_UPPER` is reproduced by the loop -- either the interception
substitutes the capture on a virtual fail-high, or the capture heads
the sorted move list (`CaptureFirst`) and cuts as the first searched
real yield, exactly.  Everywhere else the interception and the
reference verifier make identical decisions about the (shared,
definitional) pass probe (`nullArm_match`), and the folds agree
child-by-child.  By strong induction: children one level down, the
pass three levels down.  Machine-checked twin of the build battery's
`reference == production` over 9,600 probes. -/
theorem production_eq_reference (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    {B : Int} (hF : ValFloor G B) (h240 : B ≤ 240) (hNM : NoMaskedMobility G)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G)
    (hK : KillerLegal G kill) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_LOWER < gamma → gamma ≤ MATE_LOWER →
      boundKCX G guard d p gamma
        = boundD2 G guard kill d p gamma := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p gamma hg1 hg2
    cases d with
    | zero =>
      simp only [boundKCX, boundD2]
    | succ d =>
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [boundKCX_kingGone G guard (d + 1) p gamma hkg,
          boundD2_kingGone G guard kill (d + 1) p gamma hkg]
      · -- The two pass probes are the same value, by the induction
        -- hypothesis at the reduced depth (band window flip).
        have hpasseq : -(boundKCX G guard (d + 1 - 3) (G.pass p) (1 - gamma))
            = -(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)) := by
          rw [ih (d + 1 - 3) (by omega) (G.pass p) (1 - gamma) (by omega) (by omega)]
        rw [boundKCX_succ, if_neg hkg, hpasseq, boundD2_succ, if_neg hkg]
        by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · -- King-capturable: the reference's eager MATE_UPPER, reproduced.
          by_cases hnc : NCut G guard
              (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))
              (d + 1) p gamma
          · rw [if_pos hnc, if_pos hcap, if_pos hcap]
          · rw [if_neg hnc, if_pos hcap]
            obtain ⟨k, rest, hmoves, hkev⟩ := hCF p hcap
            obtain ⟨rest', hma⟩ :=
              movesAbove_cons_of_captureFirst G hV hmoves hkev (d + 1)
            have hfk : -(boundKCX G guard d k (1 - gamma))
                = MATE_UPPER := by
              rw [boundKCX_kingGone G guard d k (1 - gamma) hkev]
              omega
            have hS : searchMoves gamma
                (fun m => -(boundKCX G guard d m (1 - gamma)))
                (movesAbove G (val_lower (d + 1)) p) LOSS = MATE_UPPER := by
              rw [hma]
              simp only [searchMoves]
              rw [hfk, if_pos (by omega)]
              omega
            have hnF : nFoldKCX G guard
                (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))
                (d + 1) p gamma < gamma := by
              simp only [nFoldKCX]
              by_cases hen : guard p = true ∧ 2 < d + 1
              · rw [if_pos hen]
                by_cases hge : gamma ≤ -(boundD2 G guard kill (d + 1 - 3)
                    (G.pass p) (1 - gamma))
                · exact absurd ⟨hen, hge, Or.inl hcap⟩ hnc
                · rw [if_neg hge]; omega
              · rw [if_neg hen]; omega
            simp only [golfFix]
            rw [hS, if_neg (fun h => absurd h.1 (by omega))]
            omega
        · -- Not capturable: null arms match, folds agree by induction.
          have hcapf : hasKingCapture G.toNullGame.toGame p = false := by
            cases h : hasKingCapture G.toNullGame.toGame p
            · rfl
            · exact absurd h hcap
          obtain ⟨hiff, hval, hfold⟩ :=
            nullArm_match G guard kill
              (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))
              hK d p gamma hg1 hg2 hkg hcapf
          have hSeq : searchMoves gamma
              (fun m => -(boundKCX G guard d m (1 - gamma)))
              (movesAbove G (val_lower (d + 1)) p) LOSS
              = searchMoves gamma
                  (fun m => -(boundD2 G guard kill d m (1 - gamma)))
                  (movesAbove G (val_lower (d + 1)) p) LOSS := by
            refine searchMoves_congr gamma _ _ _ LOSS (fun m _ => ?_)
            show -(boundKCX G guard d m (1 - gamma))
                = -(boundD2 G guard kill d m (1 - gamma))
            rw [ih d (by omega) m (1 - gamma) (by omega) (by omega)]
          by_cases hnc : NCut G guard
              (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))
              (d + 1) p gamma
          · have hcut := hiff.mp hnc
            rw [if_pos hnc, if_neg hcap, if_neg hcap, if_pos hcut, hval hnc]
          · have hcut : ¬ (useD2 G guard kill
                (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))
                (d + 1) p gamma = true ∧
                gamma ≤ nullVerify G kill
                  (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma)))
                  gamma p) :=
              fun h => hnc (hiff.mpr h)
            rw [if_neg hnc, if_neg hcap, if_neg hcut, hSeq, hfold hnc]
            exact golfFix_eq_d2Fix G guard kill hB hQ hF h240 hNM d p gamma
              hg1 hg2 hkg hcap _
              (nullPartD2_ge_LOSS G guard kill hB d p gamma hg1 hg2)

/-! ### The restored invariant, and the transfers -/

/-- **KingCapturableReportsExact, RESTORED** (compare
`kingCapturableReportsExact_refuted` above, the countermodel for the
UNREPAIRED loop): the production search reports a king-capturable node
as exactly `MATE_UPPER`, at every depth and driver-range window --
by construction of the consumer, via the equivalence with the eager
reference.  This is the invariant the whole pre-kcx ledger orbited;
`CexR` remains as the record of why the repair was needed. -/
theorem kingCapturableReportsExact_restored (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    {B : Int} (hF : ValFloor G B) (h240 : B ≤ 240) (hNM : NoMaskedMobility G)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G) (hK : KillerLegal G kill)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_LOWER < gamma) (hg2 : gamma ≤ MATE_LOWER)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    boundKCX G guard d p gamma = MATE_UPPER := by
  rw [production_eq_reference G guard kill hB hQ hF h240 hNM hV hCF hK d p gamma hg1 hg2]
  exact boundD2_of_capture G guard kill d p gamma hkg hcap

/-- The layered spec, transferred to production: the shipped consumer
brackets the null-inclusive declared value function -- no null bet, one
band premise (see `bound_null_spec`). -/
theorem boundKCX_null_spec (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    {B : Int} (hF : ValFloor G B) (h240 : B ≤ 240) (hNM : NoMaskedMobility G)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G)
    (hK : KillerLegal G kill)
    (hR : NoZugzwangInMateBand G guard) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_LOWER < gamma → gamma ≤ MATE_LOWER →
      (gamma ≤ boundKCX G guard d p gamma →
        boundKCX G guard d p gamma ≤ nullValueD2 G guard d p) ∧
      (boundKCX G guard d p gamma < gamma →
        nullValueD2 G guard d p ≤ boundKCX G guard d p gamma) := by
  intro d p gamma h1 h2
  rw [production_eq_reference G guard kill hB hQ hF h240 hNM hV hCF hK d p gamma h1 h2]
  exact bound_null_spec G guard kill hB hK hR d p gamma h1 h2

/-- **VerifiedSearchNoCrossing**, production form: two driver-range
probes of the same `(pos, depth)` never store contradictory bounds --
zugzwang-free, bet-free, exactness restored.  (Table lookups and
repetition are the CanNull layer; `HistoryLegal` below closes the one
path that precedes the consumer.) -/
theorem kcx_no_crossing (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    {B : Int} (hF : ValFloor G B) (h240 : B ≤ 240) (hNM : NoMaskedMobility G)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G)
    (hK : KillerLegal G kill)
    (hR : NoZugzwangInMateBand G guard)
    (d : Nat) (p : G.Pos) (g1 g2 : Int)
    (hg1a : -MATE_LOWER < g1) (hg1b : g1 ≤ MATE_LOWER)
    (hg2a : -MATE_LOWER < g2) (hg2b : g2 ≤ MATE_LOWER)
    (hhi : g1 ≤ boundKCX G guard d p g1)
    (hlo : boundKCX G guard d p g2 < g2) :
    boundKCX G guard d p g1 ≤ boundKCX G guard d p g2 := by
  have h1 := (boundKCX_null_spec G guard kill hB hQ hF h240 hNM hV hCF hK hR
    d p g1 hg1a hg1b).1 hhi
  have h2 := (boundKCX_null_spec G guard kill hB hQ hF h240 hNM hV hCF hK hR
    d p g2 hg2a hg2b).2 hlo
  omega

/-- The chess-facing production spec: under `NoZugzwang`, production
brackets the real-move value. -/
theorem boundKCX_spec (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hQ : EvalQuiet G.toNullGame.toGame)
    {B : Int} (hF : ValFloor G B) (h240 : B ≤ 240) (hNM : NoMaskedMobility G)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G)
    (hK : KillerLegal G kill)
    (hZ : NoZugzwang G guard) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_LOWER < gamma → gamma ≤ MATE_LOWER →
      BoundSpecD2 G d p gamma (boundKCX G guard d p gamma) := by
  intro d p gamma h1 h2
  rw [production_eq_reference G guard kill hB hQ hF h240 hNM hV hCF hK d p gamma h1 h2]
  exact boundD2_spec G guard kill hB hK hZ d p gamma h1 h2

/-- The verified loop on the countermodel that broke the pre-kcx
design, production consumer: exact 0 at both windows (the +30 pass is
normalized -- positive claim at a verified terminal -- and the
correction stores the draw). -/
theorem kcx_repairs_cexT :
    boundKCX CexT (fun _ => true) 4 () 10 = 0 ∧
    boundKCX CexT (fun _ => true) 4 () 100 = 0 :=
  ⟨by decide, by decide⟩

/-! ### The history hypothesis -/

/-- **HistoryLegal** (input validity, fidelity-class like `Bounded`):
positions in the game history never have a capturable king -- every
position that was actually reached came from a legal move.  This closes
the one theoretical hole the production consumer leaves open: the
repetition check PRECEDES the consumer, so a king-capturable position
inside `history` would return the repetition 0 and evade the restored
invariant (the reference dodges it via the eager entry scan).  Under
this hypothesis the repetition-0 and the invariant's subject are
disjoint. -/
def HistoryLegal (G : NullGame) (hist : G.Pos → Bool) : Prop :=
  ∀ p, hist p = true → hasKingCapture G.toGame p = false

/-- Under `HistoryLegal` the repetition-0 can never mask the sentinel:
no king-capturable position is a repetition hit. -/
theorem repetition_never_masks (G : NullGame) (hist : G.Pos → Bool)
    (hH : HistoryLegal G hist) (p : G.Pos)
    (hcap : hasKingCapture G.toGame p = true) : hist p = false := by
  cases h : hist p with
  | false => rfl
  | true =>
    rw [hH p h] at hcap
    exact Bool.noConfusion hcap

/-! ### The reduced scan's countermodel, the warm oracle, the fast path,
and the clamped driver -/

/-- Countermodel for the premise-free reduced scan: `Q` has one
admitted-but-illegal move (`A`, whose child `KC` is a captured king)
and one legal move `F` filtered at depth 1 (value -150 < -100).  `Q`
returns the raw identity although its scan SAW the legal move; `R`
converts that to a spurious `MATE_UPPER`; at depth 3 the probe-free
scan trusts the corrupted sentinel and `P` fires a false terminal.
`NoMaskedMobility` fails exactly at `Q`. -/
inductive MPos where
  | P | R | Q | A | F | KC
  deriving DecidableEq

open MPos in
def CexM : QSGame where
  Pos := MPos
  moves := fun x => match x with
    | P => [R]
    | R => [Q]
    | Q => [A, F]
    | A => [KC]
    | _ => []
  eval := fun x => match x with
    | KC => -60000
    | _ => 0
  pass := fun x => x
  val := fun x m => match x, m with
    | Q, F => -150
    | _, _ => 0

/-- The golfed gate fires a false terminal at `P` where the oracle gate
correctly declines; `NoMaskedMobility` is violated.  The reduced scan
is NOT premise-free -- recorded next to the golf agent's own two
countermodels (the raw null fail-low outscoring a legal yield, and
depth-1 futility skipping admitted moves). -/
theorem reducedScan_needs_premise :
    boundKCX CexM (fun _ => false) 3 MPos.P 5 = 0 ∧
    boundD2 CexM (fun _ => false) (fun _ => false) 3 MPos.P 5 = -MATE_UPPER ∧
    ¬ NoMaskedMobility CexM := by
  refine ⟨by decide, by decide, fun h => ?_⟩
  have := h MPos.Q (by decide) MPos.F
    (show MPos.F ∈ [MPos.A, MPos.F] from List.mem_cons_of_mem _ (List.mem_cons_self _ _))
  exact absurd this (by decide)

/-- **The warm oracle** (both probe sites store since `c72cf6d`): the
probe key is always `(child, 0)`, and at depth 0 the declared value IS
capturability -- `MATE_UPPER` exactly at capturable children, a
sub-band static value otherwise.  So any sound table bracket of the
depth-0 value decides the probe: a stored `lower = MATE_UPPER` proves
capturable (the converse invariant), and a stored `upper < MATE_UPPER`
proves legal (write-exactness: capturable children never store one,
because every depth-0 return there is the exact sentinel).  The warm
probe therefore answers exactly like the cold one, by the same point-
spec table induction that carries `kcx_no_crossing`. -/
theorem nullValueD2_zero_MU_iff (G : QSGame) (guard : G.Pos → Bool)
    (hQ : EvalQuiet G.toNullGame.toGame) (c : G.Pos)
    (hkg : ¬ (G.eval c ≤ -MATE_LOWER)) :
    nullValueD2 G guard 0 c = MATE_UPPER
      ↔ hasKingCapture G.toNullGame.toGame c = true := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hq := hQ c hkg
  simp only [nullValueD2]
  rw [if_neg hkg]
  constructor
  · intro h
    cases hc : hasKingCapture G.toNullGame.toGame c with
    | true => rfl
    | false =>
      rw [if_neg (by simp [hc])] at h
      omega
  · intro hc
    rw [if_pos hc]

theorem warm_probe_decides (G : QSGame) (guard : G.Pos → Bool)
    (hQ : EvalQuiet G.toNullGame.toGame)
    (c : G.Pos) (hkg : ¬ (G.eval c ≤ -MATE_LOWER)) (lo hi : Int)
    (hlo : lo ≤ nullValueD2 G guard 0 c) (hhi : nullValueD2 G guard 0 c ≤ hi) :
    (MATE_UPPER ≤ lo → hasKingCapture G.toNullGame.toGame c = true) ∧
    (hi < MATE_UPPER → hasKingCapture G.toNullGame.toGame c = false) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hq := hQ c hkg
  constructor
  · intro h
    refine (nullValueD2_zero_MU_iff G guard hQ c hkg).mp ?_
    simp only [nullValueD2] at hlo hhi ⊢
    rw [if_neg hkg] at hlo hhi ⊢
    by_cases hc : hasKingCapture G.toNullGame.toGame c = true
    · rw [if_pos hc]
    · rw [if_neg hc] at hlo ⊢
      omega
  · intro h
    cases hc : hasKingCapture G.toNullGame.toGame c with
    | false => rfl
    | true =>
      exfalso
      have := (nullValueD2_zero_MU_iff G guard hQ c hkg).mpr hc
      omega

/-- **The killer fast path** (`king = self.tp_move.get(pos) or
pos.king_capture(); if king and pos.value(king) >= MATE_LOWER`): with
`KillerInv` and `KillerAtKingCapturable` as INPUTS -- they were the
lifecycle's preserved outputs; the fast path is where they become
load-bearing -- the O(1) value test decides capturability exactly.
Fidelity-class caveat: a LIVE `tp_move` carries entries the lifecycle
theorems built under the current store discipline; entries written by
other engine versions (or a corrupted table) void the input. -/
def KillerAtKingCapturable (G : QSGame) (killM : G.Pos → Option G.Pos) : Prop :=
  ∀ p k, killM p = some k → hasKingCapture G.toNullGame.toGame p = true →
    G.eval k ≤ -MATE_LOWER

theorem fastPath_decides (G : QSGame) (killM : G.Pos → Option G.Pos)
    (hV : KingCaptureValHigh G) (hHi : HighValIsKingCapture G)
    (hInv : KillerInv G killM) (hKAK : KillerAtKingCapturable G killM)
    (p : G.Pos) :
    (match killM p with
      | some k => decide (MATE_LOWER ≤ G.val p k)
      | none => hasKingCapture G.toNullGame.toGame p)
      = hasKingCapture G.toNullGame.toGame p := by
  cases hk : killM p with
  | none => rfl
  | some k =>
    obtain ⟨hm, _⟩ := hInv p k hk
    cases hc : hasKingCapture G.toNullGame.toGame p with
    | true => exact decide_eq_true (hV p k hm (hKAK p k hk hc))
    | false =>
      apply decide_eq_false
      intro hval
      have := (hasKingCapture_iff G.toNullGame.toGame p).mpr
        ⟨k, hm, hHi p k hm hval⟩
      rw [hc] at this
      exact Bool.noConfusion this

/-- A quiet killer certifies mobility, so the terminal arm may skip its
scan (`not king`). -/
theorem fastPath_skip_sound (G : QSGame) (killM : G.Pos → Option G.Pos)
    (hV : KingCaptureValHigh G) (hInv : KillerInv G killM)
    (p k : G.Pos) (hk : killM p = some k)
    (hquiet : G.val p k < MATE_LOWER) :
    allIllegalB G p = false := by
  obtain ⟨hm, hor⟩ := hInv p k hk
  cases hor with
  | inl hcapk => exact absurd (hV p k hm hcapk) (by omega)
  | inr hleg => exact allIllegalB_false_of_legal hm hleg

end Sunfish
