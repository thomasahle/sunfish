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

end Sunfish
