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

end Sunfish
