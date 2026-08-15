/-
The repetition rule, as part of the DEFINITION of the modeled game.

Until now the development described the modeled game as "chess without
draw rules" and treated `Searcher.bound`'s

    if depth > 0 and pos in self.history: return 0

as an unmodeled heuristic.  Thomas has since fixed the modeled game's
definition, and this file records it:

  **THE RULE.  A position that recurs from the GAME HISTORY SO FAR is a
  draw, valued 0.**

Under that definition the line above is not a heuristic at all: it is
EXACT TERMINAL SEMANTICS, carrying no pruning debt.  There is nothing to
widen, nothing to postpone, and no premise to discharge -- the search
returns the game's own value at such a node.

**The rule's scope, exactly** (all four clauses are load-bearing, and
each is visible in the source):

1. *Game history only.*  `self.history` is `set(history)`, assigned once
   per `search()` call from the positions the GAME has actually reached.
   The search never inserts its own nodes.  So a repetition arising
   entirely INSIDE a search line is NOT detected: only a return to a
   position of the real game triggers the rule.  `hist` below is
   therefore a fixed predicate, constant for the whole search -- which
   is also why the value function stays `(pos, depth)`-determined.
2. *One recurrence, not three.*  A single match with the game history
   draws.  This is deliberately simpler than FIDE's threefold rule; it
   is the modeled game's own definition, not an approximation of FIDE's.
   (Nor is it 50-move or insufficient material: those remain outside the
   modeled game -- see the scope note in `Classification.lean`.)
3. *Positive depth only.*  The check is skipped at `depth = 0`, where it
   would be expensive and would break futility pruning.  The model
   mirrors this: the depth-0 arm is untouched.
4. *Non-root only.*  Driver probes (`root=True`: the search root and the
   IID probe, and in PR #171 the QS-tail retry) skip the check, because
   the root is itself in `history` without being a draw.  The model's
   value function is the NON-root function, exactly as everywhere else
   in this development.

**A position is the full key.**  `Position` carries board, side to move
(by rotation), castling rights and the en-passant square, so "the same
position" in the rule means all of that agreeing -- the same notion the
table keys use.

**What this file proves** (`ValFloor` only; no chess premise, no
`EvalQuiet` needed for these arms):

* `fuelValueD2tH_of_history` -- the rule is exact: at a history
  position, at positive depth, the value IS 0.
* `repetition_not_lost` -- **the draw arm strengthens**.  A position
  from which the side to move has a LEGAL MOVE INTO THE GAME HISTORY is
  proven NOT LOST: its value is `≥ 0` at every sufficient depth.  Before
  the rule, the trichotomy could only say "no forced mate for either
  side implies the value stays inside the band"; now such a position
  carries a positive WITNESS (the repeating move) and a two-sided claim.
* `all_replies_repeat_forces_draw` -- and when every legal move repeats,
  the value is EXACTLY 0 above the fuel horizon: an outright proven
  draw, not merely the absence of a mate.
* `draw_arm_strengthened` -- the two facts stated against the
  trichotomy's own "neither" arm.

**Where the horizon matters, honestly.**  The exact-0 theorem is stated
for the real-only regime (`d ≥ 8`, `EventuallyWide.lean` Part I): below
the horizon the capped pass sits in the fold's initial accumulator and
can hold the value ABOVE 0, so only the `≥ 0` half survives there.  That
is precisely the pruning debt the fuel oracle retires -- the repetition
draw becomes exact exactly where the pass stops being a score candidate.

**What is NOT claimed here** (model-matches-code discipline).  The
`ForcedMate` / `ForcedlyMated` specs of `Liveness.lean` describe the
RULELESS game: they quantify over all legal replies with no repetition
escape.  Once the repetition rule is part of the game, a defender who
can repeat is not mated even if the ruleless spec says so, so the
finding and honesty arms of the trichotomy do NOT transfer verbatim to
`fuelValueD2tH`; they are stated and proven for `fuelValueD2t`.  The
faithful upgrade is a `ForcedMateH` spec whose defender may escape into
the history set, and re-running the two inductions against it.  That is
the recorded follow-up, deliberately not papered over: what is proven
below is exactly the draw side, which is what the rule buys.

Zero sorries; no Mathlib.
-/

import Sunfish.EventuallyWide

namespace Sunfish

/-! ### The history-augmented value function -/

/-- **The modeled game's value with the repetition rule.**  Identical to
`fuelValueD2t` except for the one new branch: at positive depth, a
position in the GAME history is a draw, valued 0.  The branch sits after
the king-capture normalization, which is immaterial on the reachable
domain (history positions are legal, so `hasKingCapture` is false there)
and keeps the sentinel exactly where every other theorem expects it.
`hist` is fixed for the whole search, so the function remains
`(pos, depth)`-determined. -/
def fuelValueD2tH (G : QSGame) (hist : G.Pos → Bool) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) : Nat → G.Pos → Int
  | 0, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if hist p = true then 0
    else if allIllegalB G p = true then terminalValue G (d + 1) p
    else if d + 1 < 8 then
      foldMax (fun m => -(fuelValueD2tH G hist guard C spend d m)) (tailList G (d + 1) p)
        (if guard p = true ∧ 2 < d + 1 then
          (if -(fuelValueD2tH G hist guard C spend (d + 1 - 5) (G.pass p)) < MATE_LOWER then
            max LOSS (-(fuelValueD2tH G hist guard C spend (d + 1 - 5) (G.pass p)))
          else LOSS)
        else LOSS)
    else
      foldMax (fun m => -(fuelValueD2tH G hist guard C spend
          (d - min (C - 1) (spend p (d + 1) m)) m))
        (tailList G (d + 1) p) LOSS
termination_by d _ => d
decreasing_by all_goals omega

/-! ### Branch lemmas -/

theorem fuelValueD2tH_kingGone (G : QSGame) (hist guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos) (h : G.eval p ≤ -MATE_LOWER) :
    fuelValueD2tH G hist guard C spend d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [fuelValueD2tH]; rw [if_pos h]
  | succ d => simp only [fuelValueD2tH]; rw [if_pos h]

theorem fuelValueD2tH_of_capture (G : QSGame) (hist guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos) (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    fuelValueD2tH G hist guard C spend d p = MATE_UPPER := by
  cases d with
  | zero => simp only [fuelValueD2tH]; rw [if_neg hkg, if_pos hcap]
  | succ d => simp only [fuelValueD2tH]; rw [if_neg hkg, if_pos hcap]

/-- **The rule is exact**: at positive depth a game-history position is
valued 0 -- no fold, no approximation, no debt. -/
theorem fuelValueD2tH_of_history (G : QSGame) (hist guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hh : hist p = true) :
    fuelValueD2tH G hist guard C spend (d + 1) p = 0 := by
  simp only [fuelValueD2tH]
  rw [if_neg hkg, if_neg hcap, if_pos hh]

theorem fuelValueD2tH_of_fold_regime (G : QSGame) (hist guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hh : hist p = false)
    (hai : allIllegalB G p = false)
    (hd : 7 ≤ d) :
    fuelValueD2tH G hist guard C spend (d + 1) p
      = foldMax (fun m => -(fuelValueD2tH G hist guard C spend
            (d - min (C - 1) (spend p (d + 1) m)) m))
          (tailList G (d + 1) p) LOSS := by
  simp only [fuelValueD2tH]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hh]), if_neg (by simp [hai]),
    if_neg (by omega)]

theorem fuelValueD2tH_of_fold_sub (G : QSGame) (hist guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hh : hist p = false)
    (hai : allIllegalB G p = false)
    (hd : d < 7) :
    fuelValueD2tH G hist guard C spend (d + 1) p
      = foldMax (fun m => -(fuelValueD2tH G hist guard C spend d m))
          (tailList G (d + 1) p)
          (if guard p = true ∧ 2 < d + 1 then
            (if -(fuelValueD2tH G hist guard C spend (d + 1 - 5) (G.pass p)) < MATE_LOWER then
              max LOSS (-(fuelValueD2tH G hist guard C spend (d + 1 - 5) (G.pass p)))
            else LOSS)
          else LOSS) := by
  simp only [fuelValueD2tH]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hh]), if_neg (by simp [hai]),
    if_pos (by omega)]

/-! ### The draw arm strengthens -/

/-- A legal move's child is neither kingless nor king-capturable, so its
history branch is the one that fires. -/
theorem legal_child_normal (G : QSGame) {p m : G.Pos}
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hm : m ∈ G.moves p)
    (hleg : hasKingCapture G.toNullGame.toGame m = false) :
    ¬ (G.eval m ≤ -MATE_LOWER) ∧ ¬ (hasKingCapture G.toNullGame.toGame m = true) := by
  refine ⟨fun hle => hcap ?_, by simp [hleg]⟩
  exact (hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, hle⟩

/-- **The side to move is NOT LOST when it can repeat.**  A legal move
into the game history gives the mover a drawing option the search
actually takes: the child is valued exactly 0 by the rule, so the fold
-- in either regime -- is at least 0.

The depth condition `C + 2 ≤ D` is what makes the child's own history
check fire: the reduced child depth is at least `D - C ≥ 2 - 1`, i.e.
positive, in the regime, and `D - 1 ≥ 1` below the horizon.

Premise: `ValFloor G 192` (fidelity, tables), which admits the repeating
move through the QS filter at nominal depth ≥ 2.  No chess premise. -/
theorem repetition_not_lost (G : QSGame) (hist guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192)
    {p m : G.Pos}
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hh : hist p = false)
    (hai : allIllegalB G p = false)
    (hm : m ∈ G.moves p)
    (hleg : hasKingCapture G.toNullGame.toGame m = false)
    (hrep : hist m = true) :
    ∀ D : Nat, C + 2 ≤ D → 0 ≤ fuelValueD2tH G hist guard C spend D p := by
  obtain ⟨hkgm, hcapm⟩ := legal_child_normal G hcap hm hleg
  intro D hD
  cases D with
  | zero => omega
  | succ d =>
    have hmem : m ∈ tailList G (d + 1) p :=
      mem_tailList_of_admitted G (mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm)
    by_cases hreg : 7 ≤ d
    · rw [fuelValueD2tH_of_fold_regime G hist guard C spend d p hkg hcap hh hai hreg]
      obtain ⟨dc, hdc⟩ : ∃ x, d - min (C - 1) (spend p (d + 1) m) = x + 1 :=
        ⟨d - min (C - 1) (spend p (d + 1) m) - 1, by omega⟩
      have hchild : fuelValueD2tH G hist guard C spend
          (d - min (C - 1) (spend p (d + 1) m)) m = 0 := by
        rw [hdc]
        exact fuelValueD2tH_of_history G hist guard C spend dc m hkgm hcapm hrep
      have hfold : -(fuelValueD2tH G hist guard C spend
            (d - min (C - 1) (spend p (d + 1) m)) m)
          ≤ foldMax (fun x => -(fuelValueD2tH G hist guard C spend
                (d - min (C - 1) (spend p (d + 1) x)) x))
              (tailList G (d + 1) p) LOSS :=
        foldMax_le_of_mem _ _ _ _ hmem
      omega
    · rw [fuelValueD2tH_of_fold_sub G hist guard C spend d p hkg hcap hh hai (by omega)]
      obtain ⟨dc, hdc⟩ : ∃ x, d = x + 1 := ⟨d - 1, by omega⟩
      have hchild : fuelValueD2tH G hist guard C spend d m = 0 := by
        rw [hdc]
        exact fuelValueD2tH_of_history G hist guard C spend dc m hkgm hcapm hrep
      have hfold : -(fuelValueD2tH G hist guard C spend d m)
          ≤ foldMax (fun x => -(fuelValueD2tH G hist guard C spend d x))
              (tailList G (d + 1) p)
              (if guard p = true ∧ 2 < d + 1 then
                (if -(fuelValueD2tH G hist guard C spend (d + 1 - 5) (G.pass p))
                    < MATE_LOWER then
                  max LOSS (-(fuelValueD2tH G hist guard C spend (d + 1 - 5) (G.pass p)))
                else LOSS)
              else LOSS) :=
        foldMax_le_of_mem _ _ _ _ hmem
      omega

/-- **An outright proven draw**: above the fuel horizon, if every legal
move repeats a game-history position then the value is EXACTLY 0.  The
upper bound needs the regime (`7 ≤ d`), where the fold starts from
`LOSS`: illegal members contribute the negated sentinel (`LOSS`), legal
ones the negated exact 0, so nothing can lift the fold above 0; the
lower bound is `repetition_not_lost`'s witness.

Below the horizon only the `≥ 0` half holds -- the capped pass sits in
the initial accumulator and can hold the value above 0.  That gap is
exactly the pruning debt the fuel oracle retires. -/
theorem all_replies_repeat_forces_draw (G : QSGame) (hist guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192)
    {p m : G.Pos}
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hh : hist p = false)
    (hai : allIllegalB G p = false)
    (hm : m ∈ G.moves p)
    (hleg : hasKingCapture G.toNullGame.toGame m = false)
    (hall : ∀ m' ∈ G.moves p,
      hasKingCapture G.toNullGame.toGame m' = false → hist m' = true) :
    ∀ d : Nat, 7 ≤ d → C + 1 ≤ d →
      fuelValueD2tH G hist guard C spend (d + 1) p = 0 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d hd hCd
  have hlow := repetition_not_lost G hist guard C spend hC hF hkg hcap hh hai hm hleg
    (hall m hm hleg) (d + 1) (by omega)
  rw [fuelValueD2tH_of_fold_regime G hist guard C spend d p hkg hcap hh hai hd]
  rw [fuelValueD2tH_of_fold_regime G hist guard C spend d p hkg hcap hh hai hd] at hlow
  have hup : foldMax (fun x => -(fuelValueD2tH G hist guard C spend
        (d - min (C - 1) (spend p (d + 1) x)) x))
      (tailList G (d + 1) p) LOSS ≤ 0 := by
    refine foldMax_le _ _ _ (fun m' hm' => ?_) (by omega)
    have hm'm : m' ∈ G.moves p := tailList_subset G _ p m' hm'
    cases hcm : hasKingCapture G.toNullGame.toGame m' with
    | true =>
      have hkgm' : ¬ (G.eval m' ≤ -MATE_LOWER) := fun hle =>
        hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m', hm'm, hle⟩)
      show -(fuelValueD2tH G hist guard C spend
        (d - min (C - 1) (spend p (d + 1) m')) m') ≤ 0
      rw [fuelValueD2tH_of_capture G hist guard C spend
        (d - min (C - 1) (spend p (d + 1) m')) m' hkgm' hcm]
      omega
    | false =>
      obtain ⟨hkgm', hcapm'⟩ := legal_child_normal G hcap hm'm hcm
      show -(fuelValueD2tH G hist guard C spend
        (d - min (C - 1) (spend p (d + 1) m')) m') ≤ 0
      obtain ⟨dc, hdc⟩ : ∃ x, d - min (C - 1) (spend p (d + 1) m') = x + 1 :=
        ⟨d - min (C - 1) (spend p (d + 1) m') - 1, by omega⟩
      rw [hdc, fuelValueD2tH_of_history G hist guard C spend dc m' hkgm' hcapm'
        (hall m' hm'm hcm)]
      omega
  omega

/-- **The trichotomy's draw arm, strengthened by the rule.**  At a
position with a repetition available, the "neither" arm's two-sided band
claim is upgraded: the value is not merely inside the band, it is at
least 0 (proven not lost, with the repeating move as witness) -- and
exactly 0 above the horizon when every legal move repeats.

Read against `eventual_classification_fuel`: positions that the ruleless
classification could only place in "no forced mate for either side" are
now, in the modeled game with the repetition rule, DRAWS WITH A
CERTIFICATE. -/
theorem draw_arm_strengthened (G : QSGame) (hist guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192)
    {p m : G.Pos}
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hh : hist p = false)
    (hai : allIllegalB G p = false)
    (hm : m ∈ G.moves p)
    (hleg : hasKingCapture G.toNullGame.toGame m = false)
    (hrep : hist m = true) :
    (∀ D : Nat, C + 2 ≤ D →
        0 ≤ fuelValueD2tH G hist guard C spend D p ∧
        ¬ (fuelValueD2tH G hist guard C spend D p ≤ -MATE_LOWER)) ∧
      ((∀ m' ∈ G.moves p,
          hasKingCapture G.toNullGame.toGame m' = false → hist m' = true) →
        ∀ d : Nat, 7 ≤ d → C + 1 ≤ d →
          fuelValueD2tH G hist guard C spend (d + 1) p = 0) := by
  have hML : MATE_LOWER = 47923 := rfl
  refine ⟨fun D hD => ?_, fun hall d hd hCd =>
    all_replies_repeat_forces_draw G hist guard C spend hC hF hkg hcap hh hai hm hleg
      hall d hd hCd⟩
  have h := repetition_not_lost G hist guard C spend hC hF hkg hcap hh hai hm hleg hrep D hD
  exact ⟨h, by omega⟩

end Sunfish
