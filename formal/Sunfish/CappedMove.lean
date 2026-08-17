/-
The unified shallow static move cap and lazy depth-one tail in Searcher.bound.

At depths zero through three, every move except a king capture passes through
one producer with cap

    min (MATE_LOWER - 1) (static + gain + (depth - 1) * QS_A).

Natural subtraction makes the margin zero at depths zero and one. There the
cap is the existing exact stand-pat futility estimate: the score identity and
`futilityOK_discharged` show that it targets the ordinary child value. At
depths two and three it instead declares the move value to be the minimum of
the cap and the full child value. If the cap lies below the window, the child
need not be searched; otherwise `WindowReport.cap` transports its report.

Moves are sorted by decreasing gain. The cap is monotone in gain, so once a
move returns a virtual cap, that report also dominates the rest of the tail.
Only king captures bypass the cap and retain the exact `MATE_UPPER` sentinel.
The selective cap can delay a shallow mate proof, but it cannot invent one,
and it disappears above depth three.

At remaining depth one, the omitted moves' best possible stand-pat is emitted
as a fail-low upper report.  If that report cannot fail low, the threshold
widens to the table-proved move floor and the complete tail is searched.  Thus
the depth-one report targets the complete move fold without paying to generate
the tail at ordinary windows.
-/

import Sunfish.CappedNull
import Sunfish.EvalBounds
import Sunfish.Stalemate

namespace Sunfish

/-- The exact fixed cap used for an eligible Python move. -/
def shallowMoveCap (static gain : Int) (depth : Nat) : Int :=
  min (MATE_LOWER - 1) (static + gain + ((depth - 1 : Nat) : Int) * QS_A)

/-- At depths zero and one, natural subtraction makes the margin vanish.
Under the ordinary-move evaluation bound, the unified cap is exactly the old
stand-pat futility estimate. -/
theorem shallowMoveCap_lowDepth (static gain : Int) (depth : Nat)
    (hdepth : depth ≤ 1) (hband : static + gain < MATE_LOWER) :
    shallowMoveCap static gain depth = static + gain := by
  have hzero : depth - 1 = 0 := by omega
  simp [shallowMoveCap, hzero, Int.min_def]
  omega

/-- The cap follows the intrinsic move ordering: a later, no-higher-gain move
has a no-higher cap. This is the algebra behind ending the sorted stream after
its first virtual capped report. -/
theorem shallowMoveCap_mono_gain (static first later : Int) (depth : Nat)
    (hgain : later ≤ first) :
    shallowMoveCap static later depth ≤ shallowMoveCap static first depth := by
  unfold shallowMoveCap
  simp only [Int.min_def]
  split <;> split <;> omega

/-- A cap below the current window is a complete fail-low report for the
capped value; no report about the full child is needed. -/
theorem cappedMove_failLow (cap gamma value : Int) (h : cap < gamma) :
    WindowReport gamma cap (min cap value) :=
  WindowReport.cap_failLow cap gamma value h

/-- When the child is searched, the generic monotone-cap theorem supplies
the report for the declared capped move value. -/
theorem cappedMove_report (cap gamma report value : Int)
    (h : WindowReport gamma report value) :
    WindowReport gamma (min cap report) (min cap value) :=
  h.cap cap gamma report value

/-- The explicit ceiling keeps every eligible cap below the positive mate
band, independent of the static score and margin. -/
theorem shallowMoveCap_below_positiveMate (static gain : Int) (depth : Nat) :
    shallowMoveCap static gain depth < MATE_LOWER := by
  unfold shallowMoveCap
  simp only [Int.min_def]
  split <;> omega

/-- The lower clamp is unnecessary.  A both-kings child gives
`-MATE_LOWER < static + gain`; the shipped positive margin can only raise
that quantity, and the positive-band ceiling is itself above `-MATE_LOWER`. -/
theorem shallowMoveCap_above_negativeMate (static gain : Int) (depth : Nat)
    (hstatic : -MATE_LOWER < static + gain) :
    -MATE_LOWER < shallowMoveCap static gain depth := by
  have hML : MATE_LOWER = 47923 := rfl
  have hnn : (0 : Int) ≤ ((depth - 1 : Nat) : Int) := Int.ofNat_nonneg _
  unfold shallowMoveCap QS_A
  simp only [Int.min_def]
  split <;> omega

/-- Capping cannot create a positive mate report: any positive mate in the
capped value was already present in the full value. -/
theorem cappedMove_positiveMate_only_from_full (cap value : Int)
    (hcap : cap < MATE_LOWER) (h : MATE_LOWER ≤ min cap value) :
    MATE_LOWER ≤ value := by
  omega

/-- A cap above the negative mate band preserves a full negative mate
exactly.  This is the mate-soundness direction used by the parent fold. -/
theorem cappedMove_preserves_negativeMate (cap value : Int)
    (hcap : -MATE_LOWER < cap) (hvalue : value ≤ -MATE_LOWER) :
    min cap value = value := by
  omega

/-- Two reports at the same window can be joined through the node's `max`.
This is the report-algebra rule used to combine the searched prefix with the
omitted tail's upper report. -/
theorem WindowReport.max (gamma ra rb a b : Int)
    (ha : WindowReport gamma ra a) (hb : WindowReport gamma rb b) :
    WindowReport gamma (max ra rb) (max a b) := by
  rcases ha with ha | ha <;> rcases hb with hb | hb <;>
    simp only [WindowReport, Int.max_def] at * <;>
    split <;> split <;> omega

/-- Integer move values below the threshold are at most `threshold - 1`.
The usual depth-one stand-pat bound therefore gives one upper report for the
entire omitted tail. -/
theorem omittedMove_le_tailCap (static value threshold moveValue : Int)
    (hvalue : value < threshold) (hmove : moveValue ≤ static + value) :
    moveValue ≤ static + threshold - 1 := by
  omega

/-- When the tail cap is below the window it is a valid fail-low report; no
omitted child needs to be generated or searched. -/
theorem lazyTail_failLow (gamma tailCap tailValue : Int)
    (hcap : tailCap < gamma) (htail : tailValue ≤ tailCap) :
    WindowReport gamma tailCap tailValue := by
  exact Or.inl ⟨hcap, htail⟩

/-- Combining the searched prefix with a safely skipped tail still reports on
the complete depth-one move fold. -/
theorem lazyTail_report (gamma prefixReport tailCap prefixValue tailValue : Int)
    (hprefix : WindowReport gamma prefixReport prefixValue)
    (hcap : tailCap < gamma) (htail : tailValue ≤ tailCap) :
    WindowReport gamma (max prefixReport tailCap) (max prefixValue tailValue) := by
  exact hprefix.max gamma prefixReport tailCap prefixValue tailValue
    (lazyTail_failLow gamma tailCap tailValue hcap htail)

/-- The fixed move set denoted by the lazy implementation: all moves at the
depth-one frontier, and the ordinary threshold above it. -/
def lazyMoves (G : QSGame) (depth : Nat) (p : G.Pos) : List G.Pos :=
  if depth = 1 then G.moves p else movesAbove G (val_lower depth) p

/-- With the shipped table floor, the lazy search denotes the complete real
move list at every positive depth. Depth one is complete by definition;
depth two and above already clear the intrinsic-value floor. -/
theorem lazyMoves_eq_moves (G : QSGame) (hF : ValFloor G 192)
    (depth : Nat) (p : G.Pos) (hdepth : 1 ≤ depth) :
    lazyMoves G depth p = G.moves p := by
  by_cases h1 : depth = 1
  · simp [lazyMoves, h1]
  · rw [lazyMoves, if_neg h1]
    exact movesAbove_all G depth p
      (depth_arm_redundant G hF (by omega) depth (by omega) p)

end Sunfish
