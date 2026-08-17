/-
The positive-depth move producer and shallow static cap in Searcher.bound.

At depth zero the producer retains the tuned quiescence threshold. At every
positive depth it emits the complete pseudo-legal move list, independently of
the window and move table. A mate-band intrinsic value is normalized directly
to `MATE_UPPER`; `HighValIsKingCapture` says that this branch is exactly a king
capture, whose recursive child would immediately return `-MATE_UPPER`.

Every other move at depths zero through three has the fixed cap

    min (MATE_LOWER - 1) (static + gain + (depth - 1) * QS_A).

Natural subtraction makes the margin zero at depths zero and one. There the
cap is the existing exact stand-pat futility estimate: the score identity and
`futilityOK_discharged` show that it targets the ordinary child value. At
depths two and three it instead declares the move value to be the minimum of
the cap and the full child value. If the cap lies below the window, the child
need not be searched; otherwise `WindowReport.cap` transports its report.

The positive-band ceiling prevents a selective cap from inventing mate. The
cap disappears above depth three.
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

/-- The exact fixed producer set in Python: tactical moves at quiescence,
and every pseudo-legal move at positive depth. -/
def producerMoves (G : QSGame) (depth : Nat) (p : G.Pos) : List G.Pos :=
  if depth = 0 then movesAbove G QS p else G.moves p

theorem producerMoves_zero (G : QSGame) (p : G.Pos) :
    producerMoves G 0 p = movesAbove G QS p := by
  simp [producerMoves]

/-- Positive-depth completeness is structural: it needs no score-floor or
window premise. In particular, a filtered legal evasion cannot fabricate a
mate at the old depth-one frontier. -/
theorem producerMoves_positive (G : QSGame) (depth : Nat) (p : G.Pos)
    (hdepth : 0 < depth) : producerMoves G depth p = G.moves p := by
  simp [producerMoves, Nat.ne_of_gt hdepth]

/-- The producer's exact report for an intrinsic mate-band move. Ordinary
move values remain unresolved until the consumer searches or caps them. -/
def producedScore (gain : Int) : Int :=
  if MATE_LOWER ≤ gain then MATE_UPPER else gain

theorem producedScore_capture (gain : Int) (hgain : MATE_LOWER ≤ gain) :
    producedScore gain = MATE_UPPER := by
  simp [producedScore, hgain]

theorem producedScore_ordinary (gain : Int) (hgain : gain < MATE_LOWER) :
    producedScore gain = gain := by
  simp [producedScore, Int.not_le.mpr hgain]

/-- Under the table-backed high-value premise, producer normalization never
turns an ordinary move into the exact king-capture sentinel. -/
theorem producedScore_exact_capture (G : QSGame) (hHi : HighValIsKingCapture G)
    (p m : G.Pos) (hm : m ∈ G.moves p) (hgain : MATE_LOWER ≤ G.val p m) :
    producedScore (G.val p m) = MATE_UPPER ∧ G.eval m ≤ -MATE_LOWER :=
  ⟨producedScore_capture (G.val p m) hgain, hHi p m hm hgain⟩

end Sunfish
