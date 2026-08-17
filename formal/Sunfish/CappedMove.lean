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
cap disappears above depth three.  The implementation evaluates this fixed
fold lazily: at a given window it searches only the moves whose cap can still
reach it, and aggregates every move below that threshold into ONE maximum-cap
report.  The cap is monotone in the intrinsic move value, so that one number
is the cap of the largest move in the tail - which, in Python's decreasing
sort, is simply the first entry past the partition.  The threshold changes
with the window, but the value being reported does not; the last section
proves the aggregate reports on the same capped fold for every threshold, and
that either evaluation order (tail first, or prefix first with the report
emitted last) is exactly the original producer fold.
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

/-! ### Lazy evaluation of the fixed capped fold -/

/-- The fixed producer floor in Python: the tactical QS threshold at depth
zero and the table sentinel at every positive depth. -/
def producerFloor (depth : Nat) : Int :=
  if depth = 0 then QS else -MATE_UPPER

/-- The intrinsic-value threshold derived by solving
`static + gain + margin < gamma` for `gain`.  The `MATE_LOWER` clamp keeps
king captures in the searched prefix.  Above the capped horizon the threshold
is just the producer floor, so the tail is empty. -/
def lazyMoveThreshold (static gamma : Int) (depth : Nat) : Int :=
  if depth ≤ 3 then
    max (producerFloor depth)
      (min MATE_LOWER
        (gamma - static - ((depth - 1 : Nat) : Int) * QS_A))
  else producerFloor depth

/-- Moves whose cap may reach the current window.  This is an evaluation
order only; membership in the declared producer remains `producerMoves`. -/
def lazyMovePrefix (G : QSGame) (static gamma : Int) (depth : Nat)
    (p : G.Pos) : List G.Pos :=
  (producerMoves G depth p).filter
    (fun m => decide (lazyMoveThreshold static gamma depth ≤ G.val p m))

/-- The complementary producer tail.  Its children need not be searched:
one maximum of their fixed caps is a valid upper report for the whole tail. -/
def lazyMoveTail (G : QSGame) (static gamma : Int) (depth : Nat)
    (p : G.Pos) : List G.Pos :=
  (producerMoves G depth p).filter
    (fun m => !(decide (lazyMoveThreshold static gamma depth ≤ G.val p m)))

/-- Every produced move clears the concrete Python `base`.  At depth zero
this is true by construction; at positive depth it follows from the shipped
move-value floor. -/
theorem producerMoves_above_floor (G : QSGame) (hF : ValFloor G 211)
    (depth : Nat) (p m : G.Pos) (hm : m ∈ producerMoves G depth p) :
    producerFloor depth ≤ G.val p m := by
  by_cases hd : depth = 0
  · subst depth
    have hmem := (mem_movesAbove.mp (by simpa [producerMoves] using hm)).2
    simpa [producerFloor] using hmem
  · have hmove : m ∈ G.moves p := by
      simpa [producerMoves, hd] using hm
    have hval := hF p m hmove
    have hMU : MATE_UPPER = 69290 := rfl
    simp only [producerFloor, if_neg hd]
    omega

/-- Membership in the lazy tail is exactly producer membership below the
derived threshold. -/
theorem mem_lazyMoveTail {G : QSGame} {static gamma : Int} {depth : Nat}
    {p m : G.Pos} :
    m ∈ lazyMoveTail G static gamma depth p ↔
      m ∈ producerMoves G depth p ∧
        G.val p m < lazyMoveThreshold static gamma depth := by
  simp [lazyMoveTail, List.mem_filter]
  omega

/-- A tail move's fixed cap lies below the current window.  This is the
arithmetic correspondence between Python's intrinsic threshold and the cap;
no chess or child-search premise is involved. -/
theorem lazyMoveTail_cap_lt_gamma (G : QSGame) (hF : ValFloor G 211)
    (static gamma : Int) (depth : Nat) (p m : G.Pos) (hdepth : depth ≤ 3)
    (hm : m ∈ lazyMoveTail G static gamma depth p) :
    shallowMoveCap static (G.val p m) depth < gamma := by
  have hmem := (mem_lazyMoveTail.mp hm).1
  have hbase := producerMoves_above_floor G hF depth p m hmem
  have htail := (mem_lazyMoveTail.mp hm).2
  let margin : Int := ((depth - 1 : Nat) : Int) * QS_A
  have ht : G.val p m < max (producerFloor depth)
      (min MATE_LOWER (gamma - static - margin)) := by
    simpa [lazyMoveThreshold, hdepth, margin] using htail
  have hmin : G.val p m < min MATE_LOWER (gamma - static - margin) := by
    by_cases hle : min MATE_LOWER (gamma - static - margin) ≤ G.val p m
    · have hmax : max (producerFloor depth)
          (min MATE_LOWER (gamma - static - margin)) ≤ G.val p m := by
        simp only [Int.max_def]
        split <;> omega
      omega
    · omega
  have hx : G.val p m < gamma - static - margin := by
    have := Int.min_le_right MATE_LOWER (gamma - static - margin)
    omega
  have hcap : shallowMoveCap static (G.val p m) depth ≤
      static + G.val p m + margin := by
    unfold shallowMoveCap
    simpa [margin] using
      (Int.min_le_right (MATE_LOWER - 1)
        (static + G.val p m + ((depth - 1 : Nat) : Int) * QS_A))
  omega

/-- `max` combines two reports at the same window into a report for the
maximum of their fixed values. -/
theorem WindowReport.max {gamma ra rb a b : Int}
    (ha : WindowReport gamma ra a) (hb : WindowReport gamma rb b) :
    WindowReport gamma (max ra rb) (max a b) := by
  rcases ha with ha | ha <;> rcases hb with hb | hb <;>
    simp only [WindowReport] at * <;> omega

/-- Pointwise reports fold to a report for the pointwise fixed-value fold. -/
theorem foldMax_windowReports {α : Type _} (gamma : Int)
    (report value : α → Int) (l : List α) (reportInit valueInit : Int)
    (hinit : WindowReport gamma reportInit valueInit)
    (hall : ∀ a ∈ l, WindowReport gamma (report a) (value a)) :
    WindowReport gamma
      (foldMax report l reportInit) (foldMax value l valueInit) := by
  induction l generalizing reportInit valueInit with
  | nil => simpa [foldMax] using hinit
  | cons a l ih =>
    simp only [foldMax]
    exact ih (max reportInit (report a)) (max valueInit (value a))
      (WindowReport.max hinit (hall a (by simp)))
      (fun x hx => hall x (List.mem_cons_of_mem a hx))

/-- The one emitted maximum-tail-cap report is valid for the fold of the
actual capped tail values. -/
theorem lazyMoveTail_report (G : QSGame) (hF : ValFloor G 211)
    (static gamma : Int) (depth : Nat) (p : G.Pos) (full : G.Pos → Int)
    (hdepth : depth ≤ 3) (hwindow : LOSS < gamma) :
    WindowReport gamma
      (foldMax (fun m => shallowMoveCap static (G.val p m) depth)
        (lazyMoveTail G static gamma depth p) LOSS)
      (foldMax (fun m => min (shallowMoveCap static (G.val p m) depth) (full m))
        (lazyMoveTail G static gamma depth p) LOSS) := by
  apply foldMax_windowReports gamma _ _ _ _ _
  · exact Or.inl ⟨hwindow, Int.le_refl LOSS⟩
  · intro m hm
    exact WindowReport.cap_failLow _ _ _
      (lazyMoveTail_cap_lt_gamma G hF static gamma depth p m hdepth hm)

/-- Evaluating the tail first and the prefix second is exactly the original
producer fold.  This is pure `max` algebra; the gamma-dependent partition
changes evaluation order, never the fixed value. -/
theorem lazyMove_partition (G : QSGame) (static gamma : Int) (depth : Nat)
    (p : G.Pos) (value : G.Pos → Int) (init : Int) (hinit : LOSS ≤ init) :
    foldMax value (lazyMovePrefix G static gamma depth p)
        (foldMax value (lazyMoveTail G static gamma depth p) init) =
      foldMax value (producerMoves G depth p) init := by
  let f := fun m => decide (lazyMoveThreshold static gamma depth ≤ G.val p m)
  have hsplit := foldMax_filter_split value f (producerMoves G depth p) init
  have hp := foldMax_init_split value ((producerMoves G depth p).filter f) init hinit
  have ht := foldMax_init_split value
    ((producerMoves G depth p).filter (fun m => !(f m))) init hinit
  have htailFloor := foldMax_ge_init value
    ((producerMoves G depth p).filter (fun m => !(f m))) init
  have hseq := foldMax_init_split value ((producerMoves G depth p).filter f)
    (foldMax value ((producerMoves G depth p).filter (fun m => !(f m))) init)
    (Int.le_trans hinit htailFloor)
  simp only [lazyMovePrefix, lazyMoveTail, f] at *
  omega

/-! ### One number for the tail, and either evaluation order -/

/-- The shallow cap is monotone in the intrinsic move value, so `max` passes
straight through it. -/
theorem shallowMoveCap_max (static a b : Int) (depth : Nat) :
    shallowMoveCap static (max a b) depth =
      max (shallowMoveCap static a depth) (shallowMoveCap static b depth) := by
  unfold shallowMoveCap
  generalize (MATE_LOWER - 1 : Int) = ceiling
  generalize ((depth - 1 : Nat) : Int) * QS_A = margin
  omega

/-- Hence the maximum of a list of caps is the cap of the maximum intrinsic
value.  This is what lets Python report the tail with ONE number instead of
folding the tail's caps: it caps the largest tail move and stops. -/
theorem foldMax_shallowMoveCap {α : Type _} (static : Int) (depth : Nat)
    (val : α → Int) : ∀ (l : List α) (init : Int),
      foldMax (fun a => shallowMoveCap static (val a) depth) l
          (shallowMoveCap static init depth)
        = shallowMoveCap static (foldMax val l init) depth := by
  intro l
  induction l with
  | nil => intro init; simp [foldMax]
  | cons a l ih =>
    intro init
    simp only [foldMax]
    rw [← shallowMoveCap_max]
    exact ih (max init (val a))

/-- The tail's aggregate cap, as Python computes it: the cap of the tail's
single largest intrinsic value.  In the decreasing sort that value is the
first entry past the partition, so the whole tail costs one lookup. -/
theorem lazyMoveTail_maxCap (G : QSGame) (static gamma : Int) (depth : Nat)
    (p : G.Pos) (init : Int) :
    foldMax (fun m => shallowMoveCap static (G.val p m) depth)
        (lazyMoveTail G static gamma depth p) (shallowMoveCap static init depth)
      = shallowMoveCap static
          (foldMax (G.val p) (lazyMoveTail G static gamma depth p) init) depth :=
  foldMax_shallowMoveCap static depth (G.val p) _ init

/-- Prefix first and tail second is also exactly the producer fold.  `max` is
commutative, so the gamma-dependent partition is an evaluation ORDER in either
direction and never a change of value.  Python emits the tail report last,
which is why a prefix cutoff skips it entirely. -/
theorem lazyMove_partition_prefixFirst (G : QSGame) (static gamma : Int)
    (depth : Nat) (p : G.Pos) (value : G.Pos → Int) (init : Int)
    (hinit : LOSS ≤ init) :
    foldMax value (lazyMoveTail G static gamma depth p)
        (foldMax value (lazyMovePrefix G static gamma depth p) init) =
      foldMax value (producerMoves G depth p) init := by
  let f := fun m => decide (lazyMoveThreshold static gamma depth ≤ G.val p m)
  have hsplit := foldMax_filter_split value f (producerMoves G depth p) init
  have hp := foldMax_init_split value ((producerMoves G depth p).filter f) init hinit
  have ht := foldMax_init_split value
    ((producerMoves G depth p).filter (fun m => !(f m))) init hinit
  have hprefixFloor := foldMax_ge_init value ((producerMoves G depth p).filter f) init
  have hseq := foldMax_init_split value
    ((producerMoves G depth p).filter (fun m => !(f m)))
    (foldMax value ((producerMoves G depth p).filter f) init)
    (Int.le_trans hinit hprefixFloor)
  simp only [lazyMovePrefix, lazyMoveTail, f] at *
  omega

/-- When the tail is empty the prefix alone is the producer fold: this is the
window at which Python emits no tail report at all. -/
theorem lazyMove_partition_emptyTail (G : QSGame) (static gamma : Int)
    (depth : Nat) (p : G.Pos) (value : G.Pos → Int) (init : Int)
    (hinit : LOSS ≤ init) (hempty : lazyMoveTail G static gamma depth p = []) :
    foldMax value (lazyMovePrefix G static gamma depth p) init =
      foldMax value (producerMoves G depth p) init := by
  have h := lazyMove_partition_prefixFirst G static gamma depth p value init hinit
  rw [hempty] at h
  simpa [foldMax] using h

end Sunfish
