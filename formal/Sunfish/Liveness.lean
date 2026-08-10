/-
Mate-in-k completeness: the development's first LIVENESS theorem.

Everything proven about the shipped search so far is SAFETY: the search
never reports a bound its declared value function does not honor
(`bound_null_spec`), and under `NoZugzwang` the declared function is the
real-move value (`nullValue_eq_realValue_of_noZugzwang`).  Safety alone
is compatible with an engine that never finds any mate.  This file adds
the other direction: if the side to move has a forced mate in at most
`k` plies -- stated as a spec, in the model's own king-capture
vocabulary (`ForcedMate` below) -- then at every remaining depth
`D ≥ k + 1` the real-move value `negamaxD2` sits in the mate band
(`forcedMate_negamaxD2`), hence under `NoZugzwang` so does the declared
value `nullValueD2` that the shipped search provably brackets
(`forcedMate_complete`), hence no driver-range probe below the band can
fail low (`forcedMate_probe_failsHigh`, both consumers): the MTD-bi
bisection is forced to certify `MATE_LOWER ≤ lower`.  The engine FINDS
the mate; it does not merely avoid lying about it.

Two points the proof turns on:

* **The QS val-filter cannot hide the mating line.**  The attacker's
  chosen move must survive `movesAbove (val_lower depth)`, and
  `ValFloor G 192` (the shipped tables' floor,
  `EvalBounds.quietDropMax`) already clears the depth-2 threshold
  `val_lower 2 = -240` -- the same arithmetic as the retired gate's
  `tables_kill_filter_at_depth2`, respent here on the liveness side.
  Every attacker node of the induction sits at remaining depth ≥ 2, so
  NOTHING is filtered there.  On the defender side no floor is needed
  at all: a filtered-away defender reply only shrinks the defender's
  fold, which is the direction the bound needs.

* **Illegal defender replies are the sentinel's job.**  `ForcedMate`
  quantifies over LEGAL defender replies only (the standard chess
  reading); a reply that leaves the defender's own king capturable is
  answered by the model's by-construction king-capture branch
  (`negamaxD2_of_capture`, the exact `MATE_UPPER`), no hypothesis
  spent.  Symmetrically, "legal move" for the attacker means
  `hasKingCapture` is false at the reached position: the move did not
  leave the attacker's own king capturable.

Premises: `ValFloor G 192` (fidelity, tables) for the spine;
`NoZugzwang` (layer 2, chess) for the transfer -- Thomas's decision:
reuse the existing layer-2 assumption rather than change the engine.
`NoZugzwang` thereby gets its SECOND consumer: accuracy
(`nullValue_eq_realValue_of_noZugzwang`) and completeness (this file),
both layer 2.  Layer 1 and table consistency still carry no chess
statement.  No new chess premise is introduced anywhere here.

Design option, recorded in formal/README.md and NOT implemented: a
depth-decaying null guard (`abs(score) < 500 - 10*depth`) would switch
the null option off at large remaining depth and make completeness
unconditional -- at the cost of `D ≥ k + 52` and a code change.
-/

import Sunfish.Stalemate
import Sunfish.Driver
import Sunfish.TableSwap

namespace Sunfish

/-! ### The spec: forced mate in the king-capture vocabulary -/

/-- **Checkmated** (defender to move): every generated move loses the
king to the immediate recapture -- `allIllegalB`, the oracle scan the
d2 correction verifies (the `all(...)` legality scan over the FULL
`gen_moves()` list) -- AND the defender is in check (`inCheckB`, the
`pos.rotate(nullmove=True).king_capture()` probe).  Exactly the
condition under which `terminalValue` assigns `-MATE_LOWER`. -/
def Checkmated (G : QSGame) (q : G.Pos) : Prop :=
  allIllegalB G q = true ∧ inCheckB G.toNullGame q = true

/-- **ForcedMate G k p**: the side to move at `p` has a forced mate in
at most `k` plies.  A spec, kept as obviously correct as possible:

* `mate` -- some LEGAL move (one that does not leave the mover's own
  king capturable: `hasKingCapture` false at the child) reaches a
  `Checkmated` position.  One ply; stated at `k + 1` for any `k`, which
  is the "at most" reading.
* `step` -- some legal move reaches a position that is NOT terminal
  (the defender has a legal reply -- excluding the stalemate that the
  king-capture vocabulary would otherwise let a vacuous quantifier
  sneak past) such that EVERY legal defender reply hands the attacker a
  forced mate in `k`; two more plies.

Both constructors require the attacker's king on the board
(`¬ eval ≤ -MATE_LOWER`, the king-capture normalization's "not already
lost").  Defender replies that are themselves illegal are deliberately
NOT quantified over: the model refutes them with the exact `MATE_UPPER`
sentinel, no spec clause needed. -/
inductive ForcedMate (G : QSGame) : Nat → G.Pos → Prop where
  | mate {k : Nat} {p m : G.Pos}
      (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
      (hm : m ∈ G.moves p)
      (hleg : hasKingCapture G.toNullGame.toGame m = false)
      (hmate : Checkmated G m) :
      ForcedMate G (k + 1) p
  | step {k : Nat} {p m : G.Pos}
      (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
      (hm : m ∈ G.moves p)
      (hleg : hasKingCapture G.toNullGame.toGame m = false)
      (hnt : allIllegalB G m = false)
      (hreply : ∀ m' ∈ G.moves m,
        hasKingCapture G.toNullGame.toGame m' = false → ForcedMate G k m') :
      ForcedMate G (k + 2) p

/-! ### The filter clears the floor -/

/-- At remaining depth ≥ 2 the QS threshold sits at or below the
shipped tables' move-value floor (`val_lower 2 = -240 ≤ -192`):
the liveness respend of `tables_kill_filter_at_depth2`. -/
theorem val_lower_le_neg_floor (d : Nat) (h : 2 ≤ d) : val_lower d ≤ -192 := by
  unfold val_lower QS QS_A
  omega

/-- Under the tables' floor, EVERY legal move survives the QS filter at
remaining depth ≥ 2 -- in particular the mating move does. -/
theorem mem_movesAbove_of_floor (G : QSGame) (hF : ValFloor G 192)
    {d : Nat} (hd : 2 ≤ d) {p m : G.Pos} (hm : m ∈ G.moves p) :
    m ∈ movesAbove G (val_lower d) p := by
  rw [mem_movesAbove]
  have h1 := hF p m hm
  have h2 := val_lower_le_neg_floor d hd
  exact ⟨hm, by omega⟩

/-! ### The spine: the real-move layer -/

/-- **Mate-in-k completeness on the real-move layer**: a forced mate in
at most `k` plies puts `negamaxD2` in the mate band at every remaining
depth `D ≥ k + 1`.  (The task's natural guess was `k + 2`; the honest
minimum came out one ply better, because the checkmated leaf needs only
remaining depth 1 for its terminal branch.)

Induction over the `ForcedMate` derivation.  At the attacker's node the
chosen move survives the filter (`mem_movesAbove_of_floor`, the one
place `ValFloor` is spent) and its child bounds the fold from below
(`foldMax_le_of_mem`).  At the defender's node the fold is bounded from
above by `-MATE_LOWER` member-by-member (`foldMax_le`): an illegal
reply is refuted by the sentinel branch, a legal one by the induction
hypothesis two plies down, and the initial accumulator is `LOSS`.  The
checkmated leaf is the terminal branch: `allIllegalB` true and
`inCheckB` true give exactly `terminalValue = -MATE_LOWER` -- the
verified correction's mate arm, `-MATE_LOWER if ... king_capture()
else 0`. -/
theorem forcedMate_negamaxD2 (G : QSGame) (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, k + 1 ≤ D → MATE_LOWER ≤ negamaxD2 G D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  induction hFM with
  | @mate k p m hkg hm hleg hmate =>
    intro D hD
    cases D with
    | zero => omega
    | succ d =>
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [negamaxD2_of_capture G (d + 1) p hkg hcap]; omega
      · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
        rw [negamaxD2_of_fold G d p hkg hcap hai]
        have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
        have hchild : negamaxD2 G d m ≤ -MATE_LOWER := by
          cases d with
          | zero => omega
          | succ d' =>
            by_cases hkgm : G.eval m ≤ -MATE_LOWER
            · rw [negamaxD2_kingGone G (d' + 1) m hkgm]; omega
            · rw [negamaxD2_of_allIllegal G d' m hkgm (by simp [hleg]) hmate.1]
              simp only [terminalValue]
              rw [if_pos hmate.2]
              omega
        have hfold : -(negamaxD2 G d m)
            ≤ foldMax (fun x => -(negamaxD2 G d x))
                (movesAbove G (val_lower (d + 1)) p) LOSS :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega
  | @step k p m hkg hm hleg hnt _hreply ih =>
    intro D hD
    cases D with
    | zero => omega
    | succ d =>
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [negamaxD2_of_capture G (d + 1) p hkg hcap]; omega
      · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
        rw [negamaxD2_of_fold G d p hkg hcap hai]
        have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
        have hchild : negamaxD2 G d m ≤ -MATE_LOWER := by
          cases d with
          | zero => omega
          | succ d' =>
            by_cases hkgm : G.eval m ≤ -MATE_LOWER
            · rw [negamaxD2_kingGone G (d' + 1) m hkgm]; omega
            · rw [negamaxD2_of_fold G d' m hkgm (by simp [hleg]) hnt]
              refine foldMax_le _ _ _ (fun m' hm' => ?_) (by omega)
              show -(negamaxD2 G d' m') ≤ -MATE_LOWER
              have hm'' : m' ∈ G.moves m :=
                movesAbove_subset G _ m m' hm'
              have hkgm' : ¬ (G.eval m' ≤ -MATE_LOWER) := by
                intro hle
                have hc : hasKingCapture G.toNullGame.toGame m = true :=
                  (hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm'', hle⟩
                rw [hleg] at hc
                exact Bool.noConfusion hc
              cases hcm : hasKingCapture G.toNullGame.toGame m' with
              | true =>
                rw [negamaxD2_of_capture G d' m' hkgm' hcm]; omega
              | false =>
                have := ih m' hm'' hcm d' (by omega)
                omega
        have hfold : -(negamaxD2 G d m)
            ≤ foldMax (fun x => -(negamaxD2 G d x))
                (movesAbove G (val_lower (d + 1)) p) LOSS :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega

/-! ### The mated side -/

/-- **ForcedlyMated** (defender to move at `q`): either checkmated
outright, or not terminal while every LEGAL reply hands the opponent a
forced mate in `k`.  The non-terminality conjunct plays the same role
as `step`'s: it keeps a king-capture-vocabulary stalemate (all replies
illegal, not in check -- value 0, a draw) from slipping in through a
vacuous quantifier. -/
def ForcedlyMated (G : QSGame) (k : Nat) (q : G.Pos) : Prop :=
  Checkmated G q ∨
  (allIllegalB G q = false ∧
    ∀ m ∈ G.moves q, hasKingCapture G.toNullGame.toGame m = false →
      ForcedMate G k m)

/-- The dual: the mated side's real-move value sits at or below
`-MATE_LOWER`.  Stated at a node the attacker reached by a LEGAL move
(`hasKingCapture` false -- the defender cannot capture the attacker's
king), which is the only way the search ever gets to ask.  A corollary
of the spine, not a second induction: legal replies invoke
`forcedMate_negamaxD2`, illegal ones the sentinel branch. -/
theorem forcedlyMated_negamaxD2 (G : QSGame) (hF : ValFloor G 192)
    {k : Nat} {q : G.Pos}
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hFL : ForcedlyMated G k q) :
    ∀ D : Nat, k + 2 ≤ D → negamaxD2 G D q ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro D hD
  cases D with
  | zero => omega
  | succ d =>
    by_cases hkg : G.eval q ≤ -MATE_LOWER
    · rw [negamaxD2_kingGone G (d + 1) q hkg]; omega
    · have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by
        simp [hcapq]
      cases hFL with
      | inl hcm =>
        rw [negamaxD2_of_allIllegal G d q hkg hcapq' hcm.1]
        simp only [terminalValue]
        rw [if_pos hcm.2]
        omega
      | inr h =>
        obtain ⟨hai, hall⟩ := h
        rw [negamaxD2_of_fold G d q hkg hcapq' hai]
        refine foldMax_le _ _ _ (fun m hm => ?_) (by omega)
        show -(negamaxD2 G d m) ≤ -MATE_LOWER
        have hm' : m ∈ G.moves q := movesAbove_subset G _ q m hm
        cases hcm : hasKingCapture G.toNullGame.toGame m with
        | true =>
          have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := by
            intro hle
            have hc : hasKingCapture G.toNullGame.toGame q = true :=
              (hasKingCapture_iff G.toNullGame.toGame q).mpr ⟨m, hm', hle⟩
            rw [hcapq] at hc
            exact Bool.noConfusion hc
          rw [negamaxD2_of_capture G d m hkgm hcm]; omega
        | false =>
          have := forcedMate_negamaxD2 G hF (hall m hm' hcm) d (by omega)
          omega

/-! ### The transfer to the declared function -/

/-- **Mate-in-k completeness, declared-function form**: under
`NoZugzwang` (its SECOND consumer -- the first is the accuracy lemma
`nullValue_eq_realValue_of_noZugzwang`, and both are layer 2; layer 1
still carries no chess statement) the declared value `nullValueD2` --
the function the shipped search provably brackets with no chess premise
at all (`bound_null_spec`) -- is in the mate band wherever a forced
mate exists. -/
theorem forcedMate_complete (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hZ : NoZugzwang G guard)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, k + 1 ≤ D → MATE_LOWER ≤ nullValueD2 G guard D p := by
  intro D hD
  rw [nullValue_eq_realValue_of_noZugzwang G guard hZ D p]
  exact forcedMate_negamaxD2 G hF hFM D hD

/-! ### The driver corollaries: probes below the band cannot fail low -/

/-- **Liveness at the probe level, reference consumer**: with a forced
mate in reach of the depth, NO driver-range window at or below
`MATE_LOWER` can fail low -- composing `bound_null_spec` (safety: a
fail-low upper-bounds the declared value) with `forcedMate_complete`
(the declared value is in the band) yields a contradiction.  So every
bisection probe below the band fails high, each fail-high certifies
`gamma ≤ lower` through the very same spec, and the driver's bracket is
driven into the band: the search REPORTS the mate it was promised. -/
theorem forcedMate_probe_failsHigh (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hK : KillerLegal G kill)
    (hF : ValFloor G 192) (hZ : NoZugzwang G guard)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p)
    (D : Nat) (hD : k + 1 ≤ D)
    (gamma : Int) (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_LOWER) :
    gamma ≤ boundD2 G guard kill D p gamma := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hval := forcedMate_complete G guard hF hZ hFM D hD
  by_cases hfl : boundD2 G guard kill D p gamma < gamma
  · have := (bound_null_spec G guard kill hB hK D p gamma hg1 (by omega)).2 hfl
    omega
  · omega

/-- The same corollary for the PRODUCTION consumer, through
`production_eq_reference`'s premises (`boundKCX_null_spec`). -/
theorem forcedMate_probe_failsHigh_kcx (G : QSGame)
    (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G)
    (hK : KillerLegal G kill)
    (hF : ValFloor G 192) (hZ : NoZugzwang G guard)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p)
    (D : Nat) (hD : k + 1 ≤ D)
    (gamma : Int) (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_LOWER) :
    gamma ≤ boundKCX G guard D p gamma := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hval := forcedMate_complete G guard hF hZ hFM D hD
  by_cases hfl : boundKCX G guard D p gamma < gamma
  · have := (boundKCX_null_spec G guard kill hB hV hCF hK D p gamma hg1
      (by omega)).2 hfl
    omega
  · omega

/-! # Milestone 2: the `search()` package

Milestone 1 above proved the engine FINDS mates.  What follows
completes the story of `Searcher.search` around `bound`:

* **A. Driver termination and convergence** -- the MTD-bi inner loop
  (`while lower < upper - EVAL_ROUGHNESS`) provably terminates, on a
  concrete budget, and exits with a bracket of width `≤ EVAL_ROUGHNESS`
  that contains the declared value.
* **B. Best-move soundness** -- the docstring's `tp_move` clause as a
  theorem: the stored move is legal (`storedMoveLegal`, cited, not
  reproven) AND attains the returned score against the declared value
  function.
* **C. No false mates** -- the converse of milestone 1: a mate-band
  declared value implies a real `ForcedMate`, with the one genuinely
  required chess premise (`NoMaskedMobility`) characterized by a
  countermodel (`CexF`).
* **D. pst-swap soundness** -- `Sunfish/TableSwap.lean`. -/

/-! ## A. Driver termination and convergence

`Driver.lean` proved what windows the bisection probes with (`dstep`,
`depthInit`, the wide invariant).  Here we close the loop itself: the
inner `while` of `search`, modeled fuel-indexed so that termination is
a THEOREM about the fuel bound rather than an assumption.

Three facts compose:

* **strictness** (`dstep_raises_lower` / `dstep_lowers_upper`): a
  fail-high sets `lower := score ≥ gamma`, and every midpoint window is
  STRICTLY inside its bracket (`driverGamma_in_bracket`), so the raised
  `lower` strictly exceeds the old one; dually for fail-low.
* **halving** (`dstep_halves`): with the window at the midpoint, ONE
  probe at least halves the bracket -- fail-soft overshoot only
  overshrinks, so no score hypothesis is needed at all.  The
  logarithmic bound is clean, so it is the one stated
  (`driverLoop_halving`); the linear ≥-1-per-probe bound
  (`dstep_strictly_narrows`) is kept as the strictness record.
* **bracket soundness** (`dstep_bracket`): fail-soft-sound scores keep
  the declared value inside `[lower, upper]`, which ALSO keeps every
  computed window in the wide range with no clamp -- the
  `driver_wide_is_now_the_range` argument, re-run through the loop.

The one honest wrinkle: the bracket resets to
`lower, upper = 1 - MATE_UPPER, MATE_UPPER`, whose lower end sits ONE
above the value band's floor.  A root whose declared value is the exact
kingless sentinel `-MATE_UPPER` therefore ends with
`lower = 1 - MATE_UPPER` one above its value -- the `max` in the
conclusion records exactly this and nothing else; everywhere else
`final.lower ≤ V`.  (A kingless root never reaches `search` from a
legal game -- `HistoryLegal`'s territory -- but the theorem does not
need to assume it.)

The budget: the full band has width `2 * MATE_UPPER - 1 = 138579`, and
`138579 ≤ EVAL_ROUGHNESS * 2^14`, so ONE carried-window probe (the
first probe of a depth inherits `gamma` from the previous depth --
`Driver.lean`'s finding) plus 14 midpoint probes always suffice:
**15 probes per depth**, after which the loop is provably idle
(`driver_probe_budget`). -/

/-- The inner MTD-bi loop of `search`, fuel-indexed: while
`lower < upper - EVAL_ROUGHNESS`, probe at the current window and
fail-soft-update (`dstep`, which also computes the next midpoint).
Runs `probe` at most `fuel` times; the convergence theorems below show
fuel 15 is enough from any per-depth reset. -/
def driverLoop (probe : Int → Int) : Nat → DState → DState
  | 0, st => st
  | n + 1, st =>
    if st.lower < st.upper - EVAL_ROUGHNESS
    then driverLoop probe n (dstep st (probe st.gamma))
    else st

/-- **Strictness, fail-high side**: a window strictly above `lower`
that fails high raises `lower` strictly -- `lower := score ≥ gamma >
lower`.  Every midpoint window qualifies (`driverGamma_in_bracket`). -/
theorem dstep_raises_lower (st : DState) (s : Int)
    (hg : st.lower < st.gamma) (hs : st.gamma ≤ s) :
    st.lower < (dstep st s).lower := by
  unfold dstep
  simp only [if_pos hs]
  omega

/-- **Strictness, fail-low side**: a window at or below `upper` that
fails low lowers `upper` strictly -- `upper := score < gamma ≤ upper`. -/
theorem dstep_lowers_upper (st : DState) (s : Int)
    (hg : st.gamma ≤ st.upper) (hs : ¬ (st.gamma ≤ s)) :
    (dstep st s).upper < st.upper := by
  unfold dstep
  simp only [if_neg hs]
  omega

/-- The two strictness facts combined: a probe at a window strictly
inside the bracket shrinks the width by at least 1, whichever way it
fails -- the linear termination argument, kept explicit. -/
theorem dstep_strictly_narrows (st : DState) (s : Int)
    (hg1 : st.lower < st.gamma) (hg2 : st.gamma ≤ st.upper) :
    (dstep st s).upper - (dstep st s).lower ≤ st.upper - st.lower - 1 := by
  unfold dstep
  by_cases hc : st.gamma ≤ s
  · simp only [if_pos hc]; omega
  · simp only [if_neg hc]; omega

/-- **Halving**: with the window at the midpoint, one probe at least
halves the bracket width.  No hypothesis at all beyond the midpoint
shape: fail-soft overshoot lands the endpoint even further inside
(possibly crossing the bracket, which only exits the loop sooner), and
the inequality holds vacuously-harder for degenerate brackets. -/
theorem dstep_halves (st : DState) (s : Int)
    (hmid : st.gamma = driverGamma st.lower st.upper) :
    (dstep st s).upper - (dstep st s).lower ≤ (st.upper - st.lower) / 2 := by
  unfold dstep
  rw [hmid]
  unfold driverGamma
  by_cases hc : (st.lower + st.upper + 1) / 2 ≤ s
  · simp only [if_pos hc]; omega
  · simp only [if_neg hc]; omega

/-- After any probe the window IS the midpoint of the new bracket, by
construction of `dstep` -- only a depth's FIRST window can be carried. -/
theorem dstep_gamma_mid (st : DState) (s : Int) :
    (dstep st s).gamma = driverGamma (dstep st s).lower (dstep st s).upper := rfl

/-- A converged state is a fixed point of the loop: once
`upper - lower ≤ EVAL_ROUGHNESS`, no fuel makes another probe. -/
theorem driverLoop_stopped (probe : Int → Int) (n : Nat) (st : DState)
    (h : ¬ (st.lower < st.upper - EVAL_ROUGHNESS)) :
    driverLoop probe n st = st := by
  cases n with
  | zero => rfl
  | succ n => simp only [driverLoop]; rw [if_neg h]

/-- Fuel composes: running `m + n` steps is running `m`, then `n`. -/
theorem driverLoop_add (probe : Int → Int) (m n : Nat) :
    ∀ st : DState,
      driverLoop probe (m + n) st
        = driverLoop probe n (driverLoop probe m st) := by
  induction m with
  | zero =>
    intro st
    rw [Nat.zero_add]
    rfl
  | succ m ih =>
    intro st
    rw [Nat.succ_add]
    by_cases hc : st.lower < st.upper - EVAL_ROUGHNESS
    · simp only [driverLoop]
      rw [if_pos hc, if_pos hc, ih]
    · simp only [driverLoop]
      rw [if_neg hc, if_neg hc, driverLoop_stopped probe n st hc]

/-- **Termination, logarithmic**: from any midpoint-windowed state,
`n` probes converge a bracket of width up to `EVAL_ROUGHNESS * 2^n` --
each probe halves (`dstep_halves`) and re-establishes the midpoint
invariant (`dstep_gamma_mid`), and the loop guard is exactly the
convergence condition. -/
theorem driverLoop_halving (probe : Int → Int) :
    ∀ (n : Nat) (st : DState),
      st.gamma = driverGamma st.lower st.upper →
      st.upper - st.lower ≤ EVAL_ROUGHNESS * ((2 ^ n : Nat) : Int) →
      (driverLoop probe n st).upper - (driverLoop probe n st).lower
        ≤ EVAL_ROUGHNESS := by
  have hE : EVAL_ROUGHNESS = 15 := rfl
  intro n
  induction n with
  | zero =>
    intro st _ hw
    have h1 : ((2 ^ 0 : Nat) : Int) = 1 := rfl
    rw [hE, h1] at hw
    simp only [driverLoop]
    omega
  | succ n ih =>
    intro st hmid hw
    rw [hE, Nat.pow_succ] at hw
    simp only [driverLoop]
    by_cases hc : st.lower < st.upper - EVAL_ROUGHNESS
    · rw [if_pos hc]
      refine ih (dstep st (probe st.gamma)) (dstep_gamma_mid st _) ?_
      have hh := dstep_halves st (probe st.gamma) hmid
      rw [hE]
      omega
    · rw [if_neg hc]
      omega

/-- The loop invariant for convergence-to-the-value: the bracket
endpoints keep their reset-side bounds (which keeps every computed
window in the wide range, `driver_wide_is_now_the_range`'s argument),
the declared value stays inside the bracket -- up to the one-off
`1 - MATE_UPPER` reset floor recorded by the `max` -- and the current
window is a wide-range window. -/
def BracketOK (V : Int) (st : DState) : Prop :=
  1 - MATE_UPPER ≤ st.lower ∧ st.upper ≤ MATE_UPPER ∧
    st.lower ≤ max V (1 - MATE_UPPER) ∧ V ≤ st.upper ∧
    -MATE_UPPER < st.gamma ∧ st.gamma ≤ MATE_UPPER

/-- One fail-soft-sound probe preserves `BracketOK`: a fail-high score
is a valid lower bound of `V` (so raising `lower` to it is sound), a
fail-low score a valid upper bound; and the next midpoint stays in the
wide range because the value in the bracket pins the endpoints. -/
theorem dstep_bracket (V : Int) (hV1 : -MATE_UPPER ≤ V) (hV2 : V ≤ MATE_UPPER)
    (st : DState) (hst : BracketOK V st) (s : Int)
    (hs1 : st.gamma ≤ s → s ≤ V) (hs2 : s < st.gamma → V ≤ s) :
    BracketOK V (dstep st s) := by
  have hMU : MATE_UPPER = 69290 := rfl
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hst
  unfold dstep driverGamma BracketOK
  by_cases hc : st.gamma ≤ s
  · have := hs1 hc
    simp only [if_pos hc]
    omega
  · have := hs2 (by omega)
    simp only [if_neg hc]
    omega

/-- The invariant holds through any run of the loop, given a probe that
is fail-soft-sound for `V` at every wide-range window (`bound`'s layer-1
spec shape). -/
theorem driverLoop_bracket (probe : Int → Int) (V : Int)
    (hV1 : -MATE_UPPER ≤ V) (hV2 : V ≤ MATE_UPPER)
    (hspec : ∀ g, -MATE_UPPER < g → g ≤ MATE_UPPER →
      (g ≤ probe g → probe g ≤ V) ∧ (probe g < g → V ≤ probe g)) :
    ∀ (n : Nat) (st : DState), BracketOK V st →
      BracketOK V (driverLoop probe n st) := by
  intro n
  induction n with
  | zero => intro st hst; exact hst
  | succ n ih =>
    intro st hst
    simp only [driverLoop]
    by_cases hc : st.lower < st.upper - EVAL_ROUGHNESS
    · rw [if_pos hc]
      have hg := hspec st.gamma hst.2.2.2.2.1 hst.2.2.2.2.2
      exact ih (dstep st (probe st.gamma))
        (dstep_bracket V hV1 hV2 st hst (probe st.gamma) hg.1 hg.2)
    · rw [if_neg hc]
      exact hst

/-- **The package, abstract form**: for any band-bounded value `V` and
any probe fail-soft-sound for `V` at wide-range windows, 15 probes from
the per-depth reset (`lower, upper = 1 - MATE_UPPER, MATE_UPPER`, the
carried window inherited) provably exit the inner loop with

* a converged bracket: `upper - lower ≤ EVAL_ROUGHNESS`, and
* the declared value inside it: `lower ≤ max V (1 - MATE_UPPER)` and
  `V ≤ upper` (the `max` is the reset-floor wrinkle documented above).

One carried probe (which cannot widen anything the invariant tracks)
plus 14 halvings of the width-138579 band. -/
theorem driver_depth_converges (probe : Int → Int) (V : Int)
    (hV1 : -MATE_UPPER ≤ V) (hV2 : V ≤ MATE_UPPER)
    (hspec : ∀ g, -MATE_UPPER < g → g ≤ MATE_UPPER →
      (g ≤ probe g → probe g ≤ V) ∧ (probe g < g → V ≤ probe g))
    (carried : Int)
    (hc1 : -MATE_UPPER < carried) (hc2 : carried ≤ MATE_UPPER) :
    (driverLoop probe 15 (depthInit carried)).upper
        - (driverLoop probe 15 (depthInit carried)).lower ≤ EVAL_ROUGHNESS ∧
      (driverLoop probe 15 (depthInit carried)).lower
          ≤ max V (1 - MATE_UPPER) ∧
        V ≤ (driverLoop probe 15 (depthInit carried)).upper := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hE : EVAL_ROUGHNESS = 15 := rfl
  have h0 : BracketOK V (depthInit carried) := by
    simp only [BracketOK, depthInit]
    omega
  have hcond : (depthInit carried).lower
      < (depthInit carried).upper - EVAL_ROUGHNESS := by
    show 1 - MATE_UPPER < MATE_UPPER - EVAL_ROUGHNESS
    omega
  have hone : driverLoop probe 1 (depthInit carried)
      = dstep (depthInit carried) (probe carried) := by
    show driverLoop probe (0 + 1) (depthInit carried) = _
    simp only [driverLoop]
    rw [if_pos hcond]
    rfl
  have hstep : driverLoop probe 15 (depthInit carried)
      = driverLoop probe 14 (dstep (depthInit carried) (probe carried)) := by
    rw [show (15 : Nat) = 1 + 14 from rfl, driverLoop_add, hone]
  have hg := hspec carried hc1 hc2
  have hb1 : BracketOK V (dstep (depthInit carried) (probe carried)) :=
    dstep_bracket V hV1 hV2 (depthInit carried) h0 (probe carried) hg.1 hg.2
  have hw1 : (dstep (depthInit carried) (probe carried)).upper
      - (dstep (depthInit carried) (probe carried)).lower
        ≤ EVAL_ROUGHNESS * ((2 ^ 14 : Nat) : Int) := by
    obtain ⟨a1, a2, _, _, _, _⟩ := hb1
    have h14 : ((2 ^ 14 : Nat) : Int) = 16384 := rfl
    rw [hE, h14]
    omega
  have hconv := driverLoop_halving probe 14
    (dstep (depthInit carried) (probe carried))
    (dstep_gamma_mid (depthInit carried) (probe carried)) hw1
  have hbr := driverLoop_bracket probe V hV1 hV2 hspec 14
    (dstep (depthInit carried) (probe carried)) hb1
  rw [hstep]
  exact ⟨hconv, hbr.2.2.1, hbr.2.2.2.1⟩

/-- **Termination as a fixed point**: any fuel beyond the 15-probe
budget changes nothing -- the loop is provably idle after 15 probes, so
`while lower < upper - EVAL_ROUGHNESS` makes at most 15 probes per
depth. -/
theorem driver_probe_budget (probe : Int → Int) (V : Int)
    (hV1 : -MATE_UPPER ≤ V) (hV2 : V ≤ MATE_UPPER)
    (hspec : ∀ g, -MATE_UPPER < g → g ≤ MATE_UPPER →
      (g ≤ probe g → probe g ≤ V) ∧ (probe g < g → V ≤ probe g))
    (carried : Int)
    (hc1 : -MATE_UPPER < carried) (hc2 : carried ≤ MATE_UPPER) :
    ∀ k : Nat, driverLoop probe (15 + k) (depthInit carried)
      = driverLoop probe 15 (depthInit carried) := by
  intro k
  rw [driverLoop_add]
  refine driverLoop_stopped probe k _ ?_
  have h := (driver_depth_converges probe V hV1 hV2 hspec carried hc1 hc2).1
  omega

/-- **A, instantiated for the reference consumer**: probing the root
with `boundD2` at depth `D`, from any wide-range carried window, the
inner loop converges in 15 probes to a bracket of width
`≤ EVAL_ROUGHNESS` containing the declared value `nullValueD2` -- the
composition of the loop package with `bound_null_spec` (layer 1: no
chess premise) and `nullValueD2_bounded`. -/
theorem search_inner_loop_converges (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hK : KillerLegal G kill)
    (D : Nat) (p : G.Pos) (carried : Int)
    (hc1 : -MATE_UPPER < carried) (hc2 : carried ≤ MATE_UPPER) :
    (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).upper
        - (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).lower
          ≤ EVAL_ROUGHNESS ∧
      (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).lower
          ≤ max (nullValueD2 G guard D p) (1 - MATE_UPPER) ∧
        nullValueD2 G guard D p
          ≤ (driverLoop (fun g => boundD2 G guard kill D p g) 15 (depthInit carried)).upper := by
  have hV := nullValueD2_bounded G guard hB D p
  exact driver_depth_converges (fun g => boundD2 G guard kill D p g)
    (nullValueD2 G guard D p) hV.1 hV.2
    (fun g hg1 hg2 => bound_null_spec G guard kill hB hK D p g hg1 hg2)
    carried hc1 hc2

/-- **A, instantiated for the production consumer** (`boundKCX`),
through `boundKCX_null_spec`'s premises. -/
theorem search_inner_loop_converges_kcx (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G)
    (hK : KillerLegal G kill)
    (D : Nat) (p : G.Pos) (carried : Int)
    (hc1 : -MATE_UPPER < carried) (hc2 : carried ≤ MATE_UPPER) :
    (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).upper
        - (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).lower
          ≤ EVAL_ROUGHNESS ∧
      (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).lower
          ≤ max (nullValueD2 G guard D p) (1 - MATE_UPPER) ∧
        nullValueD2 G guard D p
          ≤ (driverLoop (fun g => boundKCX G guard D p g) 15 (depthInit carried)).upper := by
  have hVb := nullValueD2_bounded G guard hB D p
  exact driver_depth_converges (fun g => boundKCX G guard D p g)
    (nullValueD2 G guard D p) hVb.1 hVb.2
    (fun g hg1 hg2 => boundKCX_null_spec G guard kill hB hV hCF hK D p g hg1 hg2)
    carried hc1 hc2

/-! ## B. Best-move soundness: the stored move attains the report

The docstring of `bound` promises, for a root fail-high, that
`self.tp_move[pos]` holds a LEGAL move that ACHIEVES the returned score
`r ≥ gamma`.  Legality is `storedMoveLegal` (cited below, not
reproven).  This section adds ATTAINMENT, against the declared value
function `nullValueD2` -- the strong form, not just the fold the search
evaluated: a fail-high yield `-(bound child) ≥ gamma` is a fail-LOW of
the child probe at the flipped window, and `bound_null_spec` at the
child certifies `nullValueD2 child ≤ bound child`, so the negated
DECLARED child value is at least the yield.

The three store species (`KillStore`) are covered:

* the searched real winner (the `best >= gamma` break with a truthy
  move): `storedMove_attains` at the store site,
  `boundD2_failHigh_attained` at the node -- the loop's cutting yield
  IS the returned report (`searchMoves_failHigh_exact`), so the stored
  move attains exactly `r`;
* the kcx SUBSTITUTION store and the mate-case futility store: both
  store a move whose child has lost its king, and the sentinel is the
  attainment (`substitution_attains` -- `-(nullValueD2 child)` is the
  full `MATE_UPPER`);
* eviction stores nothing.

The killer's own yield (`yield killer, -self.bound(pos.move(killer), 1
- gamma, depth - 1)`) is a searched real yield of an admitted move --
move ordering, which the model does not order, so it is one of the fold
members covered by the first bullet. -/

/-- A fail-high loop from a below-window seed returns the CUTTING
yield exactly: fail-soft `best` on the break is `max best (score m)`
with the running `best` still below the window, i.e. the breaking
move's own score.  (Strengthens `searchMoves_failHigh_witness` with the
equality, which is what "achieves the returned score" needs.) -/
theorem searchMoves_failHigh_exact {α : Type _} (gamma : Int) (f : α → Int) :
    ∀ (ms : List α) (b : Int), b < gamma → gamma ≤ searchMoves gamma f ms b →
      ∃ m ∈ ms, searchMoves gamma f ms b = f m ∧ gamma ≤ f m := by
  intro ms
  induction ms with
  | nil =>
    intro b hb hge
    simp only [searchMoves] at hge
    omega
  | cons a ms ih =>
    intro b hb hge
    simp only [searchMoves] at hge ⊢
    by_cases hcut : gamma ≤ max b (f a)
    · rw [if_pos hcut] at hge ⊢
      exact ⟨a, List.mem_cons_self a ms, by omega, by omega⟩
    · rw [if_neg hcut] at hge ⊢
      obtain ⟨m, hm, heq, hf⟩ := ih (max b (f a)) (by omega) hge
      exact ⟨m, List.mem_cons_of_mem a hm, heq, hf⟩

/-- **Attainment at the store site** (companion of `storedMoveLegal`,
same hypothesis shape): a move stored on a real fail-high at a
wide-range window attains the window against the child's DECLARED
value -- the parent's fail-high yield is the child probe's fail-low,
and layer 1 at the child does the rest.  No chess premise. -/
theorem storedMove_attains (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hK : KillerLegal G kill)
    (d : Nat) (m : G.Pos) (gamma : Int)
    (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_UPPER)
    (hhi : gamma ≤ -(boundD2 G guard kill d m (1 - gamma))) :
    gamma ≤ -(nullValueD2 G guard d m) := by
  have h := (bound_null_spec G guard kill hB hK d m (1 - gamma)
    (by omega) (by omega)).2 (by omega)
  omega

/-- The substitution / mate-case-futility stores' attainment: the
stored move wins the king outright, so the child is the kingless
sentinel and its negation is the full `MATE_UPPER` -- above any
wide-range window.  (This is why `KillerAtKingCapturable` entries are
sound for the pv too.) -/
theorem substitution_attains (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (m : G.Pos) (hkev : G.eval m ≤ -MATE_LOWER)
    (gamma : Int) (hg2 : gamma ≤ MATE_UPPER) :
    gamma ≤ -(nullValueD2 G guard d m) := by
  rw [nullValueD2_kingGone G guard d m hkev]
  omega

/-- **The docstring's `tp_move` clause, as a theorem**: at a
non-capturable interior node where the null option did not cut and the
real-move loop failed high, the search's report IS the yield of one
searched move `m`, and that move is

* generated and admitted (`searchedAt`, hence in `G.moves p`),
* LEGAL (`storedMoveLegal`, cited), and
* ATTAINING: the negated declared value of its child is at least the
  returned report (hence at least `gamma`).

Exactly the state in which the loop stores `m` in `tp_move` -- the
break with a truthy move.  The two virtual accumulators cannot own the
report here: the futility term is below the window by construction
(`futTerm_lt_gamma`) and a surviving null yield in this branch failed
the cut test, so both sit strictly below `gamma ≤ S`. -/
theorem boundD2_failHigh_attained (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hK : KillerLegal G kill)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_UPPER)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hnc : ¬ (useD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - MATE_LOWER)) (d + 1) p gamma = true ∧
        gamma ≤ nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma (boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - MATE_LOWER)) p))
    (hS : gamma ≤ searchMoves gamma
        (fun m => -(boundD2 G guard kill d m (1 - gamma)))
        (searchedAt G (d + 1) gamma p) LOSS) :
    ∃ m ∈ searchedAt G (d + 1) gamma p,
      hasKingCapture G.toNullGame.toGame m = false ∧
      boundD2 G guard kill (d + 1) p gamma
        = -(boundD2 G guard kill d m (1 - gamma)) ∧
      boundD2 G guard kill (d + 1) p gamma ≤ -(nullValueD2 G guard d m) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  obtain ⟨m, hmem, heq, hf⟩ := searchMoves_failHigh_exact gamma
    (fun x => -(boundD2 G guard kill d x (1 - gamma)))
    (searchedAt G (d + 1) gamma p) LOSS (by omega) hS
  have hret : boundD2 G guard kill (d + 1) p gamma
      = -(boundD2 G guard kill d m (1 - gamma)) := by
    rw [boundD2_succ, if_neg hkg, if_neg hcap, if_neg hnc]
    have hfut := futTerm_lt_gamma G (d + 1) gamma p hg1
    have hnp : nullPartD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - MATE_LOWER)) (d + 1) p gamma < gamma := by
      simp only [nullPartD2]
      by_cases hu : useD2 G guard kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) (boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - MATE_LOWER)) (d + 1) p gamma = true
      · rw [if_pos hu]
        by_cases hv : gamma ≤ nullVerify G kill (-(boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - gamma))) gamma (boundD2 G guard kill (d + 1 - 3) (G.pass p) (1 - MATE_LOWER)) p
        · exact absurd ⟨hu, hv⟩ hnc
        · omega
      · rw [if_neg hu]
        omega
    simp only [termFix]
    rw [if_neg (fun h => absurd h.1 (by omega))]
    omega
  have hmm : m ∈ G.moves p :=
    movesAbove_subset G _ p m (searchedAt_subset G (d + 1) gamma p m hmem)
  have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
    hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hmm, hh⟩)
  have hleg := storedMoveLegal G guard kill d m gamma hg1 hkgm hf
  have hatt := (bound_null_spec G guard kill hB hK d m (1 - gamma)
    (by omega) (by omega)).2 (by omega)
  exact ⟨m, hmem, hleg, hret, by omega⟩

/-! ## C. No false mates: the converse of milestone 1

Milestone 1: a real forced mate puts the declared value in the band.
This section proves the PRIZE direction: a mate-band declared value
implies a real `ForcedMate` -- the driver's certified `MATE_LOWER ≤
lower` is never a lie.

**What made it (almost) unconditional.**  Band values of `nullValueD2`
have exactly three origins -- the king-capture sentinel, the verified
mate correction, and negations of `-MATE_LOWER`-or-below children --
and the PASS TERM is not one of them: the A1 suppression is baked into
the declared function (`nullTermD2` withdraws any mate-band pass
yield), so a band-value fold names a REAL witnessing move
(`nullTermD2_lt_ML` + `foldMax_failHigh_witness`).  The induction then
inverts milestone 1's, and `NoZugzwang` is NOWHERE needed: no-false-
mates holds for the declared function itself, not just its zugzwang-
free validity region.

**The one genuine obstruction: the defender-side QS filter at the
frontier.**  The value's defender fold ranges over `movesAbove`, but
`ForcedMate` quantifies over ALL legal defender replies -- filtering
shrinks the defender's options in the VALUE but not in real chess.  At
defender remaining depth ≥ 2 the shipped tables' floor clears the
threshold (`mem_movesAbove_of_floor`, `ValFloor G 192` vs `val_lower 2
= -240`) and nothing is filtered; at defender remaining depth EXACTLY
1 the threshold is `val_lower 1 = -100` and a legal reply valued in
`[-192, -100)` is invisible to the fold.  That is not a proof gap but
a FALSITY: `CexF` below is a machine-checked countermodel -- all
fidelity premises hold, the depth-2 declared value is the full
`MATE_UPPER`, and there is NO forced mate; the defender's one escape
(a ≥100cp-dropping move to stalemate) is QS-filtered exactly at the
frontier, and one more ply of depth dissolves the phantom
(`cexF_false_mate_at_frontier`).

The premise that closes the frontier is one this development already
knows: `NoMaskedMobility` ("a position whose every depth-1-admitted
move is illegal has no legal move at all") -- retired with the rejected
reduced scan, resurrected here as a live layer-2 premise with its first
real consumer.  Under it, a frontier defender fold at or below
`-MATE_LOWER` forces `allIllegalB`, contradicting the fold branch; the
band value then cannot reach the frontier shape at all.  Premises of
the spine: `ValFloor G 192` + `EvalQuiet` (fidelity, tables) +
`NoMaskedMobility` (chess, layer 2) + root legality
(`hasKingCapture p = false`, the `HistoryLegal` shape -- a root the
side to move could win the king from is `MATE_UPPER` by construction
without being a mate).  `NoZugzwang` is NOT among them. -/

/-- A fold at or above `B` from an initial accumulator below `B` names
a member at or above `B` (the `foldMax` mirror of
`searchMoves_failHigh_witness`). -/
theorem foldMax_failHigh_witness {α : Type _} (w : α → Int) {B : Int} :
    ∀ (ms : List α) (acc : Int), acc < B → B ≤ foldMax w ms acc →
      ∃ m ∈ ms, B ≤ w m := by
  intro ms
  induction ms with
  | nil =>
    intro acc hacc hge
    simp only [foldMax] at hge
    omega
  | cons a ms ih =>
    intro acc hacc hge
    simp only [foldMax] at hge
    by_cases hwa : B ≤ w a
    · exact ⟨a, List.mem_cons_self a ms, hwa⟩
    · obtain ⟨m, hm, h⟩ := ih (max acc (w a)) (by omega) hge
      exact ⟨m, List.mem_cons_of_mem a hm, h⟩

/-- **The suppression, spent**: the pass term of the declared function
is ALWAYS below the mate band -- a surviving pass yield is sub-band by
the withdrawal's own condition, and the fold identity is `LOSS`.  So a
band-value fold cannot originate in the pass: its witness is a real
move. -/
theorem nullTermD2_lt_ML (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) : nullTermD2 G guard d p < MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  simp only [nullTermD2]
  by_cases h1 : guard p = true ∧ 2 < d + 1
  · rw [if_pos h1]
    by_cases h2 : -(nullValueD2 G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
    · rw [if_pos h2]
      omega
    · rw [if_neg h2]
      omega
  · rw [if_neg h1]
    omega

/-- **The frontier, closed by `NoMaskedMobility`**: at a defender node
(reached by a legal attacker move) whose depth-1 fold sits at or below
`-MATE_LOWER`, every ADMITTED reply must be an illegal one refuted by
the sentinel -- a legal admitted reply's depth-0 value is its static
eval, quiet below the band (`EvalQuiet`), and a kingless reply would
contradict the attacker move's own legality.  `NoMaskedMobility` then
promotes "every admitted reply illegal" to "every reply illegal",
contradicting the fold branch's `allIllegalB = false`. -/
theorem frontier_filtered_escape (G : QSGame) (guard : G.Pos → Bool)
    (hQ : EvalQuiet G.toNullGame.toGame) (hNM : NoMaskedMobility G)
    (m : G.Pos)
    (hcapm : ¬ (hasKingCapture G.toNullGame.toGame m = true))
    (hai : allIllegalB G m = false)
    (hrep : ∀ m' ∈ movesAbove G (val_lower 1) m,
      MATE_LOWER ≤ nullValueD2 G guard 0 m') :
    False := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hallcap : ∀ m' ∈ movesAbove G (val_lower 1) m,
      hasKingCapture G.toNullGame.toGame m' = true := by
    intro m' hm'
    have hv := hrep m' hm'
    have hm'' : m' ∈ G.moves m := movesAbove_subset G _ m m' hm'
    by_cases hkgm' : G.eval m' ≤ -MATE_LOWER
    · exact absurd
        ((hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm'', hkgm'⟩) hcapm
    · cases hcm' : hasKingCapture G.toNullGame.toGame m' with
      | true => rfl
      | false =>
        exfalso
        have hval : nullValueD2 G guard 0 m' = G.eval m' := by
          simp only [nullValueD2]
          rw [if_neg hkgm', if_neg (by simp [hcm'])]
        rw [hval] at hv
        have := hQ m' hkgm'
        omega
  rw [allIllegalB_true_iff.mpr (hNM m hallcap)] at hai
  exact Bool.noConfusion hai

/-- **No false mates, declared-function form -- the prize**: a
mate-band declared value at a legally-reached root IS a forced mate in
at most `D` plies.  The suppression makes the witness real
(`nullTermD2_lt_ML`), the sentinel and king-gone branches are refuted
or converted (an illegal defender reply is the spec's non-case, a
kingless witness would contradict root legality), the checkmated leaf
is the verified correction's mate arm read backwards, and the deep
defender folds quantify over ALL legal replies because nothing is
filtered at remaining depth ≥ 2 (`ValFloor`).  The frontier arm is
`frontier_filtered_escape`.  NO `NoZugzwang` anywhere. -/
theorem forcedMate_of_nullValueD2 (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G) :
    ∀ (D : Nat) (p : G.Pos),
      hasKingCapture G.toNullGame.toGame p = false →
      MATE_LOWER ≤ nullValueD2 G guard D p →
      ForcedMate G D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro D
  induction D using Nat.strongRecOn with
  | _ D ih =>
    intro p hcapf hband
    have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by
      simp [hcapf]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · -- (`absurd`, not a bare `omega`: an omega closing a
      -- non-arithmetic goal routes through `Classical.byContradiction`,
      -- and this theorem is choice-free without it.)
      rw [nullValueD2_kingGone G guard D p hkg] at hband
      exact absurd hband (by omega)
    cases D with
    | zero =>
      exfalso
      have hval : nullValueD2 G guard 0 p = G.eval p := by
        simp only [nullValueD2]
        rw [if_neg hkg, if_neg hcap]
      rw [hval] at hband
      have := hQ p hkg
      omega
    | succ d =>
      cases hai : allIllegalB G p with
      | true =>
        exfalso
        rw [nullValueD2_of_allIllegal G guard d p hkg hcap hai] at hband
        simp only [terminalValue] at hband
        by_cases hic : inCheckB G.toNullGame p = true
        · rw [if_pos hic] at hband
          omega
        · rw [if_neg hic] at hband
          omega
      | false =>
        rw [nullValueD2_of_fold G guard d p hkg hcap hai] at hband
        obtain ⟨m, hmem, hmv⟩ :=
          foldMax_failHigh_witness (fun x => -(nullValueD2 G guard d x))
            (movesAbove G (val_lower (d + 1)) p) (nullTermD2 G guard d p)
            (nullTermD2_lt_ML G guard d p) hband
        have hm : m ∈ G.moves p := movesAbove_subset G _ p m hmem
        have hchild : nullValueD2 G guard d m ≤ -MATE_LOWER := by omega
        have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
          hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, hh⟩)
        have hlegm : hasKingCapture G.toNullGame.toGame m = false := by
          cases hcm : hasKingCapture G.toNullGame.toGame m with
          | false => rfl
          | true =>
            exfalso
            rw [nullValueD2_of_capture G guard d m hkgm hcm] at hchild
            omega
        have hcapm : ¬ (hasKingCapture G.toNullGame.toGame m = true) := by
          simp [hlegm]
        cases d with
        | zero =>
          exfalso
          have hval : nullValueD2 G guard 0 m = G.eval m := by
            simp only [nullValueD2]
            rw [if_neg hkgm, if_neg hcapm]
          rw [hval] at hchild
          exact hkgm hchild
        | succ d' =>
          cases hai' : allIllegalB G m with
          | true =>
            rw [nullValueD2_of_allIllegal G guard d' m hkgm hcapm hai'] at hchild
            by_cases hic : inCheckB G.toNullGame m = true
            · exact ForcedMate.mate (k := d' + 1) hkg hm hlegm ⟨hai', hic⟩
            · exfalso
              simp only [terminalValue] at hchild
              rw [if_neg hic] at hchild
              omega
          | false =>
            rw [nullValueD2_of_fold G guard d' m hkgm hcapm hai'] at hchild
            have hrep : ∀ m' ∈ movesAbove G (val_lower (d' + 1)) m,
                MATE_LOWER ≤ nullValueD2 G guard d' m' := by
              intro m' hm'
              have hle : -(nullValueD2 G guard d' m')
                  ≤ foldMax (fun x => -(nullValueD2 G guard d' x))
                      (movesAbove G (val_lower (d' + 1)) m)
                      (nullTermD2 G guard d' m) :=
                foldMax_le_of_mem _ _ _ m' hm'
              omega
            cases d' with
            | zero =>
              exact (frontier_filtered_escape G guard hQ hNM m hcapm hai' hrep).elim
            | succ d'' =>
              refine ForcedMate.step (k := d'' + 1) hkg hm hlegm hai' ?_
              intro m' hm' hleg'
              have hmem' : m' ∈ movesAbove G (val_lower (d'' + 1 + 1)) m :=
                mem_movesAbove_of_floor G hF (d := d'' + 1 + 1) (by omega) hm'
              exact ih (d'' + 1) (by omega) m' hleg' (hrep m' hmem')

/-- **The mated-side dual**: a declared value at or below `-MATE_LOWER`
at a legally-reached defender node whose king is on the board means the
defender is FORCEDLY MATED -- checkmated outright (the correction's
mate arm read backwards) or every legal reply hands the attacker a
forced mate (`forcedMate_of_nullValueD2` at the replies; depth ≥ 2 at
the node means nothing was filtered, and the depth-1 case is the
frontier again, closed the same way). -/
theorem forcedlyMated_of_nullValueD2 (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G)
    (D : Nat) (q : G.Pos)
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hkgq : ¬ (G.eval q ≤ -MATE_LOWER))
    (hlo : nullValueD2 G guard (D + 1) q ≤ -MATE_LOWER) :
    ForcedlyMated G D q := by
  have hML : MATE_LOWER = 47923 := rfl
  have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by
    simp [hcapq]
  cases hai : allIllegalB G q with
  | true =>
    rw [nullValueD2_of_allIllegal G guard D q hkgq hcapq' hai] at hlo
    by_cases hic : inCheckB G.toNullGame q = true
    · exact Or.inl ⟨hai, hic⟩
    · exfalso
      simp only [terminalValue] at hlo
      rw [if_neg hic] at hlo
      omega
  | false =>
    rw [nullValueD2_of_fold G guard D q hkgq hcapq' hai] at hlo
    have hrep : ∀ m' ∈ movesAbove G (val_lower (D + 1)) q,
        MATE_LOWER ≤ nullValueD2 G guard D m' := by
      intro m' hm'
      have hle : -(nullValueD2 G guard D m')
          ≤ foldMax (fun x => -(nullValueD2 G guard D x))
              (movesAbove G (val_lower (D + 1)) q) (nullTermD2 G guard D q) :=
        foldMax_le_of_mem _ _ _ m' hm'
      omega
    cases D with
    | zero =>
      exact (frontier_filtered_escape G guard hQ hNM q hcapq' hai hrep).elim
    | succ D' =>
      refine Or.inr ⟨hai, fun m' hm' hleg' => ?_⟩
      have hmem' : m' ∈ movesAbove G (val_lower (D' + 1 + 1)) q :=
        mem_movesAbove_of_floor G hF (d := D' + 1 + 1) (by omega) hm'
      exact forcedMate_of_nullValueD2 G guard hF hQ hNM (D' + 1) m'
        hleg' (hrep m' hmem')

/-! ### The probe corollaries: a certified mate report is a real mate -/

/-- **Reference consumer**: a fail-high at a window at or above
`MATE_LOWER` -- exactly how the driver's bracket certifies
`MATE_LOWER ≤ lower` -- implies a real forced mate within the probed
depth, by `bound_null_spec` (layer 1) + the spine. -/
theorem mate_report_honest (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hK : KillerLegal G kill)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G)
    (D : Nat) (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (gamma : Int) (hg1 : MATE_LOWER ≤ gamma) (hg2 : gamma ≤ MATE_UPPER)
    (hhi : gamma ≤ boundD2 G guard kill D p gamma) :
    ForcedMate G D p := by
  have hML : MATE_LOWER = 47923 := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  have h := (bound_null_spec G guard kill hB hK D p gamma (by omega) hg2).1 hhi
  exact forcedMate_of_nullValueD2 G guard hF hQ hNM D p hcapf (by omega)

/-- **Production consumer**, through `boundKCX_null_spec`'s premises. -/
theorem mate_report_honest_kcx (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G)
    (hK : KillerLegal G kill)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G)
    (D : Nat) (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (gamma : Int) (hg1 : MATE_LOWER ≤ gamma) (hg2 : gamma ≤ MATE_UPPER)
    (hhi : gamma ≤ boundKCX G guard D p gamma) :
    ForcedMate G D p := by
  have hML : MATE_LOWER = 47923 := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  have h := (boundKCX_null_spec G guard kill hB hV hCF hK D p gamma
    (by omega) hg2).1 hhi
  exact forcedMate_of_nullValueD2 G guard hF hQ hNM D p hcapf (by omega)

/-- The mated-side probe corollary, reference consumer: a report at or
below `-MATE_LOWER` from a window above the mated band's edge is a
fail-low, hence an upper bound of the declared value, hence a real
forced loss.  (The window premise is genuinely needed: at
`gamma ≤ -MATE_LOWER` such a report could be a fail-HIGH, which bounds
the value from the other side and certifies nothing about being mated.
The kcx twin is the same rewrite through `boundKCX_null_spec`.) -/
theorem mated_report_honest (G : QSGame) (guard kill : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hK : KillerLegal G kill)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (hNM : NoMaskedMobility G)
    (D : Nat) (q : G.Pos)
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hkgq : ¬ (G.eval q ≤ -MATE_LOWER))
    (gamma : Int) (hg1 : -MATE_LOWER < gamma) (hg2 : gamma ≤ MATE_UPPER)
    (hr : boundD2 G guard kill (D + 1) q gamma ≤ -MATE_LOWER) :
    ForcedlyMated G D q := by
  have hML : MATE_LOWER = 47923 := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  have h := (bound_null_spec G guard kill hB hK (D + 1) q gamma
    (by omega) hg2).2 (by omega)
  exact forcedlyMated_of_nullValueD2 G guard hF hQ hNM D q hcapq hkgq (by omega)

/-! ### The countermodel: the frontier premise is genuinely needed

`CexF`: root `R` (attacker) -- `M` (defender) -- with `M`'s two
replies an illegal `I` (admitted at depth 1) and a legal escape `E` to
stalemate, valued at a 150cp drop: below the depth-1 threshold
(`val_lower 1 = -100`), above the depth-2 one and the tables' -192
floor.  At depth 2 the declared value of `R` is the full `MATE_UPPER`
-- the depth-1 defender fold sees only the refuted `I` -- yet NO forced
mate exists at any `k`: `E` escapes.  One more ply and the phantom
dissolves (depth 3 computes the honest 0 through the stalemate).  All
the spine's fidelity premises hold; what fails is exactly
`NoMaskedMobility` at `M`.  So the frontier premise is not an artifact
of this proof: without it, the theorem is FALSE. -/

inductive FPos where
  | R | M | I | E | X
  deriving DecidableEq

open FPos in
/-- `R -> {M}`; `M -> {I, E}`; `I -> {X}` (the recapture that proves
`I` illegal); `E` and `X` terminal.  `E` is a stalemate: moveless, not
in check (`pass` is the identity and `E` has no king capture). -/
def CexF : QSGame where
  Pos := FPos
  moves := fun p => match p with
    | R => [M]
    | M => [I, E]
    | I => [X]
    | _ => []
  eval := fun p => match p with
    | X => -MATE_UPPER
    | _ => 0
  pass := fun p => p
  val := fun p m => match p, m with
    | M, E => -150
    | I, X => MATE_LOWER
    | _, _ => 0

instance : DecidableEq CexF.Pos := inferInstanceAs (DecidableEq FPos)

theorem cexF_bounded : Bounded CexF.toNullGame.toGame := by
  intro p
  cases p <;> decide

theorem cexF_quiet : EvalQuiet CexF.toNullGame.toGame := by
  intro p
  cases p <;> decide

theorem cexF_floor : ValFloor CexF 192 := by
  intro p m _
  cases p <;> cases m <;> decide

theorem cexF_valHigh : KingCaptureValHigh CexF := by
  intro p m hm hev
  cases p <;> cases m <;>
    first
      | decide
      | exact absurd hm (by decide)
      | exact absurd hev (by decide)

theorem cexF_root_legal : hasKingCapture CexF.toNullGame.toGame FPos.R = false := by
  decide

/-- The defender's depth-1 fold sees only the illegal `I`: the escape
`E` is QS-filtered, so `M`'s depth-1 declared value is the untouched
`LOSS` sentinel. -/
theorem cexF_M_depth1 :
    nullValueD2 CexF (fun _ => false) 1 FPos.M = -MATE_UPPER := by
  rw [nullValueD2_of_fold CexF (fun _ => false) 0 FPos.M (by decide) (by decide)
    (by decide)]
  have hma : movesAbove CexF (val_lower 1) FPos.M = [FPos.I] := by decide
  have hT : nullTermD2 CexF (fun _ => false) 0 FPos.M = LOSS := by
    simp only [nullTermD2]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
  have hI : nullValueD2 CexF (fun _ => false) 0 FPos.I = MATE_UPPER :=
    nullValueD2_of_capture CexF (fun _ => false) 0 FPos.I (by decide) (by decide)
  rw [hma, hT]
  simp only [foldMax]
  rw [hI]
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  omega

/-- The phantom: at depth 2 the root's declared value is the full
sentinel -- in the mate band. -/
theorem cexF_bandValue :
    MATE_LOWER ≤ nullValueD2 CexF (fun _ => false) 2 FPos.R := by
  rw [nullValueD2_of_fold CexF (fun _ => false) 1 FPos.R (by decide) (by decide)
    (by decide)]
  have hma : movesAbove CexF (val_lower 2) FPos.R = [FPos.M] := by decide
  have hT : nullTermD2 CexF (fun _ => false) 1 FPos.R = LOSS := by
    simp only [nullTermD2]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
  rw [hma, hT]
  simp only [foldMax]
  rw [cexF_M_depth1]
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  omega

/-- One ply deeper the filter admits `E`, and `M`'s value is the honest
stalemate draw. -/
theorem cexF_M_depth2 :
    nullValueD2 CexF (fun _ => false) 2 FPos.M = 0 := by
  rw [nullValueD2_of_fold CexF (fun _ => false) 1 FPos.M (by decide) (by decide)
    (by decide)]
  have hma : movesAbove CexF (val_lower 2) FPos.M = [FPos.I, FPos.E] := by decide
  have hT : nullTermD2 CexF (fun _ => false) 1 FPos.M = LOSS := by
    simp only [nullTermD2]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
  have hI : nullValueD2 CexF (fun _ => false) 1 FPos.I = MATE_UPPER :=
    nullValueD2_of_capture CexF (fun _ => false) 1 FPos.I (by decide) (by decide)
  have hE : nullValueD2 CexF (fun _ => false) 1 FPos.E = 0 := by
    rw [nullValueD2_of_allIllegal CexF (fun _ => false) 0 FPos.E (by decide)
      (by decide) (by decide)]
    simp only [terminalValue]
    rw [if_neg (by decide)]
  rw [hma, hT]
  simp only [foldMax]
  rw [hI, hE]
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  omega

/-- The phantom dissolves at depth 3. -/
theorem cexF_deeper :
    nullValueD2 CexF (fun _ => false) 3 FPos.R = 0 := by
  rw [nullValueD2_of_fold CexF (fun _ => false) 2 FPos.R (by decide) (by decide)
    (by decide)]
  have hma : movesAbove CexF (val_lower 3) FPos.R = [FPos.M] := by decide
  have hT : nullTermD2 CexF (fun _ => false) 2 FPos.R = LOSS := by
    simp only [nullTermD2]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
  rw [hma, hT]
  simp only [foldMax]
  rw [cexF_M_depth2]
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  omega

/-- No forced mate exists at `R`, at ANY `k`: the only move reaches
`M`, which is neither checkmated (`E` is a legal reply) nor step-mated
(`E` leads to a moveless spec-non-position -- `ForcedMate` needs a
move, and `E` has none). -/
theorem cexF_no_forcedMate : ∀ k, ¬ ForcedMate CexF k FPos.R := by
  intro k h
  cases h with
  | @mate k' p m hkg hm hleg hmate =>
    have hmM : m = FPos.M := List.mem_singleton.mp hm
    subst hmM
    exact absurd hmate.1 (by decide)
  | @step k' p m hkg hm hleg hnt hreply =>
    have hmM : m = FPos.M := List.mem_singleton.mp hm
    subst hmM
    have hE : ForcedMate CexF k' FPos.E :=
      hreply FPos.E (by decide) (by decide)
    cases hE with
    | @mate _ _ m2 _ hm2 _ _ => exact absurd hm2 (List.not_mem_nil m2)
    | @step _ _ m2 _ hm2 _ _ _ => exact absurd hm2 (List.not_mem_nil m2)

/-- ... and the premise that fails is exactly `NoMaskedMobility`, at
`M`: every depth-1-admitted reply is illegal (only `I`), yet the legal
`E` exists. -/
theorem cexF_masked_mobility : ¬ NoMaskedMobility CexF := by
  intro h
  have hpre : ∀ m ∈ movesAbove CexF (val_lower 1) FPos.M,
      hasKingCapture CexF.toNullGame.toGame m = true := by
    have hma : movesAbove CexF (val_lower 1) FPos.M = [FPos.I] := by decide
    rw [hma]
    intro m hm
    have hmI : m = FPos.I := List.mem_singleton.mp hm
    subst hmI
    decide
  exact absurd (h FPos.M hpre FPos.E (by decide)) (by decide)

/-- **The finding, bundled**: all fidelity premises hold and the root
is legally reached, the depth-2 declared value is in the mate band, no
forced mate exists, and one more ply computes the honest draw.  The
frontier chess premise is therefore REQUIRED: no-false-mates is false
without `NoMaskedMobility`. -/
theorem cexF_false_mate_at_frontier :
    MATE_LOWER ≤ nullValueD2 CexF (fun _ => false) 2 FPos.R ∧
      (∀ k, ¬ ForcedMate CexF k FPos.R) ∧
        nullValueD2 CexF (fun _ => false) 3 FPos.R = 0 :=
  ⟨cexF_bandValue, cexF_no_forcedMate, cexF_deeper⟩

end Sunfish
