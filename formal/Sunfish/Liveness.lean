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

end Sunfish
