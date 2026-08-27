/-
Sharp mate-depth accounting: how much depth a forced mate really costs.

`forcedMate_fuelValueD2` (EventuallyWide.lean) proves mate-in-k
completeness at `D ≥ C*k + 4`, instantiated by `forcedMate_intrinsicValue`
(IntrinsicLMR.lean) to the shipped `C = 3`, i.e. `D ≥ 3k + 4`; the dual
`forcedlyMated_fuelValueD2` needs `D ≥ C*k + C + 4` (`3k + 7`).  Those
constants are NOT sharp.  This file replaces them with the exact ones and
certifies the exactness with a countermodel.

WHERE THE OLD SLACK WAS.  The old induction demanded the real-only
regime (nominal depth `≥ 6`, where the pass is not a score candidate) at
EVERY node of the mating line, and charged a full `C` for the edge into
the mate.  Two of those charges are not needed:

* an ATTACKER node is safe at any depth `≥ 2`.  Its fold is a MAX, the
  mating child is admitted by `ValFloor` from remaining depth 2 on
  (`mem_movesAbove_of_floor`), and the sub-horizon pass term enters the
  same max -- `foldMax_le_of_mem` ignores the accumulator, so a pass
  candidate can never pull the maximum DOWN.  The horizon is irrelevant
  to the attacker.  (Below the horizon the shipped code also reduces
  nothing: intrinsic LMR starts at depth 7 and the fuel probe at depth 6,
  so each sub-horizon edge costs
  exactly one ply, as `fuelValueD2`'s sub-horizon branch records.)
* the CHECKMATED leaf is classified by the depth-gated terminal
  correction at any depth `≥ 1` (`fuelValueD2_checkmated`).

Only the DEFENDER nodes need the horizon: their fold is bounded ABOVE,
so the sub-horizon pass term -- the capped `min(pos.score + EVAL_ROUGHNESS, ...)`
candidate that the guard `not root and 2 < depth < 6 and ...` admits --
can mask the mate.  The mating line's last two plies are an attacker node
and the leaf, so the horizon has to be reached two plies BEFORE the end:

    D ≥ C*(k-2) + 6        (k ≥ 3),        D ≥ 2        (k ≤ 2).

For the shipped `C = 3` that is `D ≥ 3k`, four plies below `3k + 4`, and
the dual becomes `D ≥ C*(k-1) + 6` = `3k + 3`, four below `3k + 7`.

WHY IT IS SHARP.  `MDG` below is a ten-position game inside the theorem's
hypothesis class (`sharp_valFloor : ValFloor MDG 192`) with a forced mate
in 3 plies at `MDPos.A1` and in 5 plies at `MDPos.A2`.  One defender node sits at
nominal depth 5 -- inside the sub-horizon window -- where the pass is
worth 0; the mate is masked there, and the masking propagates to the
root as the value 0:

    sharp_mate3_at_8  : ForcedMate MDG 3 MDPos.A1, value at D = 8  is 0 < MATE_LOWER
    sharp_mate5_at_14 : ForcedMate MDG 5 MDPos.A2, value at D = 14 is 0 < MATE_LOWER
    sharp_mated3_at_11: ForcedlyMated MDG 3 MDPos.D2, value at D = 11 is 0 > -MATE_LOWER

`8 = 3*3 - 1` and `14 = 3*5 - 1` pin BOTH the constant and the slope:
the two witnesses are 6 = 2*C plies apart, so no bound of slope < C can
hold, and `11 = 3*3 + 2` pins the dual.  The certificates use the
maximal edge spend (`spend ≡ 2`, i.e. `min (C-1) 2 = 2`, the hot bit plus
the intrinsic-LMR bit) at every regime node: exactly the schedule the
shipped code realizes when the fuel probe fails high and the mating move
is a quiet one (`val < LMR`).

Zero sorries, no Mathlib, no audit-surface change.

MODEL DRIFT -- validity against post-#218 master (d0687b9)
==========================================================

This file was written against pre-#216 master (e499dae).  Since then #215
("Unify shallow pruning under lazy caps", 42c0816) and #218 ("Unify real-move
scoring in the search consumer", c01915f) rewrote the producer/consumer split
in `Searcher.bound`.  Every mechanism this file prices was re-checked line by
line against d0687b9.  VERDICT: all four uniform bounds, the three CI
corollaries and all four sharpness certificates STAND AS PROVED.  The file
compiles unchanged on top of d0687b9's `formal/` (axioms `[propext,
Quot.sound]`, zero sorries).  The one fidelity edit the audit turned up --
`capClamp`'s band -- has been taken here; it moves no value, as recorded at
the end.

MECHANISM MAP (pre-#216 line -> post-#218 line).

* fuel probe / hot bit: `:406-411` -> `:380-387`.  MOVED ONLY, out of the
  `moves()` generator into `bound()`'s own scope.  Body byte-identical then;
  4673322 has since renamed the locals only (`target` -> `t`, `nullpos`
  inlined at its single use, the bool subtraction spelled `int(...)`), so the
  current text reads
  `calm = abs(pos.score) < 750 and any(c in pos.board ...)`,
  `t = pos.score + NULL_MARGIN`,
  `nmr = calm and depth >= 6 and -self.bound(pos.rotate(nullmove=True),
  1 - t, depth - 7) >= t`. Still ONE ply (`int(nmr)`), still ONE probe,
  still at the fixed target
  `pos.score + NULL_MARGIN` and depth `depth - 7`.  `NULL_MARGIN = -200` and
  its tuner range `(-400, 800)` are untouched; the reworded comment's "burn
  two plies" is the TOTAL real-move reduction (base ply + hot bit), not a
  second probe ply.  `C = 3` is intact.

* intrinsic-LMR bit: `:439` -> `:437`.  MOVED ONLY, out of the deleted
  `score_move` helper into the consumer loop.  Expression byte-identical,
  `move_depth = depth - 1 - (guard and depth >= 7 and val < LMR) - int(nmr)`,
  and `val` is
  still `pos.value(move)` (the consumer only enters that branch when the
  produced score is below `MATE_LOWER`, where it equals the intrinsic value).
  `intrinsicSpend`, `intrinsic_child_depth` and `spend <= 2` unchanged.

* sub-horizon pass: `:395-402` -> `:406-412`.  Guard identical
  (`not root and 2 < depth < 6 and ...`), pass child now at `depth - 4`.
  Two local changes, neither reaching this file: (i) a sub-window
  short-circuit `score = cap if (cap := pos.score + EVAL_ROUGHNESS) < gamma
  else min(cap, ...)`, which is the same declared value `min(cap, P)` with a
  lazier report -- master's `WindowReport.cap_failLow` discharges it; (ii)
  `proof` now reads `pos.king_capture()` instead of
  `self.tp_move.get(pos) or pos.king_capture()`, moving the `MATE_UPPER`
  substitution off the move table and onto the position predicate, i.e.
  strictly CLOSER to the model's node-level `hasKingCapture` branch.  The
  proved bounds never touch the pass term anyway: it is confined to nominal
  depths 3-5, and every defender node in both inductions is required to sit
  at nominal depth >= 6 (`fuelValueD2C_of_fold_regime`, `hd : 5 <= d`).

* shallow static cap: the band now extends through depth 4. The arithmetic is
  unchanged in shape, while the tuned `QS_A = 180` makes its largest margin
  `3 * QS_A = 540`. `capClamp` carries the widened band -- see the closing
  section. The cap's guard
  `val < MATE_LOWER` is unchanged; its `MATE_LOWER - 1` ceiling has since
  been dropped from the code as dead under the both-kings material
  invariant, so `shallowMoveCap_below_positiveMate` now applies through
  `CapInBand` (localized here by `capClamp_eq_shipped`).

* QS admission: `val_lower = QS - depth * QS_A` (`:421`) plus the depth-one
  lazy tail (`:422-429`) -> a fixed producer floor of `QS` at depth zero and
  `-MATE_UPPER` above it.  The current `val_lower` may rise with `gamma`, but
  only as a lazy partition of that fixed producer: `lazyMoveTail_report` and
  `lazyMove_partition` prove that the aggregated cap tail followed by the
  searched prefix reports on the identical capped fold.  The depth-one tail
  and the `depth <= 1` futility break
  (`:455-469`) are GONE, folded into the unified cap.  `val_lower 0 = QS` is
  unchanged, and under `ValFloor G 192` the model's `val_lower d <= -192`
  from `d = 2` on (`val_lower_le_neg_floor`), so `movesAbove G (val_lower d)`
  already denotes the complete move list at every depth the code now uses
  unconditionally.  Model and code therefore agree at depth 0 and at every
  depth >= 2, and differ only at nominal depth 1, where the model still
  filters at `val_lower 1 = -100`.
  THIS FILE NEVER FOLDS AT NOMINAL DEPTH 1.  Part I enters a fold only at
  `D >= 2` and Part II only at `D >= 4`, and both inductions carry that
  floor down every edge.  Beyond that the two directions are structurally
  insensitive to the admitted set: completeness needs only that the mating
  move IS admitted (`mem_movesAbove_of_floor`, `2 <= d`) and then
  `foldMax_le_of_mem`, which a WIDER set can only help; the dual bounds every
  member through `movesAbove_subset` into `G.moves`, which a wider set still
  satisfies.  Consequence: `ValFloor G 192` is now sufficient but no longer
  necessary for the completeness direction -- the shipped producer admits
  every pseudo-legal move at positive depth outright
  (master's `producerMoves_positive`).  Dropping the premise is an available
  strengthening, not a repair.

* king captures resolved statically (`:419-425`, `:431`): a move with
  `pos.value(move) >= MATE_LOWER` is now reported `MATE_UPPER` without
  recursion.  Value-identical -- the old code searched it and the kingless
  child returned `-MATE_UPPER` -- so nothing here changes.  `live` and the
  `tp_move` store behave as before, and the terminal finalizer (`:456-472`)
  is untouched.

CURRENT RE-AUDIT. The tuned cap extends through depth 4, so `capClamp` now
guards on `d <= 4` and becomes the identity from depth 5. The proof spine
uses only that identity, `capClamp_le`, and the mate-band separation. Its
uniform bounds therefore rise by one ply. The witness was extended one node:
at `D = 10` the last attacker lands at depth 4 and is capped at 540, below
`MATE_LOWER`; at `D = 11` the proof clears the cap. Thus `3k + 2` is sharp.

The executable regression depths move with that theorem: mate-in-1/2/3 use
5/11/17. All formal modules compile, the terminal battery passes at its new
depth-5 observation horizon, and the uncapped Part I certificates are
unchanged.
-/

import Sunfish.IntrinsicLMR
import Sunfish.CappedMove

namespace Sunfish

/-! # Part I: the sharpened uniform bounds -/

/-- **The last ply, alone**: an attacker node whose mating move is
available needs the ADMISSION depth 2 and nothing else -- no horizon.
This is the `mate` constructor's case of the induction below, extracted
because it is also the exact `k ≤ 2` bound.  Above the horizon the edge
into the mate costs at most `C` and the leaf is still classified
(`fuelValueD2_checkmated` wants only depth ≥ 1); below it the edge costs
exactly one ply and the pass candidate shares the attacker's MAX, where
`foldMax_le_of_mem` ignores it. -/
theorem forcedMate_leaf_fuelValueD2 (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC1 : 1 ≤ C) (hC4 : C ≤ 4)
    (hF : ValFloor G 192) {p m : G.Pos}
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) (hm : m ∈ G.moves p)
    (hleg : hasKingCapture G.toNullGame.toGame m = false) (hmate : Checkmated G m) :
    ∀ D : Nat, 2 ≤ D → MATE_LOWER ≤ fuelValueD2 G guard C spend D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro D h2
  cases D with
  | zero => omega
  | succ d =>
    by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
    · rw [fuelValueD2_of_capture G guard C spend (d + 1) p hkg hcap]; omega
    · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
      have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
      by_cases hreg : 5 ≤ d
      · rw [fuelValueD2_of_fold_regime G guard C spend d p hkg hcap hai hreg]
        have hmin := Nat.min_le_left (C - 1) (spend p (d + 1) m)
        have hchild : fuelValueD2 G guard C spend
            (d - min (C - 1) (spend p (d + 1) m)) m ≤ -MATE_LOWER :=
          fuelValueD2_checkmated G guard C spend hleg hmate _ (by omega)
        have hfold : -(fuelValueD2 G guard C spend
              (d - min (C - 1) (spend p (d + 1) m)) m)
            ≤ foldMax (fun x => -(fuelValueD2 G guard C spend
                  (d - min (C - 1) (spend p (d + 1) x)) x))
                (movesAbove G (val_lower (d + 1)) p) LOSS :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega
      · rw [fuelValueD2_of_fold_sub G guard C spend d p hkg hcap hai (by omega)]
        have hchild : fuelValueD2 G guard C spend d m ≤ -MATE_LOWER :=
          fuelValueD2_checkmated G guard C spend hleg hmate d (by omega)
        have hfold : -(fuelValueD2 G guard C spend d m)
            ≤ foldMax (fun x => -(fuelValueD2 G guard C spend d x))
                (movesAbove G (val_lower (d + 1)) p)
                (if guard p = true ∧ 2 < d + 1 then
                  (if -(fuelValueD2 G guard C spend (d + 1 - 4) (G.pass p)) < MATE_LOWER then
                    max LOSS (-(fuelValueD2 G guard C spend (d + 1 - 4) (G.pass p)))
                  else LOSS)
                else LOSS) :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega

/-- **Mate-in-k completeness, sharp.**  `ValFloor` alone (fidelity,
tables -- no chess premise, exactly as `forcedMate_fuelValueD2`) puts the
declared value in the mate band at every

    D ≥ 2   with   C*k + 6 ≤ D + 2*C,

i.e. `D ≥ max 2 (C*(k-2) + 6)`: the horizon is charged only up to the
last-but-one ply, and the final attacker node needs nothing but the
admission depth 2.  For every edge-cost selector, and for every
reduction cap `1 ≤ C ≤ 4` (shipped: `C = 3`, giving `D ≥ 3k`). -/
theorem forcedMate_fuelValueD2_sharp (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC1 : 1 ≤ C) (hC4 : C ≤ 4)
    (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 2 ≤ D → C * k + 6 ≤ D + 2 * C →
      MATE_LOWER ≤ fuelValueD2 G guard C spend D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  induction hFM with
  | @mate k p m hkg hm hleg hmate =>
    intro D h2 _
    exact forcedMate_leaf_fuelValueD2 G guard C spend hC1 hC4 hF hkg hm hleg hmate D h2
  | @step k p m hkg hm hleg hnt hreply ih =>
    intro D h2 hD
    rcases Nat.eq_zero_or_pos k with hk0 | hk1
    · -- `ForcedMate G 0` is uninhabited, so `step` at index 2 is vacuous
      subst hk0
      obtain ⟨m', hm', hleg'⟩ := legal_of_allIllegalB_false hnt
      exact absurd (hreply m' hm' hleg') (fun h => by cases h)
    · -- k ≥ 1: `D ≥ C*k + 6 ≥ 6`, so the top two plies are both in the regime
      have hDk : C * k + 6 ≤ D := by
        have : C * (k + 2) = C * k + 2 * C := by
          rw [Nat.mul_add]; omega
        omega
      have hCk : C ≤ C * k := Nat.le_mul_of_pos_right C hk1
      cases D with
      | zero => omega
      | succ d =>
        by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [fuelValueD2_of_capture G guard C spend (d + 1) p hkg hcap]; omega
        · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
          have hmin := Nat.min_le_left (C - 1) (spend p (d + 1) m)
          rw [fuelValueD2_of_fold_regime G guard C spend d p hkg hcap hai (by omega)]
          have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
          have hchild : fuelValueD2 G guard C spend
              (d - min (C - 1) (spend p (d + 1) m)) m ≤ -MATE_LOWER := by
            -- the defender node sits at depth ≥ D - C ≥ 6: regime fold, LOSS seed
            obtain ⟨dm, hdm⟩ : ∃ x, d - min (C - 1) (spend p (d + 1) m) = x + 1 :=
              ⟨d - min (C - 1) (spend p (d + 1) m) - 1, by omega⟩
            rw [hdm]
            by_cases hkgm : G.eval m ≤ -MATE_LOWER
            · rw [fuelValueD2_kingGone G guard C spend (dm + 1) m hkgm]; omega
            · rw [fuelValueD2_of_fold_regime G guard C spend dm m hkgm
                (by simp [hleg]) hnt (by omega)]
              refine foldMax_le _ _ _ (fun m' hm' => ?_) (by omega)
              show -(fuelValueD2 G guard C spend
                (dm - min (C - 1) (spend m (dm + 1) m')) m') ≤ -MATE_LOWER
              have hm'' : m' ∈ G.moves m := movesAbove_subset G _ m m' hm'
              have hkgm' : ¬ (G.eval m' ≤ -MATE_LOWER) := by
                intro hle
                have hc : hasKingCapture G.toNullGame.toGame m = true :=
                  (hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm'', hle⟩
                rw [hleg] at hc
                exact Bool.noConfusion hc
              cases hcm : hasKingCapture G.toNullGame.toGame m' with
              | true =>
                rw [fuelValueD2_of_capture G guard C spend _ m' hkgm' hcm]; omega
              | false =>
                have := ih m' hm'' hcm
                  (dm - min (C - 1) (spend m (dm + 1) m')) (by omega) (by omega)
                omega
          have hfold : -(fuelValueD2 G guard C spend
                (d - min (C - 1) (spend p (d + 1) m)) m)
              ≤ foldMax (fun x => -(fuelValueD2 G guard C spend
                    (d - min (C - 1) (spend p (d + 1) x)) x))
                  (movesAbove G (val_lower (d + 1)) p) LOSS :=
            foldMax_le_of_mem _ _ _ _ hmem
          omega

/-- The one-ply corner, exactly: a mate that is already available needs
nothing but the admission depth.  (`k ≤ 2` because `ForcedMate G 2` can
only be the `mate` constructor -- `step` at index 2 would need
`ForcedMate G 0`.) -/
theorem forcedMate_fuelValueD2_short (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC1 : 1 ≤ C) (hC4 : C ≤ 4)
    (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hk : k ≤ 2) (hFM : ForcedMate G k p) :
    ∀ D : Nat, 2 ≤ D → MATE_LOWER ≤ fuelValueD2 G guard C spend D p := by
  intro D h2
  cases hFM with
  | @mate k' p' m hkg hm hleg hmate =>
    exact forcedMate_leaf_fuelValueD2 G guard C spend hC1 hC4 hF hkg hm hleg hmate D h2
  | @step k' p' m hkg hm hleg hnt hreply =>
    obtain ⟨m', hm', hleg'⟩ := legal_of_allIllegalB_false hnt
    have hk' : k' = 0 := by omega
    subst hk'
    exact absurd (hreply m' hm' hleg') (fun h => by cases h)

/-- The dual, sharp: the mated side's declared value sits at or below
`-MATE_LOWER` at every `D ≥ 6` with `C*k + 6 ≤ D + C`, i.e.
`D ≥ max 6 (C*(k-1) + 6)`.  The root is a defender node here, so its own
fold has to be regime-seeded -- one ply of horizon more than the mate
side, and no more.  Shipped `C = 3`: `D ≥ 3k + 3`. -/
theorem forcedlyMated_fuelValueD2_sharp (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC1 : 1 ≤ C) (hC4 : C ≤ 4)
    (hF : ValFloor G 192)
    {k : Nat} {q : G.Pos}
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hFL : ForcedlyMated G k q) :
    ∀ D : Nat, 6 ≤ D → C * k + 6 ≤ D + C →
      fuelValueD2 G guard C spend D q ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro D h6 hD
  cases D with
  | zero => omega
  | succ d =>
    by_cases hkg : G.eval q ≤ -MATE_LOWER
    · rw [fuelValueD2_kingGone G guard C spend (d + 1) q hkg]; omega
    · have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by
        simp [hcapq]
      cases hFL with
      | inl hcm =>
        rw [fuelValueD2_of_allIllegal G guard C spend d q hkg hcapq' hcm.1]
        have := terminalValue_mate G (d + 1) q hcm.2
        omega
      | inr h =>
        obtain ⟨hai, hall⟩ := h
        rw [fuelValueD2_of_fold_regime G guard C spend d q hkg hcapq' hai (by omega)]
        refine foldMax_le _ _ _ (fun m hm => ?_) (by omega)
        show -(fuelValueD2 G guard C spend
            (d - min (C - 1) (spend q (d + 1) m)) m)
          ≤ -MATE_LOWER
        have hm' : m ∈ G.moves q := movesAbove_subset G _ q m hm
        have hmin := Nat.min_le_left (C - 1) (spend q (d + 1) m)
        cases hcm : hasKingCapture G.toNullGame.toGame m with
        | true =>
          have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := by
            intro hle
            have hc : hasKingCapture G.toNullGame.toGame q = true :=
              (hasKingCapture_iff G.toNullGame.toGame q).mpr ⟨m, hm', hle⟩
            rw [hcapq] at hc
            exact Bool.noConfusion hc
          rw [fuelValueD2_of_capture G guard C spend _ m hkgm hcm]; omega
        | false =>
          have := forcedMate_fuelValueD2_sharp G guard C spend hC1 hC4 hF
            (hall m hm' hcm) (d - min (C - 1) (spend q (d + 1) m))
            (by omega) (by omega)
          omega

/-! ## The shipped instantiation: intrinsic LMR, `C = 3` -/

/-- **Completeness as shipped, sharp: `D ≥ 3k`** (and `D ≥ 2`), against
`3k + 4`.  Same premise as `forcedMate_intrinsicValue`: `ValFloor` only. -/
theorem forcedMate_intrinsicValue_sharp (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 2 ≤ D → 3 * k ≤ D →
      MATE_LOWER ≤ fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) D p :=
  fun D h2 hD => forcedMate_fuelValueD2_sharp G guard 3
    (intrinsicEdgeSpend G hot eligible low) (by omega) (by omega) hF hFM D h2 (by omega)

/-- The mated dual as shipped, sharp: `D ≥ 3k + 3`, against `3k + 7`. -/
theorem forcedlyMated_intrinsicValue_sharp (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {k : Nat} {q : G.Pos}
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hFL : ForcedlyMated G k q) :
    ∀ D : Nat, 6 ≤ D → 3 * k + 3 ≤ D →
      fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) D q ≤ -MATE_LOWER :=
  fun D h6 hD => forcedlyMated_fuelValueD2_sharp G guard 3
    (intrinsicEdgeSpend G hot eligible low) (by omega) (by omega) hF hcapq hFL D h6 (by omega)

/-! ## The CI table

The suite convention (`tools/quick_tests.sh`): mate-in-`n` moves is
`k = 2n - 1` plies. These are the cap-free comparison depths; the executable
suite uses the capped depths 5/11/17 below. -/

/-- mate-in-1 (`k = 1`): `D = 2` suffices (capped suite: 5). -/
theorem ci_mate_in_1 (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {p : G.Pos} (hFM : ForcedMate G 1 p) :
    MATE_LOWER ≤ fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) 2 p :=
  forcedMate_fuelValueD2_short G guard 3 (intrinsicEdgeSpend G hot eligible low)
    (by omega) (by omega) hF (by omega) hFM 2 (by omega)

/-- mate-in-2 (`k = 3`): `D = 9` suffices (capped suite: 11). -/
theorem ci_mate_in_2 (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {p : G.Pos} (hFM : ForcedMate G 3 p) :
    MATE_LOWER ≤ fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) 9 p :=
  forcedMate_intrinsicValue_sharp G guard hot eligible low hF hFM 9 (by omega) (by omega)

/-- mate-in-3 (`k = 5`): `D = 15` suffices (capped suite: 17). -/
theorem ci_mate_in_3 (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {p : G.Pos} (hFM : ForcedMate G 5 p) :
    MATE_LOWER ≤ fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) 15 p :=
  forcedMate_intrinsicValue_sharp G guard hot eligible low hF hFM 15 (by omega) (by omega)


/-! ## Menu option M1: delete the sub-horizon pass

The dominant driver of the constant is the sub-horizon window
`not root and 2 < depth < 6 and ...`, where the pass is still a SCORE
candidate: it is the only reason a defender node on the mating line has to
sit at nominal depth `≥ 6`, and therefore the reason the whole line has to
stay above the horizon and pay `C` per ply.  Deleting that block is
`guard ≡ false` in the model -- the deep fuel probe (`spend`) is untouched.
The line may then fall below the horizon, where every edge costs exactly
one ply, and the bound drops by five (shipped `C = 3`):

    D ≥ max 2 (C*k + 4 - 3*C)       -- `3k - 5`, against `3k`.

`defender_le_of_replies` below is the shared step: a defender node reports
at or below `-MATE_LOWER` as soon as its fold carries NO pass term, which
is now true at every depth, not only above the horizon. -/

/-- A defender node whose every legal reply hands the attacker a mate,
at every depth its own edges can reach, reports at or below `-MATE_LOWER`
-- provided its fold carries no pass term: either the real-only regime
(`6 ≤ dm`) or the guard off. -/
theorem defender_le_of_replies (G : QSGame) (guard : G.Pos → Bool) (C : Nat)
    (spend : G.Pos → Nat → G.Pos → Nat) (hC1 : 1 ≤ C) (hC3 : C ≤ 3)
    {m : G.Pos} {dm : Nat} (hdm : 3 ≤ dm)
    (hleg : hasKingCapture G.toNullGame.toGame m = false)
    (hnt : allIllegalB G m = false)
    (hnoseed : 6 ≤ dm ∨ guard m = false)
    (hrep : ∀ m' ∈ G.moves m, hasKingCapture G.toNullGame.toGame m' = false →
      ∀ e : Nat, dm ≤ e + C → 2 ≤ e → MATE_LOWER ≤ fuelValueD2 G guard C spend e m') :
    fuelValueD2 G guard C spend dm m ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  obtain ⟨d, rfl⟩ : ∃ d, dm = d + 1 := ⟨dm - 1, by omega⟩
  by_cases hkgm : G.eval m ≤ -MATE_LOWER
  · rw [fuelValueD2_kingGone G guard C spend (d + 1) m hkgm]; omega
  · have hkey : ∀ (m' : G.Pos) (e : Nat), m' ∈ G.moves m → d + 1 ≤ e + C → 2 ≤ e →
        -(fuelValueD2 G guard C spend e m') ≤ -MATE_LOWER := by
      intro m' e hm' he h2e
      have hkgm' : ¬ (G.eval m' ≤ -MATE_LOWER) := by
        intro hle
        have hc : hasKingCapture G.toNullGame.toGame m = true :=
          (hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm', hle⟩
        rw [hleg] at hc
        exact Bool.noConfusion hc
      cases hcm : hasKingCapture G.toNullGame.toGame m' with
      | true => rw [fuelValueD2_of_capture G guard C spend e m' hkgm' hcm]; omega
      | false => have := hrep m' hm' hcm e he h2e; omega
    by_cases hreg : 5 ≤ d
    · rw [fuelValueD2_of_fold_regime G guard C spend d m hkgm (by simp [hleg]) hnt hreg]
      refine foldMax_le _ _ _ (fun m' hm' => ?_) (by omega)
      have hmin := Nat.min_le_left (C - 1) (spend m (d + 1) m')
      exact hkey m' _ (movesAbove_subset G _ m m' hm') (by omega) (by omega)
    · have hgf : guard m = false := by
        rcases hnoseed with h6 | hgf
        · omega
        · exact hgf
      rw [fuelValueD2_of_fold_sub G guard C spend d m hkgm (by simp [hleg]) hnt (by omega),
        if_neg (by simp [hgf])]
      refine foldMax_le _ _ _ (fun m' hm' => ?_) (by omega)
      exact hkey m' d (movesAbove_subset G _ m m' hm') (by omega) (by omega)

/-- **M1's bound**: with the sub-horizon pass gone, mate-in-k completeness
holds from `D ≥ max 2 (C*k + 4 - 3*C)` -- `3k - 5` at the shipped `C = 3`,
five plies below the sharpened `3k` and nine below the shipped `3k + 4`.
The mating line is allowed to sink below the horizon, where each edge costs
exactly one ply. -/
theorem forcedMate_fuelValueD2_noSubPass (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC1 : 1 ≤ C) (hC3 : C ≤ 3)
    (hg : ∀ q, guard q = false) (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 2 ≤ D → C * k + 4 ≤ D + 3 * C →
      MATE_LOWER ≤ fuelValueD2 G guard C spend D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  induction hFM with
  | @mate k p m hkg hm hleg hmate =>
    intro D h2 _
    exact forcedMate_leaf_fuelValueD2 G guard C spend hC1 (by omega) hF hkg hm hleg hmate D h2
  | @step k p m hkg hm hleg hnt hreply ih =>
    intro D h2 hD
    rcases Nat.eq_zero_or_pos k with hk0 | hk1
    · subst hk0
      obtain ⟨m', hm', hleg'⟩ := legal_of_allIllegalB_false hnt
      exact absurd (hreply m' hm' hleg') (fun h => by cases h)
    · have hexp : C * (k + 2) = C * k + 2 * C := by rw [Nat.mul_add]; omega
      have hCk : C ≤ C * k := Nat.le_mul_of_pos_right C hk1
      cases D with
      | zero => omega
      | succ d =>
        by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [fuelValueD2_of_capture G guard C spend (d + 1) p hkg hcap]; omega
        · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
          have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
          by_cases hreg : 5 ≤ d
          · -- the attacker node is above the horizon: the edge costs up to C
            rw [fuelValueD2_of_fold_regime G guard C spend d p hkg hcap hai hreg]
            have hmin := Nat.min_le_left (C - 1) (spend p (d + 1) m)
            have hchild : fuelValueD2 G guard C spend
                (d - min (C - 1) (spend p (d + 1) m)) m ≤ -MATE_LOWER :=
              defender_le_of_replies G guard C spend hC1 hC3 (by omega) hleg hnt
                (Or.inr (hg m))
                (fun m' hm' hcm e he h2e => ih m' hm' hcm e h2e (by omega))
            have hfold : -(fuelValueD2 G guard C spend
                  (d - min (C - 1) (spend p (d + 1) m)) m)
                ≤ foldMax (fun x => -(fuelValueD2 G guard C spend
                      (d - min (C - 1) (spend p (d + 1) x)) x))
                    (movesAbove G (val_lower (d + 1)) p) LOSS :=
              foldMax_le_of_mem _ _ _ _ hmem
            omega
          · -- below the horizon: the edge costs exactly one ply
            rw [fuelValueD2_of_fold_sub G guard C spend d p hkg hcap hai (by omega)]
            have hchild : fuelValueD2 G guard C spend d m ≤ -MATE_LOWER :=
              defender_le_of_replies G guard C spend hC1 hC3 (by omega) hleg hnt
                (Or.inr (hg m))
                (fun m' hm' hcm e he h2e => ih m' hm' hcm e h2e (by omega))
            have hfold : -(fuelValueD2 G guard C spend d m)
                ≤ foldMax (fun x => -(fuelValueD2 G guard C spend d x))
                    (movesAbove G (val_lower (d + 1)) p)
                    (if guard p = true ∧ 2 < d + 1 then
                      (if -(fuelValueD2 G guard C spend (d + 1 - 4) (G.pass p)) < MATE_LOWER then
                        max LOSS (-(fuelValueD2 G guard C spend (d + 1 - 4) (G.pass p)))
                      else LOSS)
                    else LOSS) :=
              foldMax_le_of_mem _ _ _ _ hmem
            omega

/-- M1 at the CI depths: mate-in-2 (`k = 3`) at `D = 4`, mate-in-3
(`k = 5`) at `D = 10` -- against 9 and 15 today. -/
theorem ci_mate_in_2_noSubPass (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hg : ∀ q, guard q = false) (hF : ValFloor G 192)
    {p : G.Pos} (hFM : ForcedMate G 3 p) :
    MATE_LOWER ≤ fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) 4 p :=
  forcedMate_fuelValueD2_noSubPass G guard 3 (intrinsicEdgeSpend G hot eligible low)
    (by omega) (by omega) hg hF hFM 4 (by omega) (by omega)

theorem ci_mate_in_3_noSubPass (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hg : ∀ q, guard q = false) (hF : ValFloor G 192)
    {p : G.Pos} (hFM : ForcedMate G 5 p) :
    MATE_LOWER ≤ fuelValueD2 G guard 3 (intrinsicEdgeSpend G hot eligible low) 10 p :=
  forcedMate_fuelValueD2_noSubPass G guard 3 (intrinsicEdgeSpend G hot eligible low)
    (by omega) (by omega) hg hF hFM 10 (by omega) (by omega)

/-- **Menu option M2, free**: one reduction bit instead of two (`C = 2`)
is the generic theorem's `C = 2` instance -- `D ≥ 2k + 2` against `3k`.
`M3` (`C = 1`, no reductions) gives `D ≥ k + 4`. Both are corollaries of
`forcedMate_fuelValueD2_sharp`; only the edge-cost cap changes. -/
theorem forcedMate_fuelValueD2_sharp_C2 (G : QSGame) (guard : G.Pos → Bool)
    (spend : G.Pos → Nat → G.Pos → Nat) (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 2 ≤ D → 2 * k + 2 ≤ D →
      MATE_LOWER ≤ fuelValueD2 G guard 2 spend D p :=
  fun D h2 hD => forcedMate_fuelValueD2_sharp G guard 2 spend (by omega) (by omega)
    hF hFM D h2 (by omega)

theorem forcedMate_fuelValueD2_sharp_C1 (G : QSGame) (guard : G.Pos → Bool)
    (spend : G.Pos → Nat → G.Pos → Nat) (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 2 ≤ D → k + 4 ≤ D →
      MATE_LOWER ≤ fuelValueD2 G guard 1 spend D p :=
  fun D h2 hD => forcedMate_fuelValueD2_sharp G guard 1 spend (by omega) (by omega)
    hF hFM D h2 (by omega)

/-! # Part III: the shipped shallow cap, and the depth it really costs

`fuelValueD2` -- and therefore `forcedMate_fuelValueD2`, its `3k + 4`, and
Part I's sharpened `3k` -- does NOT model one mechanism the shipped search
has: the shallow static cap

    ceiling = lambda v: MATE_UPPER if depth > 4 or v >= MATE_LOWER else pos.score + v + max(depth - 1, 0) * QS_A
    if (cap := ceiling(val)) < gamma: best = max(best, cap); break
    score = min(cap, -self.bound(pos.move(move), 1 - gamma, move_depth))

At every nominal depth `≤ 4` EVERY move that is not a king capture reports
at most `shallowMoveCap`, which `shallowMoveCap_below_positiveMate`
(CappedMove.lean) puts strictly below `MATE_LOWER` under the both-kings
material invariant `CapInBand` - the spine below consumes it through
`capClamp`'s envelope, which needs no premise (`capClamp_eq_shipped`
localizes the invariant).  Such a node cannot
report a mate at all -- the cap "can delay a mate proof found exactly at
the selective frontier", as formal/README.md puts it, and this is the
delay, priced.  The cap only ever LOWERS a report, so it cannot hurt a
defender node (whose fold needs an upper bound); it binds exactly at the
ATTACKER nodes of the mating line, whose admission floor therefore rises
from 2 to 5.

The two ends of the band are not the same mechanism.  At depths 2 through 4 the
clamp is the SELECTIVE cap and it binds.  At depths 0 and 1 natural
subtraction flattens the margin and the clamp is the old stand-pat futility
estimate (`shallowMoveCap_lowDepth`), where it is mate-neutral -- a fold
weight can only reach the positive mate band through a child whose king is
gone, and such a parent fires the node-level `hasKingCapture` branch before
any fold is taken.  So `capClamp` carries the shipped `depth ≤ 4` band
exactly, and the floor is 5.

Consequence for the uniform bound (`fuelValueD2C` below is `fuelValueD2`
plus the clamp):

    D ≥ 5,   C*k + 5 ≤ D + C,   C*k + 7 ≤ D + 2*C

i.e. `D ≥ max 5 (C*(k-1) + 5) (C*(k-2) + 7)`, which at the shipped `C = 3`
is `D ≥ 3k + 2` for `k ≥ 3` and `D ≥ 5` for `k ≤ 2`. The one-ply corner
is exactly what the suite shows: mate-in-1 needs `D = 5`.

MEASUREMENT CORRECTION (2026-08-17).  The cheap way to buy Part I's
cap-free `3k` in the shipped engine -- keep the clamp, but exempt a child
report that came back at or above `MATE_LOWER`, about ten bytes -- is
REFUTED, on correctness rather than on strength.  `capClamp_le` below is
true and stays true: the exemption only RAISES the declared value, and
`forcedMate_fuelValueD2_sharp` really does bound that value by `3k`.  The
declared value is not what breaks.  The exemption drops the clamp on the
SEARCHED branch while the cap's fail-low branch
(`if cap < gamma: move, score = None, cap`) still claims the static
estimate for the SAME `(pos, depth)` key, so `bound()` becomes
gamma-dependent and the table stores a contradiction: measured,
`Entry(lower = 47938, upper = 1204)` on one key at depth 2, with twelve
terminal-bench positions and a tt-consistency fortress failing on "ladder
crossing".  (The mates do arrive -- `mate1.fen` 0/8 to 6/8 at depths 2 and
3, node battery byte-identical -- so the price is exactly the invariant.)
Generally: a cap may be dropped on a searched report only if it is also
dropped on the unsearched one, and no static rule can know that an
unsearched child mates; under a `(pos, depth)`-keyed table shallow
futility and shallow mate detection are mutually exclusive.  The sound
instances are deleting the cap on BOTH branches -- Part I's `3k`, priced
at -60.41 ± 26.61 Elo, so Elo-inadmissible -- or a
`(pos, depth, bound-type)`-aware table, which is unpriced.  Ledger:
`measure/search-features-ledger` at 0af3507, arm `exp/mate-band-exempt`. -/

/-- The shipped cap as a fold weight transformer, under the model's band
ENVELOPE `min (MATE_LOWER - 1) ·`.  The band is the consumer's own
`depth > 4` test, so it covers depths 0 through 4.  The code carries no
such envelope; `capClamp_eq_shipped` proves the two agree under the
material invariant `CapInBand`, which localizes the both-kings premise to
that one lemma and lets the whole mate-depth spine below keep consuming
the envelope's unconditional band-safety. -/
def capClamp (G : QSGame) (p : G.Pos) (d : Nat) (m : G.Pos) (x : Int) : Int :=
  if d ≤ 4 ∧ G.val p m < MATE_LOWER then
    min (min (MATE_LOWER - 1) (shallowMoveCap (G.eval p) (G.val p m) d)) x
  else x

/-- Under the both-kings material invariant the envelope is dead and the
model's fold weight is exactly the shipped `min(cap, child)`. -/
theorem capClamp_eq_shipped (G : QSGame) (p : G.Pos) (d : Nat) (m : G.Pos)
    (x : Int) (hband : CapInBand (G.eval p) (G.val p m) d) :
    capClamp G p d m x =
      if d ≤ 4 ∧ G.val p m < MATE_LOWER then
        min (shallowMoveCap (G.eval p) (G.val p m) d) x
      else x := by
  have hb : G.eval p + G.val p m + ((d - 1 : Nat) : Int) * QS_A
      < MATE_LOWER := hband
  unfold capClamp shallowMoveCap
  split
  · omega
  · rfl

/-- The cap only lowers: the defender side never pays for it.  Monotonicity
of the transformer is NOT a licence to drop the clamp in the code on the
searched branch alone -- see the measurement correction in Part III's
header: that keeps this theorem true and still breaks the `(pos, depth)`
table. -/
theorem capClamp_le (G : QSGame) (p : G.Pos) (d : Nat) (m : G.Pos) (x : Int) :
    capClamp G p d m x ≤ x := by
  unfold capClamp
  split
  · exact Int.min_le_right _ _
  · exact Int.le_refl x

/-- Above depth four the clamp is the identity. -/
theorem capClamp_of_deep (G : QSGame) (p : G.Pos) {d : Nat} (hd : 5 ≤ d)
    (m : G.Pos) (x : Int) : capClamp G p d m x = x := by
  unfold capClamp
  rw [if_neg (fun h => by have := h.1; omega)]

/-- At any depth in the band no ordinary move can report a mate. -/
theorem capClamp_lt_ML (G : QSGame) (p : G.Pos) {d : Nat} (hd4 : d ≤ 4)
    (m : G.Pos) (hval : G.val p m < MATE_LOWER) (x : Int) :
    capClamp G p d m x < MATE_LOWER := by
  unfold capClamp
  rw [if_pos ⟨hd4, hval⟩]
  have h1 := Int.min_le_left (MATE_LOWER - 1)
    (shallowMoveCap (G.eval p) (G.val p m) d)
  have h2 := Int.min_le_left
    (min (MATE_LOWER - 1) (shallowMoveCap (G.eval p) (G.val p m) d)) x
  omega

/-- **The fuel-shaped declared value WITH the shipped shallow cap.**
Identical to `fuelValueD2` except that every fold weight passes through
`capClamp` -- the code's `min(cap, ...)`. -/
def fuelValueD2C (G : QSGame) (guard : G.Pos → Bool) (C : Nat)
    (spend : G.Pos → Nat → G.Pos → Nat) : Nat → G.Pos → Int
  | 0, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if allIllegalB G p = true then terminalValue G (d + 1) p
    else if d + 1 < 6 then
      foldMax (fun m => capClamp G p (d + 1) m (-(fuelValueD2C G guard C spend d m)))
        (movesAbove G (val_lower (d + 1)) p)
        (if guard p = true ∧ 2 < d + 1 then
          (if -(fuelValueD2C G guard C spend (d + 1 - 4) (G.pass p)) < MATE_LOWER then
            max LOSS (-(fuelValueD2C G guard C spend (d + 1 - 4) (G.pass p)))
          else LOSS)
        else LOSS)
    else
      foldMax (fun m => capClamp G p (d + 1) m (-(fuelValueD2C G guard C spend
          (d - min (C - 1) (spend p (d + 1) m)) m)))
        (movesAbove G (val_lower (d + 1)) p) LOSS
termination_by d _ => d
decreasing_by all_goals omega

theorem fuelValueD2C_kingGone (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos) (h : G.eval p ≤ -MATE_LOWER) :
    fuelValueD2C G guard C spend d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [fuelValueD2C]; rw [if_pos h]
  | succ d => simp only [fuelValueD2C]; rw [if_pos h]

theorem fuelValueD2C_of_capture (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos) (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    fuelValueD2C G guard C spend d p = MATE_UPPER := by
  cases d with
  | zero => simp only [fuelValueD2C]; rw [if_neg hkg, if_pos hcap]
  | succ d => simp only [fuelValueD2C]; rw [if_neg hkg, if_pos hcap]

theorem fuelValueD2C_of_allIllegal (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true) :
    fuelValueD2C G guard C spend (d + 1) p = terminalValue G (d + 1) p := by
  simp only [fuelValueD2C]
  rw [if_neg hkg, if_neg hcap, if_pos hai]

theorem fuelValueD2C_of_fold_regime (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hd : 5 ≤ d) :
    fuelValueD2C G guard C spend (d + 1) p
      = foldMax (fun m => capClamp G p (d + 1) m (-(fuelValueD2C G guard C spend
          (d - min (C - 1) (spend p (d + 1) m)) m)))
          (movesAbove G (val_lower (d + 1)) p) LOSS := by
  simp only [fuelValueD2C]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_neg (by omega)]

theorem fuelValueD2C_of_fold_sub (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hd : d < 5) :
    fuelValueD2C G guard C spend (d + 1) p
      = foldMax (fun m => capClamp G p (d + 1) m (-(fuelValueD2C G guard C spend d m)))
          (movesAbove G (val_lower (d + 1)) p)
          (if guard p = true ∧ 2 < d + 1 then
            (if -(fuelValueD2C G guard C spend (d + 1 - 4) (G.pass p)) < MATE_LOWER then
              max LOSS (-(fuelValueD2C G guard C spend (d + 1 - 4) (G.pass p)))
            else LOSS)
          else LOSS) := by
  simp only [fuelValueD2C]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_pos (by omega)]

theorem fuelValueD2C_checkmated (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) {m : G.Pos}
    (hcap : hasKingCapture G.toNullGame.toGame m = false)
    (hmate : Checkmated G m) :
    ∀ d : Nat, 1 ≤ d → fuelValueD2C G guard C spend d m ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro d hd
  cases d with
  | zero => omega
  | succ d' =>
    by_cases hkgm : G.eval m ≤ -MATE_LOWER
    · rw [fuelValueD2C_kingGone G guard C spend (d' + 1) m hkgm]; omega
    · rw [fuelValueD2C_of_allIllegal G guard C spend d' m hkgm (by simp [hcap]) hmate.1]
      have := terminalValue_mate G (d' + 1) m hmate.2
      omega

/-- The last ply under the cap: the attacker node needs depth `≥ 5`. -/
theorem forcedMate_leaf_fuelValueD2C (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC1 : 1 ≤ C) (hC4 : C ≤ 4)
    (hF : ValFloor G 192) {p m : G.Pos}
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) (hm : m ∈ G.moves p)
    (hleg : hasKingCapture G.toNullGame.toGame m = false) (hmate : Checkmated G m) :
    ∀ D : Nat, 5 ≤ D → MATE_LOWER ≤ fuelValueD2C G guard C spend D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro D h5
  cases D with
  | zero => omega
  | succ d =>
    by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
    · rw [fuelValueD2C_of_capture G guard C spend (d + 1) p hkg hcap]; omega
    · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
      have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
      by_cases hreg : 5 ≤ d
      · rw [fuelValueD2C_of_fold_regime G guard C spend d p hkg hcap hai hreg]
        have hchild : fuelValueD2C G guard C spend
            (d - min (C - 1) (spend p (d + 1) m)) m ≤ -MATE_LOWER :=
          fuelValueD2C_checkmated G guard C spend hleg hmate _ (by omega)
        have hcl := capClamp_of_deep G p (d := d + 1) (by omega) m
          (-(fuelValueD2C G guard C spend (d - min (C - 1) (spend p (d + 1) m)) m))
        have hfold : capClamp G p (d + 1) m (-(fuelValueD2C G guard C spend
              (d - min (C - 1) (spend p (d + 1) m)) m))
            ≤ foldMax (fun x => capClamp G p (d + 1) x (-(fuelValueD2C G guard C spend
                (d - min (C - 1) (spend p (d + 1) x)) x)))
                (movesAbove G (val_lower (d + 1)) p) LOSS :=
          foldMax_le_of_mem _ _ _ _ hmem
        rw [hcl] at hfold
        omega
      · rw [fuelValueD2C_of_fold_sub G guard C spend d p hkg hcap hai (by omega)]
        have hchild : fuelValueD2C G guard C spend d m ≤ -MATE_LOWER :=
          fuelValueD2C_checkmated G guard C spend hleg hmate d (by omega)
        have hcl := capClamp_of_deep G p (d := d + 1) (by omega) m
          (-(fuelValueD2C G guard C spend d m))
        have hfold : capClamp G p (d + 1) m (-(fuelValueD2C G guard C spend d m))
            ≤ foldMax (fun x => capClamp G p (d + 1) x (-(fuelValueD2C G guard C spend d x)))
                (movesAbove G (val_lower (d + 1)) p)
                (if guard p = true ∧ 2 < d + 1 then
                  (if -(fuelValueD2C G guard C spend (d + 1 - 4) (G.pass p)) < MATE_LOWER then
                    max LOSS (-(fuelValueD2C G guard C spend (d + 1 - 4) (G.pass p)))
                  else LOSS)
                else LOSS) :=
          foldMax_le_of_mem _ _ _ _ hmem
        rw [hcl] at hfold
        omega

/-- **Mate-in-k completeness for the SHIPPED search, sharp**: the cap costs
two plies of slope-independent depth and raises the one-ply corner to 5.
Shipped `C = 3`: `D ≥ 3k + 2` (`k ≥ 3`), `D ≥ 5` (`k ≤ 2`).  `ValFloor`
only -- still no chess premise. -/
theorem forcedMate_fuelValueD2C_sharp (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC1 : 1 ≤ C) (hC4 : C ≤ 4)
    (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 5 ≤ D → C * k + 5 ≤ D + C → C * k + 7 ≤ D + 2 * C →
      MATE_LOWER ≤ fuelValueD2C G guard C spend D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  induction hFM with
  | @mate k p m hkg hm hleg hmate =>
    intro D h5 _ _
    exact forcedMate_leaf_fuelValueD2C G guard C spend hC1 hC4 hF hkg hm hleg hmate D h5
  | @step k p m hkg hm hleg hnt hreply ih =>
    intro D h4 hD1 hD2
    rcases Nat.eq_zero_or_pos k with hk0 | hk1
    · subst hk0
      obtain ⟨m', hm', hleg'⟩ := legal_of_allIllegalB_false hnt
      exact absurd (hreply m' hm' hleg') (fun h => by cases h)
    · have hexp : C * (k + 2) = C * k + 2 * C := by rw [Nat.mul_add]; omega
      have hCk : C ≤ C * k := Nat.le_mul_of_pos_right C hk1
      cases D with
      | zero => omega
      | succ d =>
        by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [fuelValueD2C_of_capture G guard C spend (d + 1) p hkg hcap]; omega
        · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
          rw [fuelValueD2C_of_fold_regime G guard C spend d p hkg hcap hai (by omega)]
          have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
          have hchild : fuelValueD2C G guard C spend
              (d - min (C - 1) (spend p (d + 1) m)) m ≤ -MATE_LOWER := by
            obtain ⟨dm, hdm⟩ : ∃ x, d - min (C - 1) (spend p (d + 1) m) = x + 1 :=
              ⟨d - min (C - 1) (spend p (d + 1) m) - 1, by omega⟩
            rw [hdm]
            by_cases hkgm : G.eval m ≤ -MATE_LOWER
            · rw [fuelValueD2C_kingGone G guard C spend (dm + 1) m hkgm]; omega
            · rw [fuelValueD2C_of_fold_regime G guard C spend dm m hkgm
                (by simp [hleg]) hnt (by omega)]
              refine foldMax_le _ _ _ (fun m' hm' => ?_) (by omega)
              have hm'' : m' ∈ G.moves m := movesAbove_subset G _ m m' hm'
              have hkgm' : ¬ (G.eval m' ≤ -MATE_LOWER) := by
                intro hle
                have hc : hasKingCapture G.toNullGame.toGame m = true :=
                  (hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm'', hle⟩
                rw [hleg] at hc
                exact Bool.noConfusion hc
              refine Int.le_trans (capClamp_le G m (dm + 1) m' _) ?_
              cases hcm : hasKingCapture G.toNullGame.toGame m' with
              | true =>
                rw [fuelValueD2C_of_capture G guard C spend _ m' hkgm' hcm]
                omega
              | false =>
                have := ih m' hm'' hcm
                  (dm - min (C - 1) (spend m (dm + 1) m')) (by omega) (by omega) (by omega)
                omega
          have hcl := capClamp_of_deep G p (d := d + 1) (by omega) m
            (-(fuelValueD2C G guard C spend (d - min (C - 1) (spend p (d + 1) m)) m))
          have hfold : capClamp G p (d + 1) m (-(fuelValueD2C G guard C spend
                (d - min (C - 1) (spend p (d + 1) m)) m))
              ≤ foldMax (fun x => capClamp G p (d + 1) x (-(fuelValueD2C G guard C spend
                  (d - min (C - 1) (spend p (d + 1) x)) x)))
                  (movesAbove G (val_lower (d + 1)) p) LOSS :=
            foldMax_le_of_mem _ _ _ _ hmem
          rw [hcl] at hfold
          omega

/-- The mated dual under the cap: the cap only lowers reports, so the
defender side pays nothing for it beyond the mate side's own floor.
Shipped `C = 3`: `D ≥ 3k + 5`. -/
theorem forcedlyMated_fuelValueD2C_sharp (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC1 : 1 ≤ C) (hC4 : C ≤ 4)
    (hF : ValFloor G 192)
    {k : Nat} {q : G.Pos}
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hFL : ForcedlyMated G k q) :
    ∀ D : Nat, 7 ≤ D → C * k + 5 ≤ D → C * k + 7 ≤ D + C →
      fuelValueD2C G guard C spend D q ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro D h7 hD1 hD2
  cases D with
  | zero => omega
  | succ d =>
    by_cases hkg : G.eval q ≤ -MATE_LOWER
    · rw [fuelValueD2C_kingGone G guard C spend (d + 1) q hkg]; omega
    · have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by
        simp [hcapq]
      cases hFL with
      | inl hcm =>
        rw [fuelValueD2C_of_allIllegal G guard C spend d q hkg hcapq' hcm.1]
        have := terminalValue_mate G (d + 1) q hcm.2
        omega
      | inr h =>
        obtain ⟨hai, hall⟩ := h
        obtain ⟨m0, hm0, hleg0⟩ := legal_of_allIllegalB_false hai
        have hk1 : 1 ≤ k := by
          rcases Nat.eq_zero_or_pos k with hk0 | hk1
          · subst hk0; exact absurd (hall m0 hm0 hleg0) (fun h => by cases h)
          · exact hk1
        have hCk : C ≤ C * k := Nat.le_mul_of_pos_right C hk1
        rw [fuelValueD2C_of_fold_regime G guard C spend d q hkg hcapq' hai (by omega)]
        refine foldMax_le _ _ _ (fun m hm => ?_) (by omega)
        have hm' : m ∈ G.moves q := movesAbove_subset G _ q m hm
        have hmin := Nat.min_le_left (C - 1) (spend q (d + 1) m)
        refine Int.le_trans (capClamp_le G q (d + 1) m _) ?_
        cases hcm : hasKingCapture G.toNullGame.toGame m with
        | true =>
          have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := by
            intro hle
            have hc : hasKingCapture G.toNullGame.toGame q = true :=
              (hasKingCapture_iff G.toNullGame.toGame q).mpr ⟨m, hm', hle⟩
            rw [hcapq] at hc
            exact Bool.noConfusion hc
          rw [fuelValueD2C_of_capture G guard C spend _ m hkgm hcm]
          omega
        | false =>
          have := forcedMate_fuelValueD2C_sharp G guard C spend hC1 hC4 hF
            (hall m hm' hcm) (d - min (C - 1) (spend q (d + 1) m))
            (by omega) (by omega) (by omega)
          omega

/-! ## The CI table for the SHIPPED search (`C = 3`, cap included)

The suite convention (`tools/quick_tests.sh`): mate-in-`n` moves is
`k = 2n - 1` plies. These are both the sharp proved depths and the executable
suite depths for the engine as it stands. -/

/-- mate-in-1 (`k = 1`): `D = 5`. -/
theorem ci_code_mate_in_1 (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {p : G.Pos} (hFM : ForcedMate G 1 p) :
    MATE_LOWER ≤ fuelValueD2C G guard 3 (intrinsicEdgeSpend G hot eligible low) 5 p :=
  forcedMate_fuelValueD2C_sharp G guard 3 (intrinsicEdgeSpend G hot eligible low)
    (by omega) (by omega) hF hFM 5 (by omega) (by omega) (by omega)

/-- mate-in-2 (`k = 3`): `D = 11`. -/
theorem ci_code_mate_in_2 (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {p : G.Pos} (hFM : ForcedMate G 3 p) :
    MATE_LOWER ≤ fuelValueD2C G guard 3 (intrinsicEdgeSpend G hot eligible low) 11 p :=
  forcedMate_fuelValueD2C_sharp G guard 3 (intrinsicEdgeSpend G hot eligible low)
    (by omega) (by omega) hF hFM 11 (by omega) (by omega) (by omega)

/-- mate-in-3 (`k = 5`): `D = 17`. -/
theorem ci_code_mate_in_3 (G : QSGame) (guard : G.Pos → Bool)
    (hot eligible : G.Pos → Nat → Bool) (low : G.Pos → Nat → G.Pos → Bool)
    (hF : ValFloor G 192) {p : G.Pos} (hFM : ForcedMate G 5 p) :
    MATE_LOWER ≤ fuelValueD2C G guard 3 (intrinsicEdgeSpend G hot eligible low) 17 p :=
  forcedMate_fuelValueD2C_sharp G guard 3 (intrinsicEdgeSpend G hot eligible low)
    (by omega) (by omega) hF hFM 17 (by omega) (by omega) (by omega)

/-- Menu options at a glance, capped model: one reduction bit (`C = 2`)
gives `D ≥ 2k + 3`; no reductions (`C = 1`) gives `D ≥ k + 5`.  Both are
instances -- only the edge-cost cap changes. -/
theorem forcedMate_fuelValueD2C_C2 (G : QSGame) (guard : G.Pos → Bool)
    (spend : G.Pos → Nat → G.Pos → Nat) (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 5 ≤ D → 2 * k + 3 ≤ D →
      MATE_LOWER ≤ fuelValueD2C G guard 2 spend D p :=
  fun D h5 hD => forcedMate_fuelValueD2C_sharp G guard 2 spend (by omega) (by omega)
    hF hFM D h5 (by omega) (by omega)

theorem forcedMate_fuelValueD2C_C1 (G : QSGame) (guard : G.Pos → Bool)
    (spend : G.Pos → Nat → G.Pos → Nat) (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 5 ≤ D → k + 5 ≤ D →
      MATE_LOWER ≤ fuelValueD2C G guard 1 spend D p :=
  fun D h5 hD => forcedMate_fuelValueD2C_sharp G guard 1 spend (by omega) (by omega)
    hF hFM D h5 (by omega) (by omega)

/-! # Part II: sharpness

A concrete game in the hypothesis class where the sharpened depth cannot
be lowered by even one ply.  The spine is

    A2 -> D2 -> A1 -> D1 -> A0 -> LF

with `LF` checkmated (`XI` is `LF`'s only, self-destructing, move; `YC`
witnesses the check).  `ZP` is the pass target: a moveless, not-in-check
position, so the terminal correction values it 0 -- the defender's escape
hatch when it is reached inside the sub-horizon window. -/

/-- The witness positions.  `KG` is the only kingless one: it is what makes
`LF`'s single move illegal (`allIllegalB LF`) and `YC` the check witness
(`inCheckB LF`).  `pass LF = YC` makes `LF` mated rather than stalemated;
`pass _ = ZP` is the masking pass -- `ZP` is moveless and not in check, so
the terminal correction values it 0.  `val ≡ 0` clears the `ValFloor 192`
bar and admits every move at every positive depth. -/
inductive MDPos where
  | A2 | D2 | A1 | D1 | A0 | LF | XI | KG | YC | ZP
  deriving DecidableEq

open MDPos in
def MDG : QSGame where
  Pos := MDPos
  moves := fun x => match x with
    | A2 => [D2]
    | D2 => [A1]
    | A1 => [D1]
    | D1 => [A0]
    | A0 => [LF]
    | LF => [XI]
    | XI => [KG]
    | KG => []
    | YC => [KG]
    | ZP => []
  eval := fun x => match x with
    | KG => -MATE_UPPER
    | _ => 0
  pass := fun x => match x with
    | LF => YC
    | _ => ZP
  val := fun _ _ => 0

/-- Inside the theorem's hypothesis class. -/
theorem sharp_valFloor : ValFloor MDG 192 := by
  intro p m _; simp [MDG]

/-- The maximal spend: `min (C-1) 2 = 2` at `C = 3`, so every regime edge
costs three plies -- the hot bit plus the intrinsic-LMR bit. -/
def mdSpend : MDPos → Nat → MDPos → Nat := fun _ _ _ => 2

/-- Guard on: the shipped `abs(pos.score) < 750 and any(c in pos.board ...)`
holds at a quiet position with pieces. -/
def mdGuard : MDPos → Bool := fun _ => true

theorem sharp_checkmated_LF : Checkmated MDG MDPos.LF := ⟨by decide, by decide⟩

theorem sharp_forcedMate_1_A0 : ForcedMate MDG 1 MDPos.A0 :=
  ForcedMate.mate (k := 0) (m := MDPos.LF) (by decide)
    (show MDPos.LF ∈ [MDPos.LF] from List.mem_cons_self _ _)
    (by decide) sharp_checkmated_LF

theorem sharp_forcedMate_3_A1 : ForcedMate MDG 3 MDPos.A1 :=
  ForcedMate.step (k := 1) (m := MDPos.D1) (by decide)
    (show MDPos.D1 ∈ [MDPos.D1] from List.mem_cons_self _ _)
    (by decide) (by decide)
    (fun m' hm' _ => by
      have h1 : m' ∈ [MDPos.A0] := hm'
      have h2 : m' = MDPos.A0 := by simpa using h1
      subst h2; exact sharp_forcedMate_1_A0)

theorem sharp_forcedMate_5_A2 : ForcedMate MDG 5 MDPos.A2 :=
  ForcedMate.step (k := 3) (m := MDPos.D2) (by decide)
    (show MDPos.D2 ∈ [MDPos.D2] from List.mem_cons_self _ _)
    (by decide) (by decide)
    (fun m' hm' _ => by
      have h1 : m' ∈ [MDPos.A1] := hm'
      have h2 : m' = MDPos.A1 := by simpa using h1
      subst h2; exact sharp_forcedMate_3_A1)

theorem sharp_forcedlyMated_3_D2 : ForcedlyMated MDG 3 MDPos.D2 :=
  Or.inr ⟨by decide, fun m' hm' _ => by
    have h1 : m' ∈ [MDPos.A1] := hm'
    have h2 : m' = MDPos.A1 := by simpa using h1
    subst h2; exact sharp_forcedMate_3_A1⟩

/-- The sub-horizon pass term, evaluated: a pass worth 0 leaves the
defender's fold seeded at 0 instead of `LOSS`.  This is the masking
channel, in one equation. -/
theorem sub_seed_of_pass_zero (G : QSGame) (guard : G.Pos → Bool) (C : Nat)
    (spend : G.Pos → Nat → G.Pos → Nat) (d : Nat) (p : G.Pos)
    (hg : guard p = true) (h2 : 2 < d + 1)
    (hpass : fuelValueD2 G guard C spend (d + 1 - 4) (G.pass p) = 0) :
    (if guard p = true ∧ 2 < d + 1 then
      (if -(fuelValueD2 G guard C spend (d + 1 - 4) (G.pass p)) < MATE_LOWER then
        max LOSS (-(fuelValueD2 G guard C spend (d + 1 - 4) (G.pass p)))
      else LOSS)
    else LOSS) = 0 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  rw [if_pos (And.intro hg h2), hpass]
  rw [if_pos (show -(0 : Int) < MATE_LOWER by omega)]
  omega

/-- The pass target is worth 0 at every positive depth: moveless, hence
`allIllegalB`, and not in check, hence the terminal correction's draw. -/
theorem sharp_ZP (d : Nat) : fuelValueD2 MDG mdGuard 3 mdSpend (d + 1) MDPos.ZP = 0 := by
  rw [fuelValueD2_of_allIllegal MDG mdGuard 3 mdSpend d MDPos.ZP
    (by decide) (by decide) (by decide)]
  have hic : inCheckB MDG.toNullGame MDPos.ZP = false := by decide
  simp [terminalValue, hic]

/-- The mate is there at depth 4: an attacker node two plies below the
horizon still sees it, because attacker nodes never needed the horizon. -/
theorem sharp_A0_4 : MATE_LOWER ≤ fuelValueD2 MDG mdGuard 3 mdSpend 4 MDPos.A0 :=
  forcedMate_fuelValueD2_short MDG mdGuard 3 mdSpend (by omega) (by omega)
    sharp_valFloor (by omega) sharp_forcedMate_1_A0 4 (by omega)

/-- **The masking step.**  At nominal depth 5 the defender node `D1` is
inside the sub-horizon window `2 < depth < 6`, where the pass is still a
score candidate.  Passing is worth 0, the fold is a max, and the mate --
correctly seen one ply below (`sharp_A0_4`) -- is masked. -/
theorem sharp_D1_5 : fuelValueD2 MDG mdGuard 3 mdSpend 5 MDPos.D1 = 0 := by
  have hML : MATE_LOWER = 47923 := rfl
  have hma : movesAbove MDG (val_lower 5) MDPos.D1 = [MDPos.A0] :=
    movesAbove_all MDG 5 MDPos.D1 (by decide)
  have hpassZ : fuelValueD2 MDG mdGuard 3 mdSpend (4 + 1 - 4) (MDG.pass MDPos.D1) = 0 :=
    sharp_ZP 0
  have hA0 := sharp_A0_4
  rw [show (5 : Nat) = 4 + 1 from rfl,
    fuelValueD2_of_fold_sub MDG mdGuard 3 mdSpend 4 MDPos.D1
      (by decide) (by decide) (by decide) (by omega),
    hma,
    sub_seed_of_pass_zero MDG mdGuard 3 mdSpend 4 MDPos.D1 (by decide) (by omega) hpassZ]
  simp only [foldMax]
  omega

/-- **Sharpness of `D ≥ 3k` at `k = 3`**: one ply less than `3*3` and the
mate in 3 plies is invisible -- the value is 0, not a mate score. -/
theorem sharp_mate3_at_8 : fuelValueD2 MDG mdGuard 3 mdSpend 8 MDPos.A1 = 0 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hma : movesAbove MDG (val_lower 8) MDPos.A1 = [MDPos.D1] :=
    movesAbove_all MDG 8 MDPos.A1 (by decide)
  have hD1 := sharp_D1_5
  rw [show (8 : Nat) = 7 + 1 from rfl,
    fuelValueD2_of_fold_regime MDG mdGuard 3 mdSpend 7 MDPos.A1
      (by decide) (by decide) (by decide) (by omega),
    hma]
  simp only [foldMax, mdSpend, show (7 - min (3 - 1) 2 : Nat) = 5 from rfl]
  rw [hD1]
  omega

/-- The defender node one ply up escapes with it: the dual bound `3k + 3`
is sharp at `k = 3` (`11 = 3*3 + 2`). -/
theorem sharp_mated3_at_11 : fuelValueD2 MDG mdGuard 3 mdSpend 11 MDPos.D2 = 0 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hma : movesAbove MDG (val_lower 11) MDPos.D2 = [MDPos.A1] :=
    movesAbove_all MDG 11 MDPos.D2 (by decide)
  have h8 := sharp_mate3_at_8
  rw [show (11 : Nat) = 10 + 1 from rfl,
    fuelValueD2_of_fold_regime MDG mdGuard 3 mdSpend 10 MDPos.D2
      (by decide) (by decide) (by decide) (by omega),
    hma]
  simp only [foldMax, mdSpend, show (10 - min (3 - 1) 2 : Nat) = 8 from rfl]
  rw [h8]
  omega

/-- **Sharpness of the slope**: `14 = 3*5 - 1`.  Two plies of mating line
cost `2*C = 6` plies of depth -- the witnesses at 8 and 14 are exactly
that far apart, so no bound with a slope below `C` can hold either. -/
theorem sharp_mate5_at_14 : fuelValueD2 MDG mdGuard 3 mdSpend 14 MDPos.A2 = 0 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hma : movesAbove MDG (val_lower 14) MDPos.A2 = [MDPos.D2] :=
    movesAbove_all MDG 14 MDPos.A2 (by decide)
  have h11 := sharp_mated3_at_11
  rw [show (14 : Nat) = 13 + 1 from rfl,
    fuelValueD2_of_fold_regime MDG mdGuard 3 mdSpend 13 MDPos.A2
      (by decide) (by decide) (by decide) (by omega),
    hma]
  simp only [foldMax, mdSpend, show (13 - min (3 - 1) 2 : Nat) = 11 from rfl]
  rw [h11]
  omega

/-! ## The certificates, as statements about the bounds

Each says: the sharpened bound is ATTAINED -- one ply below it there is a
game in the hypothesis class, with an edge-spend schedule the shipped code
can realize, whose declared value misses the mate band. -/

theorem mate_depth_bound_sharp_k3 :
    ∃ (G : QSGame) (guard : G.Pos → Bool) (spend : G.Pos → Nat → G.Pos → Nat) (p : G.Pos),
      ValFloor G 192 ∧ ForcedMate G 3 p ∧
        fuelValueD2 G guard 3 spend (3 * 3 - 1) p < MATE_LOWER :=
  ⟨MDG, mdGuard, mdSpend, MDPos.A1, sharp_valFloor, sharp_forcedMate_3_A1, by
    have hML : MATE_LOWER = 47923 := rfl
    have h : fuelValueD2 MDG mdGuard 3 mdSpend (3 * 3 - 1) MDPos.A1 = 0 := sharp_mate3_at_8
    omega⟩

theorem mate_depth_bound_sharp_k5 :
    ∃ (G : QSGame) (guard : G.Pos → Bool) (spend : G.Pos → Nat → G.Pos → Nat) (p : G.Pos),
      ValFloor G 192 ∧ ForcedMate G 5 p ∧
        fuelValueD2 G guard 3 spend (3 * 5 - 1) p < MATE_LOWER :=
  ⟨MDG, mdGuard, mdSpend, MDPos.A2, sharp_valFloor, sharp_forcedMate_5_A2, by
    have hML : MATE_LOWER = 47923 := rfl
    have h : fuelValueD2 MDG mdGuard 3 mdSpend (3 * 5 - 1) MDPos.A2 = 0 := sharp_mate5_at_14
    omega⟩

theorem mated_depth_bound_sharp_k3 :
    ∃ (G : QSGame) (guard : G.Pos → Bool) (spend : G.Pos → Nat → G.Pos → Nat) (q : G.Pos),
      ValFloor G 192 ∧ hasKingCapture G.toNullGame.toGame q = false ∧
        ForcedlyMated G 3 q ∧
        -MATE_LOWER < fuelValueD2 G guard 3 spend (3 * 3 + 2) q :=
  ⟨MDG, mdGuard, mdSpend, MDPos.D2, sharp_valFloor, by decide, sharp_forcedlyMated_3_D2, by
    have hML : MATE_LOWER = 47923 := rfl
    have h : fuelValueD2 MDG mdGuard 3 mdSpend (3 * 3 + 2) MDPos.D2 = 0 := sharp_mated3_at_11
    omega⟩


/-! ## Sharpness of the capped bound

The same game certifies the two plies the widened cap costs. At
`D = 10 = 3*3+1`, the mating line's last attacker node lands at nominal
depth 4, where the shipped cap clamps its report to `shallowMoveCap = 540`.
The mate never leaves that node, so `3k + 2` is the exact uniform bound and
`3k + 1` is one ply too few. -/

theorem sharp_cap_ZP0 (g : MDPos → Bool) : fuelValueD2C MDG g 3 mdSpend 0 MDPos.ZP = 0 := by
  simp only [fuelValueD2C]
  rw [if_neg (by decide), if_neg (by decide)]
  rfl

theorem sharp_cap_ZP1 (g : MDPos → Bool) : fuelValueD2C MDG g 3 mdSpend 1 MDPos.ZP = 0 := by
  rw [fuelValueD2C_of_allIllegal MDG g 3 mdSpend 0 MDPos.ZP
    (by decide) (by decide) (by decide)]
  have hic : inCheckB MDG.toNullGame MDPos.ZP = false := by decide
  simp [terminalValue, hic]

/-- The capped last attacker node: at nominal depth 4 the mate it can see
one ply below is clamped to the static cap. -/
theorem sharp_cap_A0_4 (g : MDPos → Bool) : fuelValueD2C MDG g 3 mdSpend 4 MDPos.A0 ≤ 540 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hma : movesAbove MDG (val_lower 4) MDPos.A0 = [MDPos.LF] :=
    movesAbove_all MDG 4 MDPos.A0 (by decide)
  have hZ : fuelValueD2C MDG g 3 mdSpend (3 + 1 - 4) (MDG.pass MDPos.A0) = 0 :=
    sharp_cap_ZP0 g
  rw [show (4 : Nat) = 3 + 1 from rfl,
    fuelValueD2C_of_fold_sub MDG g 3 mdSpend 3 MDPos.A0
      (by decide) (by decide) (by decide) (by omega),
    hma, hZ]
  refine foldMax_le _ _ _ (fun m hm => ?_) (by split <;> (try split) <;> omega)
  have hm' : m = MDPos.LF := by simpa using hm
  subst hm'
  have hcap : capClamp MDG MDPos.A0 (3 + 1) MDPos.LF
      (-(fuelValueD2C MDG g 3 mdSpend 3 MDPos.LF))
      ≤ shallowMoveCap (MDG.eval MDPos.A0) (MDG.val MDPos.A0 MDPos.LF) (3 + 1) := by
    unfold capClamp
    rw [if_pos (And.intro (by omega) (by decide))]
    exact Int.min_le_left _ _
  have hval : shallowMoveCap (MDG.eval MDPos.A0) (MDG.val MDPos.A0 MDPos.LF) (3 + 1) = 540 := by
    show shallowMoveCap (0 : Int) 0 4 = 540
    unfold shallowMoveCap QS_A
    omega
  omega

/-- The defender node above it keeps its escape, one ply of depth cheaper
than in the cap-free model. -/
theorem sharp_cap_D1_7 (g : MDPos → Bool) : -540 ≤ fuelValueD2C MDG g 3 mdSpend 7 MDPos.D1 := by
  have hma : movesAbove MDG (val_lower 7) MDPos.D1 = [MDPos.A0] :=
    movesAbove_all MDG 7 MDPos.D1 (by decide)
  have hA0 := sharp_cap_A0_4 g
  rw [show (7 : Nat) = 6 + 1 from rfl,
    fuelValueD2C_of_fold_regime MDG g 3 mdSpend 6 MDPos.D1
      (by decide) (by decide) (by decide) (by omega),
    hma]
  have hcl := capClamp_of_deep MDG MDPos.D1 (d := 6 + 1) (by omega) MDPos.A0
    (-(fuelValueD2C MDG g 3 mdSpend (6 - min (3 - 1) (mdSpend MDPos.D1 (6 + 1) MDPos.A0))
      MDPos.A0))
  have hfold : capClamp MDG MDPos.D1 (6 + 1) MDPos.A0
        (-(fuelValueD2C MDG g 3 mdSpend
          (6 - min (3 - 1) (mdSpend MDPos.D1 (6 + 1) MDPos.A0)) MDPos.A0))
      ≤ foldMax (fun x => capClamp MDG MDPos.D1 (6 + 1) x
          (-(fuelValueD2C MDG g 3 mdSpend
            (6 - min (3 - 1) (mdSpend MDPos.D1 (6 + 1) x)) x)))
          [MDPos.A0] LOSS :=
    foldMax_le_of_mem _ _ _ _ (List.mem_cons_self _ _)
  rw [hcl] at hfold
  have hA0' : fuelValueD2C MDG g 3 mdSpend
      (6 - min (3 - 1) (mdSpend MDPos.D1 (6 + 1) MDPos.A0)) MDPos.A0 ≤ 540 := hA0
  omega

/-- **The cap's ply, certified**: `ForcedMate MDG 3 A1` holds, and at
`D = 10` the declared value of the shipped-shaped search is at most 540 --
far below `MATE_LOWER`. So `3k + 1` does NOT license the CI depth;
`3k + 2` is sharp. -/
theorem sharp_cap_mate3_at_10 (g : MDPos → Bool) :
    fuelValueD2C MDG g 3 mdSpend 10 MDPos.A1 < MATE_LOWER := by
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  have hma : movesAbove MDG (val_lower 10) MDPos.A1 = [MDPos.D1] :=
    movesAbove_all MDG 10 MDPos.A1 (by decide)
  have hD1 := sharp_cap_D1_7 g
  rw [show (10 : Nat) = 9 + 1 from rfl,
    fuelValueD2C_of_fold_regime MDG g 3 mdSpend 9 MDPos.A1
      (by decide) (by decide) (by decide) (by omega),
    hma]
  refine Int.lt_of_le_of_lt (foldMax_le _ _ _ (fun m hm => ?_) (by omega)) (by omega : (540:Int) < MATE_LOWER)
  have hm' : m = MDPos.D1 := by simpa using hm
  subst hm'
  have hcl := capClamp_of_deep MDG MDPos.A1 (d := 9 + 1) (by omega) MDPos.D1
    (-(fuelValueD2C MDG g 3 mdSpend (9 - min (3 - 1) (mdSpend MDPos.A1 (9 + 1) MDPos.D1))
      MDPos.D1))
  rw [hcl]
  simp only [mdSpend, show (9 - min (3 - 1) 2 : Nat) = 7 from rfl]
  omega

theorem code_mate_depth_bound_sharp_k3 :
    ∃ (G : QSGame) (guard : G.Pos → Bool) (spend : G.Pos → Nat → G.Pos → Nat) (p : G.Pos),
      ValFloor G 192 ∧ ForcedMate G 3 p ∧
        fuelValueD2C G guard 3 spend (3 * 3 + 1) p < MATE_LOWER :=
  ⟨MDG, mdGuard, mdSpend, MDPos.A1, sharp_valFloor, sharp_forcedMate_3_A1,
    sharp_cap_mate3_at_10 mdGuard⟩

/-- **Menu option M1 alone buys nothing**: with the shallow cap in place, the
witness masks at the CAPPED attacker node, not at the pass -- so deleting the
sub-horizon pass (`guard ≡ false`) leaves `D = 9` short of the mate at `k = 3`
just the same.  The two mechanisms have to go together to move the bound. -/
theorem code_mate_depth_bound_sharp_k3_guardOff :
    ∃ (G : QSGame) (guard : G.Pos → Bool) (spend : G.Pos → Nat → G.Pos → Nat) (p : G.Pos),
      ValFloor G 192 ∧ (∀ q, guard q = false) ∧ ForcedMate G 3 p ∧
        fuelValueD2C G guard 3 spend (3 * 3 + 1) p < MATE_LOWER :=
  ⟨MDG, fun _ => false, mdSpend, MDPos.A1, sharp_valFloor, fun _ => rfl,
    sharp_forcedMate_3_A1, sharp_cap_mate3_at_10 (fun _ => false)⟩

end Sunfish
