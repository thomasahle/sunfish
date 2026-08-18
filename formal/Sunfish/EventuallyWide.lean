/-
Finite pruning debt / eventual widening.

THE CONTRACT (Thomas's design): a heuristic may postpone a real branch
boundedly (reductions whose child depths tend to infinity: fine
forever), or omit it for finitely many remaining depths; only exact
bounds may suppress a branch forever.  "Eventual exhaustiveness" is the
conjunction:

  (i)   every real move is eventually admitted at every node;
  (ii)  child depths tend to infinity;
  (iii) every virtual candidate (the null option) is eventually LOSS
        (the fold identity) or exactly dominated.

CENTERPIECE -- NULL AS A FUEL ORACLE (Thomas's plan, superseding the
horizon-credit design below as the primary target).  Below depth 6 the
capped null stays a score candidate, verbatim.  From depth 6 on the
pass is NEVER in the max: one probe at the fixed target
`pos.score + NULL_MARGIN` -- a window keyed by `(pos, depth)` alone --
decides whether the real moves recurse at `depth - 2` (hot) or
`depth - 1`.  Admission (`val_lower`) and the tables stay keyed by
NOMINAL depth; only the recursion is shortened.  So above the horizon
the declared value is a fold over REAL MOVES ONLY, with the existing
terminal finalizer, and every real edge consumes between 1 and C
units of depth (C = 2 shipped).  Consequences proven here:

  * `WindowReport.side_exact` / `hot_bit_determined` / `hot_bit_stable`:
    a fail-soft report is side-exact at any fixed window, so the hot
    bit is `(pos, depth)`-determined REGARDLESS of table state -- the
    probe only ever selects between two structurally recursive folds.
  * `fuelValueD2`: the declared value, general in the edge-cost
    selector (`spend : Pos → Nat → Pos → Nat`, clamped so each edge costs
    `1 .. C`) -- Thomas's statement is heuristic-independent, so the
    theorem quantifies over ALL selectors and instantiates H = 6,
    C = 2 (`hotSpend`, `hotSpend_child_depth` pins the code shape).
  * `forcedMate_fuelValueD2`: mate-in-k completeness at `D ≥ C·k + 4`
    (so `D ≥ 2k + 4` as shipped), from `ValFloor` -- fidelity, tables
    -- and NOTHING else.  `NoZugzwang` and every mate-band-agreement
    premise appear NOWHERE in this chain: every proof-tree node stays
    in the real-only regime `d ≥ 6`, the terminal child is classified
    exactly by the finalizer at positive depth, and defender folds
    start from `LOSS` with no pass term to displace them.
    `forcedlyMated_fuelValueD2` is the dual; the
    `finite_mates_eventually_recognized` wrapper is the contract's
    payoff sentence: every finite forced-mate proof is eventually
    recognized, with no assumption on the edge-cost selector.
  * Layer 1 is STATED (`FuelBracketSpec`), not yet proven: the bracket
    of `bound()` against `fuelValueD2` needs the `boundD2` mirror.
    The enabling step -- the probe's gamma-independence -- is
    `hot_bit_stable` above; the rest is the mechanical re-run of
    `Stalemate.lean`'s layer-1 against the shaped fold, recorded as
    the follow-up, not silently assumed.

END STATE (the goal, for the record): the fuel oracle retires the
deep-null zugzwang debt (this file), and the FRONTIER TAIL
(`Classification.lean` Part B, the fttail arm) retires
`NoMaskedMobility` on the honesty side.  With BOTH landed, eventual
classification carries fidelity premises (`ValFloor`, `EvalQuiet`,
tables) only -- no chess assumption anywhere in the trichotomy.

DEMOTED TO EXPERIMENTAL CONTROLS -- the horizon-credit design (the
second half of this file): a null guard with a FINITE HORIZON `H`
(`guard p d = false` for `d ≥ H`) also makes completeness
unconditional, at `D ≥ k + H + O(1)`: the `ForcedMate` induction never
leaves the depth window `[D - k, D]`, where the horizon makes every
pass term the fold identity.  The credit variants stay expressible
(`guard : Pos → Nat → Bool`), `nullValueD2G` generalizes `nullValueD2`
conservatively (`nullValueD2G_depthBlind`), and `NoZugzwangG` restates
the layer-2 premise over the depth-keyed guard with the accuracy
transfer and `D ≥ k + 1` completeness mirrored.  The three demoted
code arms:

  * arm A (control)      `2 < depth < 48`            -- `fixedHorizonGuard`,   horizon 48;
  * arm B (smooth)       `|score| + depth < 500`     -- `smoothCreditGuard`,   horizon 500;
  * arm C (phase)        `2 < depth < 12 * pieces`   -- `phaseAdaptiveGuard`,  horizon 180
                                                        (own non-pawn pieces ≤ 15).

The position-dependent parts (`base`, `score`, `phase`, `hot`) stay
abstract, exactly as the shipped model keeps `guard` abstract: only
the depth-dependence is load-bearing.

Zero sorries, no Mathlib, no audit-surface changes (the audited model
files are untouched; this file only adds definitions and theorems).
-/

import Sunfish.Stalemate
import Sunfish.Liveness
import Sunfish.CappedNull
import Sunfish.Classification

namespace Sunfish

/-! # Part I: null as a fuel oracle

## The probe is position-determined -/

/-- **Side-exactness of fail-soft reports**: any report valid at window
`gamma` sits on the same side of `gamma` as the value it brackets.  The
whole reason a fixed-target probe may STEER (rather than score): two
different valid reports -- say from different table states -- can
disagree on magnitude but never on the side. -/
theorem WindowReport.side_exact {gamma r v : Int}
    (h : WindowReport gamma r v) : gamma ≤ r ↔ gamma ≤ v := by
  rcases h with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> constructor <;> intro <;> omega

/-- The hot bit through the zero-window convention: the probe runs at
`1 - target` and is negated, exactly as `Searcher.bound` does with the
pass; composing `WindowReport.negate` with side-exactness, the bit
`target ≤ -report` equals `target ≤ -value` -- a `(pos, depth)`
predicate. -/
theorem hot_bit_determined {target rp vp : Int}
    (h : WindowReport (1 - target) rp vp) :
    (target ≤ -rp) ↔ (target ≤ -vp) :=
  (WindowReport.negate h).side_exact

/-- Two valid probe reports -- however the tables differ between them --
always agree on the hot bit. -/
theorem hot_bit_stable {target r1 r2 v : Int}
    (h1 : WindowReport (1 - target) r1 v)
    (h2 : WindowReport (1 - target) r2 v) :
    (target ≤ -r1) ↔ (target ≤ -r2) :=
  (hot_bit_determined h1).trans (hot_bit_determined h2).symm

/-! ## The fuel-shaped declared value -/

/-- **The fuel-shaped declared value.**  Below depth 6, verbatim
`nullValueD2` (the capped pass as a score candidate, sub-band admitted
and in-band suppressed).  From depth 6 on, a fold over REAL MOVES ONLY
-- initial accumulator `LOSS`, no pass term -- where each edge consumes
`1 + min (C-1) (spend p depth child)` plies: between 1 and `C`, for ANY
selector `spend` (the clamp bakes Thomas's "between 1 and C units"
into the definition, so the theorems are heuristic-independent).
Admission stays keyed by NOMINAL depth (`val_lower (d+1)`), matching
the decided code shape; only the recursion is shortened.  King-capture
normalization, exact sentinel, and the verified terminal finalizer are
unchanged at every depth.  `(pos, depth)`-determined and window-free
(`hot_bit_determined` is what lets the code compute `spend` with a
probe). -/
def fuelValueD2 (G : QSGame) (guard : G.Pos → Bool) (C : Nat)
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
      foldMax (fun m => -(fuelValueD2 G guard C spend d m))
        (movesAbove G (val_lower (d + 1)) p)
        (if guard p = true ∧ 2 < d + 1 then
          (if -(fuelValueD2 G guard C spend (d + 1 - 3) (G.pass p)) < MATE_LOWER then
            max LOSS (-(fuelValueD2 G guard C spend (d + 1 - 3) (G.pass p)))
          else LOSS)
        else LOSS)
    else
      foldMax (fun m => -(fuelValueD2 G guard C spend
          (d - min (C - 1) (spend p (d + 1) m)) m))
        (movesAbove G (val_lower (d + 1)) p) LOSS
termination_by d _ => d
decreasing_by all_goals omega

/-- Every regime edge consumes between 1 and `C` plies -- clause (ii)
of the contract, for any selector. -/
theorem fuel_edge_cost (C : Nat) (hC : 1 ≤ C) (s d : Nat) :
    d + 1 - C ≤ d - min (C - 1) s ∧ d - min (C - 1) s ≤ d := by
  omega

/-! ### Branch lemmas -/

theorem fuelValueD2_kingGone (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos) (h : G.eval p ≤ -MATE_LOWER) :
    fuelValueD2 G guard C spend d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [fuelValueD2]; rw [if_pos h]
  | succ d => simp only [fuelValueD2]; rw [if_pos h]

theorem fuelValueD2_of_capture (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos) (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    fuelValueD2 G guard C spend d p = MATE_UPPER := by
  cases d with
  | zero => simp only [fuelValueD2]; rw [if_neg hkg, if_pos hcap]
  | succ d => simp only [fuelValueD2]; rw [if_neg hkg, if_pos hcap]

theorem fuelValueD2_of_allIllegal (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true) :
    fuelValueD2 G guard C spend (d + 1) p = terminalValue G (d + 1) p := by
  simp only [fuelValueD2]
  rw [if_neg hkg, if_neg hcap, if_pos hai]

/-- The regime fold: at `d + 1 ≥ 8`, real moves only, `LOSS` init. -/
theorem fuelValueD2_of_fold_regime (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hd : 5 ≤ d) :
    fuelValueD2 G guard C spend (d + 1) p
      = foldMax (fun m => -(fuelValueD2 G guard C spend
          (d - min (C - 1) (spend p (d + 1) m)) m))
          (movesAbove G (val_lower (d + 1)) p) LOSS := by
  simp only [fuelValueD2]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_neg (by omega)]

/-- The sub-horizon fold: below depth 6, the shipped capped-null shape,
verbatim (children at `d`, the pass at `d + 1 - 3`). -/
theorem fuelValueD2_of_fold_sub (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hd : d < 5) :
    fuelValueD2 G guard C spend (d + 1) p
      = foldMax (fun m => -(fuelValueD2 G guard C spend d m))
          (movesAbove G (val_lower (d + 1)) p)
          (if guard p = true ∧ 2 < d + 1 then
            (if -(fuelValueD2 G guard C spend (d + 1 - 3) (G.pass p)) < MATE_LOWER then
              max LOSS (-(fuelValueD2 G guard C spend (d + 1 - 3) (G.pass p)))
            else LOSS)
          else LOSS) := by
  simp only [fuelValueD2]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_pos (by omega)]

/-- A checkmated child is classified exactly by the finalizer at ANY
positive depth (regime-independent: the terminal branch precedes the
regime split). -/
theorem fuelValueD2_checkmated (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) {m : G.Pos}
    (hcap : hasKingCapture G.toNullGame.toGame m = false)
    (hmate : Checkmated G m) :
    ∀ d : Nat, 1 ≤ d → fuelValueD2 G guard C spend d m ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro d hd
  cases d with
  | zero => omega
  | succ d' =>
    by_cases hkgm : G.eval m ≤ -MATE_LOWER
    · rw [fuelValueD2_kingGone G guard C spend (d' + 1) m hkgm]; omega
    · rw [fuelValueD2_of_allIllegal G guard C spend d' m hkgm (by simp [hcap]) hmate.1]
      have := terminalValue_mate G (d' + 1) m hmate.2
      omega

/-! ## Mate-in-k completeness with NO chess premise -/

/-- **Mate-in-k completeness for the fuel shape**: `ValFloor`
(fidelity, tables) alone puts the declared value in the mate band at
every `D ≥ C·k + 4`, for EVERY edge-cost selector.  With `C ≥ 2`,
every interior proof node sits at depth `≥ 6` (the real-only regime,
`LOSS`-seeded folds on the defender side), and the checkmated leaf at
depth `≥ 4` (the finalizer's exact `-MATE_LOWER`).  `NoZugzwang` and
mate-band agreement appear nowhere. -/
theorem forcedMate_fuelValueD2 (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, C * k + 4 ≤ D → MATE_LOWER ≤ fuelValueD2 G guard C spend D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  induction hFM with
  | @mate k p m hkg hm hleg hmate =>
    intro D hD
    have hexp : C * (k + 1) = C * k + C * 1 := Nat.mul_add C k 1
    cases D with
    | zero => omega
    | succ d =>
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [fuelValueD2_of_capture G guard C spend (d + 1) p hkg hcap]; omega
      · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
        rw [fuelValueD2_of_fold_regime G guard C spend d p hkg hcap hai (by omega)]
        have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
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
  | @step k p m hkg hm hleg hnt _hreply ih =>
    intro D hD
    have hexp : C * (k + 2) = C * k + C * 2 := Nat.mul_add C k 2
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
              rw [fuelValueD2_of_capture G guard C spend _ m' hkgm' hcm]; omega
            | false =>
              have := ih m' hm'' hcm
                (dm - min (C - 1) (spend m (dm + 1) m')) (by omega)
              omega
        have hfold : -(fuelValueD2 G guard C spend
              (d - min (C - 1) (spend p (d + 1) m)) m)
            ≤ foldMax (fun x => -(fuelValueD2 G guard C spend
                  (d - min (C - 1) (spend p (d + 1) x)) x))
                (movesAbove G (val_lower (d + 1)) p) LOSS :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega

/-- The dual: the mated side's fuel value sits at or below
`-MATE_LOWER` at every `D ≥ C·k + C + 4` (the top node is a defender
node, so its own fold must be regime-seeded too). -/
theorem forcedlyMated_fuelValueD2 (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192)
    {k : Nat} {q : G.Pos}
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hFL : ForcedlyMated G k q) :
    ∀ D : Nat, C * k + C + 4 ≤ D → fuelValueD2 G guard C spend D q ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro D hD
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
          have := forcedMate_fuelValueD2 G guard C spend hC hF
            (hall m hm' hcm) (d - min (C - 1) (spend q (d + 1) m)) (by omega)
          omega

/-- **The contract's payoff sentence**: every finite forced-mate proof
is eventually recognized -- a depth bound exists for every `k` -- with
no assumption on the edge-cost selector and no chess premise. -/
theorem finite_mates_eventually_recognized (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192) :
    ∀ k : Nat, ∃ Dk : Nat, ∀ p : G.Pos, ForcedMate G k p →
      ∀ D : Nat, Dk ≤ D → MATE_LOWER ≤ fuelValueD2 G guard C spend D p :=
  fun k => ⟨C * k + 4, fun _ hFM D hD =>
    forcedMate_fuelValueD2 G guard C spend hC hF hFM D hD⟩

/-! ## The shipped instantiation: H = 6, C = 2 -/

/-- The code's selector: the probe's hot bit spends the one extra ply. -/
def hotSpend (G : QSGame) (hot : G.Pos → Nat → Bool) : G.Pos → Nat → G.Pos → Nat :=
  fun p d _ => if hot p d then 1 else 0

/-- With `C = 2` the clamped edge cost is exactly the code's
`depth - (2 if hot else 1)` recursion. -/
theorem hotSpend_child_depth (G : QSGame) (hot : G.Pos → Nat → Bool)
    (p m : G.Pos) (d : Nat) :
    d - min (2 - 1) (hotSpend G hot p (d + 1) m)
      = d - (if hot p (d + 1) then 1 else 0) := by
  cases h : hot p (d + 1) <;> simp [hotSpend, h]

/-- Completeness as shipped: `D ≥ 2k + 4`, `ValFloor` only. -/
theorem forcedMate_fuelValueD2_code (G : QSGame) (guard : G.Pos → Bool)
    (hot : G.Pos → Nat → Bool) (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, 2 * k + 4 ≤ D →
      MATE_LOWER ≤ fuelValueD2 G guard 2 (hotSpend G hot) D p :=
  fun D hD => forcedMate_fuelValueD2 G guard 2 (hotSpend G hot)
    (by omega) hF hFM D (by omega)

/-! ## Layer 1: stated

The remaining obligation, as a named `Prop` so the statement is
reviewable now: the search with the fuel probe brackets `fuelValueD2`
at every depth and every driver-range window, with no chess premise --
the probe only selects between two structurally recursive folds, and
`hot_bit_stable` already shows the selection is independent of table
state.  The proof is the `boundD2` mirror (searchMoves specs, killer
verification, futility coverage, terminal correction) re-run against
the shaped fold: mechanical, sizable, and NOT assumed anywhere in this
file -- every theorem above is about the declared value itself. -/
def FuelBracketSpec (G : QSGame) (guard : G.Pos → Bool)
    (hot : G.Pos → Nat → Bool) (search : Nat → G.Pos → Int → Int) : Prop :=
  ∀ (d : Nat) (p : G.Pos) (gamma : Int),
    -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
    (gamma ≤ search d p gamma →
      search d p gamma ≤ fuelValueD2 G guard 2 (hotSpend G hot) d p) ∧
    (search d p gamma < gamma →
      fuelValueD2 G guard 2 (hotSpend G hot) d p ≤ search d p gamma)

/-! # Part II: the horizon-credit design (demoted to controls) -/

/-! ## The depth-keyed declared function -/

/-- `nullValueD2` with a DEPTH-KEYED guard: the pass option exists when
`guard p (d+1)` holds (the node's remaining depth, matching the code's
`(pos, depth)` keying) and `2 < d + 1`.  Everything else is verbatim
`nullValueD2`: king-capture normalization, exact sentinel, verified
terminal value, filtered fold whose initial accumulator is the pass
term with the sub-band admission and in-band suppression baked in.
Still `(pos, depth)`-determined and window-free. -/
def nullValueD2G (G : QSGame) (guard : G.Pos → Nat → Bool) : Nat → G.Pos → Int
  | 0, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else G.eval p
  | d + 1, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else if allIllegalB G p = true then terminalValue G (d + 1) p
    else
      foldMax (fun m => -(nullValueD2G G guard d m)) (movesAbove G (val_lower (d + 1)) p)
        (if guard p (d + 1) = true ∧ 2 < d + 1 then
          (if -(nullValueD2G G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER then
            max LOSS (-(nullValueD2G G guard (d + 1 - 3) (G.pass p)))
          else LOSS)
        else LOSS)
termination_by d _ => d
decreasing_by all_goals omega

/-- The depth-keyed pass term (the fold's initial accumulator), named. -/
def nullTermD2G (G : QSGame) (guard : G.Pos → Nat → Bool) (d : Nat) (p : G.Pos) : Int :=
  if guard p (d + 1) = true ∧ 2 < d + 1 then
    (if -(nullValueD2G G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER then
      max LOSS (-(nullValueD2G G guard (d + 1 - 3) (G.pass p)))
    else LOSS)
  else LOSS

/-! ### Branch lemmas (mirroring `nullValueD2`'s) -/

theorem nullValueD2G_kingGone (G : QSGame) (guard : G.Pos → Nat → Bool)
    (d : Nat) (p : G.Pos) (h : G.eval p ≤ -MATE_LOWER) :
    nullValueD2G G guard d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [nullValueD2G]; rw [if_pos h]
  | succ d => simp only [nullValueD2G]; rw [if_pos h]

theorem nullValueD2G_of_capture (G : QSGame) (guard : G.Pos → Nat → Bool)
    (d : Nat) (p : G.Pos) (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    nullValueD2G G guard d p = MATE_UPPER := by
  cases d with
  | zero => simp only [nullValueD2G]; rw [if_neg hkg, if_pos hcap]
  | succ d => simp only [nullValueD2G]; rw [if_neg hkg, if_pos hcap]

theorem nullValueD2G_of_allIllegal (G : QSGame) (guard : G.Pos → Nat → Bool)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true) :
    nullValueD2G G guard (d + 1) p = terminalValue G (d + 1) p := by
  simp only [nullValueD2G]
  rw [if_neg hkg, if_neg hcap, if_pos hai]

theorem nullValueD2G_of_fold (G : QSGame) (guard : G.Pos → Nat → Bool)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false) :
    nullValueD2G G guard (d + 1) p
      = foldMax (fun m => -(nullValueD2G G guard d m))
          (movesAbove G (val_lower (d + 1)) p) (nullTermD2G G guard d p) := by
  simp only [nullValueD2G]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai])]
  rfl

theorem nullTermD2G_ge_LOSS (G : QSGame) (guard : G.Pos → Nat → Bool)
    (d : Nat) (p : G.Pos) : LOSS ≤ nullTermD2G G guard d p := by
  simp only [nullTermD2G]
  by_cases h1 : guard p (d + 1) = true ∧ 2 < d + 1
  · rw [if_pos h1]
    by_cases h2 : -(nullValueD2G G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
    · rw [if_pos h2]; omega
    · rw [if_neg h2]; omega
  · rw [if_neg h1]; omega

/-- **Conservativity**: the shipped declared function is the depth-blind
instance -- a depth-ignoring guard reproduces `nullValueD2` pointwise.
The generalization costs nothing. -/
theorem nullValueD2G_depthBlind (G : QSGame) (guard : G.Pos → Bool) :
    ∀ (d : Nat) (p : G.Pos),
      nullValueD2G G (fun q _ => guard q) d p = nullValueD2 G guard d p := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p
    cases d with
    | zero => simp only [nullValueD2G, nullValueD2]
    | succ d =>
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [nullValueD2G_kingGone G _ (d + 1) p hkg,
          nullValueD2_kingGone G guard (d + 1) p hkg]
      · by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [nullValueD2G_of_capture G _ (d + 1) p hkg hcap,
            nullValueD2_of_capture G guard (d + 1) p hkg hcap]
        · cases hai : allIllegalB G p with
          | true =>
            rw [nullValueD2G_of_allIllegal G _ d p hkg hcap hai,
              nullValueD2_of_allIllegal G guard d p hkg hcap hai]
          | false =>
            rw [nullValueD2G_of_fold G _ d p hkg hcap hai,
              nullValueD2_of_fold G guard d p hkg hcap hai]
            have hterm : nullTermD2G G (fun q _ => guard q) d p
                = nullTermD2 G guard d p := by
              simp only [nullTermD2G, nullTermD2]
              rw [ih (d + 1 - 3) (by omega) (G.pass p)]
            rw [hterm]
            exact foldMax_congr _ _ _ _ (fun m _ => by
              show -(nullValueD2G G (fun q _ => guard q) d m) = -(nullValueD2 G guard d m)
              rw [ih d (by omega) m])

/-! ## The contract, formalized -/

/-- **(i) Every real move is eventually admitted, unconditionally**: the
val-filter is a bounded postponement, never a permanent omission.  This
used to be a limit statement about the sloped threshold `QS - d * QS_A`
decreasing without bound; since `c01915f` the postponement is at most
ONE ply, and the bound needs no hypothesis about the move's value at
all -- above the frontier the threshold is the bottom of the band.
Kept in the limit form the contract states it in, with `N = 1`.

The one thing the change COSTS: the old statement was unconditional --
a sloped threshold eventually clears any value whatsoever -- and this
one is not, because a flat threshold clears only what is above it.  The
hypothesis it needs is `ValFloor`, i.e. that move values live inside
the band at all, which every theorem in this file already assumes and
which the shipped tables satisfy with -192 against -69290.  An
unconditional reading would have to contemplate a move valued below the
reserved sentinel, which is not a move value but a token collision. -/
theorem every_move_eventually_admitted (G : QSGame) {B : Int}
    (hF : ValFloor G B) (hB : B ≤ MATE_UPPER) (p m : G.Pos)
    (hm : m ∈ G.moves p) :
    ∃ N : Nat, ∀ d : Nat, N ≤ d → m ∈ movesAbove G (val_lower d) p := by
  refine ⟨1, fun d hd => ?_⟩
  rw [mem_movesAbove]
  refine ⟨hm, ?_⟩
  rw [val_lower_pos d (by omega)]
  have := hF p m hm
  omega

/-- The floor makes the admission bound uniform, and after `c01915f` it
is uniform at the FRONTIER'S SUCCESSOR: from remaining depth 1 on,
nothing is filtered (the Liveness respend of
`tables_kill_filter_at_depth2`, cited through
`mem_movesAbove_of_floor`). -/
theorem admitted_uniformly_of_floor (G : QSGame) (hF : ValFloor G 192) :
    ∀ (d : Nat), 1 ≤ d → ∀ (p m : G.Pos), m ∈ G.moves p →
      m ∈ movesAbove G (val_lower d) p :=
  fun _ hd _ _ hm => mem_movesAbove_of_floor G hF hd hm

/-- **(iii)'s witness shape -- a finite null horizon**: the guard is off
at every remaining depth `≥ H`.  Arm A is `H = 48`, arm B `H = 500`,
arm C `H = 180`; the shipped guard has NO horizon (it is depth-blind),
which is exactly why the Liveness transfer needs `NoZugzwang`. -/
def Horizon (G : QSGame) (guard : G.Pos → Nat → Bool) (H : Nat) : Prop :=
  ∀ (p : G.Pos) (d : Nat), H ≤ d → guard p d = false

/-- Above the horizon the pass term is the fold identity: the virtual
candidate has resolved to LOSS. -/
theorem nullTermD2G_eq_LOSS_of_horizon (G : QSGame) (guard : G.Pos → Nat → Bool)
    {H : Nat} (hH : Horizon G guard H) (d : Nat) (p : G.Pos)
    (hd : H ≤ d + 1) :
    nullTermD2G G guard d p = LOSS := by
  have hg : guard p (d + 1) = false := hH p (d + 1) hd
  simp only [nullTermD2G, hg]
  rw [if_neg (fun h => Bool.noConfusion h.1)]

/-- **Eventual exhaustiveness** (the contract as one Prop):
(i) every real move eventually admitted; (ii) real-move child depths
tend to infinity (`d - 1`; the pass child `d - 3` grows the same way);
(iii) every virtual candidate eventually LOSS.  The in-band suppression
(`-(pass) ≥ MATE_LOWER → LOSS`, baked into `nullTermD2G`) is the "exactly
dominated" half of (iii): inside the mate band the virtual candidate is
never admitted at any depth. -/
structure EventuallyExhaustive (G : QSGame) (guard : G.Pos → Nat → Bool) : Prop where
  admitted : ∀ (p m : G.Pos), m ∈ G.moves p →
    ∃ N : Nat, ∀ d : Nat, N ≤ d → m ∈ movesAbove G (val_lower d) p
  childDepthsGrow : ∀ N : Nat, ∃ D : Nat, ∀ d : Nat, D ≤ d → N ≤ d - 1
  virtualsResolve : ∃ Hv : Nat, ∀ (d : Nat) (p : G.Pos),
    Hv ≤ d + 1 → nullTermD2G G guard d p = LOSS

/-- Any finite-horizon guard satisfies the contract, given the fidelity
floor that arm (i) now needs. -/
theorem eventuallyExhaustive_of_horizon (G : QSGame) (guard : G.Pos → Nat → Bool)
    {H : Nat} (hH : Horizon G guard H) (hF : ValFloor G 192) :
    EventuallyExhaustive G guard :=
  ⟨every_move_eventually_admitted G hF (by decide),
   fun N => ⟨N + 1, fun _ hd => by omega⟩,
   ⟨H, fun d p hd => nullTermD2G_eq_LOSS_of_horizon G guard hH d p hd⟩⟩

/-! ## The payoff: completeness with the horizon in place of the chess
premise

The induction never leaves the depth window `[D - k, D]`: attacker and
defender nodes of a mate-in-k line at root depth `D ≥ k + H + 1` all
sit at remaining depth `≥ H`, where `nullTermD2G` is LOSS.  Defender
folds therefore start from the same `LOSS` as the real-move spine;
attacker folds never read the accumulator (`foldMax_le_of_mem`).
Below the window the induction only meets terminal branches
(checkmated leaves, sentinel refutations), which carry no pass term. -/

/-- **Mate-in-k completeness, horizon form**: `ValFloor` (fidelity,
tables) and a finite null horizon -- NO chess premise -- put the
declared value in the mate band at every `D ≥ k + H + 1`. -/
theorem forcedMate_nullValueD2G_of_horizon (G : QSGame)
    (guard : G.Pos → Nat → Bool) {H : Nat}
    (hF : ValFloor G 192) (hH : Horizon G guard H)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, k + H + 1 ≤ D → MATE_LOWER ≤ nullValueD2G G guard D p := by
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
      · rw [nullValueD2G_of_capture G guard (d + 1) p hkg hcap]; omega
      · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
        rw [nullValueD2G_of_fold G guard d p hkg hcap hai]
        have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
        have hchild : nullValueD2G G guard d m ≤ -MATE_LOWER := by
          cases d with
          | zero => omega
          | succ d' =>
            by_cases hkgm : G.eval m ≤ -MATE_LOWER
            · rw [nullValueD2G_kingGone G guard (d' + 1) m hkgm]; omega
            · rw [nullValueD2G_of_allIllegal G guard d' m hkgm (by simp [hleg]) hmate.1]
              have := terminalValue_mate G (d' + 1) m hmate.2
              omega
        have hfold : -(nullValueD2G G guard d m)
            ≤ foldMax (fun x => -(nullValueD2G G guard d x))
                (movesAbove G (val_lower (d + 1)) p) (nullTermD2G G guard d p) :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega
  | @step k p m hkg hm hleg hnt _hreply ih =>
    intro D hD
    cases D with
    | zero => omega
    | succ d =>
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [nullValueD2G_of_capture G guard (d + 1) p hkg hcap]; omega
      · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
        rw [nullValueD2G_of_fold G guard d p hkg hcap hai]
        have hmem := mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
        have hchild : nullValueD2G G guard d m ≤ -MATE_LOWER := by
          cases d with
          | zero => omega
          | succ d' =>
            by_cases hkgm : G.eval m ≤ -MATE_LOWER
            · rw [nullValueD2G_kingGone G guard (d' + 1) m hkgm]; omega
            · rw [nullValueD2G_of_fold G guard d' m hkgm (by simp [hleg]) hnt]
              rw [nullTermD2G_eq_LOSS_of_horizon G guard hH d' m (by omega)]
              refine foldMax_le _ _ _ (fun m' hm' => ?_) (by omega)
              show -(nullValueD2G G guard d' m') ≤ -MATE_LOWER
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
                rw [nullValueD2G_of_capture G guard d' m' hkgm' hcm]; omega
              | false =>
                have := ih m' hm'' hcm d' (by omega)
                omega
        have hfold : -(nullValueD2G G guard d m)
            ≤ foldMax (fun x => -(nullValueD2G G guard d x))
                (movesAbove G (val_lower (d + 1)) p) (nullTermD2G G guard d p) :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega

/-- The mated side, horizon form: the defender's declared value sits at
or below `-MATE_LOWER` at every `D ≥ k + H + 2` -- the top node is a
defender node, so its own pass term must be off too (hence one more
ply of headroom than the mate side). -/
theorem forcedlyMated_nullValueD2G_of_horizon (G : QSGame)
    (guard : G.Pos → Nat → Bool) {H : Nat}
    (hF : ValFloor G 192) (hH : Horizon G guard H)
    {k : Nat} {q : G.Pos}
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hFL : ForcedlyMated G k q) :
    ∀ D : Nat, k + H + 2 ≤ D → nullValueD2G G guard D q ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro D hD
  cases D with
  | zero => omega
  | succ d =>
    by_cases hkg : G.eval q ≤ -MATE_LOWER
    · rw [nullValueD2G_kingGone G guard (d + 1) q hkg]; omega
    · have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by
        simp [hcapq]
      cases hFL with
      | inl hcm =>
        rw [nullValueD2G_of_allIllegal G guard d q hkg hcapq' hcm.1]
        have := terminalValue_mate G (d + 1) q hcm.2
        omega
      | inr h =>
        obtain ⟨hai, hall⟩ := h
        rw [nullValueD2G_of_fold G guard d q hkg hcapq' hai]
        rw [nullTermD2G_eq_LOSS_of_horizon G guard hH d q (by omega)]
        refine foldMax_le _ _ _ (fun m hm => ?_) (by omega)
        show -(nullValueD2G G guard d m) ≤ -MATE_LOWER
        have hm' : m ∈ G.moves q := movesAbove_subset G _ q m hm
        cases hcm : hasKingCapture G.toNullGame.toGame m with
        | true =>
          have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := by
            intro hle
            have hc : hasKingCapture G.toNullGame.toGame q = true :=
              (hasKingCapture_iff G.toNullGame.toGame q).mpr ⟨m, hm', hle⟩
            rw [hcapq] at hc
            exact Bool.noConfusion hc
          rw [nullValueD2G_of_capture G guard d m hkgm hcm]; omega
        | false =>
          have := forcedMate_nullValueD2G_of_horizon G guard hF hH
            (hall m hm' hcm) d (by omega)
          omega

/-! ## `NoZugzwang`, restated over the depth-keyed guard

The same statement, aimed at the depth-keyed functions, so the credit
variants can also be reasoned about INSIDE their null region (below the
horizon), where the horizon theorem says nothing.  The accuracy
transfer and the `D ≥ k + 1` completeness go through verbatim. -/

/-- `NoZugzwang` over `guard : Pos → Nat → Bool`: at every node where
the depth-keyed pass option exists, the raw pass term never strictly
beats the real-move fold. -/
def NoZugzwangG (G : QSGame) (guard : G.Pos → Nat → Bool) : Prop :=
  ∀ (d : Nat) (p : G.Pos),
    ¬ (G.eval p ≤ -MATE_LOWER) →
    ¬ (hasKingCapture G.toNullGame.toGame p = true) →
    allIllegalB G p = false → guard p (d + 1) = true → 2 < d + 1 →
    -(nullValueD2G G guard (d + 1 - 3) (G.pass p))
      ≤ foldMax (fun m => -(nullValueD2G G guard d m))
          (movesAbove G (val_lower (d + 1)) p) LOSS

/-- The shipped premise implies the depth-blind instance of the new
one: nothing is lost in restating. -/
theorem noZugzwangG_of_noZugzwang (G : QSGame) (guard : G.Pos → Bool)
    (hZ : NoZugzwang G guard) : NoZugzwangG G (fun q _ => guard q) := by
  intro d p hkg hcap hai hgu hd2
  have h := hZ d p hkg hcap hai hgu hd2
  have hc := foldMax_congr (fun m => -(nullValueD2G G (fun q _ => guard q) d m))
    (fun m => -(nullValueD2 G guard d m))
    (movesAbove G (val_lower (d + 1)) p) LOSS
    (fun m _ => by
      show -(nullValueD2G G (fun q _ => guard q) d m) = -(nullValueD2 G guard d m)
      rw [nullValueD2G_depthBlind G guard d m])
  rw [hc, nullValueD2G_depthBlind G guard (d + 1 - 3) (G.pass p)]
  exact h

/-- The accuracy transfer, depth-keyed: under `NoZugzwangG` the
declared function collapses onto the real-move value `negamaxD2` --
the mirror of `nullValue_eq_realValue_of_noZugzwang`. -/
theorem nullValueD2G_eq_realValue_of_noZugzwangG (G : QSGame)
    (guard : G.Pos → Nat → Bool) (hZ : NoZugzwangG G guard) :
    ∀ (d : Nat) (p : G.Pos), nullValueD2G G guard d p = negamaxD2 G d p := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p
    cases d with
    | zero =>
      simp only [nullValueD2G, negamaxD2]
    | succ d =>
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [nullValueD2G_kingGone G guard (d + 1) p hkg,
          negamaxD2_kingGone G (d + 1) p hkg]
      · by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [nullValueD2G_of_capture G guard (d + 1) p hkg hcap,
            negamaxD2_of_capture G (d + 1) p hkg hcap]
        · cases hai : allIllegalB G p with
          | true =>
            rw [nullValueD2G_of_allIllegal G guard d p hkg hcap hai,
              negamaxD2_of_allIllegal G d p hkg hcap hai]
          | false =>
            rw [nullValueD2G_of_fold G guard d p hkg hcap hai,
              negamaxD2_of_fold G d p hkg hcap hai]
            have hsplit := foldMax_init_split (fun m => -(nullValueD2G G guard d m))
              (movesAbove G (val_lower (d + 1)) p) (nullTermD2G G guard d p)
              (nullTermD2G_ge_LOSS G guard d p)
            have hcongr := foldMax_congr (fun m => -(nullValueD2G G guard d m))
              (fun m => -(negamaxD2 G d m))
              (movesAbove G (val_lower (d + 1)) p) LOSS
              (fun m _ => by
                show -(nullValueD2G G guard d m) = -(negamaxD2 G d m)
                rw [ih d (by omega) m])
            have hT : nullTermD2G G guard d p
                ≤ foldMax (fun m => -(nullValueD2G G guard d m))
                    (movesAbove G (val_lower (d + 1)) p) LOSS := by
              have hfl := foldMax_ge_init (fun m => -(nullValueD2G G guard d m))
                (movesAbove G (val_lower (d + 1)) p) LOSS
              by_cases hen : guard p (d + 1) = true ∧ 2 < d + 1
              · by_cases hml : -(nullValueD2G G guard (d + 1 - 3) (G.pass p)) < MATE_LOWER
                · have hT' : nullTermD2G G guard d p
                      = max LOSS (-(nullValueD2G G guard (d + 1 - 3) (G.pass p))) := by
                    simp only [nullTermD2G]
                    rw [if_pos hen, if_pos hml]
                  have hZ' := hZ d p hkg hcap hai hen.1 hen.2
                  rw [hT']
                  omega
                · have hT' : nullTermD2G G guard d p = LOSS := by
                    simp only [nullTermD2G]
                    rw [if_pos hen, if_neg hml]
                  rw [hT']
                  omega
              · have hT' : nullTermD2G G guard d p = LOSS := by
                  simp only [nullTermD2G]
                  rw [if_neg hen]
                rw [hT']
                omega
            omega

/-- Completeness under `NoZugzwangG`, `D ≥ k + 1`: the depth-keyed
mirror of `forcedMate_complete`.  A credit variant thus has BOTH
guarantees: this one inside its null region (chess premise), and the
horizon theorem above it (no premise). -/
theorem forcedMate_completeG (G : QSGame) (guard : G.Pos → Nat → Bool)
    (hF : ValFloor G 192) (hZ : NoZugzwangG G guard)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, k + 1 ≤ D → MATE_LOWER ≤ nullValueD2G G guard D p := by
  intro D hD
  rw [nullValueD2G_eq_realValue_of_noZugzwangG G guard hZ D p]
  exact forcedMate_negamaxD2_band G hF hFM D hD

/-! ## The three experiment arms, expressed -/

/-- Arm A (control): the shipped guard with a fixed horizon,
`2 < depth < 48` in the code.  `base` abstracts the position part
(`abs(score) < 750 and pieces on board`), exactly as the shipped model
abstracts its guard. -/
def fixedHorizonGuard (G : QSGame) (base : G.Pos → Bool) (H : Nat) :
    G.Pos → Nat → Bool :=
  fun p d => base p && decide (d < H)

theorem fixedHorizonGuard_horizon (G : QSGame) (base : G.Pos → Bool) (H : Nat) :
    Horizon G (fixedHorizonGuard G base H) H := by
  intro p d hd
  have hdec : decide (d < H) = false := decide_eq_false (by omega)
  simp only [fixedHorizonGuard, hdec, Bool.and_false]

/-- Arm B (smooth): `abs(score) + depth < 500` -- the balance budget is
spent against depth.  `score` abstracts the engine's incremental static
score; `base` the material guard. -/
def smoothCreditGuard (G : QSGame) (score : G.Pos → Int) (base : G.Pos → Bool) :
    G.Pos → Nat → Bool :=
  fun p d => base p && decide ((score p).natAbs + d < 500)

theorem smoothCreditGuard_horizon (G : QSGame) (score : G.Pos → Int)
    (base : G.Pos → Bool) :
    Horizon G (smoothCreditGuard G score base) 500 := by
  intro p d hd
  have hdec : decide ((score p).natAbs + d < 500) = false :=
    decide_eq_false (by omega)
  simp only [smoothCreditGuard, hdec, Bool.and_false]

/-- Arm C (phase-adaptive): `2 < depth < 12 * pieces` -- the horizon
scales with the mover's non-pawn material, vanishing exactly where
zugzwang lives.  `phase` abstracts the own-piece count; the code's
count is at most 15 (2R+2B+2N+1Q plus 8 promotions is impossible;
15 non-pawn uppercase pieces besides the king is the loose cap),
giving the uniform horizon 180. -/
def phaseAdaptiveGuard (G : QSGame) (phase : G.Pos → Nat) (base : G.Pos → Bool) :
    G.Pos → Nat → Bool :=
  fun p d => base p && decide (d < 12 * phase p)

theorem phaseAdaptiveGuard_horizon (G : QSGame) (phase : G.Pos → Nat)
    (base : G.Pos → Bool) (hb : ∀ p, phase p ≤ 15) :
    Horizon G (phaseAdaptiveGuard G phase base) 180 := by
  intro p d hd
  have hp := hb p
  have hdec : decide (d < 12 * phase p) = false := decide_eq_false (by omega)
  simp only [phaseAdaptiveGuard, hdec, Bool.and_false]

/-- The punchline, spelled out on arm A's shape: with the fixed horizon
`H`, mate-in-k completeness of the declared function needs `ValFloor`
ONLY -- the chess premise is gone, at a depth cost of `H` plies. -/
theorem forcedMate_complete_fixedHorizon (G : QSGame) (base : G.Pos → Bool)
    (H : Nat) (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, k + H + 1 ≤ D →
      MATE_LOWER ≤ nullValueD2G G (fixedHorizonGuard G base H) D p :=
  forcedMate_nullValueD2G_of_horizon G _ hF
    (fixedHorizonGuard_horizon G base H) hFM

/-! # Part III: the composition -- fuel oracle + frontier tail, and the
full W/D/L trichotomy

Thomas: "We don't just want mate-in-k.  We want to say that given
'enough fuel' we correctly determine any position as W/D/L."

This part composes the two devices, each of which retires one chess
premise, into the value function the composed code arm computes
(`fuelValueD2t`):

* the FUEL ORACLE (Part I) retires `NoZugzwang` on the FINDING side --
  above the horizon the pass steers instead of scoring, so no pass term
  can displace a real fold;
* the FRONTIER TAIL (`Classification.lean` Part B) retires
  `NoMaskedMobility` on the HONESTY side -- where the QS filter admits
  no legal move, the fold runs over the full list instead.

The result is the goal state named in this file's header: the whole
trichotomy on FIDELITY premises only (`ValFloor G 192` + `EvalQuiet`,
both table-checked).  No chess premise appears in ANY arm.

**Scope, honestly** (inherited verbatim from `Classification.lean` and
worth restating, since the headline claims "any position"): the game
classified is the game sunfish plays -- chess WITHOUT draw rules.
"Draw" here means no-forced-mate for either side in that ruleless
game; FIDE 50-move, threefold and insufficient-material draws are NOT
detected as 0 (in K+B vs K the value converges to a small material
score, not to 0).  The theorem says the report never enters the mate
band there -- not that the engine calls it a draw.

**The `∃ k` form of W is WLOG.**  `ForcedMate G k p` is "mate in at
most `k` plies"; the classification quantifies `∃ k`.  For a
finitely-branching game a position that is won at all is won within
SOME finite `k` (König), so `∃ k, ForcedMate G k p` is the honest
formalization of "this position is a win" -- no generality is lost by
the indexed spec.  `forcedMate_mono` below is the "at most" reading as
a lemma, and it is what lets the honesty direction's small index be
lifted to the probed depth.

**Stability is the user-facing form.**  `eventual_classification_fuel`
produces ONE `D0` such that every depth from `D0` on classifies
correctly AND agrees with every other -- the two `iff`s make the
classification read off the value exactly right, and the draw arm is
then automatic and holds at every depth (it never oscillates).  The
per-arm lemmas below are its support. -/

/-! "At most `k` plies" is `Liveness.forcedMate_mono` -- a forced mate in
`k` is a forced mate in any `k' ≥ k`.  Stated there, used here. -/

/-- The mated-side monotonicity, inherited from the mate side. -/
theorem forcedlyMated_mono (G : QSGame) {k : Nat} {q : G.Pos}
    (h : ForcedlyMated G k q) : ∀ k' : Nat, k ≤ k' → ForcedlyMated G k' q := by
  intro k' hk'
  cases h with
  | inl hcm => exact Or.inl hcm
  | inr h =>
    exact Or.inr ⟨h.1, fun m hm hleg => forcedMate_mono G (h.2 m hm hleg) k' hk'⟩

/-- The list the fold runs over: the tail trigger swaps the FULL move
list in exactly where every admitted move is illegal, and otherwise the
QS-admitted list.  Keyed by NOMINAL depth in both components -- the
decided admission fork. -/
def tailList (G : QSGame) (d : Nat) (p : G.Pos) : List G.Pos :=
  if allAdmittedIllegalB G d p = true then G.moves p
  else movesAbove G (val_lower d) p

theorem tailList_subset (G : QSGame) (d : Nat) (p : G.Pos) :
    ∀ m ∈ tailList G d p, m ∈ G.moves p := by
  intro m hm
  simp only [tailList] at hm
  by_cases h : allAdmittedIllegalB G d p = true
  · rw [if_pos h] at hm; exact hm
  · rw [if_neg h] at hm; exact movesAbove_subset G _ p m hm

/-- An admitted move is always in the folded list: if the trigger fired
the list is everything, otherwise it IS the admitted list. -/
theorem mem_tailList_of_admitted (G : QSGame) {d : Nat} {p m : G.Pos}
    (hm : m ∈ movesAbove G (val_lower d) p) : m ∈ tailList G d p := by
  simp only [tailList]
  by_cases h : allAdmittedIllegalB G d p = true
  · rw [if_pos h]; exact movesAbove_subset G _ p m hm
  · rw [if_neg h]; exact hm

/-- **The composed declared value**: the fuel shape of Part I with the
frontier tail's list in both regimes.  Below depth 6 the capped pass is
still a score candidate (verbatim `nullValueD2t` shape); from depth 6
the fold is over real moves only, at the fuel-reduced child depth.
Admission and the trigger read the NOMINAL depth; only the recursion is
shortened.  `(pos, depth)`-determined and window-free. -/
def fuelValueD2t (G : QSGame) (guard : G.Pos → Bool) (C : Nat)
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
      foldMax (fun m => -(fuelValueD2t G guard C spend d m)) (tailList G (d + 1) p)
        (if guard p = true ∧ 2 < d + 1 then
          (if -(fuelValueD2t G guard C spend (d + 1 - 3) (G.pass p)) < MATE_LOWER then
            max LOSS (-(fuelValueD2t G guard C spend (d + 1 - 3) (G.pass p)))
          else LOSS)
        else LOSS)
    else
      foldMax (fun m => -(fuelValueD2t G guard C spend
          (d - min (C - 1) (spend p (d + 1) m)) m))
        (tailList G (d + 1) p) LOSS
termination_by d _ => d
decreasing_by all_goals omega

/-- The sub-horizon pass term, named. -/
def fuelTermD2t (G : QSGame) (guard : G.Pos → Bool) (C : Nat)
    (spend : G.Pos → Nat → G.Pos → Nat) (d : Nat) (p : G.Pos) : Int :=
  if guard p = true ∧ 2 < d + 1 then
    (if -(fuelValueD2t G guard C spend (d + 1 - 3) (G.pass p)) < MATE_LOWER then
      max LOSS (-(fuelValueD2t G guard C spend (d + 1 - 3) (G.pass p)))
    else LOSS)
  else LOSS

/-! ### Branch lemmas -/

theorem fuelValueD2t_kingGone (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos) (h : G.eval p ≤ -MATE_LOWER) :
    fuelValueD2t G guard C spend d p = -MATE_UPPER := by
  cases d with
  | zero => simp only [fuelValueD2t]; rw [if_pos h]
  | succ d => simp only [fuelValueD2t]; rw [if_pos h]

theorem fuelValueD2t_of_capture (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos) (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = true) :
    fuelValueD2t G guard C spend d p = MATE_UPPER := by
  cases d with
  | zero => simp only [fuelValueD2t]; rw [if_neg hkg, if_pos hcap]
  | succ d => simp only [fuelValueD2t]; rw [if_neg hkg, if_pos hcap]

theorem fuelValueD2t_of_allIllegal (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = true) :
    fuelValueD2t G guard C spend (d + 1) p = terminalValue G (d + 1) p := by
  simp only [fuelValueD2t]
  rw [if_neg hkg, if_neg hcap, if_pos hai]

theorem fuelValueD2t_zero_eq_eval (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true)) :
    fuelValueD2t G guard C spend 0 p = G.eval p := by
  simp only [fuelValueD2t]
  rw [if_neg hkg, if_neg hcap]

theorem fuelValueD2t_of_fold_regime (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hd : 5 ≤ d) :
    fuelValueD2t G guard C spend (d + 1) p
      = foldMax (fun m => -(fuelValueD2t G guard C spend
            (d - min (C - 1) (spend p (d + 1) m)) m))
          (tailList G (d + 1) p) LOSS := by
  simp only [fuelValueD2t]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_neg (by omega)]

theorem fuelValueD2t_of_fold_sub (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (d : Nat) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hai : allIllegalB G p = false)
    (hd : d < 5) :
    fuelValueD2t G guard C spend (d + 1) p
      = foldMax (fun m => -(fuelValueD2t G guard C spend d m)) (tailList G (d + 1) p)
          (fuelTermD2t G guard C spend d p) := by
  simp only [fuelValueD2t]
  rw [if_neg hkg, if_neg hcap, if_neg (by simp [hai]), if_pos (by omega)]
  rfl

theorem fuelTermD2t_ge_LOSS (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (d : Nat) (p : G.Pos) :
    LOSS ≤ fuelTermD2t G guard C spend d p := by
  simp only [fuelTermD2t]
  by_cases h1 : guard p = true ∧ 2 < d + 1
  · rw [if_pos h1]
    by_cases h2 : -(fuelValueD2t G guard C spend (d + 1 - 3) (G.pass p)) < MATE_LOWER
    · rw [if_pos h2]; omega
    · rw [if_neg h2]; omega
  · rw [if_neg h1]; omega

/-- The suppression, spent for the composed value: the sub-horizon pass
term can never reach the mate band, so a band-value fold always names a
REAL move (in the regime the initial accumulator is `LOSS` outright). -/
theorem fuelTermD2t_lt_ML (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (d : Nat) (p : G.Pos) :
    fuelTermD2t G guard C spend d p < MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  simp only [fuelTermD2t]
  by_cases h1 : guard p = true ∧ 2 < d + 1
  · rw [if_pos h1]
    by_cases h2 : -(fuelValueD2t G guard C spend (d + 1 - 3) (G.pass p)) < MATE_LOWER
    · rw [if_pos h2]; omega
    · rw [if_neg h2]; omega
  · rw [if_neg h1]; omega

/-- A checkmated child is finalized exactly at any positive depth. -/
theorem fuelValueD2t_checkmated (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) {m : G.Pos}
    (hcap : hasKingCapture G.toNullGame.toGame m = false)
    (hmate : Checkmated G m) :
    ∀ d : Nat, 1 ≤ d → fuelValueD2t G guard C spend d m ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro d hd
  cases d with
  | zero => omega
  | succ d' =>
    by_cases hkgm : G.eval m ≤ -MATE_LOWER
    · rw [fuelValueD2t_kingGone G guard C spend (d' + 1) m hkgm]; omega
    · rw [fuelValueD2t_of_allIllegal G guard C spend d' m hkgm (by simp [hcap]) hmate.1]
      have := terminalValue_mate G (d' + 1) m hmate.2
      omega

/-- Any-branch defender bound, REGIME form: at a defender node in the
real-only regime, whichever list the trigger selects its members are
real moves, so a bound over `G.moves` at each move's selected child
depth closes the fold.  The initial accumulator is `LOSS` outright --
this is exactly what the fuel oracle bought: no pass term can hold the
defender's value above the band. -/
theorem fuelValueD2t_defender_le (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) {d : Nat} {m : G.Pos}
    (hkgm : ¬ (G.eval m ≤ -MATE_LOWER))
    (hcapm : ¬ (hasKingCapture G.toNullGame.toGame m = true))
    (hnt : allIllegalB G m = false)
    (hd : 5 ≤ d)
    (hall : ∀ m' ∈ G.moves m,
      -(fuelValueD2t G guard C spend
          (d - min (C - 1) (spend m (d + 1) m')) m')
        ≤ -MATE_LOWER) :
    fuelValueD2t G guard C spend (d + 1) m ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  rw [fuelValueD2t_of_fold_regime G guard C spend d m hkgm hcapm hnt hd]
  exact foldMax_le _ _ _
    (fun m' hm' => hall m' (tailList_subset G _ m m' hm')) (by omega)

/-! ### The finding side: completeness, `ValFloor` only -/

/-- **Mate-in-k completeness for the composed value**: `ValFloor`
alone, at every `D ≥ C·k + 4`.  The attacker's witness is admitted
(`mem_movesAbove_of_floor`), hence in the folded list whichever branch
the trigger takes (`mem_tailList_of_admitted`); the defender's fold is
bounded member-by-member over the full move list, so the tail's extra
options are harmless (they are refuted by the same derivation). -/
theorem forcedMate_fuelValueD2t (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p) :
    ∀ D : Nat, C * k + 4 ≤ D → MATE_LOWER ≤ fuelValueD2t G guard C spend D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  induction hFM with
  | @mate k p m hkg hm hleg hmate =>
    intro D hD
    have hexp : C * (k + 1) = C * k + C * 1 := Nat.mul_add C k 1
    cases D with
    | zero => omega
    | succ d =>
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [fuelValueD2t_of_capture G guard C spend (d + 1) p hkg hcap]; omega
      · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
        rw [fuelValueD2t_of_fold_regime G guard C spend d p hkg hcap hai (by omega)]
        have hmem : m ∈ tailList G (d + 1) p :=
          mem_tailList_of_admitted G (mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm)
        have hchild : fuelValueD2t G guard C spend
            (d - min (C - 1) (spend p (d + 1) m)) m ≤ -MATE_LOWER :=
          fuelValueD2t_checkmated G guard C spend hleg hmate _ (by omega)
        have hfold : -(fuelValueD2t G guard C spend
              (d - min (C - 1) (spend p (d + 1) m)) m)
            ≤ foldMax (fun x => -(fuelValueD2t G guard C spend
                  (d - min (C - 1) (spend p (d + 1) x)) x))
                (tailList G (d + 1) p) LOSS :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega
  | @step k p m hkg hm hleg hnt _hreply ih =>
    intro D hD
    have hexp : C * (k + 2) = C * k + C * 2 := Nat.mul_add C k 2
    cases D with
    | zero => omega
    | succ d =>
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [fuelValueD2t_of_capture G guard C spend (d + 1) p hkg hcap]; omega
      · have hai : allIllegalB G p = false := allIllegalB_false_of_legal hm hleg
        rw [fuelValueD2t_of_fold_regime G guard C spend d p hkg hcap hai (by omega)]
        have hmem : m ∈ tailList G (d + 1) p :=
          mem_tailList_of_admitted G (mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm)
        have hchild : fuelValueD2t G guard C spend
            (d - min (C - 1) (spend p (d + 1) m)) m ≤ -MATE_LOWER := by
          obtain ⟨dm, hdm⟩ : ∃ x, d - min (C - 1) (spend p (d + 1) m) = x + 1 :=
            ⟨d - min (C - 1) (spend p (d + 1) m) - 1, by omega⟩
          rw [hdm]
          have hdmlb : d ≤ dm + C := by omega
          by_cases hkgm : G.eval m ≤ -MATE_LOWER
          · rw [fuelValueD2t_kingGone G guard C spend (dm + 1) m hkgm]; omega
          · refine fuelValueD2t_defender_le G guard C spend hkgm (by simp [hleg]) hnt
              (by omega) (fun m' hm' => ?_)
            have hkgm' : ¬ (G.eval m' ≤ -MATE_LOWER) := by
              intro hle
              have hc : hasKingCapture G.toNullGame.toGame m = true :=
                (hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm', hle⟩
              rw [hleg] at hc
              exact Bool.noConfusion hc
            cases hcm : hasKingCapture G.toNullGame.toGame m' with
            | true =>
              rw [fuelValueD2t_of_capture G guard C spend
                (dm - min (C - 1) (spend m (dm + 1) m')) m' hkgm' hcm]
              omega
            | false =>
              have := ih m' hm' hcm
                (dm - min (C - 1) (spend m (dm + 1) m')) (by omega)
              omega
        have hfold : -(fuelValueD2t G guard C spend
              (d - min (C - 1) (spend p (d + 1) m)) m)
            ≤ foldMax (fun x => -(fuelValueD2t G guard C spend
                  (d - min (C - 1) (spend p (d + 1) x)) x))
                (tailList G (d + 1) p) LOSS :=
          foldMax_le_of_mem _ _ _ _ hmem
        omega

/-- The mated dual for the composed value. -/
theorem forcedlyMated_fuelValueD2t (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192)
    {k : Nat} {q : G.Pos}
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hFL : ForcedlyMated G k q) :
    ∀ D : Nat, C * k + C + 4 ≤ D →
      fuelValueD2t G guard C spend D q ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro D hD
  cases D with
  | zero => omega
  | succ d =>
    by_cases hkg : G.eval q ≤ -MATE_LOWER
    · rw [fuelValueD2t_kingGone G guard C spend (d + 1) q hkg]; omega
    · have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by
        simp [hcapq]
      cases hFL with
      | inl hcm =>
        rw [fuelValueD2t_of_allIllegal G guard C spend d q hkg hcapq' hcm.1]
        have := terminalValue_mate G (d + 1) q hcm.2
        omega
      | inr h =>
        obtain ⟨hai, hall⟩ := h
        refine fuelValueD2t_defender_le G guard C spend hkg hcapq' hai
          (by omega) (fun m hm => ?_)
        cases hcm : hasKingCapture G.toNullGame.toGame m with
        | true =>
          have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := by
            intro hle
            have hc : hasKingCapture G.toNullGame.toGame q = true :=
              (hasKingCapture_iff G.toNullGame.toGame q).mpr ⟨m, hm, hle⟩
            rw [hcapq] at hc
            exact Bool.noConfusion hc
          rw [fuelValueD2t_of_capture G guard C spend
            (d - min (C - 1) (spend q (d + 1) m)) m hkgm hcm]
          omega
        | false =>
          -- the reduced child depth still clears the reply's own mate
          -- bound: `d - min ≥ (d + 1) - C ≥ C * k + 4`.
          have := forcedMate_fuelValueD2t G guard C spend hC hF
            (hall m hm hcm) (d - min (C - 1) (spend q (d + 1) m)) (by omega)
          omega

/-! ### The honesty side: no false mates, `ValFloor` + `EvalQuiet` only -/

/-- The frontier, closed by CONSTRUCTION for the composed value: at a
depth-1 defender node whose folded list is entirely in the mate band,
either the trigger fired (so the list is EVERY move, forcing every move
illegal -- contradicting non-terminality) or it did not (so a legal
admitted move exists, whose quiet depth-0 value refutes the band claim
on the spot).  `NoMaskedMobility` is not assumed; `EvalQuiet` is the
only premise. -/
theorem frontier_escape_ft (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (hQ : EvalQuiet G.toNullGame.toGame)
    {m : G.Pos}
    (hcapm : ¬ (hasKingCapture G.toNullGame.toGame m = true))
    (hai : allIllegalB G m = false)
    (hrep : ∀ m' ∈ tailList G 1 m, MATE_LOWER ≤ fuelValueD2t G guard C spend 0 m') :
    False := by
  have hML : MATE_LOWER = 47923 := rfl
  by_cases hmask : allAdmittedIllegalB G 1 m = true
  · have hlist : tailList G 1 m = G.moves m := by
      simp only [tailList]; rw [if_pos hmask]
    have hallcap : ∀ m' ∈ G.moves m,
        hasKingCapture G.toNullGame.toGame m' = true := by
      intro m' hm'
      have hv := hrep m' (by rw [hlist]; exact hm')
      by_cases hkgm' : G.eval m' ≤ -MATE_LOWER
      · exact absurd
          ((hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm', hkgm'⟩) hcapm
      · cases hcm' : hasKingCapture G.toNullGame.toGame m' with
        | true => rfl
        | false =>
          exfalso
          rw [fuelValueD2t_zero_eq_eval G guard C spend m' hkgm' (by simp [hcm'])] at hv
          have := hQ m' hkgm'
          omega
    rw [allIllegalB_true_iff.mpr hallcap] at hai
    exact Bool.noConfusion hai
  · have hmask' : allAdmittedIllegalB G 1 m = false := by
      cases h : allAdmittedIllegalB G 1 m with
      | false => rfl
      | true => exact absurd h hmask
    obtain ⟨m0, hm0, hleg0⟩ := exists_legal_admitted hmask'
    have hv := hrep m0 (mem_tailList_of_admitted G hm0)
    have hm0m : m0 ∈ G.moves m := movesAbove_subset G _ m m0 hm0
    have hkg0 : ¬ (G.eval m0 ≤ -MATE_LOWER) := fun hh =>
      hcapm ((hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m0, hm0m, hh⟩)
    rw [fuelValueD2t_zero_eq_eval G guard C spend m0 hkg0 (by simp [hleg0])] at hv
    have := hQ m0 hkg0
    omega

/-- **No false mates for the composed value -- no chess premise**: a
mate-band value at a legally-reached root IS a forced mate (within the
probed depth).  The tail closes the frontier by construction
(`frontier_escape_ft`); the fuel reduction only shortens the child
depths, which weakens the INDEX of the mate found -- recovered by
`forcedMate_mono`. -/
theorem forcedMate_of_fuelValueD2t (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame) :
    ∀ (D : Nat) (p : G.Pos),
      hasKingCapture G.toNullGame.toGame p = false →
      MATE_LOWER ≤ fuelValueD2t G guard C spend D p →
      ForcedMate G D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro D
  induction D using Nat.strongRecOn with
  | _ D ih =>
    intro p hcapf hband
    have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by
      simp [hcapf]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [fuelValueD2t_kingGone G guard C spend D p hkg] at hband
      exact absurd hband (by omega)
    cases D with
    | zero =>
      exfalso
      rw [fuelValueD2t_zero_eq_eval G guard C spend p hkg hcap] at hband
      have := hQ p hkg
      omega
    | succ d =>
      cases hai : allIllegalB G p with
      | true =>
        exfalso
        rw [fuelValueD2t_of_allIllegal G guard C spend d p hkg hcap hai] at hband
        have := (terminalValue_bounds G (d + 1) p).2
        omega
      | false =>
        -- the fold's witness is a REAL move, in either regime: the
        -- regime init is LOSS, the sub-horizon term is sub-band.
        obtain ⟨dc, hdc, m, hm, hmv⟩ :
            ∃ dc, dc ≤ d ∧ ∃ m ∈ G.moves p,
              MATE_LOWER ≤ -(fuelValueD2t G guard C spend dc m) := by
          by_cases hreg : 5 ≤ d
          · rw [fuelValueD2t_of_fold_regime G guard C spend d p hkg hcap hai hreg]
              at hband
            obtain ⟨m, hmem, hmv⟩ :=
              foldMax_failHigh_witness
                (fun x => -(fuelValueD2t G guard C spend
                  (d - min (C - 1) (spend p (d + 1) x)) x))
                (tailList G (d + 1) p) LOSS (by omega) hband
            exact ⟨_, by omega, m, tailList_subset G _ p m hmem, hmv⟩
          · rw [fuelValueD2t_of_fold_sub G guard C spend d p hkg hcap hai (by omega)]
              at hband
            obtain ⟨m, hmem, hmv⟩ :=
              foldMax_failHigh_witness (fun x => -(fuelValueD2t G guard C spend d x))
                (tailList G (d + 1) p) (fuelTermD2t G guard C spend d p)
                (fuelTermD2t_lt_ML G guard C spend d p) hband
            exact ⟨d, by omega, m, tailList_subset G _ p m hmem, hmv⟩
        have hchild : fuelValueD2t G guard C spend dc m ≤ -MATE_LOWER := by omega
        have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
          hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, hh⟩)
        have hlegm : hasKingCapture G.toNullGame.toGame m = false := by
          cases hcm : hasKingCapture G.toNullGame.toGame m with
          | false => rfl
          | true =>
            exfalso
            rw [fuelValueD2t_of_capture G guard C spend dc m hkgm hcm] at hchild
            omega
        have hcapm : ¬ (hasKingCapture G.toNullGame.toGame m = true) := by
          simp [hlegm]
        -- the child is a defender node at depth `dc`
        cases dc with
        | zero =>
          exfalso
          rw [fuelValueD2t_zero_eq_eval G guard C spend m hkgm hcapm] at hchild
          exact hkgm hchild
        | succ dc' =>
          cases hai' : allIllegalB G m with
          | true =>
            rw [fuelValueD2t_of_allIllegal G guard C spend dc' m hkgm hcapm hai']
              at hchild
            by_cases hic : inCheckB G.toNullGame m = true
            · exact ForcedMate.mate (k := d) hkg hm hlegm ⟨hai', hic⟩
            · exfalso
              simp only [terminalValue] at hchild
              rw [if_neg hic] at hchild
              omega
          | false =>
            -- Every reply in the child's folded list is in the band at
            -- its own edge-selected depth.
            have hrep : ∀ m' ∈ tailList G (dc' + 1) m,
                ∃ dg, dg ≤ dc' ∧ MATE_LOWER ≤ fuelValueD2t G guard C spend dg m' := by
              by_cases hreg' : 5 ≤ dc'
              · rw [fuelValueD2t_of_fold_regime G guard C spend dc' m hkgm hcapm
                  hai' hreg'] at hchild
                intro m' hm'
                refine ⟨dc' - min (C - 1) (spend m (dc' + 1) m'), by omega, ?_⟩
                have hle : -(fuelValueD2t G guard C spend
                      (dc' - min (C - 1) (spend m (dc' + 1) m')) m')
                    ≤ foldMax (fun x => -(fuelValueD2t G guard C spend
                        (dc' - min (C - 1) (spend m (dc' + 1) x)) x))
                      (tailList G (dc' + 1) m) LOSS :=
                  foldMax_le_of_mem _ _ _ m' hm'
                omega
              · rw [fuelValueD2t_of_fold_sub G guard C spend dc' m hkgm hcapm
                  hai' (by omega)] at hchild
                intro m' hm'
                refine ⟨dc', by omega, ?_⟩
                have hle : -(fuelValueD2t G guard C spend dc' m')
                    ≤ foldMax (fun x => -(fuelValueD2t G guard C spend dc' x))
                      (tailList G (dc' + 1) m) (fuelTermD2t G guard C spend dc' m) :=
                  foldMax_le_of_mem _ _ _ m' hm'
                omega
            by_cases hdc0 : dc' = 0
            · have hrep0 : ∀ m' ∈ tailList G 1 m,
                  MATE_LOWER ≤ fuelValueD2t G guard C spend 0 m' := by
                intro m' hm'
                obtain ⟨dg, hdg, hv⟩ := hrep m' (by simpa [hdc0] using hm')
                have : dg = 0 := by omega
                simpa [this] using hv
              exact (frontier_escape_ft G guard C spend hQ hcapm hai' hrep0).elim
            · refine forcedMate_mono G
                (ForcedMate.step (k := dc') hkg hm hlegm hai' ?_) (d + 1) (by omega)
              intro m' hm' hleg'
              have hmem' : m' ∈ tailList G (dc' + 1) m :=
                mem_tailList_of_admitted G
                  (mem_movesAbove_of_floor G hF (d := dc' + 1) (by omega) hm')
              obtain ⟨dg, hdg, hv⟩ := hrep m' hmem'
              cases dg with
              | zero =>
                exfalso
                have hkgm' : ¬ (G.eval m' ≤ -MATE_LOWER) := fun hh =>
                  hcapm ((hasKingCapture_iff G.toNullGame.toGame m).mpr ⟨m', hm', hh⟩)
                rw [fuelValueD2t_zero_eq_eval G guard C spend m' hkgm'
                  (by simp [hleg'])] at hv
                have := hQ m' hkgm'
                omega
              | succ dg' =>
                exact forcedMate_mono G
                  (ih (dg' + 1) (by omega) m' hleg' hv) dc' (by omega)

/-- The mated-side honesty dual for the composed value: fidelity
premises only. -/
theorem forcedlyMated_of_fuelValueD2t (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (D : Nat) (q : G.Pos)
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hkgq : ¬ (G.eval q ≤ -MATE_LOWER))
    (hlo : fuelValueD2t G guard C spend (D + 1) q ≤ -MATE_LOWER) :
    ForcedlyMated G D q := by
  have hML : MATE_LOWER = 47923 := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by
    simp [hcapq]
  cases hai : allIllegalB G q with
  | true =>
    rw [fuelValueD2t_of_allIllegal G guard C spend D q hkgq hcapq' hai] at hlo
    by_cases hic : inCheckB G.toNullGame q = true
    · exact Or.inl ⟨hai, hic⟩
    · exfalso
      simp only [terminalValue] at hlo
      rw [if_neg hic] at hlo
      omega
  | false =>
    have hrep : ∀ m' ∈ tailList G (D + 1) q,
        ∃ dg, dg ≤ D ∧ MATE_LOWER ≤ fuelValueD2t G guard C spend dg m' := by
      by_cases hreg : 5 ≤ D
      · rw [fuelValueD2t_of_fold_regime G guard C spend D q hkgq hcapq' hai hreg]
          at hlo
        intro m' hm'
        refine ⟨D - min (C - 1) (spend q (D + 1) m'), by omega, ?_⟩
        have hle : -(fuelValueD2t G guard C spend
              (D - min (C - 1) (spend q (D + 1) m')) m')
            ≤ foldMax (fun x => -(fuelValueD2t G guard C spend
                (D - min (C - 1) (spend q (D + 1) x)) x))
              (tailList G (D + 1) q) LOSS :=
          foldMax_le_of_mem _ _ _ m' hm'
        omega
      · rw [fuelValueD2t_of_fold_sub G guard C spend D q hkgq hcapq' hai (by omega)]
          at hlo
        intro m' hm'
        refine ⟨D, by omega, ?_⟩
        have hle : -(fuelValueD2t G guard C spend D m')
            ≤ foldMax (fun x => -(fuelValueD2t G guard C spend D x))
              (tailList G (D + 1) q) (fuelTermD2t G guard C spend D q) :=
          foldMax_le_of_mem _ _ _ m' hm'
        omega
    by_cases hD0 : D = 0
    · have hrep0 : ∀ m' ∈ tailList G 1 q,
          MATE_LOWER ≤ fuelValueD2t G guard C spend 0 m' := by
        intro m' hm'
        obtain ⟨dg, hdg, hv⟩ := hrep m' (by simpa [hD0] using hm')
        have : dg = 0 := by omega
        simpa [this] using hv
      exact (frontier_escape_ft G guard C spend hQ hcapq' hai hrep0).elim
    · refine Or.inr ⟨hai, fun m' hm' hleg' => ?_⟩
      have hmem' : m' ∈ tailList G (D + 1) q :=
        mem_tailList_of_admitted G
          (mem_movesAbove_of_floor G hF (d := D + 1) (by omega) hm')
      obtain ⟨dg, hdg, hv⟩ := hrep m' hmem'
      cases dg with
      | zero =>
        exfalso
        have hkgm' : ¬ (G.eval m' ≤ -MATE_LOWER) := fun hh =>
          hcapq' ((hasKingCapture_iff G.toNullGame.toGame q).mpr ⟨m', hm', hh⟩)
        rw [fuelValueD2t_zero_eq_eval G guard C spend m' hkgm'
          (by simp [hleg'])] at hv
        have := hQ m' hkgm'
        omega
      | succ dg' =>
        exact forcedMate_mono G
          (forcedMate_of_fuelValueD2t G guard C spend hF hQ (dg' + 1) m' hleg' hv) D
          (by omega)

/-! ### The trichotomy, and the stability headline -/

/-- **Eventual classification for the composed search, per-arm form.**
The premise ledger is the goal state: `ValFloor` + `EvalQuiet`
(fidelity, table-checked) and root legality.  `NoZugzwang` is retired
by the fuel oracle, `NoMaskedMobility` by the frontier tail -- NO chess
premise appears in any arm. -/
theorem eventual_classification_fuel_arms (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    (∀ k, ForcedMate G k p →
      ∀ D, C * k + 4 ≤ D → MATE_LOWER ≤ fuelValueD2t G guard C spend D p) ∧
    (∀ k, ForcedlyMated G k p →
      ∀ D, C * k + C + 4 ≤ D → fuelValueD2t G guard C spend D p ≤ -MATE_LOWER) ∧
    ((∀ k, ¬ ForcedMate G k p) → (∀ k, ¬ ForcedlyMated G k p) →
      ∀ D, -MATE_LOWER < fuelValueD2t G guard C spend D p ∧
        fuelValueD2t G guard C spend D p < MATE_LOWER) := by
  have hML : MATE_LOWER = 47923 := rfl
  refine ⟨fun k hFM D hD => forcedMate_fuelValueD2t G guard C spend hC hF hFM D hD,
    fun k hFMd D hD => forcedlyMated_fuelValueD2t G guard C spend hC hF hcapf hFMd D hD,
    ?_⟩
  intro hnFM hnFMd D
  constructor
  · by_cases hlo : fuelValueD2t G guard C spend D p ≤ -MATE_LOWER
    · exfalso
      cases D with
      | zero =>
        rw [fuelValueD2t_zero_eq_eval G guard C spend p hkg (by simp [hcapf])] at hlo
        exact hkg hlo
      | succ D' =>
        exact hnFMd D'
          (forcedlyMated_of_fuelValueD2t G guard C spend hF hQ D' p hcapf hkg hlo)
    · omega
  · by_cases hhi : MATE_LOWER ≤ fuelValueD2t G guard C spend D p
    · exact absurd (forcedMate_of_fuelValueD2t G guard C spend hF hQ D p hcapf hhi)
        (hnFM D)
    · omega

/-- **THE HEADLINE -- eventual classification, stability form.**  Given
enough fuel, the composed search determines ANY position as W / D / L,
and keeps determining it the same way: there is ONE depth `D0` such
that at EVERY depth `D ≥ D0` the mate-band tests read off the value are
exactly the truth about the position, and the draw case sits strictly
inside the band.  Premises: `ValFloor` + `EvalQuiet` (fidelity, tables)
and root legality -- no chess premise.

The `∃ k` on both sides is the "won at all / lost at all" reading (WLOG
for a finitely-branching game, König); "draw" is the ruleless game's
no-forced-mate, per the scope note above.

Axioms: this wrapper's `D0` is chosen by case analysis on WHICH arm
holds, which is `Classical.em` (a genuine, disclosed use -- the
per-arm lemmas above and everything they rest on are choice-free). -/
theorem eventual_classification_fuel (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    ∃ D0 : Nat, ∀ D : Nat, D0 ≤ D →
      ((MATE_LOWER ≤ fuelValueD2t G guard C spend D p) ↔ (∃ k, ForcedMate G k p)) ∧
      ((fuelValueD2t G guard C spend D p ≤ -MATE_LOWER)
        ↔ (∃ k, ForcedlyMated G k p)) ∧
      ((¬ (∃ k, ForcedMate G k p)) → (¬ (∃ k, ForcedlyMated G k p)) →
        -MATE_LOWER < fuelValueD2t G guard C spend D p ∧
          fuelValueD2t G guard C spend D p < MATE_LOWER) := by
  have hML : MATE_LOWER = 47923 := rfl
  -- the honesty directions hold at every depth
  have hhonW : ∀ D, MATE_LOWER ≤ fuelValueD2t G guard C spend D p →
      ∃ k, ForcedMate G k p :=
    fun D hD => ⟨D, forcedMate_of_fuelValueD2t G guard C spend hF hQ D p hcapf hD⟩
  have hhonL : ∀ D, 1 ≤ D → fuelValueD2t G guard C spend D p ≤ -MATE_LOWER →
      ∃ k, ForcedlyMated G k p := by
    intro D hD1 hD
    cases D with
    | zero => omega
    | succ D' =>
      exact ⟨D', forcedlyMated_of_fuelValueD2t G guard C spend hF hQ D' p hcapf hkg hD⟩
  by_cases hW : ∃ k, ForcedMate G k p
  · obtain ⟨k, hk⟩ := hW
    refine ⟨C * k + 4, fun D hD => ⟨⟨hhonW D, fun _ => ?_⟩, ⟨hhonL D (by omega), ?_⟩, ?_⟩⟩
    · exact forcedMate_fuelValueD2t G guard C spend hC hF hk D hD
    · -- a position cannot be both won and lost: the two bounds collide
      rintro ⟨k', hk'⟩
      exfalso
      have h1 := forcedMate_fuelValueD2t G guard C spend hC hF hk
        (C * k + C * k' + C + 4) (by omega)
      have h2 := forcedlyMated_fuelValueD2t G guard C spend hC hF hcapf hk'
        (C * k + C * k' + C + 4) (by omega)
      omega
    · intro hnW
      exact absurd ⟨k, hk⟩ hnW
  · by_cases hL : ∃ k, ForcedlyMated G k p
    · obtain ⟨k, hk⟩ := hL
      refine ⟨C * k + C + 4, fun D hD => ⟨⟨hhonW D, ?_⟩, ⟨hhonL D (by omega), fun _ => ?_⟩, ?_⟩⟩
      · rintro hWk
        exact absurd hWk hW
      · exact forcedlyMated_fuelValueD2t G guard C spend hC hF hcapf hk D hD
      · intro _ hnL
        exact absurd ⟨k, hk⟩ hnL
    · refine ⟨1, fun D hD => ⟨⟨hhonW D, fun hWk => absurd hWk hW⟩,
        ⟨hhonL D (by omega), fun hLk => absurd hLk hL⟩, fun _ _ => ?_⟩⟩
      obtain ⟨_, _, harmN⟩ :=
        eventual_classification_fuel_arms G guard C spend hC hF hQ p hcapf hkg
      exact harmN (fun k hk => hW ⟨k, hk⟩) (fun k hk => hL ⟨k, hk⟩) D

/-! ### The driver sees it

Layer 1 for the composed search is STATED, not proven (the `boundD2`
mirror against the shaped fold -- the recorded follow-up).  The driver
corollary is therefore stated against ANY probe satisfying the bracket
spec, so that discharging layer 1 later instantiates it with no further
work.  This is the composed twin of `Classification.lean`'s
`classification_visible`. -/

/-- The bracket spec for the composed value, at a fixed depth/position:
`bound`'s fail-soft contract, no chess premise. -/
def FuelTailBracketSpec (G : QSGame) (guard : G.Pos → Bool) (C : Nat)
    (spend : G.Pos → Nat → G.Pos → Nat) (probe : Nat → G.Pos → Int → Int) : Prop :=
  ∀ (d : Nat) (p : G.Pos) (gamma : Int),
    -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
    (gamma ≤ probe d p gamma →
      probe d p gamma ≤ fuelValueD2t G guard C spend d p) ∧
    (probe d p gamma < gamma →
      fuelValueD2t G guard C spend d p ≤ probe d p gamma)

/-- The composed value stays in the score band (needed to run the
driver package). -/
theorem fuelValueD2t_bounded (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat)
    (hB : Bounded G.toNullGame.toGame) :
    ∀ (d : Nat) (p : G.Pos),
      -MATE_UPPER ≤ fuelValueD2t G guard C spend d p ∧
        fuelValueD2t G guard C spend d p ≤ MATE_UPPER := by
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
      simp only [fuelValueD2t]
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg]
        by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [if_pos hcap]; omega
        · rw [if_neg hcap]; omega
    | succ d =>
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [fuelValueD2t_kingGone G guard C spend (d + 1) p hkg]; omega
      · by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
        · rw [fuelValueD2t_of_capture G guard C spend (d + 1) p hkg hcap]; omega
        · cases hai : allIllegalB G p with
          | true =>
            rw [fuelValueD2t_of_allIllegal G guard C spend d p hkg hcap hai]
            have := terminalValue_bounds G (d + 1) p
            omega
          | false =>
            by_cases hreg : 5 ≤ d
            · rw [fuelValueD2t_of_fold_regime G guard C spend d p hkg hcap hai hreg]
              have hge := foldMax_ge_init
                (fun m => -(fuelValueD2t G guard C spend
                  (d - min (C - 1) (spend p (d + 1) m)) m))
                (tailList G (d + 1) p) LOSS
              constructor
              · omega
              · refine foldMax_le _ _ _ (fun m _ => ?_) (by omega)
                have := ih (d - min (C - 1) (spend p (d + 1) m)) (by omega) m
                omega
            · rw [fuelValueD2t_of_fold_sub G guard C spend d p hkg hcap hai (by omega)]
              have hTl := fuelTermD2t_ge_LOSS G guard C spend d p
              have hTu : fuelTermD2t G guard C spend d p ≤ MATE_UPPER := by
                simp only [fuelTermD2t]
                by_cases h1 : guard p = true ∧ 2 < d + 1
                · rw [if_pos h1]
                  have := ih (d + 1 - 3) (by omega) (G.pass p)
                  by_cases h2 : -(fuelValueD2t G guard C spend (d + 1 - 3) (G.pass p))
                      < MATE_LOWER
                  · rw [if_pos h2]; omega
                  · rw [if_neg h2]; omega
                · rw [if_neg h1]; omega
              have hge := foldMax_ge_init
                (fun m => -(fuelValueD2t G guard C spend d m))
                (tailList G (d + 1) p) (fuelTermD2t G guard C spend d p)
              constructor
              · omega
              · refine foldMax_le _ _ _ (fun m _ => ?_) hTu
                have := ih d (by omega) m
                omega

/-- **The driver sees the trichotomy** for the composed search: after
the 15-probe budget at depth `D`, the converged bracket reports the
classification -- win and loss up to the driver's own `EVAL_ROUGHNESS`
slop on the certified side, and the draw case with no slop at all.
Conditional on layer 1 for the composed search (`hspec`), which is the
one recorded open obligation; every other premise is fidelity. -/
theorem driver_sees_trichotomy_fuel (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (probe : Nat → G.Pos → Int → Int)
    (hspec : FuelTailBracketSpec G guard C spend probe)
    (hB : Bounded G.toNullGame.toGame)
    (hF : ValFloor G 192) (hQ : EvalQuiet G.toNullGame.toGame)
    (D : Nat) (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (carried : Int) (hc1 : -MATE_UPPER < carried) (hc2 : carried ≤ MATE_UPPER) :
    ((∃ k, C * k + 4 ≤ D ∧ ForcedMate G k p) →
      MATE_LOWER - EVAL_ROUGHNESS
          ≤ (driverLoop (fun g => probe D p g) 15 (depthInit carried)).lower ∧
        MATE_LOWER
          ≤ (driverLoop (fun g => probe D p g) 15 (depthInit carried)).upper) ∧
    ((∃ k, C * k + C + 4 ≤ D ∧ ForcedlyMated G k p) →
      (driverLoop (fun g => probe D p g) 15 (depthInit carried)).lower
          ≤ -MATE_LOWER ∧
        (driverLoop (fun g => probe D p g) 15 (depthInit carried)).upper
          ≤ -MATE_LOWER + EVAL_ROUGHNESS) ∧
    ((∀ k, ¬ ForcedMate G k p) → (∀ k, ¬ ForcedlyMated G k p) →
      (driverLoop (fun g => probe D p g) 15 (depthInit carried)).lower
          < MATE_LOWER ∧
        -MATE_LOWER
          < (driverLoop (fun g => probe D p g) 15 (depthInit carried)).upper) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hE : EVAL_ROUGHNESS = 15 := rfl
  have hVb := fuelValueD2t_bounded G guard C spend hB D p
  obtain ⟨hw, hlow, hup⟩ :=
    driver_depth_converges (fun g => probe D p g) (fuelValueD2t G guard C spend D p)
      hVb.1 hVb.2 (fun g hg1 hg2 => hspec D p g hg1 hg2) carried hc1 hc2
  obtain ⟨harmW, harmL, harmN⟩ :=
    eventual_classification_fuel_arms G guard C spend hC hF hQ p hcapf hkg
  refine ⟨?_, ?_, ?_⟩
  · rintro ⟨k, hkD, hFM⟩
    have hband := harmW k hFM D hkD
    omega
  · rintro ⟨k, hkD, hFMd⟩
    have hband := harmL k hFMd D hkD
    omega
  · intro hnFM hnFMd
    have hband := harmN hnFM hnFMd D
    omega

/-! # Part IV: re-anchoring to the shipped tail (PR #171's `qs_tail`)

Thomas's PR #171 ("Search filtered QS evasions before declaring mate")
is the canonical frontier-tail implementation, and the arms measured
here run HIS code.  This part verifies -- rather than assumes -- that
the model above describes it.

**His code shape.**  `bound` gains a `qs_tail` flag.  At the post-fold
correction site, when the scan finds a legal move but every legal move
sits below this node's `val_lower`, the node re-enters itself

    best = max(best, self.bound(pos, gamma, depth, root=True, qs_tail=True))

and the re-entry folds the COMPLEMENTARY list -- the admission
predicate is inverted by `((v := pos.value(m)) >= val_lower) != qs_tail`
-- unsorted, with futility, IID and the killer yield all switched off,
and `root=True` so the probe is unstored and null-free.

**Why the model still fits, in three checks.**

* *The join.*  `tailList_retry_join` below: the max of the admitted fold
  (which carries the node's virtual evidence in its initial accumulator)
  and the complementary fold (from `LOSS`) IS the full-list fold with
  that same accumulator.  That is the fold-level content of his
  `foldMax_filtered_tail_retry`, and it is exactly the masked branch of
  `tailList` / `nullValueD2t`.  So `frontier_escape_ft`,
  `forcedMate_of_fuelValueD2t` and Classification's
  `forcedMate_of_nullValueD2t` (with its `CexF` closure) apply to his
  value unchanged.
* *(a) His tail contains ILLEGAL moves; the model's `tailList` contains
  them too.*  `foldMax_legal_tail_eq` shows they are inert either way:
  from a legal parent, an illegal child reports the exact `MATE_UPPER`
  sentinel, so its contribution is `LOSS`, the fold identity.  Folding
  the full below-threshold list and folding only its legal part give the
  same number.
* *(b) The killer during the retry.*  In the shipped source the killer
  yield is guarded (`if not qs_tail and killer and ...`, sunfish.py:407),
  so no killer enters the tail probe and there is nothing to model.  Had
  it not been guarded it would still be inert, and for the record that
  argument is `foldMax_admitted_member_idem`: a killer passing
  `pos.value(killer) >= val_lower` is by definition ADMITTED, hence
  already folded in the first pass, and re-folding a value already in
  the max cannot change it.

Nothing in this part introduces a premise; every lemma is fold algebra. -/

/-- The complementary tail: the moves the QS filter REJECTED at this
threshold -- the list his `qs_tail` re-entry folds. -/
def movesBelow (G : QSGame) (thr : Int) (p : G.Pos) : List G.Pos :=
  (G.moves p).filter (fun m => !(decide (thr ≤ G.val p m)))

/-- **The retry join is the full fold.**  `max` of the admitted fold
(keeping the node's accumulator: null term or `LOSS`) and the
complementary fold (from `LOSS`) equals the fold over all moves with
that accumulator -- the model's masked `tailList` branch.  This is the
fold-level statement of PR #171's `foldMax_filtered_tail_retry`. -/
theorem tailList_retry_join (G : QSGame) (w : G.Pos → Int) (thr : Int)
    (p : G.Pos) (init : Int) (hinit : LOSS ≤ init) :
    max (foldMax w (movesAbove G thr p) init) (foldMax w (movesBelow G thr p) LOSS)
      = foldMax w (G.moves p) init := by
  have hsplit := foldMax_filter_split w (fun m => decide (thr ≤ G.val p m))
    (G.moves p) init
  have hb := foldMax_init_split w
    ((G.moves p).filter (fun m => !(decide (thr ≤ G.val p m)))) init hinit
  have hd := foldMax_ge_init w
    ((G.moves p).filter (fun m => decide (thr ≤ G.val p m))) init
  simp only [movesAbove, movesBelow]
  omega

/-- When the trigger fires, the joined value is the `tailList` fold --
so his two-probe computation and this model's one-fold definition are
the same number. -/
theorem tailList_retry_join_masked (G : QSGame) (w : G.Pos → Int) (d : Nat)
    (p : G.Pos) (init : Int) (hinit : LOSS ≤ init)
    (hmask : allAdmittedIllegalB G d p = true) :
    max (foldMax w (movesAbove G (val_lower d) p) init)
        (foldMax w (movesBelow G (val_lower d) p) LOSS)
      = foldMax w (tailList G d p) init := by
  have h : tailList G d p = G.moves p := by
    simp only [tailList]; rw [if_pos hmask]
  rw [h]
  exact tailList_retry_join G w (val_lower d) p init hinit

/-- Without the trigger no retry happens, and the admitted fold IS the
`tailList` fold -- the untouched path. -/
theorem tailList_no_retry (G : QSGame) (w : G.Pos → Int) (d : Nat)
    (p : G.Pos) (init : Int) (hmask : allAdmittedIllegalB G d p = false) :
    foldMax w (movesAbove G (val_lower d) p) init = foldMax w (tailList G d p) init := by
  simp only [tailList]
  rw [if_neg (by simp [hmask])]

/-- **(a) The illegal members of his tail are inert.**  Folding a list
and folding any sublist that keeps every member whose weight exceeds the
`LOSS` floor give the same maximum -- stated in the form used: if every
member of `l` either lies in `sub` or weighs at most `LOSS`, the two
folds agree. -/
theorem foldMax_legal_tail_eq {α : Type _} (w : α → Int) (l sub : List α)
    (hsub : ∀ a ∈ sub, a ∈ l)
    (hrest : ∀ a ∈ l, a ∈ sub ∨ w a ≤ LOSS) :
    foldMax w l LOSS = foldMax w sub LOSS := by
  have h1 : foldMax w l LOSS ≤ foldMax w sub LOSS := by
    refine foldMax_le _ _ _ (fun a ha => ?_) (foldMax_ge_init w sub LOSS)
    cases hrest a ha with
    | inl hin => exact foldMax_le_of_mem w sub LOSS a hin
    | inr hlo =>
      have := foldMax_ge_init w sub LOSS
      omega
  have h2 : foldMax w sub LOSS ≤ foldMax w l LOSS := by
    refine foldMax_le _ _ _ (fun a ha => ?_) (foldMax_ge_init w l LOSS)
    exact foldMax_le_of_mem w l LOSS a (hsub a ha)
  omega

/-- The sentinel makes an illegal child's contribution exactly `LOSS`,
which is the hypothesis `foldMax_legal_tail_eq` needs: from a legal
parent, every illegal move reports the exact `MATE_UPPER`. -/
theorem illegal_child_contributes_LOSS (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (dc : Nat) {p m : G.Pos}
    (hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true))
    (hm : m ∈ G.moves p)
    (hcm : hasKingCapture G.toNullGame.toGame m = true) :
    -(fuelValueD2t G guard C spend dc m) = LOSS := by
  have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := fun hle =>
    hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, hle⟩)
  rw [fuelValueD2t_of_capture G guard C spend dc m hkgm hcm]
  simp only [LOSS]

/-- **(b), for the record**: re-folding an element already in the list
cannot change the maximum.  (Moot for the shipped source, whose killer
yield is switched off during the retry.) -/
theorem foldMax_admitted_member_idem {α : Type _} (w : α → Int) (l : List α)
    (init : Int) (a : α) (ha : a ∈ l) :
    max (foldMax w l init) (w a) = foldMax w l init := by
  have := foldMax_le_of_mem w l init a ha
  omega

end Sunfish
