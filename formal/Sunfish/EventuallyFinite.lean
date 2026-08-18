/-
Eventual classification from a FINITENESS premise: `hFiniteDiameter`
in place of the frontier device (`NoMaskedMobility` / the #171 tail).

THE QUESTION (Thomas, to be answered BEFORE deciding PR #171).
`EventuallyWide.lean` proves the W/D/L trichotomy for `fuelValueD2t`,
the composed value whose fold list is the frontier TAIL -- that is,
with PR #171's engine change baked into the value function.  For the
UNPATCHED fuel value `fuelValueD2` (Part I: `movesAbove` only, no
tail) only the completeness arms were proven; the honesty side was
left to the frontier device.  Can a finiteness / bounded-diameter
premise replace it?

THE ANSWER (this file): YES for the eventual claim, with an EFFECTIVE
depth bound `D0 = C*N + C + 6`; and NO fixed-depth claim comes with
it, by countermodel.  The four design items:

1. THE PREMISE -- `EndsWithin G N p`: every legal play from `p`
   reaches a terminal position within `N` plies.  Stated INDUCTIVELY,
   as the well-founded budget the induction wants: a terminal position
   ends within any budget; a position ends within `n + 1` when every
   legal child ends within `n`.  Legality and terminality are the
   model's own (`hasKingCapture` false at the child; `allIllegalB`).
   The headline hypothesis `hFiniteDiameter` is one instance at the
   ROOT, `EndsWithin G N p` -- bounded remaining game length from the
   position being classified.  Every position the proof visits then
   inherits a smaller budget by inversion (`EndsWithin.children`), so
   no separate reachable-set quantifier is needed.

   Why the premise is true of the intended game (informal -- the model
   premise is the abstract bound): ADJUDICATED chess has bounded
   remaining game length.  Under the 50-move rule a game ends within
   about 5900 plies from any position, and threefold repetition under
   match adjudication likewise caps every legal continuation.  The
   RULELESS modeled game does not satisfy it -- that is exactly
   `CexE` -- so the premise is precisely the finiteness adjudication
   restores.  `Repetition.lean`'s history rule is the in-model echo of
   the same fact.

   Alternative statement, considered and set aside: "finitely many
   reachable positions + acyclicity of legal play" implies
   `EndsWithin` with `N` = the position count, by pigeonhole.  That
   form carries a set quantifier and a counting argument nothing else
   uses; the inductive budget is what both the honesty induction (on
   the budget) and the completeness transfer (structural on the spec)
   consume directly, so it is the premise.  The counting bridge could
   be added later without touching anything here.

2. COUNTERMODEL DISPOSITIONS, as lemmas, stated against the canonical
   countermodels (`Shortest.lean`'s `CexD`, `Eventual.lean`'s `CexE`
   -- imported, not duplicated):

   * `CexE` VIOLATES the premise: `cexE_not_endsWithin` /
     `cexE_not_finite`.  The infinite masked chain admits no budget at
     any node, so #181's eventual-classification countermodel is
     EXCLUDED by `hFiniteDiameter` -- which is what makes the theorem
     below possible at all.
   * `CexD` SATISFIES the premise: `cexD_endsWithin` (budget 5 at the
     root, budget 1 at the masked node `M`) -- and still lies at fixed
     depth: `cexD_fuel_M1` prices the drawable masked node `M` in the
     mated band (`-MATE_UPPER`) at remaining depth 1, while
     `cexD_M_not_mated` shows no `ForcedlyMated` index exists for it,
     and `cexD_M_eventually_classified` shows the same node correctly
     classified from the effective bound `D0 = 10` on.  So
     `hFiniteDiameter` buys NO fixed-depth honesty: below `D0` the
     frontier can still mask, and fixed-depth claims still need
     `NoMaskedMobility` or the #171 tail.  The variant's claim is
     eventual-only -- by countermodel, not by caution.

3. THE SHAPE.  For the no-tail fuel value `fuelValueD2`:

     eventual_classification_fuel_finite :
       ValFloor G 192 -> EndsWithin G N p -> (root legality) ->
         forall D >= C*N + C + 6,
           mate-band tests read off the value are exactly the truth,
           and the no-mate case sits strictly inside the band.

   `NoMaskedMobility` appears nowhere; neither does the tail, the pass
   (`NoZugzwang`), nor -- a bonus the design did not ask for --
   `EvalQuiet`: at `D >= C*N + 6` every node the classification
   depends on is reached before the frontier, so no static evaluation
   is ever read at a nonterminal leaf.  The masking sites of the tail
   proof (`frontier_escape_ft`; the two `dg = 0` branches of
   `forcedMate_of_fuelValueD2t`) are not weakened here -- they are
   UNREACHABLE: the honesty induction carries the invariant
   `C * budget + 6 <= depth`, each attacker/defender round spends two
   budget plies against at most `2C` of depth, and a node is examined
   below depth 6 only past a terminal, where the exact finalizer
   answers instead.  The induction is on the `EndsWithin` budget
   (remaining distance to terminal), not on depth.

   WHY THE FUEL SHAPE IS WHAT MAKES THIS PROVABLE (the decomposition
   that dissolves the reduced-depth trap).  A probe that runs at
   reduced depth can bottom out at the frontier at a NON-terminal
   position however large `D` is -- if its result were a score, that
   would be a masking channel `hFiniteDiameter` cannot bound, because
   the pass child is NOT a legal move: `EndsWithin` bounds no play
   through it, and the budget never decreases across a pass.  The
   fuel oracle removes exactly that channel.  Above the horizon the
   pass never returns a score -- it only selects how much depth the
   real moves spend -- and `fuelValueD2` models it as the abstract
   selector `spend`, over which every theorem here quantifies
   UNIVERSALLY.  Frontier masking inside the engine's probe subtree
   can therefore only change WHICH selector the code computes, never
   what any selector's fold is worth; it is absorbed by the
   quantifier, and its worst case is already priced into the bound as
   the `C` of `D0 = C*N + C + 6` (every edge at full spend).  The
   sub-horizon pass (depths 3..5, probing at depth <= 2) IS a score
   candidate, and sits strictly below the invariant's floor,
   unreachable.  So the value-correctness argument needs
   frontier-freedom only on the real-move tree, which is precisely
   what the budget buys.

   Lemma ledger: no existing lemma gains the new hypothesis and no
   existing induction is restructured -- the layer-1 machinery is
   untouched, and both new inductions mirror existing patterns (the
   completeness spine; the tail honesty's witness extraction).  The
   completeness arms are reused VERBATIM (`forcedMate_fuelValueD2`,
   `forcedlyMated_fuelValueD2` -- they never needed the frontier),
   bridged by `forcedMate_le_budget` / `forcedlyMated_le_budget`
   ("won at all implies won within the budget"), which is what makes
   `D0` a computable function of `N` rather than a classical `exists`.

4. SCOPE, honestly.  NOT granted: any fixed-depth claim (`CexD`
   above -- a lie at depth 1 inside a game of diameter 5).  The depth
   bound is explicit -- `D0 = C*N + C + 6`, i.e. `2N + 8` as shipped
   -- so "eventually" names a depth computable from the adjudication
   bound, not a classical existence; the corridor costs `C + 6`
   (`6` the fuel horizon, below which the capped pass is still a
   score candidate; `C` the worst single-edge spend).  Unchanged:
   the game classified is the ruleless one (`ForcedMate` /
   `ForcedlyMated`; `Repetition.lean`'s scope note applies verbatim),
   and layer 1 (`FuelBracketSpec`) stays the recorded open obligation
   exactly as in `EventuallyWide.lean`.

Zero sorries, no Mathlib, no audit-surface changes (this file only
adds definitions and theorems; `sunfish.py` is untouched).  Every
theorem in this file -- the headline included -- depends only on
`propext` and `Quot.sound` (machine-checked with `#print axioms`):
unlike `eventual_classification_fuel`, whose `D0` is picked by case
analysis on which arm holds (`Classical.em`, disclosed there), the
effective bound needs no choice anywhere.
-/

import Sunfish.EventuallyWide
import Sunfish.Eventual

namespace Sunfish

/-! # The premise: bounded remaining game length -/

/-- **`EndsWithin G n p`** -- every legal play from `p` reaches a
terminal position within `n` plies: the position is terminal already,
or every legal move leads to a position that ends within one ply less.
The constructor set is deliberately the weakest that says it: the
`terminal` constructor takes ANY budget (so callers never need
monotonicity to assemble one), and `step` does not demand
nonterminality (so a hypothesis is as easy to hold as possible --
which makes the theorems below stronger). -/
inductive EndsWithin (G : QSGame) : Nat → G.Pos → Prop where
  | terminal {n : Nat} {p : G.Pos} (h : allIllegalB G p = true) : EndsWithin G n p
  | step {n : Nat} {p : G.Pos}
      (h : ∀ m ∈ G.moves p, hasKingCapture G.toNullGame.toGame m = false →
        EndsWithin G n m) :
      EndsWithin G (n + 1) p

/-- A budget can only be relaxed. -/
theorem EndsWithin.mono {G : QSGame} {n : Nat} {p : G.Pos}
    (h : EndsWithin G n p) : ∀ n' : Nat, n ≤ n' → EndsWithin G n' p := by
  induction h with
  | terminal ht => exact fun n' _ => EndsWithin.terminal ht
  | step _ ih =>
    intro n' hn'
    obtain ⟨n'', rfl⟩ : ∃ n'', n' = n'' + 1 := ⟨n' - 1, by omega⟩
    exact EndsWithin.step (fun m hm hleg => ih m hm hleg n'' (by omega))

/-- Budget `0` forces terminality -- the base inversion. -/
theorem EndsWithin.terminal_of_zero {G : QSGame} {p : G.Pos}
    (h : EndsWithin G 0 p) : allIllegalB G p = true := by
  cases h with
  | terminal ht => exact ht

/-- The working inversion: at a NONterminal position the budget is
positive, and every legal child inherits its predecessor. -/
theorem EndsWithin.children {G : QSGame} {n : Nat} {p : G.Pos}
    (h : EndsWithin G n p) (hnt : allIllegalB G p = false) :
    ∃ n₀, n = n₀ + 1 ∧ ∀ m ∈ G.moves p,
      hasKingCapture G.toNullGame.toGame m = false → EndsWithin G n₀ m := by
  cases h with
  | terminal ht => rw [ht] at hnt; exact Bool.noConfusion hnt
  | step hch => exact ⟨_, rfl, hch⟩

/-! # "Won at all" is "won within the budget" -/

/-- `ForcedMate` COMPRESSES to the game bound: under `EndsWithin G n p`
a forced mate at ANY index is a forced mate within `n + 1` (the `+ 1`
absorbs parity -- mate distances are odd).  This is what turns the
classification's `∃ k` into the effective depth bound: the budget caps
the index, so `D0` is a function of `N` alone and no case analysis on
which arm holds is ever needed. -/
theorem forcedMate_le_budget (G : QSGame) :
    ∀ {k : Nat} {p : G.Pos}, ForcedMate G k p →
      ∀ n : Nat, EndsWithin G n p → ForcedMate G (n + 1) p := by
  intro k p hFM
  induction hFM with
  | @mate _ p m hkg hm hleg hmate =>
    exact fun n _ => ForcedMate.mate hkg hm hleg hmate
  | @step _ p m hkg hm hleg hnt _hreply ih =>
    intro n hE
    obtain ⟨n₀, rfl, hch⟩ := hE.children (allIllegalB_false_of_legal hm hleg)
    obtain ⟨n₁, rfl, hch'⟩ := (hch m hm hleg).children hnt
    exact ForcedMate.step hkg hm hleg hnt
      (fun m' hm' hleg' => ih m' hm' hleg' n₁ (hch' m' hm' hleg'))

/-- The mated-side compression: under `EndsWithin G n q` a forced loss
at any index is a forced loss within `n`. -/
theorem forcedlyMated_le_budget (G : QSGame) {k n : Nat} {q : G.Pos}
    (hFL : ForcedlyMated G k q) (hE : EndsWithin G n q) :
    ForcedlyMated G n q := by
  cases hFL with
  | inl hcm => exact Or.inl hcm
  | inr h =>
    obtain ⟨hai, hall⟩ := h
    obtain ⟨n₀, rfl, hch⟩ := hE.children hai
    exact Or.inr ⟨hai, fun m hm hleg =>
      forcedMate_le_budget G (hall m hm hleg) n₀ (hch m hm hleg)⟩

/-! # The honesty side under the budget: the frontier is never reached -/

/-- **No false mates for the unpatched fuel value, from finiteness.**
A mate-band value at a legally reached position IS a forced mate --
with `EndsWithin` standing where the tail proof used the frontier
device.  The induction is on the BUDGET, carrying the invariant
`C * budget + 6 ≤ depth`: each attacker/defender round spends two
budget plies against at most `2C` of depth, so every examined node
stays in the real-only regime (`≥ 6`), every fold list is unfiltered
(`ValFloor` from depth 2), terminal children are finalized exactly,
and the masking sites of `forcedMate_of_fuelValueD2t` -- the
`frontier_escape_ft` call and both `dg = 0` branches -- are
unreachable.  `EvalQuiet` is not needed: no static evaluation is ever
read at a node this argument visits. -/
theorem forcedMate_of_fuelValueD2_ends (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 1 ≤ C)
    (hF : ValFloor G 192) :
    ∀ (n D : Nat) (p : G.Pos),
      EndsWithin G n p →
      hasKingCapture G.toNullGame.toGame p = false →
      C * n + 6 ≤ D →
      MATE_LOWER ≤ fuelValueD2 G guard C spend D p →
      ForcedMate G D p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro n
  induction n using Nat.strongRecOn with
  | _ n ih =>
    intro D p hE hcapf hD hband
    have hcap : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by simp [hcapf]
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · exfalso
      rw [fuelValueD2_kingGone G guard C spend D p hkg] at hband
      omega
    obtain ⟨d, rfl⟩ : ∃ d, D = d + 1 := ⟨D - 1, by omega⟩
    cases hai : allIllegalB G p with
    | true =>
      -- a terminal cannot carry a mate-band-high value.  (The explicit
      -- `exfalso`s here and above keep `omega` off its classical
      -- fallback for non-arithmetic goals; casing on the finalizer's
      -- branches rather than using `terminalValue_bounds` avoids the
      -- `Classical.choice` already inside that lemma.)
      exfalso
      rw [fuelValueD2_of_allIllegal G guard C spend d p hkg hcap hai] at hband
      by_cases hic : inCheckB G.toNullGame p = true
      · have := terminalValue_mate G (d + 1) p hic
        omega
      · simp only [terminalValue] at hband
        rw [if_neg hic] at hband
        omega
    | false =>
      obtain ⟨n', rfl, hch⟩ := hE.children hai
      have hexp1 : C * (n' + 1) = C * n' + C * 1 := Nat.mul_add C n' 1
      -- the invariant keeps the node in the real-only regime
      rw [fuelValueD2_of_fold_regime G guard C spend d p hkg hcap hai (by omega)] at hband
      -- the band witness is a real move (the regime init is LOSS)
      obtain ⟨m, hm, hmv⟩ : ∃ m ∈ G.moves p,
          MATE_LOWER ≤ -(fuelValueD2 G guard C spend
            (d - min (C - 1) (spend p (d + 1) m)) m) := by
        obtain ⟨m, hmem, hmv⟩ :=
          foldMax_failHigh_witness
            (fun x => -(fuelValueD2 G guard C spend
              (d - min (C - 1) (spend p (d + 1) x)) x))
            (movesAbove G (val_lower (d + 1)) p) LOSS (by omega) hband
        exact ⟨m, movesAbove_subset G _ p m hmem, hmv⟩
      have hchild : fuelValueD2 G guard C spend
          (d - min (C - 1) (spend p (d + 1) m)) m ≤ -MATE_LOWER := by omega
      have hkgm : ¬ (G.eval m ≤ -MATE_LOWER) := fun hh =>
        hcap ((hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, hh⟩)
      -- ... and a LEGAL one: an illegal witness would carry the sentinel
      have hlegm : hasKingCapture G.toNullGame.toGame m = false := by
        cases hcm : hasKingCapture G.toNullGame.toGame m with
        | false => rfl
        | true =>
          exfalso
          rw [fuelValueD2_of_capture G guard C spend
            (d - min (C - 1) (spend p (d + 1) m)) m hkgm hcm] at hchild
          omega
      have hEm := hch m hm hlegm
      obtain ⟨dc, hdcv⟩ : ∃ x, d - min (C - 1) (spend p (d + 1) m) = x + 1 :=
        ⟨d - min (C - 1) (spend p (d + 1) m) - 1, by omega⟩
      rw [hdcv] at hchild
      have hdc4 : C * n' + 4 ≤ dc + 1 := by omega
      cases hai' : allIllegalB G m with
      | true =>
        -- terminal defender: the exact finalizer answers
        rw [fuelValueD2_of_allIllegal G guard C spend dc m hkgm (by simp [hlegm]) hai']
          at hchild
        by_cases hic : inCheckB G.toNullGame m = true
        · exact ForcedMate.mate hkg hm hlegm ⟨hai', hic⟩
        · exfalso
          simp only [terminalValue] at hchild
          rw [if_neg hic] at hchild
          omega
      | false =>
        -- nonterminal defender: budget and regime both persist
        obtain ⟨n'', rfl, hch'⟩ := hEm.children hai'
        have hexp2 : C * (n'' + 1) = C * n'' + C * 1 := Nat.mul_add C n'' 1
        rw [fuelValueD2_of_fold_regime G guard C spend dc m hkgm (by simp [hlegm]) hai'
          (by omega)] at hchild
        -- every legal reply is folded (nothing is filtered at depth ≥ 2)
        -- and hence in the band at its own edge-selected depth
        have hrep : ∀ m' ∈ G.moves m, hasKingCapture G.toNullGame.toGame m' = false →
            MATE_LOWER ≤ fuelValueD2 G guard C spend
              (dc - min (C - 1) (spend m (dc + 1) m')) m' := by
          intro m' hm' _hleg'
          have hmem' : m' ∈ movesAbove G (val_lower (dc + 1)) m :=
            mem_movesAbove_of_floor G hF (d := dc + 1) (by omega) hm'
          have hle : -(fuelValueD2 G guard C spend
                (dc - min (C - 1) (spend m (dc + 1) m')) m')
              ≤ foldMax (fun x => -(fuelValueD2 G guard C spend
                    (dc - min (C - 1) (spend m (dc + 1) x)) x))
                  (movesAbove G (val_lower (dc + 1)) m) LOSS :=
            foldMax_le_of_mem _ _ _ m' hmem'
          omega
        refine forcedMate_mono G
          (ForcedMate.step (k := dc) hkg hm hlegm hai'
            (fun m' hm' hleg' => forcedMate_mono G
              (ih n'' (by omega) (dc - min (C - 1) (spend m (dc + 1) m')) m'
                (hch' m' hm' hleg') hleg' (by omega) (hrep m' hm' hleg'))
              dc (by omega)))
          (d + 1) (by omega)

/-- The mated-side honesty dual under the budget: `ValFloor` and
`EndsWithin` only.  The outer shell of the same argument -- the
defender's regime fold puts every legal reply in the band at its
edge-selected child depth, and the mate-side theorem prices each of them. -/
theorem forcedlyMated_of_fuelValueD2_ends (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 1 ≤ C)
    (hF : ValFloor G 192)
    {n D : Nat} {q : G.Pos}
    (hE : EndsWithin G n q)
    (hcapq : hasKingCapture G.toNullGame.toGame q = false)
    (hkgq : ¬ (G.eval q ≤ -MATE_LOWER))
    (hD : C * n + 6 ≤ D)
    (hlo : fuelValueD2 G guard C spend D q ≤ -MATE_LOWER) :
    ForcedlyMated G D q := by
  have hML : MATE_LOWER = 47923 := rfl
  have hcapq' : ¬ (hasKingCapture G.toNullGame.toGame q = true) := by simp [hcapq]
  obtain ⟨d, rfl⟩ : ∃ d, D = d + 1 := ⟨D - 1, by omega⟩
  cases hai : allIllegalB G q with
  | true =>
    rw [fuelValueD2_of_allIllegal G guard C spend d q hkgq hcapq' hai] at hlo
    by_cases hic : inCheckB G.toNullGame q = true
    · exact Or.inl ⟨hai, hic⟩
    · exfalso
      simp only [terminalValue] at hlo
      rw [if_neg hic] at hlo
      omega
  | false =>
    obtain ⟨n', rfl, hch⟩ := hE.children hai
    have hexp1 : C * (n' + 1) = C * n' + C * 1 := Nat.mul_add C n' 1
    rw [fuelValueD2_of_fold_regime G guard C spend d q hkgq hcapq' hai (by omega)] at hlo
    refine Or.inr ⟨hai, fun m hm hleg => ?_⟩
    have hmem : m ∈ movesAbove G (val_lower (d + 1)) q :=
      mem_movesAbove_of_floor G hF (d := d + 1) (by omega) hm
    have hle : -(fuelValueD2 G guard C spend
          (d - min (C - 1) (spend q (d + 1) m)) m)
        ≤ foldMax (fun x => -(fuelValueD2 G guard C spend
              (d - min (C - 1) (spend q (d + 1) x)) x))
            (movesAbove G (val_lower (d + 1)) q) LOSS :=
      foldMax_le_of_mem _ _ _ m hmem
    have hband : MATE_LOWER ≤ fuelValueD2 G guard C spend
        (d - min (C - 1) (spend q (d + 1) m)) m := by omega
    exact forcedMate_mono G
      (forcedMate_of_fuelValueD2_ends G guard C spend hC hF n'
        (d - min (C - 1) (spend q (d + 1) m)) m (hch m hm hleg) hleg (by omega) hband)
      (d + 1) (by omega)

/-! # The headline: eventual classification from finiteness -/

/-- **Eventual W/D/L classification for the UNPATCHED fuel value, from
finiteness, with an EFFECTIVE bound.**  Premises: `ValFloor` (tables),
`EndsWithin G N p` -- `hFiniteDiameter` at the root -- and root
legality.  `NoMaskedMobility`, the #171 tail, `NoZugzwang` and
`EvalQuiet` appear nowhere.  Unlike `eventual_classification_fuel`'s
classical `∃ D0`, the bound is the explicit `D0 = C*N + C + 6`
(`2N + 8` as shipped): "eventually" names a depth computable from the
adjudication bound.  The wrapper is also choice-free -- no case
analysis on which arm holds is needed, because the budget compresses
every mate index below `N + 1`. -/
theorem eventual_classification_fuel_finite (G : QSGame) (guard : G.Pos → Bool)
    (C : Nat) (spend : G.Pos → Nat → G.Pos → Nat) (hC : 2 ≤ C)
    (hF : ValFloor G 192)
    {N : Nat} (p : G.Pos)
    (hE : EndsWithin G N p)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    ∀ D : Nat, C * N + C + 6 ≤ D →
      ((MATE_LOWER ≤ fuelValueD2 G guard C spend D p) ↔ (∃ k, ForcedMate G k p)) ∧
      ((fuelValueD2 G guard C spend D p ≤ -MATE_LOWER) ↔ (∃ k, ForcedlyMated G k p)) ∧
      ((¬ (∃ k, ForcedMate G k p)) → (¬ (∃ k, ForcedlyMated G k p)) →
        -MATE_LOWER < fuelValueD2 G guard C spend D p ∧
          fuelValueD2 G guard C spend D p < MATE_LOWER) := by
  have hML : MATE_LOWER = 47923 := rfl
  intro D hD
  have hexp : C * (N + 1) = C * N + C * 1 := Nat.mul_add C N 1
  have hhonW : MATE_LOWER ≤ fuelValueD2 G guard C spend D p → ∃ k, ForcedMate G k p :=
    fun hb => ⟨D, forcedMate_of_fuelValueD2_ends G guard C spend (by omega) hF
      N D p hE hcapf (by omega) hb⟩
  have hhonL : fuelValueD2 G guard C spend D p ≤ -MATE_LOWER →
      ∃ k, ForcedlyMated G k p :=
    fun hb => ⟨D, forcedlyMated_of_fuelValueD2_ends G guard C spend (by omega) hF
      hE hcapf hkg (by omega) hb⟩
  refine ⟨⟨hhonW, ?_⟩, ⟨hhonL, ?_⟩, ?_⟩
  · rintro ⟨k, hk⟩
    exact forcedMate_fuelValueD2 G guard C spend hC hF
      (forcedMate_le_budget G hk N hE) D (by omega)
  · rintro ⟨k, hk⟩
    exact forcedlyMated_fuelValueD2 G guard C spend hC hF hcapf
      (forcedlyMated_le_budget G hk hE) D (by omega)
  · intro hnW hnL
    by_cases h1 : MATE_LOWER ≤ fuelValueD2 G guard C spend D p
    · exact absurd (hhonW h1) hnW
    by_cases h2 : fuelValueD2 G guard C spend D p ≤ -MATE_LOWER
    · exact absurd (hhonL h2) hnL
    exact ⟨by omega, by omega⟩

/-! # Countermodel dispositions

The two countermodels of the frontier premise, and where they stand
against `EndsWithin`.  `CexD` and `CexE` are the canonical games of
`Shortest.lean` and `Eventual.lean` (this file adds only the
`EndsWithin`-level dispositions; the stacks' value-level lemmas stay
where they are). -/

/-- **`CexE` violates the premise, at every node and every budget.**
The only legal move from `C n` is `C (n + 1)` and no `C i` is
terminal, so no finite budget can be spent down.  #181's
eventual-classification countermodel is thereby EXCLUDED by
`hFiniteDiameter`.  (This is the formal shape of "the frontier renews
the phantom at every horizon": renewal needs an infinite legal
nonterminal chain, and the premise is exactly its negation.) -/
theorem cexE_not_endsWithin : ∀ N n : Nat, ¬ EndsWithin CexE N (EPos.C n) := by
  intro N
  induction N with
  | zero =>
    intro n h
    have h0 := h.terminal_of_zero
    rw [cexE_ai_C] at h0
    exact Bool.noConfusion h0
  | succ N ih =>
    intro n h
    cases h with
    | terminal ht =>
      rw [cexE_ai_C] at ht
      exact Bool.noConfusion ht
    | step hch =>
      exact ih (n + 1)
        (hch (EPos.C (n + 1)) (by rw [cexE_moves_C]; simp) (cexE_cap_C (n + 1)))

/-- The named disposition: `CexE` is NOT a finite game -- `hFinite`
fails on it outright. -/
theorem cexE_not_finite : ¬ ∃ N, EndsWithin CexE N (EPos.C 0) :=
  fun ⟨N, h⟩ => cexE_not_endsWithin N 0 h

/-- The masked node itself has budget 1: its one legal move is the
stalemate `S`. -/
theorem cexD_M_endsWithin : EndsWithin CexD 1 DPos.M := by
  refine EndsWithin.step (fun m hm hleg => ?_)
  have hmv : CexD.moves DPos.M = [DPos.X, DPos.S] := rfl
  rw [hmv] at hm
  have hm' : m = DPos.X ∨ m = DPos.S := by simpa using hm
  rcases hm' with rfl | rfl
  · exact absurd hleg (by decide)
  · exact EndsWithin.terminal (by decide)

/-- **`CexD` satisfies the premise** (budget 5 at the root): the
finite fixed-depth countermodel SURVIVES `hFiniteDiameter`, so the
premise cannot rescue any fixed-depth claim -- see `cexD_fuel_M1`. -/
theorem cexD_endsWithin : EndsWithin CexD 5 DPos.Q := by
  have hCn : ∀ n, EndsWithin CexD n DPos.C := fun _ => EndsWithin.terminal (by decide)
  have hP : EndsWithin CexD 2 DPos.P := by
    refine EndsWithin.step (fun m hm _ => ?_)
    have hmv : CexD.moves DPos.P = [DPos.M, DPos.C] := rfl
    rw [hmv] at hm
    have hm' : m = DPos.M ∨ m = DPos.C := by simpa using hm
    rcases hm' with rfl | rfl
    · exact cexD_M_endsWithin
    · exact hCn 1
  have hE : EndsWithin CexD 3 DPos.E := by
    refine EndsWithin.step (fun m hm _ => ?_)
    have hmv : CexD.moves DPos.E = [DPos.P] := rfl
    rw [hmv] at hm
    have hm' : m = DPos.P := by simpa using hm
    subst hm'; exact hP
  have hD : EndsWithin CexD 4 DPos.D := by
    refine EndsWithin.step (fun m hm _ => ?_)
    have hmv : CexD.moves DPos.D = [DPos.E] := rfl
    rw [hmv] at hm
    have hm' : m = DPos.E := by simpa using hm
    subst hm'; exact hE
  have hB : EndsWithin CexD 4 DPos.B := by
    refine EndsWithin.step (fun m hm _ => ?_)
    have hmv : CexD.moves DPos.B = [DPos.C] := rfl
    rw [hmv] at hm
    have hm' : m = DPos.C := by simpa using hm
    subst hm'; exact hCn 3
  refine EndsWithin.step (fun m hm _ => ?_)
  have hmv : CexD.moves DPos.Q = [DPos.D, DPos.B] := rfl
  rw [hmv] at hm
  have hm' : m = DPos.D ∨ m = DPos.B := by simpa using hm
  rcases hm' with rfl | rfl
  · exact hD
  · exact hB

/-- **The fixed-depth lie does not survive `c01915f`.**  At remaining
depth 1 the once-masked node `M` -- whose only legal move is the
stalemate escape `S`, filtered at the PRE-`c01915f` depth-1 threshold
while the illegal `X` was admitted -- is now priced at the honest `0`,
strictly inside the band, for EVERY edge-cost selector.  (`guard` off,
matching the stacks' countermodels; depth 1 is below the fuel horizon,
so this is the shipped sub-horizon shape.)

This theorem used to read `-MATE_UPPER` and carried the disposition
"the finiteness variant buys the EVENTUAL claim only, and fixed-depth
honesty still needs `NoMaskedMobility` or the #171 tail".  Both halves
of that are retired: the admission change delivers fixed-depth honesty
at this node, and `NoMaskedMobility` is a theorem
(`noMaskedMobility_of_valFloor`).  What remains true, and is what the
variant is actually for, is that FUEL EXHAUSTION -- not masking -- is
the reason a fuel-bounded value can be sub-horizon inaccurate. -/
theorem cexD_fuel_M1 (spend : CexD.Pos → Nat → CexD.Pos → Nat) :
    fuelValueD2 CexD (fun _ => false) 2 spend 1 DPos.M = 0 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  rw [fuelValueD2_of_fold_sub CexD (fun _ => false) 2 spend 0 DPos.M
    (by decide) (by decide) (by decide) (by omega)]
  have hma : movesAbove CexD (val_lower 1) DPos.M = [DPos.X, DPos.S] := by decide
  have hS : fuelValueD2 CexD (fun _ => false) 2 spend 0 DPos.S = 0 := by
    simp only [fuelValueD2]
    rw [if_neg (by decide), if_neg (by decide)]
    rfl
  rw [hma, if_neg (by simp)]
  simp only [foldMax]
  rw [fuelValueD2_of_capture CexD (fun _ => false) 2 spend 0 DPos.X
    (by decide) (by decide), hS]
  omega

/-- No mate can be launched FROM the moveless stalemate `S`. -/
theorem cexD_S_not_mating (k : Nat) : ¬ ForcedMate CexD k DPos.S := by
  intro h
  cases h with
  | mate hkg hm hleg hmate =>
    rw [show CexD.moves DPos.S = [] from rfl] at hm
    cases hm
  | step hkg hm hleg hnt hreply =>
    rw [show CexD.moves DPos.S = [] from rfl] at hm
    cases hm

/-- The masked node is not mating at any index either: `X` is illegal,
and `S` is neither checkmated nor nonterminal. -/
theorem cexD_M_not_mating (k : Nat) : ¬ ForcedMate CexD k DPos.M := by
  intro h
  cases h with
  | @mate _ _ m hkg hm hleg hmate =>
    rw [show CexD.moves DPos.M = [DPos.X, DPos.S] from rfl] at hm
    have hm' : m = DPos.X ∨ m = DPos.S := by simpa using hm
    rcases hm' with rfl | rfl
    · exact absurd hleg (by decide)
    · exact absurd hmate.2 (by decide)
  | @step _ _ m hkg hm hleg hnt hreply =>
    rw [show CexD.moves DPos.M = [DPos.X, DPos.S] from rfl] at hm
    have hm' : m = DPos.X ∨ m = DPos.S := by simpa using hm
    rcases hm' with rfl | rfl
    · exact absurd hleg (by decide)
    · exact absurd hnt (by decide)

/-- ...nor mated at any index: the stalemate escape is a legal reply
from which no mate exists.  `M` is a DRAW of the ruleless game, and
`cexD_fuel_M1` prices it in the mated band at depth 1. -/
theorem cexD_M_not_mated (k : Nat) : ¬ ForcedlyMated CexD k DPos.M := by
  intro h
  cases h with
  | inl hcm => exact absurd hcm.1 (by decide)
  | inr h' => exact cexD_S_not_mating k (h'.2 DPos.S (by decide) (by decide))

/-- The eventual side of the disposition, ON the countermodel: from
the effective bound `D0 = C*N + C + 6 = 10` the once-masked node is
classified correctly -- strictly inside the band, as befits a draw of
the ruleless game -- for every edge-cost selector.  Since `c01915f`
depth 1 agrees with it (`cexD_fuel_M1` is `0`), so this pair is no
longer a lie/truth contrast but a consistency check: the eventual
bound and the frontier now say the same thing about this node. -/
theorem cexD_M_eventually_classified (spend : CexD.Pos → Nat → CexD.Pos → Nat)
    (D : Nat) (hD : 12 ≤ D) :
    -MATE_LOWER < fuelValueD2 CexD (fun _ => false) 2 spend D DPos.M ∧
      fuelValueD2 CexD (fun _ => false) 2 spend D DPos.M < MATE_LOWER := by
  have h := (eventual_classification_fuel_finite CexD (fun _ => false) 2 spend
    (by omega) cexD_floor DPos.M cexD_M_endsWithin (by decide) (by decide)
    D (by omega)).2.2
  refine h ?_ ?_
  · rintro ⟨k, hk⟩
    exact cexD_M_not_mating k hk
  · rintro ⟨k, hk⟩
    exact cexD_M_not_mated k hk

end Sunfish
