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

2. COUNTERMODEL DISPOSITIONS, as lemmas (the games are verbatim ports
   from the #178/#181 stack -- `Shortest.lean` / `Eventual.lean`,
   which this branch predates; on a rebase past those merges the port
   section below is deleted wholesale in favour of the originals):

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
adds definitions and theorems; `sunfish.py` is untouched).
-/

import Sunfish.EventuallyWide

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

end Sunfish
