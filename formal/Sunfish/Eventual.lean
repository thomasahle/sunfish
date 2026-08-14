/-
EVENTUAL CLASSIFICATION: how much of the trichotomy's premise ledger the
"for all large enough depths" weakening can pay off.

`Classification.lean`'s `eventual_classification` states the trichotomy
at EVERY depth and spends four premises: `ValFloor` and `EvalQuiet`
(fidelity, tables), `NoZugzwang` (chess, layer 2) and
`NoMaskedMobility` (chess, layer 2, the model-side stand-in for the
engine fix in #171).  The question this module answers is whether
weakening the conclusion from "at every depth" to "at every depth from
some `D0` on" -- which is all the driver's `range(1, 1000)` deepening
ever needs -- lets the last of those be dropped.

**The hope, and why it was reasonable.**  Masking is a FRONTIER effect.
The QS val-filter admits `val >= QS - depth * QS_A`, so at remaining
depth 2 the threshold is already -240, below the shipped tables' move
value floor of -192, and every legal move survives
(`filter_identity_off_frontier`).  Masking therefore exists ONLY at
remaining depth 1 -- roughly `D` plies from the root.  A phantom found
there is `D` negations away from the root, so it looked like it should
either be absorbed as `D` grows (`CexF`'s phantom is: it dissolves at
depth 3) or arrive with nothing left to claim.

**Why the hope fails, in one sentence.**  A masked node does not report
a shallow mate; it reports the OFF-LADDER sentinel
(`maskedFrontier_value`: the filtered fold has no admitted legal move,
so nothing displaces the initial `LOSS` accumulator and the parent
negates it to `MATE_UPPER`).  The mate ladder decays one
`EVAL_ROUGHNESS` rung per ply of unspent depth; `MATE_UPPER` decays not
at all.  So a phantom's contribution survives an arbitrary number of
negations undiminished, and the frontier RENEWS it at every horizon: at
depth `D` the phantom sits at ply `D - 1`, at depth `D + 2` at ply
`D + 1`.  Absorbing the old one buys nothing when a new one arrives.

`CexE` (section 3) is that argument turned into a machine-checked
countermodel, and it is not a near miss: the root is a draw whose
declared value is `MATE_UPPER` at every even depth and `-MATE_UPPER` at
every odd one.  The classification does not merely fail to settle -- it
oscillates between the two false extremes forever.  Neither acyclicity
(the model is a strictly increasing chain, no position ever repeats)
nor a read-time clamp at `MATE_UPPER - 1` touches it.  The frontier-tail
variant of `Classification.lean` Part B values the same root at an
honest `0` at every depth, so what retires `NoMaskedMobility` is the
engine change and nothing else.

**What the weakening does buy, and it is not nothing.**  Both
COMPLETENESS arms -- a real forced mate reaches the band, a real forced
loss reaches the mated band -- never needed the frontier premise at all
(`eventual_completeness_without_frontier`).  `NoMaskedMobility` pays
for exactly one thing: HONESTY, the "neither" arm.  A phantom can only
invent a mate that is not there; it can never hide one that is.

**And `NearMaximalChoice` is free at this granularity** (section 2).
The driver stops bisecting at `upper - lower <= EVAL_ROUGHNESS = 15`,
so the shipped root can settle for a value within 15 of the best.  The
declared value's range avoids the whole corridor between the static
score range and the band edge (`nullValueD2_offCorridor`: an induction
whose only cases are the sentinel, the terminal ladder, the static
eval, and closure of the corridor complement under negation and `max`),
and for the shipped tables that corridor is 32,486 points wide --
`shipped_band_gap_wide`, more than two THOUSAND stopping tolerances.
So no 15-point slack can move a value across a band edge
(`bandOf_eq_of_slack`), and a near-maximal choice lands in the same
band as an exact argmax (`nearMaximal_band_exact`).  This is the band
level counterpart of `Shortest.lean`'s rung-level result: there parity
refunds the tolerance one rung at a time, here the tolerance is simply
three orders of magnitude too small to matter.  Both readings rest on
the same decision -- one `EVAL_ROUGHNESS` of mate tempo per ply (#172).

Nothing here changes `sunfish.py`; `CexE` is a model, and the
frontier-tail variant it vindicates is the proof-first `negamaxD2t`
that Part B of `Classification.lean` already carries.
-/

import Sunfish.Classification
import Sunfish.Shortest

namespace Sunfish

/-! ## 1. The frontier is local; the sentinel it prints is not -/

/-- Every child of a node with no king capture still has its own king:
the parent's `hasKingCapture` scan is exactly that test. -/
theorem eval_child_live (G : QSGame) {p m : G.Pos}
    (hcap : hasKingCapture G.toNullGame.toGame p = false) (hm : m ∈ G.moves p) :
    ¬ (G.eval m ≤ -MATE_LOWER) := by
  intro h
  have hk : hasKingCapture G.toNullGame.toGame p = true :=
    (hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, h⟩
  rw [hcap] at hk
  exact Bool.noConfusion hk

/-- **(a) Masking is confined to remaining depth 1.**  `val_lower 2 =
40 - 2 * 140 = -240` is already below the tables' move-value floor of
-192, so from remaining depth 2 up the QS filter is the identity on the
legal move list and no move can be masked.  A restatement of
`movesAbove_all` at the constants, named because the whole question of
this module is where masking can live. -/
theorem filter_identity_off_frontier (G : QSGame) (hF : ValFloor G 192)
    {d : Nat} (hd : 2 ≤ d) (p : G.Pos) :
    movesAbove G (val_lower d) p = G.moves p :=
  movesAbove_all G d p (allAboveB_of_floor G hF d p (val_lower_le_neg_floor d hd))

/-- **The phantom is the sentinel, exactly.**  At a node that HAS a
legal move but whose every depth-1-admitted move is illegal, the fold
runs over refuted moves only: each contributes `-(MATE_UPPER) = LOSS`,
the pass term is `LOSS` (the null option is off below remaining depth
3), and nothing displaces the initial accumulator.  So the node reports
`-MATE_UPPER` and its parent reports `MATE_UPPER` -- a value strictly
above every rung the mate ladder can produce, and one that does not
decay with distance from the horizon.

This is why the depth-decay argument that retires a phantom mate
cannot retire a phantom sentinel: there is no rung to outrank. -/
theorem maskedFrontier_value (G : QSGame) (guard : G.Pos → Bool) (p : G.Pos)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER))
    (hcap : hasKingCapture G.toNullGame.toGame p = false)
    (hai : allIllegalB G p = false)
    (hmask : ∀ m ∈ movesAbove G (val_lower 1) p,
      hasKingCapture G.toNullGame.toGame m = true) :
    nullValueD2 G guard 1 p = -MATE_UPPER := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hcap' : ¬ (hasKingCapture G.toNullGame.toGame p = true) := by rw [hcap]; simp
  show nullValueD2 G guard (0 + 1) p = -MATE_UPPER
  rw [nullValueD2_of_fold G guard 0 p hkg hcap' hai, show nullTermD2 G guard 0 p = LOSS from by
    simp [nullTermD2]]
  have hall : ∀ m ∈ movesAbove G (val_lower (0 + 1)) p, -(nullValueD2 G guard 0 m) ≤ LOSS := by
    intro m hm
    have hmm := movesAbove_subset G _ p m hm
    have hcm := hmask m hm
    rw [nullValueD2_of_capture G guard 0 m (eval_child_live G hcap hmm) hcm]
    omega
  have h1 := foldMax_le (fun m => -(nullValueD2 G guard 0 m))
    (movesAbove G (val_lower (0 + 1)) p) LOSS hall (Int.le_refl _)
  have h2 := foldMax_ge_init (fun m => -(nullValueD2 G guard 0 m))
    (movesAbove G (val_lower (0 + 1)) p) LOSS
  omega

/-- **(b) Completeness never needed the frontier premise.**  Both
"finding" arms of `eventual_classification` -- a forced mate in `k` puts
the declared value in the band from depth `k + 1` on, a forced loss puts
it at or below `-MATE_LOWER` from depth `k + 2` on -- go through
`forcedMate_complete_band` and `forcedlyMated_negamaxD2_band`, which
spend only `ValFloor` and `NoZugzwang`.  A phantom adds spurious HIGH
values to the side that finds it; it can invent a mate that is not
there, never hide one that is.

So the premise ledger is sharper than the trichotomy's statement
suggests: `NoMaskedMobility` buys the honesty arm and nothing else. -/
theorem eventual_completeness_without_frontier (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hZ : NoZugzwang G guard) (p : G.Pos)
    (hcapf : hasKingCapture G.toNullGame.toGame p = false) :
    (∀ k, ForcedMate G k p → ∀ D, k + 1 ≤ D → MATE_LOWER ≤ nullValueD2 G guard D p) ∧
    (∀ k, ForcedlyMated G k p → ∀ D, k + 2 ≤ D → nullValueD2 G guard D p ≤ -MATE_LOWER) := by
  refine ⟨fun k hFM D hD => forcedMate_complete_band G guard hF hZ hFM D hD, fun k hFMd D hD => ?_⟩
  rw [nullValue_eq_realValue_of_noZugzwang G guard hZ D p]
  exact forcedlyMated_negamaxD2_band G hF hcapf hFMd D hD

/-! ## 2. The driver's stopping tolerance is free at band granularity

`NearMaximalChoice` is the honest form of the engine's move rule:
`search` bisects only until `upper - lower <= EVAL_ROUGHNESS`, so the
move left in `tp_move` is guaranteed only to be within 15 of the best.
`Shortest.lean` pays for that slack one RUNG at a time, and needs
parity to refund it.  At CLASSIFICATION granularity no refund is
needed, because the question is only which of three bands the value
falls in and the bands are three orders of magnitude apart.

The statement that makes this a theorem rather than an appeal to
magnitudes is that the declared value's range avoids the corridor
between the static score range and the band edge entirely.  That is an
induction with four cases and no chess content: the sentinel, the
terminal ladder, the static eval, and the fact that the corridor's
complement is closed under negation and `max`.
-/

/-- **EvalBand B**: outside the king-gone zone the static score stays
within `±B`.  The shipped tables give `B = EvalBounds.evalBound =
15437`, the same table arithmetic `EvalQuiet` reads one-sidedly; the
link from board strings to tables is unmodelled exactly as for
`Bounded`. -/
def EvalBand (G : Game) (B : Int) : Prop :=
  ∀ p, ¬ (G.eval p ≤ -MATE_LOWER) → -B ≤ G.eval p ∧ G.eval p ≤ B

/-- **OffCorridor B v**: `v` is not in the no-man's-land between the
static range and either band edge.  Every value the search can declare
is off-corridor (`nullValueD2_offCorridor`), and that is what makes a
15-point tolerance unable to move a value across a band edge. -/
def OffCorridor (B v : Int) : Prop :=
  v ≤ -MATE_LOWER ∨ (-B ≤ v ∧ v ≤ B) ∨ MATE_LOWER ≤ v

theorem offCorridor_neg {B v : Int} (h : OffCorridor B v) : OffCorridor B (-v) := by
  rcases h with h | ⟨h1, h2⟩ | h
  · exact Or.inr (Or.inr (by omega))
  · exact Or.inr (Or.inl ⟨by omega, by omega⟩)
  · exact Or.inl (by omega)

theorem offCorridor_max {B v w : Int} (hv : OffCorridor B v) (hw : OffCorridor B w) :
    OffCorridor B (max v w) := by
  rcases Int.le_total v w with h | h
  · rw [show max v w = w from by omega]; exact hw
  · rw [show max v w = v from by omega]; exact hv

theorem offCorridor_LOSS {B : Int} : OffCorridor B LOSS := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  exact Or.inl (by omega)

theorem foldMax_offCorridor {α : Type _} (w : α → Int) {B : Int} :
    ∀ (ms : List α) (acc : Int), (∀ m ∈ ms, OffCorridor B (w m)) → OffCorridor B acc →
      OffCorridor B (foldMax w ms acc) := by
  intro ms
  induction ms with
  | nil => intro acc _ hacc; exact hacc
  | cons a ms ih =>
    intro acc hall hacc
    exact ih (max acc (w a)) (fun x hx => hall x (List.mem_cons_of_mem a hx))
      (offCorridor_max hacc (hall a (List.mem_cons_self a ms)))

/-- The terminal correction is off-corridor at every depth: checkmate
lands on the ladder at or below `-MATE_LOWER` (the clamp keeps it above
the illegal-move sentinel), stalemate is exactly `0`. -/
theorem terminalValue_offCorridor (G : QSGame) {B : Int} (hB : 0 ≤ B) (d : Nat) (p : G.Pos) :
    OffCorridor B (terminalValue G d p) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hnn : (0 : Int) ≤ (d : Int) := Int.ofNat_nonneg d
  have hmul : (0 : Int) ≤ (d : Int) * EVAL_ROUGHNESS :=
    Int.mul_nonneg hnn (by decide)
  unfold terminalValue
  split
  · exact Or.inl (by omega)
  · exact Or.inr (Or.inl ⟨by omega, hB⟩)

/-- **The declared value never enters the corridor.**  Four cases, none
of them about chess: the two sentinel branches are `±MATE_UPPER`, the
terminal branch is the ladder or `0`, depth 0 is the static eval, and
the fold is a `max` of negated off-corridor values seeded with an
off-corridor pass term.  The corridor's complement is closed under both
operations, so the induction is immediate. -/
theorem nullValueD2_offCorridor (G : QSGame) (guard : G.Pos → Bool) {B : Int}
    (hB : 0 ≤ B) (hE : EvalBand G.toNullGame.toGame B) :
    ∀ (d : Nat) (p : G.Pos), OffCorridor B (nullValueD2 G guard d p) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [nullValueD2_kingGone G guard d p hkg]
      unfold OffCorridor; omega
    · by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [nullValueD2_of_capture G guard d p hkg hcap]
        unfold OffCorridor; omega
      · cases d with
        | zero =>
          have hval : nullValueD2 G guard 0 p = G.eval p := by
            simp only [nullValueD2]; rw [if_neg hkg, if_neg hcap]
          rw [hval]
          have := hE p hkg
          unfold OffCorridor; omega
        | succ d' =>
          by_cases hai : allIllegalB G p = true
          · rw [nullValueD2_of_allIllegal G guard d' p hkg hcap hai]
            exact terminalValue_offCorridor G hB (d' + 1) p
          · have hai' : allIllegalB G p = false := by
              cases h : allIllegalB G p with
              | false => rfl
              | true => exact absurd h hai
            rw [nullValueD2_of_fold G guard d' p hkg hcap hai']
            refine foldMax_offCorridor _ _ _
              (fun m _ => offCorridor_neg (ih d' (by omega) m)) ?_
            unfold nullTermD2
            split
            · split
              · exact offCorridor_max offCorridor_LOSS
                  (offCorridor_neg (ih (d' + 1 - 3) (by omega) (G.pass p)))
              · exact offCorridor_LOSS
            · exact offCorridor_LOSS

/-- The classification a value carries: `-1` mated, `0` neither, `1`
mating.  Exactly the three arms of `eventual_classification`, read off
the value by its band. -/
def bandOf (v : Int) : Int :=
  if v ≤ -MATE_LOWER then -1 else if MATE_LOWER ≤ v then 1 else 0

/-- **The band-granularity lemma.**  If `v` is at most `w` plus the
driver's whole stopping tolerance, and neither sits in the corridor,
then `v`'s band is at most `w`'s.  The engine MINIMISES the value it
moves to, so this is the direction that matters: the near-maximal
choice is never classified worse than the alternative it may have
displaced. -/
theorem bandOf_mono_of_slack {B v w : Int} (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    (hv : OffCorridor B v) (hw : OffCorridor B w) (h : v ≤ w + EVAL_ROUGHNESS) :
    bandOf v ≤ bandOf w := by
  have hML : MATE_LOWER = 47923 := rfl
  have hER : EVAL_ROUGHNESS = 15 := rfl
  unfold OffCorridor at hv hw
  unfold bandOf
  split <;> (try split) <;> (try split) <;> (try split) <;> omega

/-- Two values the driver's tolerance can conflate lie in the same
band. -/
theorem bandOf_eq_of_slack {B v w : Int} (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    (hv : OffCorridor B v) (hw : OffCorridor B w)
    (h1 : v ≤ w + EVAL_ROUGHNESS) (h2 : w ≤ v + EVAL_ROUGHNESS) :
    bandOf v = bandOf w :=
  Int.le_antisymm (bandOf_mono_of_slack hgap hv hw h1) (bandOf_mono_of_slack hgap hw hv h2)

section
set_option maxRecDepth 4096

/-- The shipped tables clear the gap condition. -/
theorem shipped_band_gap : EvalBounds.evalBound + EVAL_ROUGHNESS < MATE_LOWER := by decide

/-- ... and not narrowly: the corridor is 32,486 points wide, more than
two thousand stopping tolerances.  This is the quantitative content of
"`NearMaximalChoice` is irrelevant at classification level". -/
theorem shipped_band_gap_wide :
    2000 * EVAL_ROUGHNESS < MATE_LOWER - EvalBounds.evalBound := by decide

end

/-- **`NearMaximalChoice` is exact at band granularity.**  The move the
shipped root leaves in `tp_move` is classified no worse than any
admitted alternative -- so replacing the idealised exact argmax by the
driver's actual stopping behaviour cannot change which of the three
classes the chosen move belongs to.  No parity, no rungs, no chess
premise; only the width of the corridor. -/
theorem nearMaximal_band_exact (G : QSGame) (guard : G.Pos → Bool) {B : Int}
    (hB : 0 ≤ B) (hE : EvalBand G.toNullGame.toGame B)
    (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    {d : Nat} {ch : G.Pos → G.Pos} (hch : NearMaximalChoice G guard d ch)
    {p m : G.Pos} (hai : allIllegalB G p = false)
    (hm : m ∈ movesAbove G (val_lower (d + 1)) p) :
    bandOf (nullValueD2 G guard d (ch p)) ≤ bandOf (nullValueD2 G guard d m) :=
  bandOf_mono_of_slack hgap (nullValueD2_offCorridor G guard hB hE d (ch p))
    (nullValueD2_offCorridor G guard hB hE d m) ((hch p hai).2 m hm)

/-- The reading that matters for play: a near-maximal choice never
misses a mate.  If some admitted move reaches a child valued at or
below `-MATE_LOWER`, so does the move the driver actually keeps. -/
theorem nearMaximal_keeps_mate (G : QSGame) (guard : G.Pos → Bool) {B : Int}
    (hB : 0 ≤ B) (hE : EvalBand G.toNullGame.toGame B)
    (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    {d : Nat} {ch : G.Pos → G.Pos} (hch : NearMaximalChoice G guard d ch)
    {p m : G.Pos} (hai : allIllegalB G p = false)
    (hm : m ∈ movesAbove G (val_lower (d + 1)) p)
    (hmate : nullValueD2 G guard d m ≤ -MATE_LOWER) :
    nullValueD2 G guard d (ch p) ≤ -MATE_LOWER := by
  have hb := nearMaximal_band_exact G guard hB hE hgap hch hai hm
  have hoc := nullValueD2_offCorridor G guard hB hE d (ch p)
  unfold bandOf at hb
  rw [if_pos hmate] at hb
  by_cases h : nullValueD2 G guard d (ch p) ≤ -MATE_LOWER
  · exact h
  · rw [if_neg h] at hb
    split at hb <;> omega

/-- **The driver's own stopping rule, read at band granularity.**
`search` returns when `upper - lower <= EVAL_ROUGHNESS`; both ends
bracket the same off-corridor value, so the converged bracket cannot
straddle a band edge.  Whatever the bisection settles on, the
classification it reports is the one the value has. -/
theorem driver_stop_band_stable {B lower upper : Int}
    (hgap : B + EVAL_ROUGHNESS < MATE_LOWER)
    (hl : OffCorridor B lower) (hu : OffCorridor B upper)
    (hle : lower ≤ upper) (hstop : upper - lower ≤ EVAL_ROUGHNESS) :
    bandOf lower = bandOf upper := by
  have hER : (0 : Int) ≤ EVAL_ROUGHNESS := by decide
  exact bandOf_eq_of_slack hgap hl hu (by omega) (by omega)

/-! ## 3. `CexE`: the eventual weakening does not retire the frontier

The countermodel.  An infinite chain `C 0 -> C 1 -> C 2 -> ...` in
which EVERY node is masked in the sense of `NoMaskedMobility`: each
`C n` generates an illegal `X` (valued 0, so admitted at every depth)
and the legal continuation `C (n+1)` (valued -150, so filtered at the
depth-1 threshold `val_lower 1 = -100` and admitted from depth 2 on).
`X` loses the king to `K`; nothing else happens anywhere.

The root is a DRAW in the strongest possible sense -- no forced mate at
any budget, for either side (`cexE_no_forcedMate`,
`cexE_no_forcedlyMated`), because the only legal move from any node is
the next link in the chain and nobody is ever checkmated or stalemated.

Its declared value, however:

```text
D:        0     1     2     3     4     5    ...
value:    0   -MU    +MU   -MU   +MU   -MU   ...
```

`cexE_ladder`.  Odd depths report `-MATE_UPPER` -- "I am mated"; even
depths from 2 on report `MATE_UPPER` -- "I mate".  The value never
enters the band at any depth past 0, so there is no `D0` beyond which
the classification is right; it is not merely unsettled but WRONG in
both directions, alternately, forever.

Three things this settles that the `CexF` / `CexD` countermodels did
not.

* **Depth does not help, and this is the eventual statement, not the
  fixed-depth one.**  `CexF`'s phantom dissolves at depth 3, which is
  exactly why the eventual weakening looked promising.  Here the
  phantom is renewed: at horizon `D` it sits at `C (D-1)`, at horizon
  `D+2` at `C (D+1)`.  The frontier travels with the search and there
  is always a masked node on it.
* **Acyclicity does not help.**  `cexE_acyclic`: the only legal move
  from `C n` is `C (n+1)`, so the chain index strictly increases and no
  position ever repeats.  No repetition rule, no well-founded descent,
  no draw-by-cycle argument touches this.
* **A read-time clamp does not help.**  `cexE_clamp_no_help`: the
  reported value is exactly `MATE_UPPER`, so clamping the root read at
  `MATE_UPPER - 1` still leaves 69,289 -- far inside the mate band.
  Only refusing to trust the sentinel at all would work, and that is
  the engine change.

And one thing it settles in the other direction: the FRONTIER-TAIL
variant of `Classification.lean` Part B values this root at an honest
`0` at every depth (`cexE_t_honest`).  When every admitted move is
illegal it unfilters and the legal continuation is seen.  So `CexE`,
like `CexF`, becomes a positive test for the proposed engine change --
which is now the only route to retiring `NoMaskedMobility`.
-/

/-- `C n` is the chain; `X` is the illegal move admitted at the
frontier; `K` is the captured king that makes `X` illegal. -/
inductive EPos where
  | C : Nat → EPos
  | X : EPos
  | K : EPos
  deriving DecidableEq

open EPos in
/-- `C n -> {X, C (n+1)}` with `val (C n) X = 0` and
`val (C n) (C (n+1)) = -150`: the legal continuation drops more than
`QS_A - QS = 100` of table value, so it is masked at remaining depth 1
and admitted from remaining depth 2 on -- the shape `NoMaskedMobility`
forbids and `ValFloor 192` permits, repeated at every ply. -/
def CexE : QSGame where
  Pos := EPos
  moves := fun p => match p with
    | C n => [X, C (n + 1)]
    | X => [K]
    | K => []
  eval := fun p => match p with
    | K => -MATE_UPPER
    | _ => 0
  pass := fun p => p
  val := fun p m => match p, m with
    | C _, C _ => -150
    | X, K => MATE_LOWER
    | _, _ => 0

instance : DecidableEq CexE.Pos := inferInstanceAs (DecidableEq EPos)

theorem cexE_moves_C (n : Nat) : CexE.moves (EPos.C n) = [EPos.X, EPos.C (n + 1)] := rfl

theorem cexE_eval_C (n : Nat) : CexE.eval (EPos.C n) = 0 := rfl

theorem cexE_eval_X : CexE.eval EPos.X = 0 := rfl

theorem cexE_cap_C (n : Nat) :
    hasKingCapture CexE.toNullGame.toGame (EPos.C n) = false := rfl

theorem cexE_cap_X : hasKingCapture CexE.toNullGame.toGame EPos.X = true := by decide

theorem cexE_ai_C (n : Nat) : allIllegalB CexE (EPos.C n) = false := rfl

/-! ### Every fidelity premise holds -/

theorem cexE_floor : ValFloor CexE 192 := by
  intro p m _
  cases p <;> cases m <;>
    first
      | decide
      | exact (by decide : (-192 : Int) ≤ -150)
      | exact (by decide : (-192 : Int) ≤ 0)
      | exact (by decide : (-192 : Int) ≤ MATE_LOWER)

theorem cexE_quiet : EvalQuiet CexE.toNullGame.toGame := by
  intro p
  cases p <;>
    first
      | decide
      | exact fun _ => (by decide : (0 : Int) < MATE_LOWER)

theorem cexE_bounded : Bounded CexE.toNullGame.toGame := by
  intro p
  cases p <;>
    first
      | decide
      | exact (by decide : (-MATE_UPPER : Int) ≤ 0 ∧ (0 : Int) ≤ MATE_UPPER)

/-- The band premises of section 2 hold at every width: every live
static score in `CexE` is exactly `0`.  So the countermodel is not a
granularity failure -- the tolerance machinery is fully available and
still does not save the classification. -/
theorem cexE_evalBand {B : Int} (hB : 0 ≤ B) : EvalBand CexE.toNullGame.toGame B := by
  intro p hkg
  cases p with
  | C n => rw [cexE_eval_C]; exact ⟨by omega, hB⟩
  | X => rw [cexE_eval_X]; exact ⟨by omega, hB⟩
  | K => exact absurd (by decide : CexE.eval EPos.K ≤ -MATE_LOWER) hkg

/-- The null option is switched off throughout, so `NoZugzwang` is
vacuous: the failure has nothing to do with the pass. -/
theorem cexE_nozug : NoZugzwang CexE (fun _ => false) := by
  intro _ _ _ _ _ hg
  exact absurd hg (by simp)

/-- **No position repeats.**  The only legal move from `C n` is
`C (n+1)`, so the chain index strictly increases along every play: this
countermodel is a strictly descending tree, not a cycle. -/
theorem cexE_acyclic (n : Nat) (m : EPos) (hm : m ∈ CexE.moves (EPos.C n))
    (hleg : hasKingCapture CexE.toNullGame.toGame m = false) : m = EPos.C (n + 1) := by
  rw [cexE_moves_C] at hm
  rcases List.mem_cons.mp hm with rfl | hm'
  · rw [cexE_cap_X] at hleg; exact Bool.noConfusion hleg
  · rcases List.mem_cons.mp hm' with rfl | hm''
    · rfl
    · cases hm''

/-! ### The two folds -/

theorem cexE_term (d : Nat) (p : EPos) : nullTermD2 CexE (fun _ => false) d p = LOSS := by
  simp [nullTermD2]

theorem cexE_X (d : Nat) : nullValueD2 CexE (fun _ => false) d EPos.X = MATE_UPPER :=
  nullValueD2_of_capture CexE _ d EPos.X (by decide) (by decide)

/-- At the frontier the filter keeps only the illegal move. -/
theorem cexE_ma1 (n : Nat) : movesAbove CexE (val_lower 1) (EPos.C n) = [EPos.X] := by
  unfold movesAbove
  rw [cexE_moves_C]
  rfl

/-- One ply deeper it keeps everything (`filter_identity_off_frontier`
at the concrete constants). -/
theorem cexE_ma2 (d n : Nat) :
    movesAbove CexE (val_lower (d + 2)) (EPos.C n) = [EPos.X, EPos.C (n + 1)] := by
  rw [movesAbove_all CexE (d + 2) (EPos.C n)
    (allAboveB_of_floor CexE cexE_floor (d + 2) (EPos.C n)
      (val_lower_le_neg_floor (d + 2) (by omega))), cexE_moves_C]

/-- ... and the premise that does fail is exactly `NoMaskedMobility`. -/
theorem cexE_masked : ¬ NoMaskedMobility CexE := by
  intro h
  have hpre : ∀ m ∈ movesAbove CexE (val_lower 1) (EPos.C 0),
      hasKingCapture CexE.toNullGame.toGame m = true := by
    rw [cexE_ma1]
    intro m hm
    rcases List.mem_cons.mp hm with rfl | hm'
    · exact cexE_cap_X
    · cases hm'
  have hleg := h (EPos.C 0) hpre (EPos.C 1) (by rw [cexE_moves_C]; simp)
  rw [cexE_cap_C] at hleg
  exact Bool.noConfusion hleg

/-- **The phantom**, `maskedFrontier_value` at `C n`. -/
theorem cexE_C1 (n : Nat) : nullValueD2 CexE (fun _ => false) 1 (EPos.C n) = -MATE_UPPER := by
  show nullValueD2 CexE (fun _ => false) (0 + 1) (EPos.C n) = -MATE_UPPER
  rw [nullValueD2_of_fold CexE _ 0 (EPos.C n) (by rw [cexE_eval_C]; decide)
    (by rw [cexE_cap_C]; decide) (cexE_ai_C n), cexE_term, cexE_ma1]
  show max LOSS (-(nullValueD2 CexE (fun _ => false) 0 EPos.X)) = -MATE_UPPER
  rw [cexE_X]
  decide

/-- Off the frontier the recursion is honest, and the illegal `X`
contributes the sentinel from the other side. -/
theorem cexE_Cstep (d n : Nat) :
    nullValueD2 CexE (fun _ => false) (d + 2) (EPos.C n)
      = max (-MATE_UPPER) (-(nullValueD2 CexE (fun _ => false) (d + 1) (EPos.C (n + 1)))) := by
  show nullValueD2 CexE (fun _ => false) ((d + 1) + 1) (EPos.C n) = _
  rw [nullValueD2_of_fold CexE _ (d + 1) (EPos.C n) (by rw [cexE_eval_C]; decide)
    (by rw [cexE_cap_C]; decide) (cexE_ai_C n), cexE_term]
  show foldMax _ (movesAbove CexE (val_lower (d + 2)) (EPos.C n)) LOSS = _
  rw [cexE_ma2 d n]
  show max (max LOSS (-(nullValueD2 CexE (fun _ => false) (d + 1) EPos.X)))
      (-(nullValueD2 CexE (fun _ => false) (d + 1) (EPos.C (n + 1)))) = _
  rw [cexE_X]
  have hLOSS : LOSS = -MATE_UPPER := rfl
  rw [hLOSS]
  omega

/-- **The oscillation.**  Uniform in the chain position, because every
node of the chain has the same shape -- which is the whole point: the
frontier always finds a fresh masked node however deep it goes. -/
theorem cexE_ladder :
    ∀ k n, nullValueD2 CexE (fun _ => false) (2 * k + 1) (EPos.C n) = -MATE_UPPER
      ∧ nullValueD2 CexE (fun _ => false) (2 * k + 2) (EPos.C n) = MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  intro k
  induction k with
  | zero =>
    intro n
    refine ⟨cexE_C1 n, ?_⟩
    show nullValueD2 CexE (fun _ => false) (0 + 2) (EPos.C n) = MATE_UPPER
    rw [cexE_Cstep 0 n, cexE_C1 (n + 1)]
    omega
  | succ k ih =>
    have hodd : ∀ n, nullValueD2 CexE (fun _ => false) (2 * (k + 1) + 1) (EPos.C n)
        = -MATE_UPPER := by
      intro n
      have hidx : 2 * (k + 1) + 1 = (2 * k + 1) + 2 := by omega
      rw [hidx, cexE_Cstep (2 * k + 1) n, show (2 * k + 1) + 1 = 2 * k + 2 from by omega,
        (ih (n + 1)).2]
      omega
    intro n
    refine ⟨hodd n, ?_⟩
    have hidx : 2 * (k + 1) + 2 = (2 * k + 2) + 2 := by omega
    rw [hidx, cexE_Cstep (2 * k + 2) n, show (2 * k + 2) + 1 = 2 * (k + 1) + 1 from by omega,
      hodd (n + 1)]
    omega

/-! ### ... and no forced mate, in either direction -/

/-- The only legal move from `C n` is `C (n+1)`, which is never
checkmated, so the spec's `mate` leaf is unreachable and its `step`
constructor just walks down the chain with a smaller budget. -/
theorem cexE_no_forcedMate : ∀ k n, ¬ ForcedMate CexE k (EPos.C n) := by
  intro k
  induction k using Nat.strongRecOn with
  | _ k ih =>
    intro n h
    cases h with
    | mate hkg hm hleg hmate =>
      rw [cexE_moves_C] at hm
      rcases List.mem_cons.mp hm with rfl | hm'
      · rw [cexE_cap_X] at hleg; exact Bool.noConfusion hleg
      · rcases List.mem_cons.mp hm' with rfl | hm''
        · have hai := hmate.1; rw [cexE_ai_C] at hai; exact Bool.noConfusion hai
        · cases hm''
    | @step k' _ m hkg hm hleg hnt hreply =>
      rw [cexE_moves_C] at hm
      rcases List.mem_cons.mp hm with rfl | hm'
      · rw [cexE_cap_X] at hleg; exact Bool.noConfusion hleg
      · rcases List.mem_cons.mp hm' with rfl | hm''
        · refine ih k' (by omega) (n + 1 + 1) (hreply (EPos.C (n + 1 + 1)) ?_ (cexE_cap_C _))
          rw [cexE_moves_C]; exact List.mem_cons_of_mem _ (List.mem_cons_self _ _)
        · cases hm''

theorem cexE_no_forcedlyMated : ∀ k n, ¬ ForcedlyMated CexE k (EPos.C n) := by
  intro k n h
  cases h with
  | inl hcm => have hai := hcm.1; rw [cexE_ai_C] at hai; exact Bool.noConfusion hai
  | inr h =>
    refine cexE_no_forcedMate k (n + 1) (h.2 (EPos.C (n + 1)) ?_ (cexE_cap_C _))
    rw [cexE_moves_C]; exact List.mem_cons_of_mem _ (List.mem_cons_self _ _)

/-! ### The readings, at arbitrarily large depth -/

theorem cexE_mate_reading (D0 : Nat) :
    ∃ D, D0 ≤ D ∧ nullValueD2 CexE (fun _ => false) D (EPos.C 0) = MATE_UPPER :=
  ⟨2 * D0 + 2, by omega, (cexE_ladder D0 0).2⟩

theorem cexE_mated_reading (D0 : Nat) :
    ∃ D, D0 ≤ D ∧ nullValueD2 CexE (fun _ => false) D (EPos.C 0) = -MATE_UPPER :=
  ⟨2 * D0 + 1, by omega, (cexE_ladder D0 0).1⟩

/-- **A read-time clamp does not help.**  The phantom arrives as the
exact sentinel, so clamping the root read at `MATE_UPPER - 1` leaves
69,289 -- still 21,366 points inside the mate band. -/
theorem cexE_clamp_no_help (D0 : Nat) :
    ∃ D, D0 ≤ D ∧
      MATE_LOWER ≤ min (nullValueD2 CexE (fun _ => false) D (EPos.C 0)) (MATE_UPPER - 1) := by
  obtain ⟨D, hD, hval⟩ := cexE_mate_reading D0
  refine ⟨D, hD, ?_⟩
  rw [hval]
  decide

/-! ### The frontier-tail variant values the same root honestly -/

theorem cexE_admitted_masked (n : Nat) : allAdmittedIllegalB CexE 1 (EPos.C n) = true := by
  refine allAdmittedIllegalB_true_iff.mpr ?_
  rw [cexE_ma1]
  intro m hm
  rcases List.mem_cons.mp hm with rfl | hm'
  · exact cexE_cap_X
  · cases hm'

theorem cexE_admitted_open (d n : Nat) :
    allAdmittedIllegalB CexE (d + 2) (EPos.C n) = false :=
  allAdmittedIllegalB_false_of_legal
    (by rw [cexE_ma2 d n]; exact List.mem_cons_of_mem _ (List.mem_cons_self _ _))
    (cexE_cap_C (n + 1))

/-- Both branches of the t-recursion fold the same list: at the
frontier because the unfilter trigger fires, deeper because the filter
was already the identity. -/
theorem cexE_t_fold (d n : Nat) :
    nullValueD2t CexE (fun _ => false) (d + 1) (EPos.C n)
      = foldMax (fun m => -(nullValueD2t CexE (fun _ => false) d m))
          [EPos.X, EPos.C (n + 1)] LOSS := by
  have hterm : nullTermD2t CexE (fun _ => false) d (EPos.C n) = LOSS := by simp [nullTermD2t]
  cases d with
  | zero =>
    rw [nullValueD2t_of_masked CexE _ 0 (EPos.C n) (by rw [cexE_eval_C]; decide)
      (by rw [cexE_cap_C]; decide) (cexE_ai_C n) (cexE_admitted_masked n), hterm, cexE_moves_C]
  | succ d' =>
    rw [nullValueD2t_of_fold CexE _ (d' + 1) (EPos.C n) (by rw [cexE_eval_C]; decide)
      (by rw [cexE_cap_C]; decide) (cexE_ai_C n) (cexE_admitted_open d' n), hterm]
    show foldMax _ (movesAbove CexE (val_lower (d' + 2)) (EPos.C n)) LOSS = _
    rw [cexE_ma2 d' n]

/-- **`CexE` becomes a positive test for the frontier tail.**  The
t-variant reports the honest `0` at every depth and every chain
position: no phantom, no oscillation, correct classification. -/
theorem cexE_t_honest : ∀ (d n : Nat),
    nullValueD2t CexE (fun _ => false) d (EPos.C n) = 0 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d with
  | zero =>
    intro n
    simp only [nullValueD2t]
    rw [if_neg (by rw [cexE_eval_C]; decide), if_neg (by rw [cexE_cap_C]; decide), cexE_eval_C]
  | succ d ih =>
    intro n
    rw [cexE_t_fold d n]
    show max (max LOSS (-(nullValueD2t CexE (fun _ => false) d EPos.X)))
        (-(nullValueD2t CexE (fun _ => false) d (EPos.C (n + 1)))) = 0
    rw [ih (n + 1), show nullValueD2t CexE (fun _ => false) d EPos.X = MATE_UPPER from
      nullValueD2t_of_capture CexE _ d EPos.X (by decide) (by decide)]
    omega

/-! ## 4. The verdict -/

section
set_option maxRecDepth 4096

/-- **(c) The eventual trichotomy is FALSE without `NoMaskedMobility`.**

Stated against the strongest hypotheses available: every fidelity
premise (`ValFloor`, `EvalQuiet`, and section 2's two-sided `EvalBand`
with its gap condition), `NoZugzwang`, root legality, a live king, and
no forced mate for either side at any budget -- and the conclusion
weakened all the way to "from SOME depth on".  It still fails.

The shipped value function offers no route to the honesty arm.  What
retires the premise is the engine change (`cexE_t_honest`). -/
theorem eventual_classification_needs_frontier :
    ¬ (∀ (G : QSGame) (guard : G.Pos → Bool) (B : Int), 0 ≤ B →
        ValFloor G 192 → EvalQuiet G.toNullGame.toGame → EvalBand G.toNullGame.toGame B →
        B + EVAL_ROUGHNESS < MATE_LOWER → NoZugzwang G guard →
        ∀ p : G.Pos, hasKingCapture G.toNullGame.toGame p = false →
          ¬ (G.eval p ≤ -MATE_LOWER) →
          (∀ k, ¬ ForcedMate G k p) → (∀ k, ¬ ForcedlyMated G k p) →
          ∃ D0 : Nat, ∀ D, D0 ≤ D →
            -MATE_LOWER < nullValueD2 G guard D p ∧ nullValueD2 G guard D p < MATE_LOWER) := by
  intro h
  obtain ⟨D0, hD0⟩ := h CexE (fun _ => false) EvalBounds.evalBound (by decide)
    cexE_floor cexE_quiet (cexE_evalBand (by decide)) shipped_band_gap cexE_nozug
    (EPos.C 0) (cexE_cap_C 0) (by rw [cexE_eval_C]; decide)
    (fun k => cexE_no_forcedMate k 0) (fun k => cexE_no_forcedlyMated k 0)
  obtain ⟨D, hDge, hval⟩ := cexE_mate_reading D0
  have hlt := (hD0 D hDge).2
  rw [hval] at hlt
  exact absurd hlt (by decide)

/-- The countermodel's whole content in one statement: a drawn root
whose shipped-value classification is wrong in BOTH directions at
arbitrarily large depth, and right at every depth under the
frontier-tail variant. -/
theorem eventual_classification_verdict :
    (∀ k, ¬ ForcedMate CexE k (EPos.C 0)) ∧
    (∀ k, ¬ ForcedlyMated CexE k (EPos.C 0)) ∧
    (∀ D0, ∃ D, D0 ≤ D ∧ MATE_LOWER ≤ nullValueD2 CexE (fun _ => false) D (EPos.C 0)) ∧
    (∀ D0, ∃ D, D0 ≤ D ∧ nullValueD2 CexE (fun _ => false) D (EPos.C 0) ≤ -MATE_LOWER) ∧
    (∀ d n, nullValueD2t CexE (fun _ => false) d (EPos.C n) = 0) := by
  refine ⟨fun k => cexE_no_forcedMate k 0, fun k => cexE_no_forcedlyMated k 0, ?_, ?_,
    cexE_t_honest⟩
  · intro D0
    obtain ⟨D, hD, hval⟩ := cexE_mate_reading D0
    exact ⟨D, hD, by rw [hval]; decide⟩
  · intro D0
    obtain ⟨D, hD, hval⟩ := cexE_mated_reading D0
    exact ⟨D, hD, by rw [hval]; decide⟩

end

end Sunfish
