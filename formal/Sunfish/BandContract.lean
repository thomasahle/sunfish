/-
Can the two EXACT king-capture clauses of `Searcher.bound`'s docstring be
weakened to band membership?

    - our own king already captured: r = -MATE_UPPER.          (a)
    - if depth >= 1:
        - if the opponent king capturable: r = MATE_UPPER      (b)

The proposed weakening is `r <= -MATE_LOWER` for (a) and `MATE_LOWER <= r`
for (b); the engine-side prize is deleting the mate-band normalization from
the producer, `yield move, MATE_UPPER if val >= MATE_LOWER else val` becoming
`yield move, val` (a king capture's `pos.value` is about 60000, comfortably
inside `[MATE_LOWER, MATE_UPPER]`).

**Answer: no.**  Boundedness already gives `r <= MATE_UPPER` for free
(`boundD2_bounded`), so clause (b) is exactly the extra fact `MATE_UPPER <= r`
-- and `MATE_UPPER <= r` is the ONLY half any consumer uses.  The proposed
band `MATE_LOWER <= r` is the half that was free.  The weakening therefore
deletes the whole content of the clause, and each of four consumers breaks:

1. **`BoundSpec` itself** (`capture_report_must_fail_high`).  At a
   king-capturable node the declared value is `MATE_UPPER`
   (`nullValueD2_of_capture`), so no spec-valid report can fail low.  A band
   report may (`band_report_can_fail_low`), and a fail-low report is STORED as
   an upper bound -- the crossed entry the table forbids.
2. **The `live` bit** (`bandLeaf_correction_misses`).  `live |= move is not
   None and score > -MATE_UPPER` calls a move legal exactly when its yield
   clears the fold identity; an illegal move's yield is `-r`, and only `r =
   MATE_UPPER` keeps it AT the identity (`yield_at_identity_iff_exact`).  With
   a band report the yield is about `-60000`, `live` goes true at a mated or
   stalemated node, the `if depth and not live` gate never opens and the exact
   terminal value is never assigned.
3. **`storedMoveLegal`** (`storedMoveLegal_band_refuted`).  Its proof is
   "an illegal move's yield is `-MATE_UPPER`, which cannot reach an in-band
   `gamma`".  Under the band the yield is `-r >= -MATE_UPPER + 1`, which
   reaches every `gamma <= -r`: an illegal move can fail high and be written
   to `tp_move` as the score witness -- the engine's own answer.
4. **The fold identity** (`sentinel_is_fold_identity`).  `LOSS = -MATE_UPPER`
   is the seed of every fold in the development; an exact illegal yield is
   therefore NEUTRAL in the max.  A band yield is not, and clamps a
   sufficiently deep mate score up to about `-60000`
   (`bandYield_clamps_mate_score`).

Measured on the shipped engine, both branches of the same experiment:
`yield move, val` alone fails 104 of the 149 `tests/test_terminal_bench.py`
cases (the stalemate witness `8/3p4/3P4/3P4/3P4/1pb5/1Nk5/K7 w` returns -59977
instead of 0, gamma-dependently, leaving the crossed entry
`Entry(lower=0, upper=-200)`); with the live test relaxed to `score >
-MATE_LOWER` all 149 pass again, but `test_qs_stratified_contract`,
`test_legality_oracle_vs_python_chess` and
`test_king_capture_keeps_exact_sentinel` still fail, and soundness then rests
on a numeric margin no proof pins (`storedMoveLegal_band_refuted`).

**Why the mate-distance zone is different.**  Mated values are already
non-exact in-band values, `-MATE_LOWER - depth * EVAL_ROUGHNESS`, and they are
safe.  The distinction is what the consumer does with them: every consumer of
the distance zone is an ORDER consumer (`max`, `>=`, the driver's bracket), and
order is all the zone promises.  The two band EDGES are the development's only
IDENTITY consumers -- `score > -MATE_UPPER` and `bound(child, MATE_UPPER, 0) ==
MATE_UPPER` compare for equality with a reserved token.  Variation is free
where the reader compares magnitudes and fatal where the reader compares
identities; the golf moves a value off a token.

Nothing here changes the shipped search.  The file records what the two
clauses buy, so the next agent to eye them can read the answer instead of
re-deriving it.
-/

import Sunfish.CappedMove

namespace Sunfish

/-! ### The band contract, as a predicate -/

/-- The proposed weakening of clause (b): the report is somewhere in the
positive mate band, rather than exactly at its top. -/
def BandReport (r : Int) : Prop := MATE_LOWER ≤ r ∧ r ≤ MATE_UPPER

/-- The proposed weakening of clause (a). -/
def BandReportNeg (r : Int) : Prop := -MATE_UPPER ≤ r ∧ r ≤ -MATE_LOWER

/-- Clause (b) implies its weakening -- the direction that is never in
doubt. -/
theorem bandReport_of_exact : BandReport MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  exact ⟨by omega, by omega⟩

theorem bandReportNeg_of_exact : BandReportNeg (-MATE_UPPER) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  exact ⟨by omega, by omega⟩

/-! ### Which half of the clause the consumers use

Every consumer reads the child report `r` through the yield `-r` and asks
whether it stayed at the fold identity `LOSS = -MATE_UPPER`.  Boundedness
(`boundD2_bounded`) already gives `r ≤ MATE_UPPER`, so that question is
precisely "is `r` exactly `MATE_UPPER`". -/

/-- A one-move loop is a plain `max`: both branches of the cutoff test
return the same thing, so the cutoff never changes a singleton fold. -/
theorem searchMoves_singleton {α : Type _} (gamma : Int) (f : α → Int)
    (m : α) (b : Int) : searchMoves gamma f [m] b = max b (f m) := by
  simp only [searchMoves]
  split <;> rfl

/-- **The half the fold consumes.**  Under boundedness, "this child's yield
stayed at the fold identity" and "this child reported the exact sentinel" are
the SAME statement.  `live` reads the left side; clause (b) supplies the
right. -/
theorem yield_at_identity_iff_exact {r : Int} (hb : r ≤ MATE_UPPER) :
    -r ≤ LOSS ↔ r = MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  omega

/-- Dually for clause (a): the searched king-capture child hands its parent
the full `MATE_UPPER` only if it reported the exact `-MATE_UPPER`. -/
theorem kingGone_yield_exact_iff {r : Int} (hb : -MATE_UPPER ≤ r) :
    MATE_UPPER ≤ -r ↔ r = -MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  omega

/-- **The band is the half that was already free.**  It admits reports whose
yield sits a whole zone above the fold identity -- 60000 is the shipped
magnitude of `pos.value` on a king capture. -/
theorem band_leaves_yield_above_identity :
    ∃ r : Int, BandReport r ∧ LOSS < -r := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  exact ⟨60000, ⟨by omega, by omega⟩, by omega⟩

/-! ### The docstring's own bracket, and the two forms the model uses -/

/-- **The docstring and `BoundSpec` say the same thing.**  The docstring
splits on where `s*` sits ("if gamma > s* ... if gamma <= s* ..."), the proved
spec splits on where `r` sits; each derives the other, so the shipped wording
is the theorem and not a paraphrase of it. -/
theorem boundSpec_iff_docstring (gamma r s : Int) :
    ((gamma ≤ r → r ≤ s) ∧ (r < gamma → s ≤ r)) ↔
      ((s < gamma → s ≤ r ∧ r < gamma) ∧ (gamma ≤ s → gamma ≤ r ∧ r ≤ s)) := by
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨fun hs => ?_, fun hs => ?_⟩ <;>
      by_cases hr : gamma ≤ r
    · have := h1 hr; omega
    · have := h2 (by omega); omega
    · have := h1 hr; omega
    · have := h2 (by omega); omega
  · rintro ⟨h1, h2⟩
    refine ⟨fun hr => ?_, fun hr => ?_⟩ <;>
      by_cases hs : gamma ≤ s
    · have := h2 hs; omega
    · have := h1 (by omega); omega
    · have := h2 hs; omega
    · have := h1 (by omega); omega

/-- And `WindowReport`, the disjunctive form the capped-null transport
lemmas are stated in, is the same predicate once more. -/
theorem windowReport_iff_boundSpec (gamma r s : Int) :
    WindowReport gamma r s ↔ ((gamma ≤ r → r ≤ s) ∧ (r < gamma → s ≤ r)) := by
  unfold WindowReport
  constructor
  · rintro (⟨h1, h2⟩ | ⟨h1, h2⟩) <;> exact ⟨fun _ => by omega, fun _ => by omega⟩
  · rintro ⟨h1, h2⟩
    by_cases hr : gamma ≤ r
    · exact Or.inr ⟨hr, h1 hr⟩
    · exact Or.inl ⟨by omega, h2 (by omega)⟩

/-! ### Consumer 1: the fail-soft bracket itself -/

/-- At a king-capturable node the declared value is `MATE_UPPER`
(`nullValueD2_of_capture`, `negamaxD2_of_capture`, `fuelValueD2C_of_capture`),
so `BoundSpec`'s fail-low branch demands `MATE_UPPER ≤ r`: **any spec-valid
report at such a node fails high.**  This is upstream of every mate constant:
it is the docstring's own bracket. -/
theorem capture_report_must_fail_high {gamma r : Int} (hg : gamma ≤ MATE_UPPER)
    (hspec : r < gamma → MATE_UPPER ≤ r) : gamma ≤ r := by omega

/-- The band admits reports the bracket forbids.  The witness is the whole
open half of the band: at `gamma = MATE_UPPER` every report below the top
fails low, and a fail-low report is what the table stores as an UPPER bound --
below the value a later fail-high stores as a lower bound.  That is the
crossing `tests/test_terminal_bench.py` reproduces. -/
theorem band_report_can_fail_low :
    ∃ gamma r : Int, -MATE_UPPER < gamma ∧ gamma ≤ MATE_UPPER ∧
      BandReport r ∧ r < gamma := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  exact ⟨MATE_UPPER, MATE_LOWER, by omega, by omega, ⟨by omega, by omega⟩, by omega⟩

/-! ### Consumer 2: `live`, and the terminal override

`termFix2` fires on `S = LOSS ∧ allIllegalB p`, and the model's own reading of
the first conjunct is the code's `not live`: "no searched real yield ever beat
the sentinel".  The countermodel below is the smallest game that separates the
exact clause from its band weakening, with everything else -- the position, the
window, the move list, the oracle scan -- held fixed. -/

/-- The depth-0 leaf of the shipped reference (`boundD2''`), with the exact
king-capture branch replaced by a parameter.  `kcLeafExact` below shows that
`r = MATE_UPPER` is the shipped leaf on the nose. -/
def kcLeaf (G : QSGame) (r : Int) (p : G.Pos) : Int :=
  if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
  else if hasKingCapture G.toNullGame.toGame p = true then r
  else G.eval p

/-- At `r = MATE_UPPER` the parametrized leaf IS the shipped one. -/
theorem kcLeafExact (G : QSGame) (guard : G.Pos → Bool) (p : G.Pos) (gamma : Int) :
    kcLeaf G MATE_UPPER p = boundD2'' G guard 0 p gamma := rfl

/-- Whatever `r` is, the leaf keeps clause (a): a kingless node reports the
sentinel.  Only clause (b) is under test. -/
theorem kcLeaf_kingGone (G : QSGame) (r : Int) (p : G.Pos)
    (h : G.eval p ≤ -MATE_LOWER) : kcLeaf G r p = -MATE_UPPER := by
  simp only [kcLeaf]; rw [if_pos h]

/-- The countermodel's positions: `BP` is stalemated (its one pseudo-move
`BM` is illegal, and `BP` is not in check), `BM` can capture the king, `BK`
is the kingless position after the capture. -/
inductive BPos where
  | BP | BM | BK
  deriving DecidableEq

open BPos in
/-- Every fidelity hypothesis of the development holds here: the king
capture is valued in the mate band (`KingCaptureValHigh`), no move value is
below -192 (`ValFloor`), every both-kings evaluation is quiet (`EvalQuiet`),
and the pass is the identity, so `BP` is not in check.  The only thing under
test is what a king-capturable node reports. -/
def CexB : QSGame where
  Pos := BPos
  moves := fun x => match x with
    | BP => [BM]
    | BM => [BK]
    | BK => []
  eval := fun x => match x with
    | BK => -60000
    | _ => 0
  pass := fun x => x
  val := fun _ m => match m with
    | BK => 60000
    | _ => 0

theorem cexB_valFloor : ValFloor CexB 192 := by
  intro p; cases p <;> decide

theorem cexB_kingCaptureValHigh : KingCaptureValHigh CexB := by
  intro p; cases p <;> decide

theorem cexB_evalQuiet : EvalQuiet CexB.toNullGame.toGame := by
  intro p; cases p <;> decide

theorem cexB_bounded : Bounded CexB.toNullGame.toGame := by
  intro p; cases p <;> decide

/-- `BP` is oracle-terminal and not in check: a genuine STALEMATE, whose
exact value is 0 at every depth. -/
theorem cexB_stalemate :
    allIllegalB CexB BPos.BP = true ∧
    inCheckB CexB.toNullGame BPos.BP = false ∧
    terminalValue CexB 1 BPos.BP = 0 := by
  refine ⟨by decide, by decide, by decide⟩

/-- At `gamma = 0` the one pseudo-move is SEARCHED -- futility does not fire,
so this is not the `stratLeaf_needs_futility` hole reappearing; the parent
really does consume the child's report. -/
theorem cexB_searched : searchedAt CexB 1 0 BPos.BP = [BPos.BM] := rfl

/-- **The exact clause: the correction fires.**  The illegal move's yield is
the fold identity, `S = LOSS`, and the override assigns the exact draw. -/
theorem exactLeaf_correction_fires :
    kcLeaf CexB MATE_UPPER BPos.BM = MATE_UPPER ∧
    searchMoves 0 (fun m => -(kcLeaf CexB MATE_UPPER m))
      (searchedAt CexB 1 0 BPos.BP) LOSS = LOSS ∧
    termFix2 CexB 1 LOSS LOSS BPos.BP = 0 := by
  refine ⟨by decide, by decide, by decide⟩

/-- **The band clause: the correction misses.**  Same game, same window, same
oracle scan; the child reports 60000 instead of 69290 -- legal under
`BandReport` -- the fold lands at -60000, `S ≠ LOSS` is the code's `live`
going true on an ILLEGAL move, the override cannot fire, and a stalemate is
reported as a six-queen loss.  (The shipped engine measures -59977 on
`8/3p4/3P4/3P4/3P4/1pb5/1Nk5/K7 w`: the same number, less the position's
static score.) -/
theorem bandLeaf_correction_misses :
    BandReport 60000 ∧
    kcLeaf CexB 60000 BPos.BM = 60000 ∧
    searchMoves 0 (fun m => -(kcLeaf CexB 60000 m))
      (searchedAt CexB 1 0 BPos.BP) LOSS = -60000 ∧
    termFix2 CexB 1 (-60000) (-60000) BPos.BP = -60000 ∧
    (-60000 : Int) ≠ terminalValue CexB 1 BPos.BP := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  refine ⟨⟨by omega, by omega⟩, by decide, by decide, by decide, by decide⟩

/-- **The gate fires exactly at the exact clause.**  On the countermodel the
correction is available for one value of the report and no other: the
implication "band ⟹ terminal correction still works" fails for every report
below the top of the band, not just for the witness above. -/
theorem correction_fires_iff_exact (r : Int) (hb : r ≤ MATE_UPPER) :
    searchMoves 0 (fun m => -(kcLeaf CexB r m))
        (searchedAt CexB 1 0 BPos.BP) LOSS = LOSS ↔ r = MATE_UPPER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hleaf : kcLeaf CexB r BPos.BM = r := by
    simp only [kcLeaf]
    rw [if_neg (by decide), if_pos (by decide)]
  rw [cexB_searched, searchMoves_singleton, hleaf]
  constructor
  · intro h; omega
  · intro h; omega

/-! ### Consumer 3: `tp_move` legality

`storedMoveLegal` says a fail-high real yield certifies its move legal, and
its whole proof is: the yield of an illegal move is exactly `-MATE_UPPER`,
which no in-band `gamma` can reach.  Isolated, that step is one `omega`. -/

/-- The exact clause, as `storedMoveLegal` consumes it. -/
theorem illegal_yield_cannot_cut {gamma : Int} (hg : -MATE_UPPER < gamma) :
    ¬ (gamma ≤ -MATE_UPPER) := by omega

/-- **Under the band it can.**  A report of 60000 at an illegal child gives
the parent -60000, which is a fail-high for every `gamma ≤ -60000` -- and a
real-move fail-high at positive depth is written to `tp_move`, which is the
move `go_loop` plays.  The engine's own bracket keeps `|gamma|` below about
58606 today, so this is a margin and not a live bug; the margin is unproven,
tuning-dependent (it shrinks as `EVAL_ROUGHNESS` grows) and is exactly what
the exact clause replaced with a structural fact. -/
theorem storedMoveLegal_band_refuted :
    ∃ gamma r : Int, -MATE_UPPER < gamma ∧ gamma ≤ MATE_UPPER ∧
      BandReport r ∧ gamma ≤ -r := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  exact ⟨-60000, 60000, by omega, by omega, ⟨by omega, by omega⟩, by omega⟩

/-! ### Consumer 4: the sentinel is the fold's identity element -/

/-- An exact illegal yield is NEUTRAL: it can never raise `best`, at any
node, at any depth.  That is why illegal pseudo-moves cost the fold nothing
and why a mate score survives a node full of them. -/
theorem sentinel_is_fold_identity {b : Int} (hb : LOSS ≤ b) : max b LOSS = b := by
  omega

/-- A band yield is not neutral, and the value it clamps is a mate score: a
mated node at unspent depth 806 is worth `-MATE_LOWER - 806 * EVAL_ROUGHNESS =
-60013`, which an illegal sibling reporting 60000 would raise to -60000.  The
mate-distance ordering (`dtm_optimal`, `leastMate_value_separation`) is lost
at exactly the depth where the two zones meet. -/
theorem bandYield_clamps_mate_score :
    ∃ (b r : Int), LOSS ≤ b ∧ BandReport r ∧ b < -r ∧
      b = -MATE_LOWER - 806 * EVAL_ROUGHNESS := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hER : EVAL_ROUGHNESS = 15 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  exact ⟨-MATE_LOWER - 806 * EVAL_ROUGHNESS, 60000, by omega,
    ⟨by omega, by omega⟩, by omega, rfl⟩

/-! ### The reservation the whole scheme rests on

`-MATE_UPPER` is a TOKEN, not a score: the code hands it out only at the
kingless early return, and every other return path is kept strictly above it.
The two candidates for a leak are the depth-0 leaf (the finalizer is gated on
`if depth`, so a mated QS node is never rewritten) and the finalizer's own
mate value (which is negated twice on the way up).  Both are closed here. -/

/-- **Leak candidate 1, closed: the QS leaf.**  A depth-0 node whose own king
is on the board never returns the reserved sentinel -- and the reason is not
the correction (which does not run at depth 0) but the early return one line
above it: the stand-pat floor `pos.score` is already above `-MATE_LOWER`,
because `pos.score <= -MATE_LOWER` returned. -/
theorem qsLeaf_reserves_sentinel (G : QSGame) (guard : G.Pos → Bool)
    (p : G.Pos) (gamma : Int) (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    -MATE_UPPER < boundD2'' G guard 0 p gamma := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  simp only [boundD2'']
  rw [if_neg hkg]
  split <;> omega

/-- **Leak candidate 2, closed: the finalizer.**  `mate = max(1 - MATE_UPPER,
-MATE_LOWER - depth * EVAL_ROUGHNESS)` -- the floor is one point above the
sentinel, and that one point is the reservation.  A node valued `-MATE_UPPER`
would reach its GRANDPARENT as exactly the illegal-move sentinel. -/
theorem terminalValue_reserves_sentinel (G : QSGame) (d : Nat) (p : G.Pos) :
    -MATE_UPPER < terminalValue G d p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have h := (terminalValue_bounds G d p).1
  omega

/-- The margin is a whole zone at every reachable depth, not one point: the
floor binds only past unspent depth 1424, three orders of magnitude beyond the
driver's `range(1, 1000)`.  Below it the mated value is `-MATE_LOWER - d *
EVAL_ROUGHNESS`, which clears the sentinel by more than 21000. -/
theorem terminalValue_margin (G : QSGame) (d : Nat) (p : G.Pos)
    (hd : (d : Int) * EVAL_ROUGHNESS ≤ 21366) :
    -MATE_UPPER + 1 ≤ terminalValue G d p := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have h := (terminalValue_bounds G d p).1
  omega

/-! ### The reservation at the RECURSION level

The two endpoint lemmas close the two ways `-MATE_UPPER` could be
MANUFACTURED at a node.  Neither says it cannot arrive from BELOW, and
that is the load-bearing claim: `-MATE_UPPER` is what an illegal move's
child reports, so `score > -MATE_UPPER` is a legality test one ply up
only if no LEGAL line ever produces it.

That is a statement about the recursion, and it comes in a dual pair,
because the two directions feed each other:

* **reserved below** -- a node whose own king is on the board returns
  strictly above `-MATE_UPPER`;
* **reserved above** -- a node that cannot capture the enemy king
  returns strictly below `MATE_UPPER`.

The lower arm needs a searched legal child to report below
`MATE_UPPER`, so that its negation lifts the accumulator off the
sentinel.  The upper arm needs every searched child to report above
`-MATE_UPPER`, so that no negation reaches the positive token.  Each
holds one ply down by the other, so they are proven together by
induction on depth, with `qsLeaf_reserves_sentinel` and
`terminalValue_reserves_sentinel` as the base and terminal cases.

**The lower arm is what `c01915f` unblocked.**  Its live case has to
produce a legal move that reaches the real accumulator, and under the
pre-`c01915f` sloped admission a legal move could be missing from
`searchedAt` outright -- filtered at remaining depth 1 with nothing
else to displace the accumulator, which is exactly `CexE`'s and
`CexF`'s phantom.  The model then could not prove the arm, and the
audit recorded it as blocked.  Now `movesAbove_pos` puts every legal
move in the admitted list at every positive depth, and the only thing
that can still keep one out of the REAL accumulator is futility -- which
pays for itself: a futile move's own stand-pat enters `futTerm`, and a
live child's stand-pat is below `MATE_LOWER` under `EvalQuiet`, so the
accumulator is lifted either way.

Premises are fidelity only: a move-value floor inside the band
(`ValFloor`, tables -192) and the quiet-eval bound (`EvalQuiet`).  No
chess-side premise, no window premise beyond the docstring's own
`1 - MATE_UPPER < gamma <= MATE_UPPER`. -/

/-- **The dual reservation, reference consumer.**  Proven as a pair
because neither arm is available without the other one ply down. -/
theorem boundD2''_reserves_pair (G : QSGame) (guard : G.Pos → Bool)
    {B : Int} (hF : ValFloor G B) (hBMU : B ≤ MATE_UPPER)
    (hQ : EvalQuiet G.toNullGame.toGame) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int), -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      ((¬ (G.eval p ≤ -MATE_LOWER)) → -MATE_UPPER < boundD2'' G guard d p gamma) ∧
      (hasKingCapture G.toNullGame.toGame p = false →
        boundD2'' G guard d p gamma < MATE_UPPER) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  intro d
  induction d with
  | zero =>
    intro p gamma hg1 hg2
    refine ⟨fun hkg => qsLeaf_reserves_sentinel G guard p gamma hkg, fun hcap => ?_⟩
    simp only [boundD2'']
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · rw [if_pos hkg]; omega
    · rw [if_neg hkg, if_neg (by rw [hcap]; exact fun h => Bool.noConfusion h)]
      have := hQ p hkg
      omega
  | succ d ih =>
    intro p gamma hg1 hg2
    -- the child window, and the child liveness both arms consume
    have hw1 : -MATE_UPPER < 1 - gamma := by omega
    have hw2 : 1 - gamma ≤ MATE_UPPER := by omega
    have hchild : ∀ m, hasKingCapture G.toNullGame.toGame p = false → m ∈ G.moves p →
        ¬ (G.eval m ≤ -MATE_LOWER) := by
      intro m hcapf hm hev
      have hk : hasKingCapture G.toNullGame.toGame p = true :=
        (hasKingCapture_iff G.toNullGame.toGame p).mpr ⟨m, hm, hev⟩
      rw [hcapf] at hk
      exact Bool.noConfusion hk
    have hSge := searchMoves_ge_init gamma
      (fun m => -(boundD2'' G guard d m (1 - gamma))) (searchedAt G (d + 1) gamma p) LOSS
    have hfutge := futTerm_ge_LOSS G (d + 1) gamma p
    have hfutlt := futTerm_lt_gamma G (d + 1) gamma p hg1
    rw [boundD2''_succ]
    constructor
    · -- **reserved below**
      intro hkg
      rw [if_neg hkg]
      by_cases hcap : hasKingCapture G.toNullGame.toGame p = true
      · rw [if_pos hcap]; omega
      · rw [if_neg hcap]
        have hcapf : hasKingCapture G.toNullGame.toGame p = false := by
          cases h : hasKingCapture G.toNullGame.toGame p with
          | false => rfl
          | true => exact absurd h hcap
        split
        · -- the null cut: a cut is at least `gamma`, and a verified terminal
          -- routes through the finalizer
          next hcut =>
            split
            · exact terminalValue_reserves_sentinel G (d + 1) p
            · have := hcut.2; omega
        · -- the fold
          next hcut =>
            simp only [termFix2]
            split
            · exact terminalValue_reserves_sentinel G (d + 1) p
            · next hfire =>
              by_cases hai : allIllegalB G p = true
              · -- the scan says terminal, so the override declined only because
                -- a searched move already beat the sentinel
                by_cases hSeq : searchMoves gamma
                    (fun m => -(boundD2'' G guard d m (1 - gamma)))
                    (searchedAt G (d + 1) gamma p) LOSS = LOSS
                · exact absurd ⟨hSeq, hai⟩ hfire
                · omega
              · -- a legal move exists; it is ADMITTED (`movesAbove_pos`), and
                -- either searched or futile -- both lift the accumulator
                have hai' : allIllegalB G p = false := by
                  cases h : allIllegalB G p with
                  | false => rfl
                  | true => exact absurd h hai
                obtain ⟨m, hm, hleg⟩ := exists_legal_of_allIllegalB_false hai'
                have hmlive := hchild m hcapf hm
                have hmma : m ∈ movesAbove G (val_lower (d + 1)) p := by
                  rw [movesAbove_pos G hF hBMU (d + 1) (by omega) p]; exact hm
                by_cases hfut : futileAt G (d + 1) gamma p m = true
                · have hmem : m ∈ (movesAbove G (val_lower (d + 1)) p).filter
                      (fun x => futileAt G (d + 1) gamma p x) :=
                    List.mem_filter.mpr ⟨hmma, hfut⟩
                  have hle : -(G.eval m) ≤ futTerm G (d + 1) gamma p := by
                    simp only [futTerm]
                    exact foldMax_le_of_mem _ _ _ m hmem
                  have := hQ m hmlive
                  omega
                · have hmem : m ∈ searchedAt G (d + 1) gamma p :=
                    List.mem_filter.mpr ⟨hmma, by simp [hfut]⟩
                  have hup := (ih m (1 - gamma) hw1 hw2).2 hleg
                  have hlow := searchMoves_ge_min_of_mem gamma
                    (fun x => -(boundD2'' G guard d x (1 - gamma)))
                    (searchedAt G (d + 1) gamma p) LOSS m hmem
                  simp only at hlow
                  omega
    · -- **reserved above**
      intro hcap
      have hall : ∀ m ∈ searchedAt G (d + 1) gamma p,
          -(boundD2'' G guard d m (1 - gamma)) ≤ MATE_UPPER - 1 := by
        intro m hm
        have hmm : m ∈ G.moves p :=
          movesAbove_subset G _ p m (searchedAt_subset G (d + 1) gamma p m hm)
        have hlow := (ih m (1 - gamma) hw1 hw2).1 (hchild m hcap hmm)
        omega
      have hSle := searchMoves_le_max gamma
        (fun m => -(boundD2'' G guard d m (1 - gamma))) (searchedAt G (d + 1) gamma p)
        LOSS (MATE_UPPER - 1) hall (by omega)
      by_cases hkg : G.eval p ≤ -MATE_LOWER
      · rw [if_pos hkg]; omega
      · rw [if_neg hkg, if_neg (by rw [hcap]; exact fun h => Bool.noConfusion h)]
        split
        · -- the cut path is capped by the mate-band decline the gate carries
          next hcut =>
            have hband : nullVerify'' G
                (-(boundD2'' G guard (d + 1 - 3) (G.pass p) (1 - gamma))) gamma
                (boundD2'' G guard (d + 1 - 3) (G.pass p) (1 - MATE_LOWER)) p
                  < MATE_LOWER := by
              have huse := hcut.1
              simp only [useD2'', Bool.and_eq_true, decide_eq_true_eq] at huse
              exact huse.2 hcut.2
            split
            · have := (terminalValue_bounds G (d + 1) p).2; omega
            · omega
        · next hcut =>
            have hnull : nullPartD2'' G guard
                (-(boundD2'' G guard (d + 1 - 3) (G.pass p) (1 - gamma)))
                (boundD2'' G guard (d + 1 - 3) (G.pass p) (1 - MATE_LOWER))
                (d + 1) p gamma ≤ MATE_UPPER - 1 := by
              simp only [nullPartD2'']
              split
              · next huse =>
                have hlt : ¬ (gamma ≤ nullVerify'' G
                    (-(boundD2'' G guard (d + 1 - 3) (G.pass p) (1 - gamma))) gamma
                    (boundD2'' G guard (d + 1 - 3) (G.pass p) (1 - MATE_LOWER)) p) :=
                  fun h => hcut ⟨huse, h⟩
                omega
              · omega
            simp only [termFix2]
            split
            · have := (terminalValue_bounds G (d + 1) p).2; omega
            · omega

/-- **Leak candidate 3, closed: the recursion.**  A node whose own king
is on the board never returns the reserved sentinel, at ANY depth.
This is the fact `score > -MATE_UPPER` is a legality test BY, and the
docstring's "Only a searched real move sets `live`" is its consumer. -/
theorem boundD2''_reserves_sentinel (G : QSGame) (guard : G.Pos → Bool)
    {B : Int} (hF : ValFloor G B) (hBMU : B ≤ MATE_UPPER)
    (hQ : EvalQuiet G.toNullGame.toGame)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_UPPER)
    (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    -MATE_UPPER < boundD2'' G guard d p gamma :=
  (boundD2''_reserves_pair G guard hF hBMU hQ d p gamma hg1 hg2).1 hkg

/-- The positive token is reserved the same way: only a node that can
actually take the king reports `MATE_UPPER`, which is the docstring's
"an exact `MATE_UPPER` proves a king capture" -- here as a property of
the recursion rather than of one branch. -/
theorem boundD2''_reserves_positive (G : QSGame) (guard : G.Pos → Bool)
    {B : Int} (hF : ValFloor G B) (hBMU : B ≤ MATE_UPPER)
    (hQ : EvalQuiet G.toNullGame.toGame)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_UPPER)
    (hcap : hasKingCapture G.toNullGame.toGame p = false) :
    boundD2'' G guard d p gamma < MATE_UPPER :=
  (boundD2''_reserves_pair G guard hF hBMU hQ d p gamma hg1 hg2).2 hcap

/-- **The legality test, as the code uses it.**  `score > -MATE_UPPER`
at a child report is EXACTLY "the move was legal": the sentinel is
returned at a king-gone node and nowhere else.  One direction is
`boundD2''_kingGone`, the other is the recursion-level reservation. -/
theorem boundD2''_live_iff_legal (G : QSGame) (guard : G.Pos → Bool)
    {B : Int} (hF : ValFloor G B) (hBMU : B ≤ MATE_UPPER)
    (hQ : EvalQuiet G.toNullGame.toGame)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_UPPER) :
    (-MATE_UPPER < boundD2'' G guard d p gamma) ↔ ¬ (G.eval p ≤ -MATE_LOWER) := by
  constructor
  · intro h hev
    rw [boundD2''_kingGone G guard d p gamma hev] at h
    omega
  · exact boundD2''_reserves_sentinel G guard hF hBMU hQ d p gamma hg1 hg2

/-- **The production consumer inherits it.**  `boundKCX''` is the
shipped shape; `production''_eq_reference''` transports both arms
verbatim. -/
theorem boundKCX''_reserves_sentinel (G : QSGame) (guard : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) (hV : KingCaptureValHigh G)
    (hCF : CaptureFirst G)
    {B : Int} (hF : ValFloor G B) (hBMU : B ≤ MATE_UPPER)
    (hQ : EvalQuiet G.toNullGame.toGame)
    (d : Nat) (p : G.Pos) (gamma : Int)
    (hg1 : -MATE_UPPER < gamma) (hg2 : gamma ≤ MATE_UPPER) :
    ((¬ (G.eval p ≤ -MATE_LOWER)) → -MATE_UPPER < boundKCX'' G guard d p gamma) ∧
    (hasKingCapture G.toNullGame.toGame p = false →
      boundKCX'' G guard d p gamma < MATE_UPPER) := by
  rw [production''_eq_reference'' G guard hB hV hCF d p gamma hg1 hg2]
  exact boundD2''_reserves_pair G guard hF hBMU hQ d p gamma hg1 hg2

/-! ### The direction of the mate value, and the alternative that inverts it

`-MATE_LOWER - depth * EVAL_ROUGHNESS` with `depth` the UNSPENT depth makes the
mater's yield `MATE_LOWER + depth * EVAL_ROUGHNESS`: rising in unspent depth,
so a mate found with more depth left -- a FASTER mate -- outscores a slower
one, which is what `dtm_optimal` and `leastMate_value_separation` turn into
"the engine plays the shortest mate and defends the longest".  The floated
alternative `-MATE_UPPER + depth * EVAL_ROUGHNESS` is monotone the other way,
and inverts both halves. -/

/-- The shipped mated value, as a function of unspent depth. -/
def matedShipped (d : Int) : Int := max (1 - MATE_UPPER) (-MATE_LOWER - d * EVAL_ROUGHNESS)

/-- The floated alternative. -/
def matedAlt (d : Int) : Int := -MATE_UPPER + d * EVAL_ROUGHNESS

/-- **Shipped: deeper detection is worse for the mated side**, so its
negation -- what the mating side scores -- is better for a faster mate. -/
theorem matedShipped_anti {d e : Int} (h : d ≤ e) : matedShipped e ≤ matedShipped d := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold matedShipped EVAL_ROUGHNESS
  omega

/-- **The alternative inverts it**: the mated side gains by having MORE depth
left, so the mating side's yield `MATE_UPPER - d * EVAL_ROUGHNESS` falls as the
mate gets faster -- the shuffling failure mode, machine-checked. -/
theorem matedAlt_mono {d e : Int} (h : d ≤ e) : matedAlt d ≤ matedAlt e := by
  have hMU : MATE_UPPER = 69290 := rfl
  unfold matedAlt EVAL_ROUGHNESS
  omega

/-- The inversion, at the two distances the driver actually separates: a mate
in one (found with 3 unspent plies) must outscore a mate in two (found with 1),
and under the alternative it does not. -/
theorem matedAlt_inverts_preference :
    -matedShipped 1 < -matedShipped 3 ∧ -matedAlt 3 < -matedAlt 1 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold matedShipped matedAlt EVAL_ROUGHNESS
  omega

/-- And the alternative spends the reservation margin down to a single
`EVAL_ROUGHNESS`: at the shallowest depth the finalizer runs, its value is one
step above the sentinel, where the shipped form clears it by 21352 -- the whole
mate-distance zone. -/
theorem matedAlt_margin_is_one_step :
    matedAlt 1 - (-MATE_UPPER) = EVAL_ROUGHNESS ∧
    matedShipped 1 - (-MATE_UPPER) = 21352 := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  unfold matedAlt matedShipped EVAL_ROUGHNESS
  exact ⟨by omega, by omega⟩

/-! ### Band cohabitation: mate distance and king capture share the band

The mate-distance zone `[-MATE_LOWER - d*EVAL_ROUGHNESS, -MATE_LOWER]` and its
positive mirror live INSIDE the same band whose edges are the king-capture
tokens.  They never get mixed up, and the separation is structural in four
places rather than numeric in one.

* **Yield-layer exclusion.**  A yielded score is one of exactly three species,
  and none of them can land in the open gap `(MATE_LOWER, MATE_UPPER)`: the
  producer's normalization emits `MATE_UPPER` or a raw `pos.value` (below
  `MATE_LOWER`, `val_lower_lt_ML`); the QS stand-pat emits `pos.score` (below
  `MATE_LOWER`, `EvalQuiet`); and the pass emits at most `pos.score +
  EVAL_ROUGHNESS` under a `abs(pos.score) < 750` guard.  Gap values enter only
  as SEARCHED scores, `min(cap, -child)`, which carry a move.
* **Searched scores stay below the token** (`searched_score_below_MU`): the
  finalizer's floor `1 - MATE_UPPER` on the child makes `-child ≤ MATE_UPPER -
  1`, so `MATE_UPPER` is unreachable on any searched path.  Hence
  MU-exactness has KING-CAPTURE PROVENANCE (`MU_provenance`): a fold that
  reaches the token names a yield that was the token.
* **The negative token does appear as a null yield** -- at an in-check node the
  pass is king-capturable, so `-bound(pass) = -MATE_UPPER` exactly
  (`nullAtMateD2`, another consumer of clause (b)).  That is harmless because
  the null yields with `move = None`, and the live test's `move is not None`
  conjunct is what makes it so.
* **The table cannot smuggle a token across contexts.**  Its key is
  `(pos, depth)` and capture-ness is a function of `pos` alone, so an entry
  holding `MATE_UPPER` belongs to a capturable position permanently;
  `tp_score` is cleared at the top of every `search()`; and the seed entry
  `Entry(-MATE_UPPER, MATE_UPPER)` can never be RETURNED, because the
  driver's window range excludes both comparisons
  (`tt_sentinel_defaults_never_returned`). -/

/-- **Searched scores never reach the positive token.**  `score = min(cap,
-child)` and the terminal floor gives `1 - MATE_UPPER ≤ child`. -/
theorem searched_score_below_MU {cap child : Int} (hfloor : 1 - MATE_UPPER ≤ child) :
    min cap (-child) < MATE_UPPER := by omega

/-- **MU-exactness ⟺ king-capture provenance** (the positive-side dual of the
sentinel reservation): a loop that reports the token names a yield that WAS
the token, and by the previous lemma no searched yield can be. -/
theorem MU_provenance {α : Type _} (gamma : Int) (f : α → Int) (ms : List α)
    (h : MATE_UPPER ≤ searchMoves gamma f ms LOSS) :
    ∃ m ∈ ms, MATE_UPPER ≤ f m := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  exact searchMoves_exists_ge gamma f ms LOSS MATE_UPPER (by omega) h

/-- **The seed entry is unreturnable.**  `Entry(-MATE_UPPER, MATE_UPPER)` is
the table's default, and the two probe tests are `entry.lower >= gamma` and
`entry.upper < gamma`; the driver's own window range refutes both, so a
sentinel can never leave the table it never entered. -/
theorem tt_sentinel_defaults_never_returned {gamma : Int}
    (h1 : -MATE_UPPER < gamma) (h2 : gamma ≤ MATE_UPPER) :
    ¬ (gamma ≤ -MATE_UPPER) ∧ ¬ (MATE_UPPER < gamma) := ⟨by omega, by omega⟩

/-- **The window range is self-reproducing under the null-window flip**, with
exactly one point to spare: `gamma = MATE_UPPER` sends `1 - gamma = 1 -
MATE_UPPER` to the child, which is the mate floor and the first legal window
above the forbidden sentinel.  The half-open interval in the docstring is the
unique one closed under `gamma ↦ 1 - gamma`. -/
theorem window_flip_preserves_range {gamma : Int}
    (h1 : -MATE_UPPER < gamma) (h2 : gamma ≤ MATE_UPPER) :
    -MATE_UPPER < 1 - gamma ∧ 1 - gamma ≤ MATE_UPPER := ⟨by omega, by omega⟩

/-- **The intentional asymmetry, in one statement.**  At depth ≤ 4 a SEARCHED
move is clamped strictly below the mate band, while a king capture bypasses the
clamp entirely and reports the token: the two never overlap, so a shallow
search can neither invent a mate nor hide a capture. -/
theorem shallowCap_and_capture_disjoint (static gain : Int) (depth : Nat)
    (hband : CapInBand static gain depth) {v : Int} (hv : MATE_LOWER ≤ v) :
    shallowMoveCap static gain depth < MATE_LOWER ∧
    shallowMoveCap static gain depth < producedScore v :=
  ⟨shallowMoveCap_below_positiveMate static gain depth hband, by
    have hMU : MATE_UPPER = 69290 := rfl
    have hML : MATE_LOWER = 47923 := rfl
    have h := shallowMoveCap_below_positiveMate static gain depth hband
    rw [producedScore_capture v hv]
    omega⟩

/-! ### What the weakening WOULD have been enough for

Not every consumer of the producer's normalization needs the exact value.
These are the ones a band contract would carry unchanged -- recorded so the
census is two-sided. -/

/-- The band restatement of `producedScore_exact_capture`: it holds, it is
just not what anyone needs.  (`producedScore` is `MATE_UPPER if val >=
MATE_LOWER else val`, the line the golf would delete.) -/
theorem producedScore_capture_band (G : QSGame) (hHi : HighValIsKingCapture G)
    (p m : G.Pos) (hm : m ∈ G.moves p) (hgain : MATE_LOWER ≤ G.val p m) :
    BandReport (producedScore (G.val p m)) ∧ G.eval m ≤ -MATE_LOWER := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  refine ⟨?_, hHi p m hm hgain⟩
  rw [producedScore_capture (G.val p m) hgain]
  exact ⟨by omega, by omega⟩

/-- Mate-band completeness (`forcedMate_leaf_fuelValueD2` and the whole
`MateDepth` spine) consumes the capture branch only through `MATE_LOWER ≤ ·`:
its step is this, and the band supplies it. -/
theorem mateBand_step_needs_only_band {r acc : Int} (h : BandReport r) :
    MATE_LOWER ≤ max acc r := by
  obtain ⟨h1, _⟩ := h
  omega

/-- The shallow cap's separation lemmas (`shallowMoveCap_below_positiveMate`,
`cappedMove_preserves_negativeMate`) are already band-shaped: they only ever
compare against `±MATE_LOWER`, never against the token.  Since the clamp was
dropped they carry the material invariant `CapInBand` instead of a syntactic
ceiling. -/
theorem cap_separation_is_band_shaped (static gain : Int) (depth : Nat)
    (hband : CapInBand static gain depth) :
    shallowMoveCap static gain depth < MATE_LOWER :=
  shallowMoveCap_below_positiveMate static gain depth hband

end Sunfish
