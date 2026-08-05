/-
HISTORICAL (commits 2c95ab0..7f9f164): the clamp documented here
shipped alongside re-search LMR and was removed together with it when
master switched to deterministic LMR (7f9f164) -- under single-function
bounds the clamp is a provable no-op (`clamp_noop_high`/`clamp_noop_low`
in `Sunfish/LmrDet.lean`).  Kept as the formal record: any future
gamma-dependent evaluation choice must reinstate BOTH this clamp and
the interval spec of `Sunfish/Lmr.lean`.

The clamped store, commit 2c95ab0 ("Never store contradictory
transposition entries"), sunfish.py lines 435-443 as of that commit:

    if best >= gamma:
        self.tp_score[...] = Entry(best, max(entry.upper, best))
    if best < gamma:
        self.tp_score[...] = Entry(min(entry.lower, best), best)

Context: under LMR the two sides of a `tp_score` entry are facts about
different value functions -- fail highs bound `Vhi`, fail lows bound `Vlo`
(`Sunfish/Lmr.lean`, `boundLmr_spec`), and `lmr_tt_crossing` exhibits a
real position where a fail-high report strictly exceeds a fail-low
report.  The pre-2c95ab0 store (`tablePart2` in `Sunfish/Tricks.lean`)
would record that as `Entry(lower > upper)` -- a syntactic contradiction
that downstream consumers (the MTD-bi bracket, lines 449-461) were never
written to expect.  The clamp widens the *stale* side instead.

This file proves the two facts that make the clamp the right fix:

* `IntervalTableOK` is the honest post-LMR table invariant -- every
  stored entry satisfies `lower ≤ Vhi(pos, depth)` and
  `Vlo(pos, depth) ≤ upper` -- and the clamped store PRESERVES it, given
  only that the incoming bound is sound on its own side (which is
  exactly what `boundLmr_spec` returns).  The key step: when a new lower
  `L` exceeds the stored upper `U`, the clamped entry `(L, max U L)`
  still satisfies both sides -- `L ≤ Vhi` from the new bound, and
  `Vlo ≤ U ≤ max U L` from the old entry.  Symmetrically on the other
  side.  (The proof needs no case split: `max`/`min` monotonicity covers
  the non-crossing stores too.)

* `clamp_no_crossing`: clamped entries satisfy `lower ≤ upper` *by
  construction*.  So post-2c95ab0 the search-level crossing PHENOMENON
  remains (`lmr_tt_crossing` is about reports, and still holds), but the
  table-level CONTRADICTION is gone: a crossing can no longer be stored,
  and what is stored remains an honest interval claim.

Measured in sunfish: crossings at ~0.016% of nodes; the clamp at
+12 ± 28 ELO vs storing them (commit message of 2c95ab0).
-/

import Sunfish.Tricks
import Sunfish.Lmr

namespace Sunfish

/-- Fail-high clamped entry: `Entry(best, max(entry.upper, best))`
(sunfish.py line 441). -/
def clampHigh (e : Int × Int) (best : Int) : Int × Int :=
  (best, max e.2 best)

/-- Fail-low clamped entry: `Entry(min(entry.lower, best), best)`
(sunfish.py line 443). -/
def clampLow (e : Int × Int) (best : Int) : Int × Int :=
  (min e.1 best, best)

/-- **(b) Clamped entries never cross**: `lower ≤ upper` by construction,
whatever the old entry and the new bound were.  Contrast `lmr_tt_crossing`
(`Sunfish/Lmr.lean`): the crossing *reports* still occur; post-2c95ab0
they can no longer be *stored*. -/
theorem clamp_no_crossing (e : Int × Int) (best : Int) :
    (clampHigh e best).1 ≤ (clampHigh e best).2 ∧
    (clampLow e best).1 ≤ (clampLow e best).2 := by
  refine ⟨?_, ?_⟩
  · show best ≤ max e.2 best
    omega
  · show min e.1 best ≤ best
    omega

/-- **(a) The honest post-LMR table invariant**: every stored entry's
lower bound is a claim about `Vhi` and its upper bound a claim about
`Vlo` (the `lmrVal` pair of `Sunfish/Lmr.lean`).  By `lmrVal_sandwich`
(`Vlo ≤ negamax ≤ Vhi`) such an entry still confines the full-depth
value from whichever side it reports -- it is the interval weakening of
`TableOK`, exactly parallel to how `boundLmr_spec` weakens `bound_spec`. -/
def IntervalTableOK (G : Game) (red : Nat → Nat → G.Pos → Bool)
    (t : Table G) : Prop :=
  ∀ (d : Nat) (p : G.Pos) (lo hi : Int),
    t.find d p = some (lo, hi) →
      lo ≤ lmrVal G red d true p ∧ lmrVal G red d false p ≤ hi

/-- Storing an interval-valid entry preserves the invariant. -/
theorem intervalTableOK_store {G : Game} (red : Nat → Nat → G.Pos → Bool)
    [DecidableEq G.Pos] {t : Table G} (ht : IntervalTableOK G red t)
    {d : Nat} {p : G.Pos} {lo hi : Int}
    (h1 : lo ≤ lmrVal G red d true p) (h2 : lmrVal G red d false p ≤ hi) :
    IntervalTableOK G red (Table.store t d p (lo, hi)) := by
  intro d' p' lo' hi' hfind
  simp only [Table.store] at hfind
  by_cases hdp : d' = d ∧ p' = p
  · rw [if_pos hdp] at hfind
    injection hfind with he
    rw [Prod.mk.injEq] at he
    rw [hdp.1, hdp.2, ← he.1, ← he.2]
    exact ⟨h1, h2⟩
  · rw [if_neg hdp] at hfind
    exact ht d' p' lo' hi' hfind

/-- The upper component of the current entry (stored or the fresh
`Entry(-MATE_UPPER, MATE_UPPER)` default of line 308) is a valid `Vlo`
upper bound; needs `Bounded` for the default, via the sandwich. -/
theorem entry_upper_valid (G : Game) (red : Nat → Nat → G.Pos → Bool)
    (hb : Bounded G) {t : Table G} (ht : IntervalTableOK G red t)
    (d : Nat) (p : G.Pos) :
    lmrVal G red d false p ≤ ((t.find d p).getD (LOSS, MATE_UPPER)).2 := by
  cases hfind : t.find d p with
  | none =>
    have hs := (lmrVal_sandwich G red d p).1
    have hband := (negamax_bounded G hb d p).2
    show lmrVal G red d false p ≤ MATE_UPPER
    omega
  | some e =>
    exact (ht d p e.1 e.2 (by rw [hfind])).2

/-- Symmetric: the lower component of the current entry is a valid `Vhi`
lower bound. -/
theorem entry_lower_valid (G : Game) (red : Nat → Nat → G.Pos → Bool)
    (hb : Bounded G) {t : Table G} (ht : IntervalTableOK G red t)
    (d : Nat) (p : G.Pos) :
    ((t.find d p).getD (LOSS, MATE_UPPER)).1 ≤ lmrVal G red d true p := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  cases hfind : t.find d p with
  | none =>
    have hs := (lmrVal_sandwich G red d p).2
    have hband := (negamax_bounded G hb d p).1
    show LOSS ≤ lmrVal G red d true p
    omega
  | some e =>
    exact (ht d p e.1 e.2 (by rw [hfind])).1

/-- **(a) The fail-high clamped store preserves `IntervalTableOK`**,
given only that the incoming bound is sound on its own side
(`best ≤ Vhi`, which is `boundLmr_spec`'s fail-high clause).  In
particular when `best` exceeds the stored upper `U`: the entry becomes
`(best, max U best)`, whose lower is fine by the new bound and whose
upper is fine because `Vlo ≤ U ≤ max U best` from the OLD entry. -/
theorem intervalTableOK_clampHigh (G : Game) (red : Nat → Nat → G.Pos → Bool)
    [DecidableEq G.Pos] (hb : Bounded G) {t : Table G}
    (ht : IntervalTableOK G red t) (d : Nat) (p : G.Pos) {best : Int}
    (hbest : best ≤ lmrVal G red d true p) :
    IntervalTableOK G red
      (Table.store t d p (clampHigh ((t.find d p).getD (LOSS, MATE_UPPER)) best)) := by
  have hup := entry_upper_valid G red hb ht d p
  show IntervalTableOK G red
    (Table.store t d p (best, max ((t.find d p).getD (LOSS, MATE_UPPER)).2 best))
  exact intervalTableOK_store red ht hbest (by omega)

/-- **(a) The fail-low clamped store preserves `IntervalTableOK`**,
symmetrically, given `Vlo ≤ best` (`boundLmr_spec`'s fail-low clause). -/
theorem intervalTableOK_clampLow (G : Game) (red : Nat → Nat → G.Pos → Bool)
    [DecidableEq G.Pos] (hb : Bounded G) {t : Table G}
    (ht : IntervalTableOK G red t) (d : Nat) (p : G.Pos) {best : Int}
    (hbest : lmrVal G red d false p ≤ best) :
    IntervalTableOK G red
      (Table.store t d p (clampLow ((t.find d p).getD (LOSS, MATE_UPPER)) best)) := by
  have hlo := entry_lower_valid G red hb ht d p
  show IntervalTableOK G red
    (Table.store t d p (min ((t.find d p).getD (LOSS, MATE_UPPER)).1 best, best))
  exact intervalTableOK_store red ht (by omega) hbest

/-- Tables whose entries never cross, the syntactic sanity property the
clamp guarantees. -/
def NoCrossingTable (G : Game) (t : Table G) : Prop :=
  ∀ (d : Nat) (p : G.Pos) (lo hi : Int), t.find d p = some (lo, hi) → lo ≤ hi

/-- Any store of a non-crossing entry -- in particular any clamped store,
by `clamp_no_crossing` -- preserves `NoCrossingTable`. -/
theorem noCrossingTable_store {G : Game} [DecidableEq G.Pos] {t : Table G}
    (ht : NoCrossingTable G t) {d : Nat} {p : G.Pos} {lo hi : Int}
    (h : lo ≤ hi) : NoCrossingTable G (Table.store t d p (lo, hi)) := by
  intro d' p' lo' hi' hfind
  simp only [Table.store] at hfind
  by_cases hdp : d' = d ∧ p' = p
  · rw [if_pos hdp] at hfind
    injection hfind with he
    rw [Prod.mk.injEq] at he
    rw [← he.1, ← he.2]
    exact h
  · rw [if_neg hdp] at hfind
    exact ht d' p' lo' hi' hfind

end Sunfish
