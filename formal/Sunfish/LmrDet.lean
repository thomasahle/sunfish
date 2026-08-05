/-
Deterministic Late Move Reductions (commit 7f9f164, current master):

    LMR = int(depth >= 4 and i_m >= 8 and val < 0)
    yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1 - LMR)

The re-search is gone: the reduction depends only on (depth, index,
value) -- NEVER on gamma -- so each move is searched once, at a
per-move depth fixed before any window is consulted.  This restores the
maintainer's design principle by construction ("gamma may shape
termination, and may trigger shortcuts whose value provably
one-side-bounds the same function; gamma must never select between
incomparable evaluations of a move"), and with it the POINT spec:

* `negamaxDet` is the single value function -- negamax where reducible
  moves are valued one ply shallower, RECURSIVELY (children of a
  reduced move follow the same rule at their own depths).
* `boundLmrDet_spec` (proven, unconditional): the deterministic-LMR
  search satisfies the original docstring against `negamaxDet` --
  fail-soft point bounds on one function, no interval, no mutual
  recursion.  Simply `bound_spec` with per-move depths.
* `boundLmrDet_no_crossing` (proven): fail-high reports never exceed
  fail-low reports at the same `(pos, depth)` -- `bound_no_crossing`
  generalizes verbatim, so contradiction-free transposition entries
  come back by construction.
* `clamp_noop_high` / `clamp_noop_low` (proven): under single-function
  bounds the 2c95ab0 store clamp is a NO-OP -- `max(entry.upper, best)`
  is `entry.upper` and `min(entry.lower, best)` is `entry.lower`
  whenever both the entry and the new bound are honest about the same
  value.  This is the formal justification for removing the clamp in
  7f9f164; any future gamma-dependent evaluation choice must reinstate
  it (and the interval spec of `Sunfish/Lmr.lean`).

The ELO ledger, for the record: the gamma-adaptive re-search LMR
measured ~16 ELO stronger than this rule (-16 ± 38 direct; both large
wins over no LMR).  Consistency was chosen deliberately: every theorem
in this directory about the shipped search is now a point spec on a
single value function.

`Sunfish/Lmr.lean` (the interval spec, the `Vlo`/`Vhi` pair, the TT
crossing) and `Sunfish/TableClamp.lean` (the clamp) describe the
re-search LMR of commits 58883ea..7f9f164 only -- kept as the formal
record of a shipped-then-retired mechanism and of WHY the deterministic
rule was chosen.
-/

import Sunfish.Lmr
import Sunfish.TableClamp

namespace Sunfish

/-- The single value function of deterministic LMR: reducible moves
(`red d i m`, abstracting `depth >= 4 and i_m >= 8 and val < 0`) are
valued one ply shallower, recursively. -/
def negamaxDet (G : Game) (red : Nat → Nat → G.Pos → Bool) : Nat → G.Pos → Int
  | 0, p => G.eval p
  | 1, p => foldMax (fun m => -(G.eval m)) (G.moves p) LOSS
  | d + 2, p =>
    foldMaxIdx (fun i m =>
      if red (d + 2) i m = true then -(negamaxDet G red d m)
      else -(negamaxDet G red (d + 1) m)) 0 (G.moves p) LOSS

/-- The deterministic-LMR search: each move searched ONCE at
`depth - 1 - LMR`, the reduction decided gamma-free. -/
def boundLmrDet (G : Game) (red : Nat → Nat → G.Pos → Bool) :
    Nat → G.Pos → Int → Int
  | 0, p, _gamma => G.eval p
  | 1, p, gamma => searchMoves gamma (fun m => -(G.eval m)) (G.moves p) LOSS
  | d + 2, p, gamma =>
    searchMovesIdx gamma
      (fun i m =>
        if red (d + 2) i m = true then -(boundLmrDet G red d m (1 - gamma))
        else -(boundLmrDet G red (d + 1) m (1 - gamma)))
      0 (G.moves p) LOSS

/-- **The point spec, proven and unconditional**: deterministic LMR
satisfies the original docstring against the single value function
`negamaxDet`.  No interval, no side conditions -- `bound_spec` with
per-move depths. -/
theorem boundLmrDet_spec (G : Game) (red : Nat → Nat → G.Pos → Bool) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      (gamma ≤ boundLmrDet G red d p gamma →
        boundLmrDet G red d p gamma ≤ negamaxDet G red d p) ∧
      (boundLmrDet G red d p gamma < gamma →
        negamaxDet G red d p ≤ boundLmrDet G red d p gamma) := by
  suffices H : ∀ (n d : Nat), d ≤ n → ∀ (p : G.Pos) (gamma : Int),
      (gamma ≤ boundLmrDet G red d p gamma →
        boundLmrDet G red d p gamma ≤ negamaxDet G red d p) ∧
      (boundLmrDet G red d p gamma < gamma →
        negamaxDet G red d p ≤ boundLmrDet G red d p gamma) by
    exact fun d => H d d (Nat.le_refl d)
  intro n
  induction n with
  | zero =>
    intro d hd p gamma
    have hd0 : d = 0 := by omega
    subst hd0
    refine ⟨fun _ => ?_, fun _ => ?_⟩ <;> (simp only [boundLmrDet, negamaxDet]; omega)
  | succ n ihn =>
    intro d hd p gamma
    cases d with
    | zero =>
      refine ⟨fun _ => ?_, fun _ => ?_⟩ <;> (simp only [boundLmrDet, negamaxDet]; omega)
    | succ d' =>
      cases d' with
      | zero =>
        have h := searchMoves_spec gamma (fun m => -(G.eval m)) (fun m => -(G.eval m))
          (fun m => ⟨fun _ => Int.le_refl _, fun _ => Int.le_refl _⟩)
          (G.moves p) LOSS LOSS
          (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
        simp only [boundLmrDet, negamaxDet]
        exact h
      | succ d'' =>
        have hchild : ∀ (i : Nat) (m : G.Pos),
            (gamma ≤ (if red (d'' + 2) i m = true then
                -(boundLmrDet G red d'' m (1 - gamma))
              else -(boundLmrDet G red (d'' + 1) m (1 - gamma))) →
              (if red (d'' + 2) i m = true then
                -(boundLmrDet G red d'' m (1 - gamma))
              else -(boundLmrDet G red (d'' + 1) m (1 - gamma)))
                ≤ (if red (d'' + 2) i m = true then -(negamaxDet G red d'' m)
                  else -(negamaxDet G red (d'' + 1) m))) ∧
            ((if red (d'' + 2) i m = true then
                -(boundLmrDet G red d'' m (1 - gamma))
              else -(boundLmrDet G red (d'' + 1) m (1 - gamma))) < gamma →
              (if red (d'' + 2) i m = true then -(negamaxDet G red d'' m)
                else -(negamaxDet G red (d'' + 1) m))
                ≤ (if red (d'' + 2) i m = true then
                    -(boundLmrDet G red d'' m (1 - gamma))
                  else -(boundLmrDet G red (d'' + 1) m (1 - gamma)))) := by
          intro i m
          by_cases hr : red (d'' + 2) i m = true
          · rw [if_pos hr, if_pos hr]
            have h1 := (ihn d'' (by omega) m (1 - gamma)).1
            have h2 := (ihn d'' (by omega) m (1 - gamma)).2
            constructor
            · intro hge
              have := h2 (by omega)
              omega
            · intro hlt
              have := h1 (by omega)
              omega
          · rw [if_neg hr, if_neg hr]
            have h1 := (ihn (d'' + 1) (by omega) m (1 - gamma)).1
            have h2 := (ihn (d'' + 1) (by omega) m (1 - gamma)).2
            constructor
            · intro hge
              have := h2 (by omega)
              omega
            · intro hlt
              have := h1 (by omega)
              omega
        have h := searchMovesIdx_spec gamma
          (fun i m =>
            if red (d'' + 2) i m = true then -(boundLmrDet G red d'' m (1 - gamma))
            else -(boundLmrDet G red (d'' + 1) m (1 - gamma)))
          (fun i m =>
            if red (d'' + 2) i m = true then -(negamaxDet G red d'' m)
            else -(negamaxDet G red (d'' + 1) m))
          (fun i m =>
            if red (d'' + 2) i m = true then -(negamaxDet G red d'' m)
            else -(negamaxDet G red (d'' + 1) m))
          hchild (G.moves p) 0 LOSS LOSS LOSS
          (fun _ => Int.le_refl _) (fun _ => Int.le_refl _)
        simp only [boundLmrDet, negamaxDet]
        exact h

/-- **No crossing, restored**: with the point spec back, a fail-high
report can never exceed a fail-low report at the same `(pos, depth)` --
the generalization of `bound_no_crossing` to deterministic LMR, and the
reason contradiction-free transposition entries need no clamp. -/
theorem boundLmrDet_no_crossing (G : Game) (red : Nat → Nat → G.Pos → Bool)
    (d : Nat) (p : G.Pos) {g1 g2 r1 r2 : Int}
    (h1 : boundLmrDet G red d p g1 = r1) (hh : g1 ≤ r1)
    (h2 : boundLmrDet G red d p g2 = r2) (hl : r2 < g2) :
    r1 ≤ r2 := by
  have s1 := (boundLmrDet_spec G red d p g1).1
  have s2 := (boundLmrDet_spec G red d p g2).2
  rw [h1] at s1
  rw [h2] at s2
  exact Int.le_trans (s1 hh) (s2 hl)

/-- **The clamp is a no-op under single-function bounds** (the formal
justification for removing it in 7f9f164): if the stored entry and the
incoming fail-high bound are honest about the same value `V`, then
`max(entry.upper, best) = entry.upper` -- the clamped store IS the plain
store `Entry(best, entry.upper)`. -/
theorem clamp_noop_high (e : Int × Int) (best V : Int)
    (hb : best ≤ V) (he : V ≤ e.2) :
    clampHigh e best = (best, e.2) := by
  unfold clampHigh
  simp only [Prod.mk.injEq]
  exact ⟨trivial, by omega⟩

/-- Symmetric: an honest fail-low bound leaves `entry.lower` alone --
the clamped store IS the plain store `Entry(entry.lower, best)`. -/
theorem clamp_noop_low (e : Int × Int) (best V : Int)
    (hb : V ≤ best) (he : e.1 ≤ V) :
    clampLow e best = (e.1, best) := by
  unfold clampLow
  simp only [Prod.mk.injEq]
  exact ⟨by omega, trivial⟩

end Sunfish
