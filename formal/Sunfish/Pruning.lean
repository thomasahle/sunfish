/-
PROPOSED prunings, proof-first: soundness envelopes worked out in the
model BEFORE any engine code exists.  Both candidates are stated
against the DOUBLE-PRIMED pair (`boundD2''`/`boundKCX''`, the model of
shipped master) and its declared value function `nullValueD2` -- the
`(pos, depth)`-determined function every table entry describes
(the point-spec doctrine, formal/README.md).

# Candidate 1 -- razoring with runtime verification

The proposal (from the NNUE lane, proven here on the shared search
model): at depth `d ∈ {2, 3}`, if `pos.score + R(d) < gamma`, answer
from the depth-0 (QS) value INSTEAD of the full-width search -- but if
that razored answer would fail high (`>= gamma`), discard it and run
the full search, so razoring only ever produces fail-low answers.

VERDICT: UNSOUND AS PROPOSED, and no useful repair exists.

* The runtime verification inspects the wrong direction.  A razored
  fail-low stored at the `(pos, d)` key claims
  `nullValueD2 d p ≤ r < gamma`; what the QS answer certifies is a
  bound on the DEPTH-0 declared value.  The two are incomparable, and
  the failure mode is a FALSE FAIL-LOW -- a deep tactic (already a
  quiet mate-in-1) invisible to QS -- which the fail-high re-check by
  construction never inspects.  Detecting a false fail-low IS the
  full-width search the razor was built to skip; there is no cheap
  decisive probe here, which is exactly what separates this from the
  band-edge arm (there the suspicious claim had a boundary window at
  which both fail directions are decisive).  `razorVerified_not_sound`
  is the machine-checked countermodel, with every fidelity premise of
  the double-primed theorems satisfied; `razor_tt_crossing` upgrades
  it to a crossed table entry (`upper < lower` at one key), the same
  indictment that ended re-search LMR (`lmr_tt_crossing`, in git
  history).

* The sound form abandons the QS answer.  If a margin premise
  `RazorMargin` (`nullValueD2 d p ≤ pos.score + R(d)` at razor depths)
  is available, then `pos.score + R(d)` itself is a sound fail-low
  answer -- note the QS search contributes nothing to soundness and
  can be skipped entirely (`boundRm`, `bound_null_spec_Rm`,
  `razorMargin_no_crossing`).  An engine wanting the QS refinement may
  serve `max(qs, pos.score + R(d))` with the fail-high re-search; the
  margin premise still carries all of the soundness.

* But the margin premise is NOT a table fact -- it prices unbounded
  deep tactics, not `pos.value` deltas, so it cannot come from
  `EvalBounds`.  It fails on a quiet mate-in-1 under every margin that
  can ever fire below the mate band: `razorMargin_required` exhibits,
  for EVERY `R < 2 * MATE_LOWER - 1`, a countermodel (all fidelity
  premises satisfied) whose trigger fires at the in-band window
  `MATE_LOWER` while the margin claim is false there.  Conversely
  `razorMargin_trigger_vacuous`: with `R ≥ 2 * MATE_LOWER - 1` the
  trigger can only fire at windows STRICTLY ABOVE the mate band --
  windows the mate machinery already owns.  So razoring that prunes
  anything below the band would put a FALSE chess premise into
  LAYER 1 (the razored answer is stored), the layer the band-edge
  landing just freed of chess premises; and razoring above the band
  prunes nothing.

The countermodel `CexRz v` (parameterized by the razored node's quiet
evaluation `v`, instantiated at `razorV R = min (MATE_LOWER - R - 1) 0`
to cover every margin at once): the razored node `rz` has one QUIET
checking move to `mm`, where the defender's only reply `xx` hangs the
king (`kd`) -- a mate-in-1 whose delivering move is quiet, so depth 0
stands pat at `v` while the depth-2 declared value is the full
`MATE_LOWER`.  `cexRz_real` confirms the crossing is against the
real-move value `negamaxD2` too -- no null machinery is involved
(`guard` is irrelevant at depth 2, where `2 < depth` fails).

Position roster (`r0` belongs to candidate 2's budget theorems and is
inert here -- nothing below `rz` reaches it):

    r0 --> rz --> mm --> xx --> kd        pass rz = qq, pass mm = qq,
                                          qq --> kd (the king capture)
-/

import Sunfish.Stalemate
import Sunfish.Liveness

namespace Sunfish

/-! ### The proposed razor, modeled at the probed node

Below the razor depths there are no razor sites (children of a depth-2
node are searched at depth 1, the pass at depth 0), so at `d = 2` the
at-node wrapper IS the fully threaded search; the countermodel lives at
`d = 2` where the distinction is empty. -/

/-- Razoring exactly as proposed: at `d ∈ {2, 3}`, at a normal node
(king present, no capture pending), with the static score `R d` below
the window, serve the depth-0 answer -- unless it would fail high, in
which case run the full search (the "runtime verification").  Every
razored answer is therefore a fail-low. -/
def boundRz (G : QSGame) (guard : G.Pos → Bool) (R : Nat → Int)
    (d : Nat) (p : G.Pos) (gamma : Int) : Int :=
  if (d = 2 ∨ d = 3) ∧ ¬ (G.eval p ≤ -MATE_LOWER)
      ∧ ¬ (hasKingCapture G.toNullGame.toGame p = true)
      ∧ G.eval p + R d < gamma
      ∧ boundD2'' G guard 0 p gamma < gamma
  then boundD2'' G guard 0 p gamma
  else boundD2'' G guard d p gamma

/-! ### The countermodel -/

/-- Positions of the razoring countermodel. -/
inductive RzPos where
  | r0   -- quiet pre-root (not in check): candidate 2's contrast node
  | rz   -- the razored node: quiet eval `v`, one quiet checking move
  | mm   -- the checkmated child
  | xx   -- `mm`'s only (illegal) reply: the mover's king hangs
  | kd   -- the captured-king position
  | qq   -- pass target of `rz` and `mm`: the check, via `qq --> kd`
  deriving DecidableEq

/-- The countermodel game, parameterized by the razored node's quiet
static evaluation `v`. -/
def CexRz (v : Int) : QSGame where
  Pos := RzPos
  moves := fun x => match x with
    | RzPos.r0 => [RzPos.rz]
    | RzPos.rz => [RzPos.mm]
    | RzPos.mm => [RzPos.xx]
    | RzPos.xx => [RzPos.kd]
    | RzPos.qq => [RzPos.kd]
    | RzPos.kd => []
  eval := fun x => match x with
    | RzPos.rz => v
    | RzPos.kd => -MATE_UPPER
    | _ => 0
  pass := fun x => match x with
    | RzPos.rz => RzPos.qq
    | RzPos.mm => RzPos.qq
    | x => x
  val := fun p m => match p, m with
    | RzPos.xx, RzPos.kd => MATE_UPPER
    | RzPos.qq, RzPos.kd => MATE_UPPER
    | _, _ => 0

/-- The quiet evaluation that turns `CexRz` into a countermodel for the
margin `R`: low enough that the trigger fires at the in-band window
`MATE_LOWER`, high enough (given `R < 2 * MATE_LOWER - 1`) to stay out
of the king-gone zone. -/
def razorV (R : Int) : Int := min (MATE_LOWER - R - 1) 0

/-- The countermodel's guard: null machinery off.  (Irrelevant at the
depths used -- the pass term needs `2 < depth` -- but a named constant
keeps every statement's guard syntactically identical.) -/
def gF : RzPos → Bool := fun _ => false

/-! ### Countermodel facts

Everything below `rz` is `v`-free, so the structural facts are
definitional (`rfl`-checkable); only the branches that read
`eval rz = v` need the quietness hypotheses. -/

/-- The checkmated child: oracle-terminal, in check, exact value
`-MATE_LOWER - 15` -- the flat mate value plus the ONE ply of depth
the search still had in hand, at `EVAL_ROUGHNESS` a ply. -/
theorem cexRz_value_mm (v : Int) (g : RzPos → Bool) :
    nullValueD2 (CexRz v) g 1 RzPos.mm = -MATE_LOWER - 15 := by
  rw [nullValueD2_of_allIllegal (CexRz v) g 0 RzPos.mm
    (by show ¬ ((0 : Int) ≤ -MATE_LOWER); decide)
    (fun h => Bool.noConfusion h) rfl]
  rfl

/-- The mate-in-1 is real: the depth-2 declared value at the razored
node is `MATE_LOWER + 15` -- the mate band, one ply of distance in.
(`guard` is irrelevant: the pass term requires `2 < depth`.) -/
theorem cexRz_value_rz (v : Int) (g : RzPos → Bool)
    (hv1 : -MATE_LOWER < v) :
    nullValueD2 (CexRz v) g 2 RzPos.rz = MATE_LOWER + 15 := by
  rw [nullValueD2_of_fold (CexRz v) g 1 RzPos.rz
    (by show ¬ (v ≤ -MATE_LOWER); omega)
    (fun h => Bool.noConfusion h) rfl]
  have hT : nullTermD2 (CexRz v) g 1 RzPos.rz = LOSS := by
    simp only [nullTermD2]
    rw [if_neg (fun h => absurd h.2 (by omega))]
  have hma : movesAbove (CexRz v) (val_lower 2) RzPos.rz = [RzPos.mm] := rfl
  rw [hT, hma]
  simp only [foldMax]
  rw [cexRz_value_mm v g]
  decide

/-- The real-move value agrees: the crossing is not a null artifact. -/
theorem cexRz_real (v : Int) (hv1 : -MATE_LOWER < v) :
    negamaxD2 (CexRz v) 2 RzPos.rz = MATE_LOWER + 15 := by
  have hmm : negamaxD2 (CexRz v) 1 RzPos.mm = -MATE_LOWER - 15 := by
    rw [negamaxD2_of_allIllegal (CexRz v) 0 RzPos.mm
      (by show ¬ ((0 : Int) ≤ -MATE_LOWER); decide)
      (fun h => Bool.noConfusion h) rfl]
    rfl
  rw [negamaxD2_of_fold (CexRz v) 1 RzPos.rz
    (by show ¬ (v ≤ -MATE_LOWER); omega)
    (fun h => Bool.noConfusion h) rfl]
  have hma : movesAbove (CexRz v) (val_lower 2) RzPos.rz = [RzPos.mm] := rfl
  rw [hma]
  simp only [foldMax]
  rw [hmm]
  decide

/-- The depth-0 answer at the razored node is the quiet stand-pat:
QS sees no capture, hence no mate. -/
theorem cexRz_qs (v : Int) (g : RzPos → Bool) (gamma : Int)
    (hv1 : -MATE_LOWER < v) :
    boundD2'' (CexRz v) g 0 RzPos.rz gamma = v := by
  simp only [boundD2'']
  rw [if_neg (show ¬ ((CexRz v).eval RzPos.rz ≤ -MATE_LOWER) by
        show ¬ (v ≤ -MATE_LOWER); omega),
    if_neg (fun h => Bool.noConfusion h)]
  rfl

/-- Every fidelity premise of the double-primed theorems holds on the
countermodel: bounded quiet evals, table floor far above `-192`, king
captures valued at the top and only them, legally-reached root. -/
theorem cexRz_fidelity (v : Int) (hv1 : -MATE_LOWER < v) (hv2 : v ≤ 0) :
    Bounded (CexRz v).toNullGame.toGame ∧
    EvalQuiet (CexRz v).toNullGame.toGame ∧
    ValFloor (CexRz v) 192 ∧
    KingCaptureValHigh (CexRz v) ∧
    HighValIsKingCapture (CexRz v) ∧
    hasKingCapture (CexRz v).toNullGame.toGame RzPos.rz = false := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  refine ⟨?_, ?_, ?_, ?_, ?_, rfl⟩
  · intro p
    constructor <;> (cases p <;> simp [CexRz] <;> omega)
  · intro p hq
    cases p <;> simp_all [CexRz] <;> omega
  · intro p m hm
    cases p <;> cases m <;> simp_all [CexRz] <;> omega
  · intro p m hm hme
    cases p <;> cases m <;> simp_all [CexRz] <;> omega
  · intro p m hm hval
    cases p <;> cases m <;> simp_all [CexRz] <;> omega

/-- The full-width search at the same key and window certifies the
mate: a sound fail-high report of `MATE_LOWER`. -/
theorem cexRz_bound2 (v : Int)
    (hv1 : -MATE_LOWER < v) :
    boundD2'' (CexRz v) gF 2 RzPos.rz MATE_LOWER
      = MATE_LOWER + 15 := by
  simp only [boundD2'']
  rw [if_neg (show ¬ ((CexRz v).eval RzPos.rz ≤ -MATE_LOWER) by
        show ¬ (v ≤ -MATE_LOWER); omega),
    if_neg (fun h => Bool.noConfusion h),
    if_neg (fun h => Bool.noConfusion h.1)]
  rfl

/-! ### The verdict theorems -/

/-- **Razoring as proposed is UNSOUND, for every margin that can ever
fire below the mate band** (`R < 2 * MATE_LOWER - 1`; beyond that see
`razorMargin_trigger_vacuous`).  On `CexRz (razorV R)` -- all fidelity
premises satisfied, `cexRz_fidelity` -- the trigger fires at the
in-band window `MATE_LOWER`, the runtime verification PASSES (the
razored answer fails low, so the fail-high re-check never runs), and
the served answer's fail-low claim `nullValueD2 2 p ≤ r` is FALSE: the
declared value at the `(rz, 2)` key is the full `MATE_LOWER`
(a quiet mate-in-1, invisible at depth 0). -/
theorem razorVerified_not_sound (R : Int) (hR : R < 2 * MATE_LOWER - 1) :
    (CexRz (razorV R)).eval RzPos.rz + R < MATE_LOWER ∧
    boundRz (CexRz (razorV R)) gF (fun _ => R)
        2 RzPos.rz MATE_LOWER < MATE_LOWER ∧
    MATE_LOWER ≤ nullValueD2 (CexRz (razorV R)) gF 2 RzPos.rz := by
  have hML : MATE_LOWER = 47923 := rfl
  have hv1 : -MATE_LOWER < razorV R := by simp only [razorV]; omega
  have hv2 : razorV R ≤ 0 := by simp only [razorV]; omega
  have htrig : razorV R + R < MATE_LOWER := by simp only [razorV]; omega
  have hqs := cexRz_qs (razorV R) gF MATE_LOWER hv1
  have hrz : boundRz (CexRz (razorV R)) gF (fun _ => R)
      2 RzPos.rz MATE_LOWER = razorV R := by
    unfold boundRz
    have hcond : ((2 : Nat) = 2 ∨ (2 : Nat) = 3)
        ∧ ¬ ((CexRz (razorV R)).eval RzPos.rz ≤ -MATE_LOWER)
        ∧ ¬ (hasKingCapture (CexRz (razorV R)).toNullGame.toGame RzPos.rz = true)
        ∧ (CexRz (razorV R)).eval RzPos.rz + (fun _ => R) 2 < MATE_LOWER
        ∧ boundD2'' (CexRz (razorV R)) gF 0 RzPos.rz MATE_LOWER < MATE_LOWER := by
      refine ⟨Or.inl rfl, ?_, fun h => Bool.noConfusion h, ?_, ?_⟩
      · show ¬ (razorV R ≤ -MATE_LOWER); omega
      · show razorV R + R < MATE_LOWER; exact htrig
      · rw [hqs]; omega
    rw [if_pos hcond]
    exact hqs
  refine ⟨htrig, ?_, ?_⟩
  · rw [hrz]; omega
  · rw [cexRz_value_rz (razorV R) gF hv1]
    omega

/-- **The crossed table entry**: at the same key `(rz, 2)` and the same
window `MATE_LOWER`, the honest search fails HIGH at `MATE_LOWER`
(a sound lower bound: the mate is real, `cexRz_value_rz`/`cexRz_real`)
while the razored probe serves a fail-low upper bound below it --
an interval `(lower, upper)` with `upper < lower`, the exact shape of
`lmr_tt_crossing` that ended re-search LMR. -/
theorem razor_tt_crossing (R : Int) (hR : R < 2 * MATE_LOWER - 1) :
    MATE_LOWER ≤ boundD2'' (CexRz (razorV R)) gF
        2 RzPos.rz MATE_LOWER ∧
    boundRz (CexRz (razorV R)) gF (fun _ => R)
        2 RzPos.rz MATE_LOWER < MATE_LOWER := by
  have hML : MATE_LOWER = 47923 := rfl
  have hv1 : -MATE_LOWER < razorV R := by simp only [razorV]; omega
  refine ⟨?_, (razorVerified_not_sound R hR).2.1⟩
  rw [cexRz_bound2 (razorV R) hv1]
  omega

/-! ### The sound modified form, and why it is useless -/

/-- **RazorMargin** -- the premise a sound razor needs: at the razor
depths, the declared value never exceeds the static evaluation by more
than the margin.  A CHESS premise about deep tactics (not a table
fact): `razorMargin_required` proves it false for every useful margin,
and it would sit in LAYER 1 (razored answers are stored), the layer
the band-edge landing freed of chess premises. -/
def RazorMargin (G : QSGame) (guard : G.Pos → Bool) (R : Nat → Int) : Prop :=
  ∀ (d : Nat) (p : G.Pos), (d = 2 ∨ d = 3) →
    ¬ (G.eval p ≤ -MATE_LOWER) →
    ¬ (hasKingCapture G.toNullGame.toGame p = true) →
    nullValueD2 G guard d p ≤ G.eval p + R d

/-- The sound razor: serve `pos.score + R d` itself.  The QS answer
contributes nothing to soundness and is not consulted -- under the
margin premise the static bound IS the fail-low answer.  (An engine
may still serve `max(qs, pos.score + R d)` with the fail-high
re-search; the margin premise carries all the soundness either way.) -/
def boundRm (G : QSGame) (guard : G.Pos → Bool) (R : Nat → Int)
    (d : Nat) (p : G.Pos) (gamma : Int) : Int :=
  if (d = 2 ∨ d = 3) ∧ ¬ (G.eval p ≤ -MATE_LOWER)
      ∧ ¬ (hasKingCapture G.toNullGame.toGame p = true)
      ∧ G.eval p + R d < gamma
  then G.eval p + R d
  else boundD2'' G guard d p gamma

/-- Layer 1 for the margin razor: under `Bounded` and `RazorMargin`,
`boundRm` brackets the SAME declared function `nullValueD2` at every
key and driver-range window -- so razored and full answers can mix at
a key without contradiction.  (At-node form; children of a razored
node are never searched, and a threaded variant's children enter any
such proof only through this same bracket property.) -/
theorem bound_null_spec_Rm (G : QSGame) (guard : G.Pos → Bool)
    (R : Nat → Int)
    (hB : Bounded G.toNullGame.toGame)
    (hM : RazorMargin G guard R) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      (gamma ≤ boundRm G guard R d p gamma →
        boundRm G guard R d p gamma ≤ nullValueD2 G guard d p) ∧
      (boundRm G guard R d p gamma < gamma →
        nullValueD2 G guard d p ≤ boundRm G guard R d p gamma) := by
  intro d p gamma hg1 hg2
  unfold boundRm
  by_cases h : (d = 2 ∨ d = 3) ∧ ¬ (G.eval p ≤ -MATE_LOWER)
      ∧ ¬ (hasKingCapture G.toNullGame.toGame p = true)
      ∧ G.eval p + R d < gamma
  · rw [if_pos h]
    obtain ⟨hd, hkg, hcap, ht⟩ := h
    have hm := hM d p hd hkg hcap
    exact ⟨fun hge => absurd hge (by omega), fun _ => hm⟩
  · rw [if_neg h]
    exact bound_null_spec'' G guard hB d p gamma hg1 hg2

/-- Table consistency for the margin razor: no crossed entries, even
when one probe razored and the other ran the full search. -/
theorem razorMargin_no_crossing (G : QSGame) (guard : G.Pos → Bool)
    (R : Nat → Int)
    (hB : Bounded G.toNullGame.toGame)
    (hM : RazorMargin G guard R)
    (d : Nat) (p : G.Pos) (g1 g2 : Int)
    (hg1a : -MATE_UPPER < g1) (hg1b : g1 ≤ MATE_UPPER)
    (hg2a : -MATE_UPPER < g2) (hg2b : g2 ≤ MATE_UPPER)
    (hhi : g1 ≤ boundRm G guard R d p g1)
    (hlo : boundRm G guard R d p g2 < g2) :
    boundRm G guard R d p g1 ≤ boundRm G guard R d p g2 := by
  have h1 := (bound_null_spec_Rm G guard R hB hM d p g1 hg1a hg1b).1 hhi
  have h2 := (bound_null_spec_Rm G guard R hB hM d p g2 hg2a hg2b).2 hlo
  omega

/-- **The margin premise is required, and false for every useful
margin**: for every `R < 2 * MATE_LOWER - 1`, `CexRz (razorV R)`
satisfies all fidelity premises, its trigger fires at the in-band
window `MATE_LOWER` (first conjunct, cf. `razorVerified_not_sound`),
and `RazorMargin` fails -- the quiet mate-in-1 outruns the margin. -/
theorem razorMargin_required (R : Int) (hR : R < 2 * MATE_LOWER - 1) :
    ¬ RazorMargin (CexRz (razorV R)) gF (fun _ => R) := by
  intro hM
  have hML : MATE_LOWER = 47923 := rfl
  have hv1 : -MATE_LOWER < razorV R := by simp only [razorV]; omega
  have h := hM 2 RzPos.rz (Or.inl rfl)
    (show ¬ (razorV R ≤ -MATE_LOWER) by simp only [razorV]; omega)
    (fun h => Bool.noConfusion h)
  rw [cexRz_value_rz (razorV R) gF hv1] at h
  have hev : (CexRz (razorV R)).eval RzPos.rz = razorV R := rfl
  rw [hev] at h
  simp only [razorV] at h
  omega

/-- **Above the impossibility line the razor is vacuous below the mate
band**: with `R ≥ 2 * MATE_LOWER - 1` (the least margin
`razorMargin_required` does not refute), a trigger at any node whose
evaluation is out of the king-gone zone forces the window STRICTLY
ABOVE `MATE_LOWER` -- razoring can then only fire at mate-band
windows, where it prunes nothing the mate machinery does not already
decide.  Unconditional arithmetic. -/
theorem razorMargin_trigger_vacuous (R : Nat → Int) (d : Nat)
    (v gamma : Int)
    (hbig : 2 * MATE_LOWER - 1 ≤ R d)
    (hquiet : -MATE_LOWER < v) (htrig : v + R d < gamma) :
    MATE_LOWER < gamma := by
  have hML : MATE_LOWER = 47923 := rfl
  omega

/-! # Candidate 2 -- the check extension

The proposal: `e(pos) = 1` if the side to move is in check (the board
predicate `inCheckB`, already modeled); at node entry the effective
depth is `d' = d + e(pos)` and everything else is unchanged --
children at `d' - 1`, the entry stored at the `(pos, d)` key.

VERDICT: SOUND ONLY IN THE KEY-DETERMINED (DEPTH-SHIFT) FORM, which
extends nothing below the probed node; the RECURSIVE at-entry form has
NO declared value function at all, and the budgeted repair is sound
exactly when the budget joins the table key.

* **The key-determined form** (`boundE`/`boundKCXE` with declared
  function `nullValueE d p = nullValueD2 (d + e p) p`): `e` is
  position-derived, so `d + e p` is a function of the `(pos, d)` key
  and the `can_ext` doctrine's fold-into-depth branch applies
  verbatim -- layer 1 transfers by instantiation
  (`boundE_null_spec` / `boundKCXE_null_spec`), no-crossing holds at
  every key (`extE_no_crossing`), and the liveness bound IMPROVES at
  in-check roots (`forcedMate_ext`: mate-in-k visible at `D ≥ k`, one
  ply earlier).  The honesty clause: `nullValueE`'s fold consumes
  UNEXTENDED children (`nullValueD2 (d' - 1) m`, not the child's own
  extended value), so an engine implementing this form must search
  children FLAT -- the effective depths telescope and only the probed
  node's own horizon moves.  It is a depth relabeling, not a deeper
  search of checking lines.

* **The recursive at-entry form** -- children at `d' - 1` where each
  child re-extends at ITS entry, which is what the natural engine
  change (`depth += pos.in_check()` at the top of `bound`) does -- has
  no `(pos, depth)`-determined value function to bracket, and not for
  a fixable reason: on a mutual-check cycle (real chess: PERPETUAL
  CHECK) the would-be defining equations do not determine any function.
  `ExtValueEqns` states those equations verbatim
  (`extValueEqns_of_checkFree` sanity-checks them: on a check-free
  game the shipped `nullValueD2` satisfies them);
  `checkExt_any_value` solves them on the two-position perpetual-check
  game `CexPerp` with ARBITRARY values `x`/`-x` in the full score
  band, and `checkExt_no_declared_value` exhibits two solutions
  disagreeing at one `(pos, depth)` key.  The engine reading is
  non-termination: at `pa` the call at depth `d` recurses through `pb`
  back to `pa` at the SAME depth -- the at-entry extension erases the
  depth limit exactly where the game has a cycle, and sunfish's
  in-search recursion has no repetition stop (the history check
  guards root-line repetitions).  So there is no spec to transfer and
  no sound implementation to write: point spec, table invariant and
  termination all fail together.

* **The budgeted repair** (`nullValueEB`: a per-path extension budget
  `b`, extension consumed from it) restores well-foundedness -- the
  definition itself carries the termination proof, measure `b + d` --
  and collapses to the shipped function at budget 0
  (`nullValueEB_zero`).  But the budget is VALUE-BEARING at a fixed
  `(pos, depth)` key (`ext_budget_value_bearing`: budgets 0 and 1
  disagree at one key), so a table serving it must key on
  `(pos, depth, budget)` -- the `can_ext` doctrine's key-must-grow
  branch, the same shape as `extended_value_not_key_independent`
  (Tricks.lean).  And with ample budget the recursive extension is a
  genuinely different function from every depth-shift of the shipped
  one at the probed key (`ext_recursive_ne_shift`), so no relabeling
  recovers it.

Engine guidance is recorded in formal/README.md ("Proposed prunings:
proven envelopes"). -/

/-! ### The extension bit and the key-determined (depth-shift) form -/

/-- `e(pos)`: 1 if the side to move is in check -- position-derived,
which is what lets it fold into the depth component of the key. -/
def extB (G : QSGame) (p : G.Pos) : Nat :=
  if inCheckB G.toNullGame p = true then 1 else 0

/-- The extended declared value, depth-shift form: the SAME
`nullValueD2`, read at the effective depth `d + e p`.  A function of
the `(pos, d)` key because `e` is position-derived. -/
def nullValueE (G : QSGame) (guard : G.Pos → Bool) (d : Nat) (p : G.Pos) : Int :=
  nullValueD2 G guard (d + extB G p) p

/-- The extended reference search: the probed node's own horizon moves
by `e p`; everything below is the unchanged double-primed search
(children FLAT -- see the module comment's honesty clause). -/
def boundE (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (gamma : Int) : Int :=
  boundD2'' G guard (d + extB G p) p gamma

/-- The extended production consumer. -/
def boundKCXE (G : QSGame) (guard : G.Pos → Bool)
    (d : Nat) (p : G.Pos) (gamma : Int) : Int :=
  boundKCX'' G guard (d + extB G p) p gamma

/-- Layer 1 for the depth-shift extension, reference side: transfers
from `bound_null_spec''` by instantiation at the effective depth --
the `can_ext` fold-into-depth doctrine made formal. -/
theorem boundE_null_spec (G : QSGame) (guard : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      (gamma ≤ boundE G guard d p gamma →
        boundE G guard d p gamma ≤ nullValueE G guard d p) ∧
      (boundE G guard d p gamma < gamma →
        nullValueE G guard d p ≤ boundE G guard d p gamma) :=
  fun d p gamma h1 h2 =>
    bound_null_spec'' G guard hB (d + extB G p) p gamma h1 h2

/-- Layer 1 for the depth-shift extension, production side. -/
theorem boundKCXE_null_spec (G : QSGame) (guard : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G) :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int),
      -MATE_UPPER < gamma → gamma ≤ MATE_UPPER →
      (gamma ≤ boundKCXE G guard d p gamma →
        boundKCXE G guard d p gamma ≤ nullValueE G guard d p) ∧
      (boundKCXE G guard d p gamma < gamma →
        nullValueE G guard d p ≤ boundKCXE G guard d p gamma) :=
  fun d p gamma h1 h2 =>
    boundKCX''_null_spec G guard hB hV hCF (d + extB G p) p gamma h1 h2

/-- Table consistency at every `(pos, d)` key: both probes bracket the
one key-determined `nullValueE`, so entries never cross -- the
`(pos, depth)` table needs NO new field for this form. -/
theorem extE_no_crossing (G : QSGame) (guard : G.Pos → Bool)
    (hB : Bounded G.toNullGame.toGame)
    (hV : KingCaptureValHigh G) (hCF : CaptureFirst G)
    (d : Nat) (p : G.Pos) (g1 g2 : Int)
    (hg1a : -MATE_UPPER < g1) (hg1b : g1 ≤ MATE_UPPER)
    (hg2a : -MATE_UPPER < g2) (hg2b : g2 ≤ MATE_UPPER)
    (hhi : g1 ≤ boundKCXE G guard d p g1)
    (hlo : boundKCXE G guard d p g2 < g2) :
    boundKCXE G guard d p g1 ≤ boundKCXE G guard d p g2 := by
  have h1 := (boundKCXE_null_spec G guard hB hV hCF d p g1 hg1a hg1b).1 hhi
  have h2 := (boundKCXE_null_spec G guard hB hV hCF d p g2 hg2a hg2b).2 hlo
  omega

/-- The liveness improvement: at an in-check root the extension buys
exactly one ply -- a forced mate in ≤ k is visible in the extended
declared value at every `D ≥ k` (the unextended bound is `k + 1`,
`forcedMate_complete`).  Same premises as the unextended statement:
`ValFloor` (fidelity) + `NoZugzwang` (layer 2). -/
theorem forcedMate_ext (G : QSGame) (guard : G.Pos → Bool)
    (hF : ValFloor G 192) (hZ : NoZugzwang G guard)
    {k : Nat} {p : G.Pos} (hFM : ForcedMate G k p)
    (hic : inCheckB G.toNullGame p = true) :
    ∀ D : Nat, k ≤ D → MATE_LOWER ≤ nullValueE G guard D p := by
  intro D hD
  unfold nullValueE
  have he : extB G p = 1 := by unfold extB; rw [if_pos hic]
  rw [he]
  exact forcedMate_complete_band G guard hF hZ hFM (D + 1) (by omega)

/-! ### The recursive at-entry form has no declared value function -/

/-- The would-be defining equations of the RECURSIVE at-entry
extension's declared value: `nullValueD2`'s own equations, with the
node's branch structure read at the effective depth `d + e p` and the
children consumed one below it -- each child call re-extending at its
own entry, because `V` itself satisfies the equations there.  This is
exactly what `depth += pos.in_check()` at the top of `bound` computes.
`extValueEqns_of_checkFree` sanity-checks the shape: with no checks
anywhere the equations are `nullValueD2`'s and the shipped function
satisfies them. -/
def ExtValueEqns (G : QSGame) (guard : G.Pos → Bool)
    (V : Nat → G.Pos → Int) : Prop :=
  ∀ (d : Nat) (p : G.Pos),
    V d p =
      if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
      else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
      else match d + extB G p with
        | 0 => G.eval p
        | k + 1 =>
          if allIllegalB G p = true then terminalValue G (k + 1) p
          else
            foldMax (fun m => -(V k m))
              (movesAbove G (val_lower (k + 1)) p)
              (if guard p = true ∧ 2 < k + 1 then
                (if -(V (k + 1 - 3) (G.pass p)) < MATE_LOWER then
                  max LOSS (-(V (k + 1 - 3) (G.pass p)))
                else LOSS)
              else LOSS)

/-- Sanity: on a check-free game the equations are `nullValueD2`'s own
equations, and the shipped declared function satisfies them. -/
theorem extValueEqns_of_checkFree (G : QSGame) (guard : G.Pos → Bool)
    (hcf : ∀ p, inCheckB G.toNullGame p = false) :
    ExtValueEqns G guard (nullValueD2 G guard) := by
  intro d p
  have he : extB G p = 0 := by unfold extB; rw [hcf p]; rfl
  rw [he, Nat.add_zero]
  cases d with
  | zero => simp only [nullValueD2]
  | succ d => simp only [nullValueD2]

/-- The perpetual-check countermodel: `pa` and `pb` move only to each
other, and both are in check (their pass hands the opponent the king
capture through `qa`/`qb`).  Real chess realizes this as any perpetual
check -- the position graph of chess has cycles, and mutual-check
cycles among them. -/
inductive PPos where
  | pa | pb   -- the mutual-check cycle
  | qa | qb   -- pass targets: the opponent to move, king en prise
  | dead      -- the captured-king position
  deriving DecidableEq

def CexPerp : QSGame where
  Pos := PPos
  moves := fun x => match x with
    | PPos.pa => [PPos.pb]
    | PPos.pb => [PPos.pa]
    | PPos.qa => [PPos.dead]
    | PPos.qb => [PPos.dead]
    | PPos.dead => []
  eval := fun x => match x with
    | PPos.dead => -MATE_UPPER
    | _ => 0
  pass := fun x => match x with
    | PPos.pa => PPos.qa
    | PPos.pb => PPos.qb
    | x => x
  val := fun _ _ => 0

/-- The perpetual-check guard (named for syntactic stability). -/
def gP : PPos → Bool := fun _ => false

/-- A solution family for the extension equations on `CexPerp`:
assign `x` to `pa` and `-x` to `pb`, at EVERY depth. -/
def perpSol (x : Int) : Nat → PPos → Int := fun _ p =>
  match p with
  | PPos.pa => x
  | PPos.pb => -x
  | PPos.qa => MATE_UPPER
  | PPos.qb => MATE_UPPER
  | PPos.dead => -MATE_UPPER

/-- Every `x` in the score band solves the equations: at `pa` the
effective depth is `d + 1`, the child `pb` is consumed at depth `d`,
and `pb` symmetrically hands back `pa` at depth `d` -- the depth never
decreases around the cycle, so the equations relate the SAME keys to
each other and any consistent assignment survives. -/
theorem perpSol_satisfies (x : Int)
    (h1 : -MATE_UPPER ≤ x) (h2 : x ≤ MATE_UPPER) :
    ExtValueEqns CexPerp gP (perpSol x) := by
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hMU : MATE_UPPER = 69290 := rfl
  intro d p
  cases p
  case dead =>
    rw [if_pos (show CexPerp.eval PPos.dead ≤ -MATE_LOWER by decide)]
    rfl
  case qa =>
    rw [if_neg (show ¬ (CexPerp.eval PPos.qa ≤ -MATE_LOWER) by decide),
      if_pos (show hasKingCapture CexPerp.toNullGame.toGame PPos.qa = true
        from rfl)]
    rfl
  case qb =>
    rw [if_neg (show ¬ (CexPerp.eval PPos.qb ≤ -MATE_LOWER) by decide),
      if_pos (show hasKingCapture CexPerp.toNullGame.toGame PPos.qb = true
        from rfl)]
    rfl
  case pa =>
    rw [if_neg (show ¬ (CexPerp.eval PPos.pa ≤ -MATE_LOWER) by decide),
      if_neg (show ¬ (hasKingCapture CexPerp.toNullGame.toGame PPos.pa = true)
        from fun h => Bool.noConfusion h)]
    have he : extB CexPerp PPos.pa = 1 := rfl
    rw [he]
    show perpSol x d PPos.pa
      = if allIllegalB CexPerp PPos.pa = true then terminalValue CexPerp (d + 1) PPos.pa
        else
          foldMax (fun m => -(perpSol x d m))
            (movesAbove CexPerp (val_lower (d + 1)) PPos.pa)
            (if gP PPos.pa = true ∧ 2 < d + 1 then
              (if -(perpSol x (d + 1 - 3) (CexPerp.pass PPos.pa)) < MATE_LOWER then
                max LOSS (-(perpSol x (d + 1 - 3) (CexPerp.pass PPos.pa)))
              else LOSS)
            else LOSS)
    rw [if_neg (show ¬ (allIllegalB CexPerp PPos.pa = true)
      from fun h => Bool.noConfusion h)]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
    have hma : movesAbove CexPerp (val_lower (d + 1)) PPos.pa = [PPos.pb] := by
      unfold movesAbove
      show List.filter _ [PPos.pb] = [PPos.pb]
      rw [List.filter_cons]
      rw [if_pos (decide_eq_true (show val_lower (d + 1) ≤ CexPerp.val PPos.pa PPos.pb by
        show val_lower (d + 1) ≤ 0
        rw [val_lower_pos (d + 1) (by omega)]
        decide))]
      rfl
    rw [hma]
    simp only [foldMax]
    show x = max LOSS (-(-x))
    omega
  case pb =>
    rw [if_neg (show ¬ (CexPerp.eval PPos.pb ≤ -MATE_LOWER) by decide),
      if_neg (show ¬ (hasKingCapture CexPerp.toNullGame.toGame PPos.pb = true)
        from fun h => Bool.noConfusion h)]
    have he : extB CexPerp PPos.pb = 1 := rfl
    rw [he]
    show perpSol x d PPos.pb
      = if allIllegalB CexPerp PPos.pb = true then terminalValue CexPerp (d + 1) PPos.pb
        else
          foldMax (fun m => -(perpSol x d m))
            (movesAbove CexPerp (val_lower (d + 1)) PPos.pb)
            (if gP PPos.pb = true ∧ 2 < d + 1 then
              (if -(perpSol x (d + 1 - 3) (CexPerp.pass PPos.pb)) < MATE_LOWER then
                max LOSS (-(perpSol x (d + 1 - 3) (CexPerp.pass PPos.pb)))
              else LOSS)
            else LOSS)
    rw [if_neg (show ¬ (allIllegalB CexPerp PPos.pb = true)
      from fun h => Bool.noConfusion h)]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
    have hma : movesAbove CexPerp (val_lower (d + 1)) PPos.pb = [PPos.pa] := by
      unfold movesAbove
      show List.filter _ [PPos.pa] = [PPos.pa]
      rw [List.filter_cons]
      rw [if_pos (decide_eq_true (show val_lower (d + 1) ≤ CexPerp.val PPos.pb PPos.pa by
        show val_lower (d + 1) ≤ 0
        rw [val_lower_pos (d + 1) (by omega)]
        decide))]
      rfl
    rw [hma]
    simp only [foldMax]
    show -x = max LOSS (-(perpSol x d PPos.pa))
    show -x = max LOSS (-x)
    omega

/-- **Any value in the score band is a "declared value" at a
perpetual-check node**: the extension equations constrain nothing
there. -/
theorem checkExt_any_value (x : Int)
    (h1 : -MATE_UPPER ≤ x) (h2 : x ≤ MATE_UPPER) (d : Nat) :
    ∃ V : Nat → PPos → Int,
      ExtValueEqns CexPerp gP V ∧ V d PPos.pa = x :=
  ⟨perpSol x, perpSol_satisfies x h1 h2, rfl⟩

/-- **The recursive at-entry check extension has NO declared value
function**: two solutions of its defining equations disagree at the
same `(pos, depth)` key, so there is no function for a table entry to
describe -- the point-spec obligation fails before any theorem about
the search can even be stated.  (The engine-side reading is
non-termination on the same cycle: `pa` at depth `d` recurses through
`pb` back to `pa` at depth `d`.) -/
theorem checkExt_no_declared_value :
    ∃ V₁ V₂ : Nat → PPos → Int,
      ExtValueEqns CexPerp gP V₁ ∧ ExtValueEqns CexPerp gP V₂ ∧
      V₁ 3 PPos.pa ≠ V₂ 3 PPos.pa :=
  ⟨perpSol 0, perpSol 1,
    perpSol_satisfies 0 (by decide) (by decide),
    perpSol_satisfies 1 (by decide) (by decide),
    by decide⟩

/-! ### The budgeted repair: well-founded iff the budget joins the key -/

/-- The extension bit under a budget: fires only while budget
remains. -/
def extAt (G : QSGame) (b : Nat) (p : G.Pos) : Nat :=
  if inCheckB G.toNullGame p = true ∧ 0 < b then 1 else 0

theorem extAt_le_one (G : QSGame) (b : Nat) (p : G.Pos) :
    extAt G b p ≤ 1 := by
  unfold extAt
  split <;> omega

theorem extAt_le_budget (G : QSGame) (b : Nat) (p : G.Pos) :
    extAt G b p ≤ b := by
  unfold extAt
  split
  · next h => omega
  · omega

theorem extAt_zero (G : QSGame) (p : G.Pos) : extAt G 0 p = 0 := by
  unfold extAt
  rw [if_neg (fun h => absurd h.2 (by omega))]

/-- The BUDGETED at-entry extension's declared value: `nullValueD2`'s
recursion with the effective depth `d + e`, the extension consumed
from the budget `b`.  The definition IS the theorem the naive form
lacks: Lean accepts it because `b + d` strictly decreases -- an
extended step trades one budget for one depth, an unextended step
spends depth -- so the function exists, keyed by `(pos, d, b)`.  The
budget is genuinely part of the key: `ext_budget_value_bearing`. -/
def nullValueEB (G : QSGame) (guard : G.Pos → Bool) : Nat → Nat → G.Pos → Int
  | b, d, p =>
    if G.eval p ≤ -MATE_LOWER then -MATE_UPPER
    else if hasKingCapture G.toNullGame.toGame p = true then MATE_UPPER
    else match _hdd : d + extAt G b p with
      | 0 => G.eval p
      | k + 1 =>
        if allIllegalB G p = true then terminalValue G (k + 1) p
        else
          foldMax (fun m => -(nullValueEB G guard (b - extAt G b p) k m))
            (movesAbove G (val_lower (k + 1)) p)
            (if guard p = true ∧ 2 < k + 1 then
              (if -(nullValueEB G guard (b - extAt G b p) (k + 1 - 3) (G.pass p)) < MATE_LOWER then
                max LOSS (-(nullValueEB G guard (b - extAt G b p) (k + 1 - 3) (G.pass p)))
              else LOSS)
            else LOSS)
termination_by b d _ => b + d
decreasing_by
  all_goals
    (have _hb1 := extAt_le_one G b p
     have _hb2 := extAt_le_budget G b p
     omega)

/-- Budget 0 is the shipped declared value: the repair EXTENDS the
existing spec rather than replacing it. -/
theorem nullValueEB_zero (G : QSGame) (guard : G.Pos → Bool) :
    ∀ (d : Nat) (p : G.Pos),
      nullValueEB G guard 0 d p = nullValueD2 G guard d p := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro p
    rw [nullValueEB, extAt_zero, Nat.add_zero, Nat.sub_zero]
    cases d with
    | zero => simp only [nullValueD2]
    | succ d =>
      simp only [nullValueD2]
      rw [foldMax_congr (fun m => -(nullValueEB G guard 0 d m))
        (fun m => -(nullValueD2 G guard d m))
        (movesAbove G (val_lower (d + 1)) p) _
        (fun m _ => by
          show -(nullValueEB G guard 0 d m) = -(nullValueD2 G guard d m)
          rw [ih d (by omega) m])]
      rw [ih (d + 1 - 3) (by omega) (G.pass p)]

/-- **The budget is value-bearing at a fixed `(pos, depth)` key**: on
`CexRz 0` (the razoring game with a neutral root evaluation, whose
`rz` node is in check with a mate-in-1 below it), budgets 0 and 1
disagree at the key `(rz, 1)` -- so a table serving the budgeted
extension must key on `(pos, depth, budget)`.  The `can_ext`
doctrine's key-must-grow branch, cf.
`extended_value_not_key_independent` (Tricks.lean). -/
theorem ext_budget_value_bearing :
    nullValueEB (CexRz 0) gF 1 1 RzPos.rz = MATE_LOWER + 15 ∧
    nullValueEB (CexRz 0) gF 0 1 RzPos.rz = 0 := by
  constructor
  · have hmm : nullValueEB (CexRz 0) gF 0 1 RzPos.mm = -MATE_LOWER - 15 := by
      rw [nullValueEB_zero]
      exact cexRz_value_mm 0 gF
    rw [nullValueEB]
    rw [if_neg (by decide), if_neg (by decide)]
    have he : extAt (CexRz 0) 1 RzPos.rz = 1 := by decide
    rw [he]
    show (if allIllegalB (CexRz 0) RzPos.rz = true
        then terminalValue (CexRz 0) 2 RzPos.rz
        else
          foldMax (fun m => -(nullValueEB (CexRz 0) gF (1 - 1) 1 m))
            (movesAbove (CexRz 0) (val_lower 2) RzPos.rz)
            (if gF RzPos.rz = true ∧ 2 < 2 then
              (if -(nullValueEB (CexRz 0) gF (1 - 1) (2 - 3) ((CexRz 0).pass RzPos.rz)) < MATE_LOWER then
                max LOSS (-(nullValueEB (CexRz 0) gF (1 - 1) (2 - 3) ((CexRz 0).pass RzPos.rz)))
              else LOSS)
            else LOSS)) = MATE_LOWER + 15
    rw [if_neg (by decide)]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
    have hma : movesAbove (CexRz 0) (val_lower 2) RzPos.rz = [RzPos.mm] := rfl
    rw [hma]
    simp only [foldMax]
    show max LOSS (-(nullValueEB (CexRz 0) gF 0 1 RzPos.mm)) = MATE_LOWER + 15
    rw [hmm]
    decide
  · rw [nullValueEB_zero]
    have h1 : nullValueD2 (CexRz 0) gF 1 RzPos.rz = 0 := by
      rw [nullValueD2_of_fold (CexRz 0) gF 0 RzPos.rz (by decide)
        (fun h => Bool.noConfusion h) rfl]
      have hT : nullTermD2 (CexRz 0) gF 0 RzPos.rz = LOSS := by
        simp only [nullTermD2]
        rw [if_neg (fun h => absurd h.2 (by omega))]
      have hma : movesAbove (CexRz 0) (val_lower 1) RzPos.rz = [RzPos.mm] := rfl
      rw [hT, hma]
      simp only [foldMax]
      have hm0 : nullValueD2 (CexRz 0) gF 0 RzPos.mm = 0 := by
        simp only [nullValueD2]
        rw [if_neg (by decide), if_neg (fun h => Bool.noConfusion h)]
        rfl
      rw [hm0]
      decide
    exact h1

/-- **With budget in hand the recursive extension is a genuinely
different function from the depth shift at the probed key**: at the
quiet (not-in-check) node `r0` above the check chain, the budgeted
recursive extension (budget ample for the whole path, so the cap
never binds) sees the mate the extra ply reveals, while the
key-determined shift function -- `r0` is not in check, so its shift
is zero -- does not.  No depth relabeling recovers the recursive
extension; deepening checking LINES genuinely requires the
key-bearing budget. -/
theorem ext_recursive_ne_shift :
    nullValueEB (CexRz 0) gF 2 2 RzPos.r0 = -MATE_LOWER - 30 ∧
    nullValueE (CexRz 0) gF 2 RzPos.r0 = 0 := by
  constructor
  · have hmm : nullValueEB (CexRz 0) gF 1 1 RzPos.mm = -MATE_LOWER - 30 := by
      rw [nullValueEB]
      rw [if_neg (by decide), if_neg (by decide)]
      have he : extAt (CexRz 0) 1 RzPos.mm = 1 := by decide
      rw [he]
      show (if allIllegalB (CexRz 0) RzPos.mm = true
          then terminalValue (CexRz 0) 2 RzPos.mm
          else
            foldMax (fun m => -(nullValueEB (CexRz 0) gF (1 - 1) 1 m))
              (movesAbove (CexRz 0) (val_lower 2) RzPos.mm)
              (if gF RzPos.mm = true ∧ 2 < 2 then
                (if -(nullValueEB (CexRz 0) gF (1 - 1) (2 - 3) ((CexRz 0).pass RzPos.mm)) < MATE_LOWER then
                  max LOSS (-(nullValueEB (CexRz 0) gF (1 - 1) (2 - 3) ((CexRz 0).pass RzPos.mm)))
                else LOSS)
              else LOSS)) = -MATE_LOWER - 30
      rw [if_pos (by decide)]
      decide
    have hrz : nullValueEB (CexRz 0) gF 2 1 RzPos.rz = MATE_LOWER + 30 := by
      rw [nullValueEB]
      rw [if_neg (by decide), if_neg (by decide)]
      have he : extAt (CexRz 0) 2 RzPos.rz = 1 := by decide
      rw [he]
      show (if allIllegalB (CexRz 0) RzPos.rz = true
          then terminalValue (CexRz 0) 2 RzPos.rz
          else
            foldMax (fun m => -(nullValueEB (CexRz 0) gF (2 - 1) 1 m))
              (movesAbove (CexRz 0) (val_lower 2) RzPos.rz)
              (if gF RzPos.rz = true ∧ 2 < 2 then
                (if -(nullValueEB (CexRz 0) gF (2 - 1) (2 - 3) ((CexRz 0).pass RzPos.rz)) < MATE_LOWER then
                  max LOSS (-(nullValueEB (CexRz 0) gF (2 - 1) (2 - 3) ((CexRz 0).pass RzPos.rz)))
                else LOSS)
              else LOSS)) = MATE_LOWER + 30
      rw [if_neg (by decide)]
      rw [if_neg (fun h => Bool.noConfusion h.1)]
      have hma : movesAbove (CexRz 0) (val_lower 2) RzPos.rz = [RzPos.mm] := rfl
      rw [hma]
      simp only [foldMax]
      show max LOSS (-(nullValueEB (CexRz 0) gF 1 1 RzPos.mm)) = MATE_LOWER + 30
      rw [hmm]
      decide
    rw [nullValueEB]
    rw [if_neg (by decide), if_neg (by decide)]
    have he : extAt (CexRz 0) 2 RzPos.r0 = 0 := by decide
    rw [he]
    show (if allIllegalB (CexRz 0) RzPos.r0 = true
        then terminalValue (CexRz 0) 2 RzPos.r0
        else
          foldMax (fun m => -(nullValueEB (CexRz 0) gF (2 - 0) 1 m))
            (movesAbove (CexRz 0) (val_lower 2) RzPos.r0)
            (if gF RzPos.r0 = true ∧ 2 < 2 then
              (if -(nullValueEB (CexRz 0) gF (2 - 0) (2 - 3) ((CexRz 0).pass RzPos.r0)) < MATE_LOWER then
                max LOSS (-(nullValueEB (CexRz 0) gF (2 - 0) (2 - 3) ((CexRz 0).pass RzPos.r0)))
              else LOSS)
            else LOSS)) = -MATE_LOWER - 30
    rw [if_neg (by decide)]
    rw [if_neg (fun h => Bool.noConfusion h.1)]
    have hma : movesAbove (CexRz 0) (val_lower 2) RzPos.r0 = [RzPos.rz] := rfl
    rw [hma]
    simp only [foldMax]
    show max LOSS (-(nullValueEB (CexRz 0) gF 2 1 RzPos.rz)) = -MATE_LOWER - 30
    rw [hrz]
    decide
  · unfold nullValueE
    have he : extB (CexRz 0) RzPos.r0 = 0 := by decide
    rw [he, Nat.add_zero]
    have h1 : nullValueD2 (CexRz 0) gF 1 RzPos.rz = 0 := by
      rw [nullValueD2_of_fold (CexRz 0) gF 0 RzPos.rz (by decide)
        (fun h => Bool.noConfusion h) rfl]
      have hT : nullTermD2 (CexRz 0) gF 0 RzPos.rz = LOSS := by
        simp only [nullTermD2]
        rw [if_neg (fun h => absurd h.2 (by omega))]
      have hma : movesAbove (CexRz 0) (val_lower 1) RzPos.rz = [RzPos.mm] := rfl
      rw [hT, hma]
      simp only [foldMax]
      have hm0 : nullValueD2 (CexRz 0) gF 0 RzPos.mm = 0 := by
        simp only [nullValueD2]
        rw [if_neg (by decide), if_neg (fun h => Bool.noConfusion h)]
        rfl
      rw [hm0]
      decide
    rw [nullValueD2_of_fold (CexRz 0) gF 1 RzPos.r0 (by decide)
      (fun h => Bool.noConfusion h) rfl]
    have hT : nullTermD2 (CexRz 0) gF 1 RzPos.r0 = LOSS := by
      simp only [nullTermD2]
      rw [if_neg (fun h => absurd h.2 (by omega))]
    have hma : movesAbove (CexRz 0) (val_lower 2) RzPos.r0 = [RzPos.rz] := rfl
    rw [hT, hma]
    simp only [foldMax]
    rw [h1]
    decide

end Sunfish
