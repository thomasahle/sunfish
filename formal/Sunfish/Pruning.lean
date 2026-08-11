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
`-MATE_LOWER` -- at any guard. -/
theorem cexRz_value_mm (v : Int) (g : RzPos → Bool) :
    nullValueD2 (CexRz v) g 1 RzPos.mm = -MATE_LOWER := by
  rw [nullValueD2_of_allIllegal (CexRz v) g 0 RzPos.mm
    (by show ¬ ((0 : Int) ≤ -MATE_LOWER); decide)
    (fun h => Bool.noConfusion h) rfl]
  rfl

/-- The mate-in-1 is real: the depth-2 declared value at the razored
node is the full `MATE_LOWER`.  (`guard` is irrelevant: the pass term
requires `2 < depth`.) -/
theorem cexRz_value_rz (v : Int) (g : RzPos → Bool)
    (hv1 : -MATE_LOWER < v) :
    nullValueD2 (CexRz v) g 2 RzPos.rz = MATE_LOWER := by
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
    negamaxD2 (CexRz v) 2 RzPos.rz = MATE_LOWER := by
  have hmm : negamaxD2 (CexRz v) 1 RzPos.mm = -MATE_LOWER := by
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
      = MATE_LOWER := by
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
`Entry(lower, upper)` with `upper < lower`, the exact shape of
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

end Sunfish
