/-
The MTD-bi driver's window range (sunfish.py, `search`, the bisection
loop):

    gamma = 0                                        -- once, NOT per depth
    for depth in range(1, 1000):
        lower, upper = -MATE_LOWER, MATE_LOWER       -- per depth
        while lower < upper - EVAL_ROUGHNESS:
            score = self.bound(pos, gamma, depth, root=True)
            if score >= gamma: lower = score
            if score < gamma: upper = score
            ...
            gamma = (lower + upper + 1) // 2

This file proves what the bisection actually guarantees about the
windows `bound()` is probed with -- and machine-checks what it does
NOT.

**The finding** (discovered while discharging the "driver range"
premise of the layered kcx specs): the README's and the code comment's
claim "MTD-bi only probes gamma in (-MATE_LOWER, MATE_LOWER]" is TRUE
of every window COMPUTED at the current depth while the scores stay
strictly inside the mate band (`driver_band_invariant`), but FALSE in
general for the carried first probe of a depth: `gamma` is not reset
when `depth` increments, and a MATE-BAND score at the previous depth
moves the bracket out of the band before the last midpoint is taken --
winning by forced mate computes a carried gamma ABOVE `MATE_LOWER`,
and even a bare mated-root score parks it exactly AT `-MATE_LOWER`,
violating the open lower end (`carried_gamma_escapes_band`,
machine-checked).  What holds unconditionally, given only fail-soft
returns in the score band, is the wider invariant
`driver_wide_invariant`: every window -- carried or computed -- stays
in `(-MATE_UPPER, MATE_UPPER]`, exactly the range the wide-window
theorems (`boundStale_spec`, `boundA1_spec`, `boundTT_spec`) are
stated for.

Consequences, recorded honestly: the kcx layered theorems
(`bound_null_spec`, `production_eq_reference`, ...) are stated for
driver-range windows `(-MATE_LOWER, MATE_LOWER]`; a carried first
probe following a mate-band score is not covered by them (and the
reference/production null folds genuinely differ at such windows: a
mate-band fail-LOW pass report exists only there).  A one-line engine
clamp (`gamma = min(max(gamma, 1 - MATE_LOWER), MATE_LOWER)` after the
midpoint, or a per-depth `gamma` reset) would close the gap; that is
an engine decision, recorded here and in formal/README.md, not taken
by this file.
-/

import Sunfish.GameTree

namespace Sunfish

/-- `EVAL_ROUGHNESS` (sunfish.py line 151). -/
def EVAL_ROUGHNESS : Int := 15

/-- The next probe window: `gamma = (lower + upper + 1) // 2`.  Lean's
`Int` division agrees with Python's floor division here
(machine-checked on the negative odd case below). -/
def driverGamma (l u : Int) : Int := (l + u + 1) / 2

/-- Floor-division fidelity: the negative odd case rounds down, as in
Python. -/
theorem driverGamma_floor_fidelity :
    driverGamma (-47923) (-47923) = -47923 ∧ driverGamma 0 1 = 1 := by
  constructor <;> decide

/-- Bisection stays inside an ordered bracket: `l < gamma ≤ u`. -/
theorem driverGamma_in_bracket (l u : Int) (h : l < u) :
    l < driverGamma l u ∧ driverGamma l u ≤ u := by
  unfold driverGamma
  omega

/-- One driver iteration: probe at `st.gamma`, fail-soft update, next
midpoint.  The `while` condition only decides WHETHER another probe
happens, so proving the invariants at every reachable state -- probed
or not -- is conservative and needs no loop-condition bookkeeping. -/
structure DState where
  lower : Int
  upper : Int
  gamma : Int

def dstep (st : DState) (score : Int) : DState :=
  let l := if st.gamma ≤ score then score else st.lower
  let u := if st.gamma ≤ score then st.upper else score
  { lower := l, upper := u, gamma := driverGamma l u }

/-- The per-depth reset, as the driver runs it since `c79b39b`: the
bracket re-initializes to the FULL proven band `[1 - MATE_UPPER,
MATE_UPPER]`, and the carried gamma is inherited unchanged (`gamma = 0`
sits outside the depth loop; the clamp `c72cf6d` had added was removed
with the widening -- see `driver_wide_is_now_the_range`). -/
def depthInit (carried : Int) : DState :=
  { lower := 1 - MATE_UPPER, upper := MATE_UPPER, gamma := carried }

/-- The pre-`c79b39b` narrow bracket, kept for the historical lemmas
below. -/
def depthInitNarrow (carried : Int) : DState :=
  { lower := -MATE_LOWER, upper := MATE_LOWER, gamma := carried }

/-- The wide window invariant. -/
def WideOK (st : DState) : Prop :=
  (-MATE_UPPER < st.lower ∧ st.lower ≤ MATE_UPPER) ∧
  (-MATE_UPPER < st.upper ∧ st.upper ≤ MATE_UPPER) ∧
  (-MATE_UPPER < st.gamma ∧ st.gamma ≤ MATE_UPPER)

/-- The driver-band invariant (bracket in the band, window in the
half-open band). -/
def BandOK (st : DState) : Prop :=
  (-MATE_LOWER ≤ st.lower ∧ st.lower ≤ MATE_LOWER) ∧
  (-MATE_LOWER < st.upper ∧ st.upper ≤ MATE_LOWER) ∧
  (-MATE_LOWER < st.gamma ∧ st.gamma ≤ MATE_LOWER)

/-- One step preserves the wide invariant. -/
theorem dstep_wide (st : DState) (s : Int)
    (hst : WideOK st) (hs : -MATE_UPPER < s ∧ s ≤ MATE_UPPER) :
    WideOK (dstep st s) := by
  have hMU : MATE_UPPER = 69290 := rfl
  obtain ⟨⟨hl1, hl2⟩, ⟨hu1, hu2⟩, _⟩ := hst
  unfold dstep driverGamma WideOK
  by_cases hc : st.gamma ≤ s
  · simp only [if_pos hc]
    omega
  · simp only [if_neg hc]
    omega

/-- One step preserves the band invariant, as long as the score stays
STRICTLY inside the mate band -- the premise a mate-band return
violates. -/
theorem dstep_band (st : DState) (s : Int)
    (hst : BandOK st) (hs : -MATE_LOWER < s ∧ s < MATE_LOWER) :
    BandOK (dstep st s) := by
  have hML : MATE_LOWER = 47923 := rfl
  obtain ⟨⟨hl1, hl2⟩, ⟨hu1, hu2⟩, _⟩ := hst
  unfold dstep driverGamma BandOK
  by_cases hc : st.gamma ≤ s
  · simp only [if_pos hc]
    omega
  · simp only [if_neg hc]
    omega

/-- **The wide invariant, unconditional over fail-soft scores**: every
window the driver ever reaches -- including every carried first probe
of every depth -- stays in `(-MATE_UPPER, MATE_UPPER]`, provided
`bound()` returns scores in `(-MATE_UPPER, MATE_UPPER]` (fail-soft
returns; `boundD2_bounded` is the model-side fact).  This is the
window range the wide-window self-consistency theorems require. -/
theorem driver_wide_invariant (scores : List Int)
    (hs : ∀ s ∈ scores, -MATE_UPPER < s ∧ s ≤ MATE_UPPER) :
    ∀ st : DState, WideOK st → WideOK (scores.foldl dstep st) := by
  induction scores with
  | nil => intro st hst; exact hst
  | cons s tl ih =>
    intro st hst
    exact ih (fun x hx => hs x (List.mem_cons_of_mem s hx)) (dstep st s)
      (dstep_wide st s hst (hs s (List.mem_cons_self s tl)))

/-- The per-depth reset preserves both invariants (the bracket parts
re-initialize; only the carried gamma is inherited). -/
theorem depthInit_wide (carried : Int)
    (h : -MATE_UPPER < carried ∧ carried ≤ MATE_UPPER) :
    WideOK (depthInit carried) := by
  have hMU : MATE_UPPER = 69290 := rfl
  simp only [depthInit, WideOK]
  omega

/-- **The widened bracket IS the window range** (`c79b39b`): with the
bracket initialized to the full band, both endpoints stay inside it
without any clamp -- `lower` only ever rises to a score that met the
window, `upper` only ever falls to one that missed it -- so every probe
window the driver computes lies in `(1 - MATE_UPPER, MATE_UPPER]`, and
a carried gamma inherited from the previous depth is already in range.
This replaces the clamp: `carried_gamma_escapes_band`,
`clampGamma_in_band` and `driver_probe_in_band` are HISTORICAL, and
describe the narrow-bracket driver of `c72cf6d`. -/
theorem driver_wide_is_now_the_range (st : DState)
    (hl : 1 - MATE_UPPER ≤ st.lower) (hu : st.upper ≤ MATE_UPPER)
    (hloop : st.lower < st.upper) :
    -MATE_UPPER < driverGamma st.lower st.upper ∧
      driverGamma st.lower st.upper ≤ MATE_UPPER := by
  unfold driverGamma
  omega

/-- The endpoints keep the wide bracket's sides through any probe made
at a wide-band window -- the invariant `driver_wide_is_now_the_range`
consumes. -/
theorem dstep_wide_sides (st : DState) (s : Int)
    (hl : 1 - MATE_UPPER ≤ st.lower) (hu : st.upper ≤ MATE_UPPER)
    (hs : -MATE_UPPER < s ∧ s ≤ MATE_UPPER) :
    1 - MATE_UPPER ≤ (dstep st s).lower ∧ (dstep st s).upper ≤ MATE_UPPER := by
  unfold dstep
  by_cases hc : st.gamma ≤ s
  · simp only [if_pos hc]
    omega
  · simp only [if_neg hc]
    omega

/-- The band fold: strictly-in-band scores keep any band-satisfying
state in the band. -/
theorem driver_band_fold (scores : List Int)
    (hs : ∀ s ∈ scores, -MATE_LOWER < s ∧ s < MATE_LOWER) :
    ∀ st : DState, BandOK st → BandOK (scores.foldl dstep st) := by
  induction scores with
  | nil => intro st hst; exact hst
  | cons s tl ih =>
    intro st hst
    exact ih (fun x hx => hs x (List.mem_cons_of_mem s hx)) (dstep st s)
      (dstep_band st s hst (hs s (List.mem_cons_self s tl)))

/-- **The band invariant** -- every window computed AT the current
depth stays in the driver band `(-MATE_LOWER, MATE_LOWER]`, GIVEN (a)
strictly-in-band scores and (b) an in-band carried first window.
Premise (b) is exactly the carried-gamma condition;
`carried_gamma_escapes_band` shows it is not free. -/
theorem driver_band_invariant (scores : List Int)
    (hs : ∀ s ∈ scores, -MATE_LOWER < s ∧ s < MATE_LOWER)
    (carried : Int) (hc : -MATE_LOWER < carried ∧ carried ≤ MATE_LOWER) :
    BandOK (scores.foldl dstep (depthInitNarrow carried)) := by
  have hML : MATE_LOWER = 47923 := rfl
  refine driver_band_fold scores hs (depthInitNarrow carried) ?_
  simp only [depthInitNarrow, BandOK]
  omega

/-- **The finding, machine-checked**: the carried gamma escapes the
band.  Left: the previous depth ended with a fail-high MATE-BAND score
(a forced mate for the root, `lower = 50000`), and the carried midpoint
lands ABOVE `MATE_LOWER`.  Right: a mated root (`upper = lower =
-MATE_LOWER` after a fail-low at the exact mate value) parks the
carried gamma exactly AT `-MATE_LOWER`, violating the open end of
`(-MATE_LOWER, MATE_LOWER]`.  Both violate premise (b) of
`driver_band_invariant` for the NEXT depth's first probe. -/
theorem carried_gamma_escapes_band :
    (driverGamma 50000 MATE_LOWER = 48962 ∧ MATE_LOWER < (48962 : Int)) ∧
    driverGamma (-MATE_LOWER) (-MATE_LOWER) = -MATE_LOWER := by
  refine ⟨⟨by decide, by decide⟩, by decide⟩

/-- The clamp `c72cf6d` added on this file's finding:
`gamma = min(max(gamma, 1 - MATE_LOWER), MATE_LOWER)` before each
depth's first probe. -/
def clampGamma (g : Int) : Int := min (max g (1 - MATE_LOWER)) MATE_LOWER

theorem clampGamma_in_band (g : Int) :
    -MATE_LOWER < clampGamma g ∧ clampGamma g ≤ MATE_LOWER := by
  have hML : MATE_LOWER = 47923 := rfl
  unfold clampGamma
  omega

/-- HISTORICAL (the narrow bracket of `c72cf6d`).  **The in-band
premise, discharged for every driver probe**: with the clamp, a depth's first window is in-band unconditionally, and every
LATER probe happens only while the loop condition holds -- which forces
the bracket back inside the band (a mate-band score always breaks the
loop: `lower` only moves up from `-MATE_LOWER` and `upper` only moves
down from `MATE_LOWER`, so an out-of-band endpoint kills
`lower < upper - EVAL_ROUGHNESS`).  Hence every window `bound()` is
probed at satisfies the layered theorems' `(-MATE_LOWER, MATE_LOWER]`
hypothesis, with NO assumption on scores.
`carried_gamma_escapes_band` above is now HISTORICAL: it is the reason
the clamp exists. -/
theorem driver_probe_in_band (st : DState)
    (hl : -MATE_LOWER ≤ st.lower) (hu : st.upper ≤ MATE_LOWER)
    (hloop : st.lower < st.upper - EVAL_ROUGHNESS) :
    -MATE_LOWER < driverGamma st.lower st.upper ∧
      driverGamma st.lower st.upper ≤ MATE_LOWER := by
  have hML : MATE_LOWER = 47923 := rfl
  have hE : EVAL_ROUGHNESS = 15 := rfl
  unfold driverGamma
  omega

/-- The bracket endpoints preserve their one-sided bounds through any
probe made at an in-band window (`lower` rises to a score ≥ gamma >
-MATE_LOWER; `upper` falls to a score < gamma ≤ MATE_LOWER) -- the
invariant `driver_probe_in_band` consumes. -/
theorem dstep_bracket_sides (st : DState) (s : Int)
    (hl : -MATE_LOWER ≤ st.lower) (hu : st.upper ≤ MATE_LOWER)
    (hg : -MATE_LOWER < st.gamma ∧ st.gamma ≤ MATE_LOWER) :
    -MATE_LOWER ≤ (dstep st s).lower ∧ (dstep st s).upper ≤ MATE_LOWER := by
  unfold dstep
  by_cases hc : st.gamma ≤ s
  · simp only [if_pos hc]
    omega
  · simp only [if_neg hc]
    omega

end Sunfish
