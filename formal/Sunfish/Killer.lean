/-
KillerIsKingCapture: the killer-cutoff exception is provably impossible.

`Sunfish/Stalemate.lean` (module comment, point 2) records a potential gap
in sunfish: the killer move is yielded before the sorted moves (sunfish.py
the killer yield, sunfish.py lines 422-423), so a non-capture killer that
fails high would let `bound`
return a value `< MATE_UPPER` at a king-capturable position, breaking the
sentinel requirement of lines 398-401.  This file upgrades that exception
from "empirically absent" to impossible:

    At any position where a king capture is available, the `tp_move`
    entry for that position -- if present -- is itself a king capture.

Hence a killer cutoff at a king-capturable node always returns exactly the
`MATE_UPPER` sentinel, and the killer path cannot under-report.
(Empirical corroboration: 0 violations measured in 694,533 killer cutoffs
over 1,270 real-game positions.)

The invariant is an induction over the execution history of stores into
`tp_move`, which we model by threading the killer table through the
search (`boundKill`).  The load-bearing hypotheses:

(a) in-band windows `-MATE_UPPER < gamma ≤ MATE_UPPER` (the same interval,
    closed under `gamma ↦ 1 - gamma`, as in `Sunfish/Stalemate.lean`): a
    `MATE_UPPER`-scoring child report then always fails high, and an
    out-of-band `gamma > MATE_UPPER` could let a quiet move cut off (and
    be stored) above the capture's sentinel;
(b) value ordering (sunfish.py line 360): king captures have
    `pos.value ≥ MATE_LOWER`, strictly above every other move, so they
    sort first.  We model the consequence directly: the loop runs over
    `orderedMoves`, king captures first;
(c) the killer is POSITION-KEYED (`tp_move.get(pos)`, the killer read at
sunfish.py line 391).  This is
    what makes the induction go through: every store into `tp_move[p]`
    happens inside a call at `p` itself.  A ply-indexed killer table
    shared across positions of the same depth would break the invariant --
    a quiet killer harvested at one position could cut off at a different,
    king-capturable one;
(d) a child reached by a king capture has `score ≤ -MATE_LOWER` and
    returns exactly `-MATE_UPPER` (lines 298-303) -- the king-capture
    normalization already used by `Sunfish/Stalemate.lean`.

Store discipline (sunfish.py lines 382-387): `tp_move[pos] = move` happens
only on a fail-high cutoff and only when `move is not None`; the null-move
and stand-pat yields carry `None` and store nothing, and deletions
(eviction, line 386-387, or a new game) only forget entries, which
trivially preserves the invariant.  sunfish's IID (lines 344-346) stores
through a recursive `bound` call at the same position and smaller depth,
so it is covered by the same induction and not modeled separately.

The model omits the TT-score table and the null move (simplification 5 of
the proof architecture); the null-move residual exception is stated as its
own named condition `NullGuardBlocksAtCaptures` below.

Audit note (exactness): sunfish gates the killer yield by
`pos.value(killer) >= val_lower` (the killer val-gate, sunfish.py line
422).  The gate is not modeled
here; it CANNOT affect `boundKill_spec`, because the load-bearing killer
is a king capture with `val ≥ MATE_LOWER = 47923`, far above every
`val_lower = QS - depth * QS_A ≤ 40` -- a king-capture killer always
passes the gate, and a quiet killer that the gate suppresses only removes
a yield, which the sorted loop re-supplies.  Exactness of a full-engine
model would require it; this file's invariant does not.
-/

import Sunfish.Stalemate

namespace Sunfish

/-! ### The killer table and its invariant -/

/-- `tp_move`: a partial map from positions to the recorded cutoff move
(identified, as everywhere in this model, with the child position it
reaches). -/
def KTable (G : Game) := G.Pos → Option G.Pos

/-- `tp_move[pos] = move` (sunfish.py line 385). -/
def kstore (G : Game) [DecidableEq G.Pos] (t : KTable G) (p m : G.Pos) : KTable G :=
  fun q => if q = p then some m else t q

/-- **KillerIsKingCapture, as a table invariant**: every stored entry is a
legal move of its key position, and at a king-capturable position it is
itself a king capture. -/
def KillerOK (G : Game) (t : KTable G) : Prop :=
  ∀ p m, t p = some m →
    m ∈ G.moves p ∧ (hasKingCapture G p = true → G.eval m ≤ -MATE_LOWER)

/-- The empty table satisfies the invariant (a fresh game / cleared
`tp_move`). -/
theorem killerEmpty_OK (G : Game) : KillerOK G (fun _ => none) := by
  intro p m h
  exact Option.noConfusion h

theorem killerOK_store (G : Game) [DecidableEq G.Pos] {t : KTable G} {p m : G.Pos}
    (ht : KillerOK G t) (hmem : m ∈ G.moves p)
    (hcap : hasKingCapture G p = true → G.eval m ≤ -MATE_LOWER) :
    KillerOK G (kstore G t p m) := by
  intro q m' hq
  by_cases hqp : q = p
  · subst hqp
    unfold kstore at hq
    rw [if_pos rfl] at hq
    have hm' : m' = m := by
      injection hq with h
      exact h.symm
    subst hm'
    exact ⟨hmem, hcap⟩
  · unfold kstore at hq
    rw [if_neg hqp] at hq
    exact ht q m' hq

/-! ### Value ordering, modeled by its consequence -/

/-- Hypothesis (b) baked in: the move loop's order after the sort of
sunfish.py line 360 -- king captures (value ≥ MATE_LOWER, strictly above
all quiet values) first, everything else after. -/
def orderedMoves (G : Game) (p : G.Pos) : List G.Pos :=
  (G.moves p).filter (fun m => decide (G.eval m ≤ -MATE_LOWER))
    ++ (G.moves p).filter (fun m => !decide (G.eval m ≤ -MATE_LOWER))

theorem orderedMoves_subset (G : Game) (p : G.Pos) :
    ∀ m ∈ orderedMoves G p, m ∈ G.moves p := by
  intro m hm
  unfold orderedMoves at hm
  cases List.mem_append.mp hm with
  | inl h => exact (List.mem_filter.mp h).1
  | inr h => exact (List.mem_filter.mp h).1

/-- At a king-capturable position the ordered move list starts with a king
capture. -/
theorem orderedMoves_capture_first (G : Game) (p : G.Pos)
    (hcap : hasKingCapture G p = true) :
    ∃ c cs, orderedMoves G p = c :: cs ∧ G.eval c ≤ -MATE_LOWER := by
  cases (hasKingCapture_iff G p).mp hcap with
  | intro m hm =>
    have hmf : m ∈ (G.moves p).filter (fun x => decide (G.eval x ≤ -MATE_LOWER)) :=
      List.mem_filter.mpr ⟨hm.1, decide_eq_true hm.2⟩
    cases hfl : (G.moves p).filter (fun x => decide (G.eval x ≤ -MATE_LOWER)) with
    | nil => rw [hfl] at hmf; cases hmf
    | cons c cs =>
      refine ⟨c, cs ++ (G.moves p).filter (fun x => !decide (G.eval x ≤ -MATE_LOWER)),
        ?_, ?_⟩
      · unfold orderedMoves
        rw [hfl]
        rfl
      · have hcmem : c ∈ (G.moves p).filter (fun x => decide (G.eval x ≤ -MATE_LOWER)) := by
          rw [hfl]
          exact List.mem_cons_self c cs
        exact of_decide_eq_true (List.mem_filter.mp hcmem).2

/-! ### The search with a killer table -/

/-- The fail-soft move loop with table threading and the store-on-cutoff
of sunfish.py lines 380-387: `best = max(best, score); if best >= gamma:
tp_move[pos] = move; break`. -/
def killLoop (G : Game) [DecidableEq G.Pos] (gamma : Int)
    (f : G.Pos → KTable G → Int × KTable G) (p : G.Pos) :
    List G.Pos → Int → KTable G → Int × KTable G
  | [], best, t => (best, t)
  | m :: ms, best, t =>
    if gamma ≤ max best (f m t).1 then (max best (f m t).1, kstore G (f m t).2 p m)
    else killLoop G gamma f p ms (max best (f m t).1) (f m t).2

/-- `bound` with the killer heuristic, threading `tp_move`:

* line 302-303: king gone -> `-MATE_UPPER`, no store;
* the killer read (line 391) + the killer yield (lines 422-423): try the
  position-keyed killer first; on a fail-high
  cutoff store it back (lines 382-387) -- on a fail low, continue into the
  sorted loop with `best` updated by the killer's score;
* line 360 + 376-387: the ordered loop with store-on-cutoff.

Not modeled: the TT-score table, null move (stores nothing; residual
exception below), QS/futility (the depth ≤ 1 `MATE_UPPER` shortcut of line
371 reports the same sentinel our capture-first head does), IID (a
recursive call at the same position, covered by the depth induction). -/
def boundKill (G : Game) [DecidableEq G.Pos] :
    Nat → G.Pos → Int → KTable G → Int × KTable G
  | 0, p, _gamma, t => ((if G.eval p ≤ -MATE_LOWER then -MATE_UPPER else G.eval p), t)
  | d + 1, p, gamma, t =>
    if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
    else
      match t p with
      | some k =>
        if gamma ≤ -(boundKill G d k (1 - gamma) t).1 then
          (-(boundKill G d k (1 - gamma) t).1,
            kstore G (boundKill G d k (1 - gamma) t).2 p k)
        else
          killLoop G gamma
            (fun m t' => (-(boundKill G d m (1 - gamma) t').1,
              (boundKill G d m (1 - gamma) t').2))
            p (orderedMoves G p)
            (max LOSS (-(boundKill G d k (1 - gamma) t).1))
            (boundKill G d k (1 - gamma) t).2
      | none =>
        killLoop G gamma
          (fun m t' => (-(boundKill G d m (1 - gamma) t').1,
            (boundKill G d m (1 - gamma) t').2))
          p (orderedMoves G p) LOSS t

/-- Hypothesis (d), packaged: a king-capture child answers exactly
`-MATE_UPPER` and never touches the table. -/
theorem boundKill_kingGone (G : Game) [DecidableEq G.Pos] (d : Nat) (k : G.Pos)
    (gamma : Int) (t : KTable G) (h : G.eval k ≤ -MATE_LOWER) :
    boundKill G d k gamma t = (-MATE_UPPER, t) := by
  cases d with
  | zero => simp only [boundKill]; rw [if_pos h]
  | succ d => simp only [boundKill]; rw [if_pos h]

/-- A loop over legal moves of a NON-capturable position preserves the
invariant, provided every child call does. -/
theorem killLoop_preserves (G : Game) [DecidableEq G.Pos] (gamma : Int)
    (f : G.Pos → KTable G → Int × KTable G) (p : G.Pos)
    (hcapP : ¬ hasKingCapture G p = true)
    (hf : ∀ (m : G.Pos) (t : KTable G), KillerOK G t → KillerOK G (f m t).2) :
    ∀ (ms : List G.Pos) (best : Int) (t : KTable G),
      (∀ m ∈ ms, m ∈ G.moves p) → KillerOK G t →
      KillerOK G (killLoop G gamma f p ms best t).2 := by
  intro ms
  induction ms with
  | nil =>
    intro best t _ ht
    simp only [killLoop]
    exact ht
  | cons m ms ih =>
    intro best t hmem ht
    simp only [killLoop]
    by_cases hcut : gamma ≤ max best (f m t).1
    · rw [if_pos hcut]
      exact killerOK_store G (hf m t ht) (hmem m (by simp))
        (fun hc => absurd hc hcapP)
    · rw [if_neg hcut]
      exact ih (max best (f m t).1) (f m t).2 (fun x hx => hmem x (by simp [hx]))
        (hf m t ht)

/-- One step of the loop at a capture-headed list: the capture scores
exactly `MATE_UPPER`, which fails high for any in-band window, so the loop
cuts off immediately, stores the capture, and returns the sentinel --
nothing else can be stored first. -/
theorem killLoop_capture_head (G : Game) [DecidableEq G.Pos] (gamma : Int)
    (d : Nat) (p c : G.Pos) (cs : List G.Pos) (t : KTable G)
    (hceval : G.eval c ≤ -MATE_LOWER) (hg2 : gamma ≤ MATE_UPPER) :
    killLoop G gamma
      (fun m t' => (-(boundKill G d m (1 - gamma) t').1,
        (boundKill G d m (1 - gamma) t').2))
      p (c :: cs) LOSS t
      = (MATE_UPPER, kstore G t p c) := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  have hceq := boundKill_kingGone G d c (1 - gamma) t hceval
  simp only [killLoop, hceq]
  split
  · simp only [Prod.mk.injEq]
    exact ⟨by omega, trivial⟩
  · next h => exact absurd (by omega) h

/-! ### The main theorem -/

/-- **KillerIsKingCapture (single-step preservation + sentinel), proven.**
One call to `boundKill`, starting from any table satisfying the invariant
and probing any in-band window:

* ends with a table satisfying the invariant, and
* returns *exactly* `MATE_UPPER` whenever the position is king-capturable
  (at depth ≥ 1, king not already gone) -- on every path: the killer path
  because the invariant makes the killer itself a king capture (which the
  king-capture normalization scores at the full sentinel, forcing the
  fail-high cutoff), and the loop path because the capture sorts first and
  cuts off before anything else can be stored.

Consequently the killer path cannot under-report the sentinel: the
exception identified in `Sunfish/Stalemate.lean` (module comment, point 2)
is impossible in this model.  Applied along any execution history of
searches starting from `killerEmpty_OK`, the invariant holds forever. -/
theorem boundKill_spec (G : Game) [DecidableEq G.Pos] :
    ∀ (d : Nat) (p : G.Pos) (gamma : Int) (t : KTable G),
      -MATE_UPPER < gamma → gamma ≤ MATE_UPPER → KillerOK G t →
      KillerOK G (boundKill G d p gamma t).2 ∧
      (1 ≤ d → ¬ (G.eval p ≤ -MATE_LOWER) → hasKingCapture G p = true →
        (boundKill G d p gamma t).1 = MATE_UPPER) := by
  have hMU : MATE_UPPER = 69290 := rfl
  intro d
  induction d with
  | zero =>
    intro p gamma t _ _ ht
    exact ⟨ht, fun h1 => absurd h1 (by omega)⟩
  | succ d ih =>
    intro p gamma t hg1 hg2 ht
    by_cases hkg : G.eval p ≤ -MATE_LOWER
    · -- King gone: value -MATE_UPPER, table untouched.
      constructor
      · simp only [boundKill]
        rw [if_pos hkg]
        exact ht
      · intro _ hn _
        exact absurd hkg hn
    · have hw1 : -MATE_UPPER < 1 - gamma := by omega
      have hw2 : 1 - gamma ≤ MATE_UPPER := by omega
      have hf : ∀ (m : G.Pos) (t' : KTable G), KillerOK G t' →
          KillerOK G (boundKill G d m (1 - gamma) t').2 :=
        fun m t' ht' => (ih m (1 - gamma) t' hw1 hw2 ht').1
      cases hkiller : t p with
      | some k =>
        have hkinfo := ht p k hkiller
        by_cases hcut : gamma ≤ -(boundKill G d k (1 - gamma) t).1
        · -- Killer cutoff: the stored move is the killer itself, which by
          -- the invariant is a king capture whenever p is capturable.
          constructor
          · simp only [boundKill, if_neg hkg, hkiller]
            rw [if_pos hcut]
            exact killerOK_store G (ih k (1 - gamma) t hw1 hw2 ht).1
              hkinfo.1 hkinfo.2
          · intro _ _ hcap
            have hkeq := boundKill_kingGone G d k (1 - gamma) t (hkinfo.2 hcap)
            simp only [boundKill, if_neg hkg, hkiller]
            rw [if_pos hcut]
            simp only [hkeq]
            omega
        · -- Killer failed low: then p cannot be capturable (a king-capture
          -- killer scores MATE_UPPER ≥ gamma), so we are in the quiet case.
          have hncap : ¬ (hasKingCapture G p = true) := by
            intro hcap
            have hkeq := boundKill_kingGone G d k (1 - gamma) t (hkinfo.2 hcap)
            have : gamma ≤ -(boundKill G d k (1 - gamma) t).1 := by
              rw [hkeq]
              show gamma ≤ -(-MATE_UPPER)
              omega
            exact absurd this hcut
          constructor
          · simp only [boundKill, if_neg hkg, hkiller]
            rw [if_neg hcut]
            exact killLoop_preserves G gamma
              (fun m t' => (-(boundKill G d m (1 - gamma) t').1,
                (boundKill G d m (1 - gamma) t').2))
              p hncap hf (orderedMoves G p)
              (max LOSS (-(boundKill G d k (1 - gamma) t).1))
              (boundKill G d k (1 - gamma) t).2
              (orderedMoves_subset G p)
              (ih k (1 - gamma) t hw1 hw2 ht).1
          · intro _ _ hcap
            exact absurd hcap hncap
      | none =>
        by_cases hcap : hasKingCapture G p = true
        · -- No killer, capturable: the capture heads the ordered list and
          -- ends the loop at once with the sentinel.
          cases orderedMoves_capture_first G p hcap with
          | intro c hrest =>
            cases hrest with
            | intro cs hcc =>
              have hcmem : c ∈ G.moves p :=
                orderedMoves_subset G p c (by rw [hcc.1]; exact List.mem_cons_self c cs)
              constructor
              · simp only [boundKill, if_neg hkg, hkiller]
                rw [hcc.1, killLoop_capture_head G gamma d p c cs t hcc.2 hg2]
                exact killerOK_store G ht hcmem (fun _ => hcc.2)
              · intro _ _ _
                simp only [boundKill, if_neg hkg, hkiller]
                rw [hcc.1, killLoop_capture_head G gamma d p c cs t hcc.2 hg2]
        · -- No killer, not capturable: plain loop preservation.
          constructor
          · simp only [boundKill, if_neg hkg, hkiller]
            exact killLoop_preserves G gamma
              (fun m t' => (-(boundKill G d m (1 - gamma) t').1,
                (boundKill G d m (1 - gamma) t').2))
              p hcap hf (orderedMoves G p) LOSS t
              (orderedMoves_subset G p) ht
          · intro _ _ hc
            exact absurd hc hcap

/-! ### Residual exception and corollaries -/

/-- **The residual exception, named**: null-move pruning is the one path
that can still end the loop below `MATE_UPPER` at a king-capturable node --
its yield carries `None` (stores nothing, so `KillerOK` survives), but its
cutoff value is not the sentinel.  sunfish guards it only by
`abs(pos.score) < 500` (line 330), and its own FIXME at lines 323-329
concedes the guard is heuristic.  `NullGuardBlocksAtCaptures` is the
condition under which the guard actually closes the hole: with it, the
`MATE_UPPER`-when-capturable invariant of `boundKill_spec` extends to the
null-move-enabled search on ALL paths. -/
def NullGuardBlocksAtCaptures (G : Game) (guard : G.Pos → Bool) : Prop :=
  ∀ p, hasKingCapture G p = true → guard p = false

/-- Quiet positions evaluate strictly inside the mate band -- true in
sunfish because with both kings on the board the material sum is far below
`MATE_LOWER` (the band's whole design).  Needed so that a depth-0
stand-pat can never fake the sentinel. -/
def QuietEvalsInBand (G : Game) : Prop :=
  ∀ p, ¬ (G.eval p ≤ -MATE_LOWER) → -MATE_LOWER < G.eval p ∧ G.eval p < MATE_LOWER

/-- The loop's fail-soft result never exceeds a bound respected by the
initial `best` and by every move's report. -/
theorem killLoop_le (G : Game) [DecidableEq G.Pos] (gamma B : Int)
    (f : G.Pos → KTable G → Int × KTable G) (p : G.Pos) :
    ∀ (ms : List G.Pos), (∀ m ∈ ms, ∀ s : KTable G, (f m s).1 ≤ B) →
      ∀ (best : Int) (t : KTable G), best ≤ B →
      (killLoop G gamma f p ms best t).1 ≤ B := by
  intro ms
  induction ms with
  | nil =>
    intro _ best _t hb
    exact hb
  | cons m ms ih =>
    intro hB best t hb
    have hm := hB m (by simp) t
    simp only [killLoop]
    by_cases hcut : gamma ≤ max best (f m t).1
    · rw [if_pos hcut]
      show max best (f m t).1 ≤ B
      omega
    · rw [if_neg hcut]
      exact ih (fun x hx s => hB x (by simp [hx]) s) (max best (f m t).1) (f m t).2
        (by omega)

/-- **Proven, at the probe's depth**: the stalemate block's in-check
probe (`bound(flipped, MATE_UPPER, 0) == MATE_UPPER`, sunfish.py line
434) is a COMPLETE decision procedure for king-capturability at the
probed position.  The model's depth 1 is the probe: the real probe is a
depth-0 QS -- static leaves plus a scan of the captures -- which is
exactly what `boundKill` at depth 1 computes (eval leaves, `orderedMoves`
capture scan).  Both directions:

* `mpr` (capturable → sentinel) is `boundKill_spec` at
  `gamma = MATE_UPPER`.
* `mp` (no false positives): at a non-capturable position every child
  answer is a negated quiet eval, strictly inside the mate band by
  `QuietEvalsInBand`; the killer, legal by `KillerOK`, is quiet too, so
  its yield cannot reach the sentinel and the loop's fail-soft maximum
  stays `≤ MATE_LOWER < MATE_UPPER` (`killLoop_le`).

Why the statement is pinned to the probe depth rather than `∀ d ≥ 1`:
at deeper `d` the no-false-positives direction is NOT provable without a
sentinel-origins characterization (a "mated" killer -- all its moves
refuted at the exact sentinel -- reports `-MATE_UPPER` one level down
without any king being capturable, the same artifact family as
`boundStale_not_unconditional`).  The engine only ever runs the probe at
depth 0, so the depth-pinned statement is the faithful one. -/
theorem killer_probe_sound (G : Game) [DecidableEq G.Pos]
    (hQ : QuietEvalsInBand G) (p : G.Pos) (t : KTable G)
    (ht : KillerOK G t) (hkg : ¬ (G.eval p ≤ -MATE_LOWER)) :
    (boundKill G 1 p MATE_UPPER t).1 = MATE_UPPER ↔ hasKingCapture G p = true := by
  have hMU : MATE_UPPER = 69290 := rfl
  have hML : MATE_LOWER = 47923 := rfl
  have hLOSS : LOSS = -MATE_UPPER := rfl
  constructor
  · intro hr
    by_cases hcap : hasKingCapture G p = true
    · exact hcap
    · exfalso
      have hquiet : ∀ m ∈ G.moves p, ¬ (G.eval m ≤ -MATE_LOWER) :=
        fun m hm hkgm => hcap ((hasKingCapture_iff G p).mpr ⟨m, hm, hkgm⟩)
      have hchildB : ∀ m ∈ orderedMoves G p, ∀ s : KTable G,
          -(boundKill G 0 m (1 - MATE_UPPER) s).1 ≤ MATE_LOWER := by
        intro m hm s
        have hmm := orderedMoves_subset G p m hm
        have hq := hQ m (hquiet m hmm)
        have hb0 : boundKill G 0 m (1 - MATE_UPPER) s = (G.eval m, s) := by
          simp only [boundKill]
          rw [if_neg (hquiet m hmm)]
        rw [hb0]
        show -(G.eval m) ≤ MATE_LOWER
        omega
      have hunf : boundKill G 1 p MATE_UPPER t
          = (if G.eval p ≤ -MATE_LOWER then (-MATE_UPPER, t)
            else
              match t p with
              | some k =>
                if MATE_UPPER ≤ -(boundKill G 0 k (1 - MATE_UPPER) t).1 then
                  (-(boundKill G 0 k (1 - MATE_UPPER) t).1,
                    kstore G (boundKill G 0 k (1 - MATE_UPPER) t).2 p k)
                else
                  killLoop G MATE_UPPER
                    (fun m t' => (-(boundKill G 0 m (1 - MATE_UPPER) t').1,
                      (boundKill G 0 m (1 - MATE_UPPER) t').2))
                    p (orderedMoves G p)
                    (max LOSS (-(boundKill G 0 k (1 - MATE_UPPER) t).1))
                    (boundKill G 0 k (1 - MATE_UPPER) t).2
              | none =>
                killLoop G MATE_UPPER
                  (fun m t' => (-(boundKill G 0 m (1 - MATE_UPPER) t').1,
                    (boundKill G 0 m (1 - MATE_UPPER) t').2))
                  p (orderedMoves G p) LOSS t) := rfl
      rw [hunf, if_neg hkg] at hr
      cases hkil : t p with
      | some k =>
        have hkinfo := ht p k hkil
        have hkq : ¬ (G.eval k ≤ -MATE_LOWER) := hquiet k hkinfo.1
        have hqk := hQ k hkq
        have hb0 : boundKill G 0 k (1 - MATE_UPPER) t = (G.eval k, t) := by
          simp only [boundKill]
          rw [if_neg hkq]
        simp only [hkil, hb0] at hr
        rw [if_neg (show ¬ (MATE_UPPER ≤ -(G.eval k)) from by omega)] at hr
        have hloop := killLoop_le G MATE_UPPER MATE_LOWER
          (fun m t' => (-(boundKill G 0 m (1 - MATE_UPPER) t').1,
            (boundKill G 0 m (1 - MATE_UPPER) t').2))
          p (orderedMoves G p) hchildB
          (max LOSS (-(G.eval k))) t (by omega)
        omega
      | none =>
        simp only [hkil] at hr
        have hloop := killLoop_le G MATE_UPPER MATE_LOWER
          (fun m t' => (-(boundKill G 0 m (1 - MATE_UPPER) t').1,
            (boundKill G 0 m (1 - MATE_UPPER) t').2))
          p (orderedMoves G p) hchildB LOSS t (by omega)
        omega
  · intro hcap
    exact (boundKill_spec G 1 p MATE_UPPER t (by decide) (Int.le_refl _) ht).2
      (by omega) hkg hcap

end Sunfish
