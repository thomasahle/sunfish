# Formal verification of sunfish's search

A small Lean 4 formalization of the search algorithm in
[`sunfish.py`](../sunfish.py) — specifically of the contract stated in the
docstring of `Searcher.bound` (sunfish.py lines 301–304):

```
Let s* be the "true" score of the sub-tree we are searching.
The method returns r, where
if gamma >  s* then s* <= r < gamma  (A better upper bound)
if gamma <= s* then gamma <= r <= s* (A better lower bound)
```

This is the property the MTD-bi driver `Searcher.search` (lines 491–518)
relies on for its binary search: every call to `bound` must move one end of
the `lower <= score <= upper` bracket, whatever `gamma` it is probed with.

## The consistency decision (enforced on master)

The maintainer's design principle is **enforced on master**: *"`gamma`
may shape termination, and may trigger shortcuts whose value provably
one-side-bounds the same function; `gamma` must never select between
incomparable evaluations of a move."* As of commit `7fdd741` the engine
ships **no reductions at all**: the move loop is a single full-width
`for val, move in sorted(...)` and every move is searched at
`depth - 1`, so every pruning decision in the shipped search is
gamma-independent by construction, the engine has **point specs
end-to-end** on single value functions, transposition entries are
contradiction-free, and `bound`'s docstring is provable exactly as
written (`bound_spec` in `Sunfish/Bound.lean`). The principle is
enforced not by a carefully-shaped reduction but by the absence of any
reduction machinery. The empirical punchline that closed the question
(measured per `docs/TESTING.md`): an *honest* LMR — identical cutoffs,
min-semantics, sound bound propagation — is worth exactly 0.00 ± 34 ELO
over no LMR; the historical re-search variant's measured edge was
entirely its over-claimed bounds. See "Retired mechanisms" below.

**`formal/` has zero sorries**: every theorem is proven, either
unconditionally or under named hypotheses carried in the statement.

## Build

```sh
cd formal
lake build
```

Requires only [elan](https://github.com/leanprover/elan) (toolchain pinned in
`lean-toolchain`, Lean 4.12.0). **No Mathlib** — core Lean and its `omega`
tactic only, so a full build takes seconds.

## What is proven (sorry-free)

- **`Sunfish/GameTree.lean`** — chess-free model. A `Game` is a type of
  positions, `moves : Pos → List Pos` (empty = terminal = side to move has
  lost), and `eval : Pos → Int`. `negamax : Nat → Pos → Int` is the
  depth-limited true value `s*`: `eval` at depth 0, otherwise the max over
  moves of the negated child value, starting from `LOSS` (= `-MATE_UPPER`,
  sunfish.py line 411).

- **`Sunfish/Bound.lean`** — the core result.
  - `bound : Nat → Pos → Int → Int` is sunfish's search stripped to its
    logical skeleton: the fail-soft `best` loop with the early cutoff on
    `best >= gamma` (lines 411–420) over children searched with the flipped
    null window `1 - gamma` at `depth - 1` (line 408). No table, no null
    move, no QS, no killer/IID — see below.
  - `BoundSpec` is the docstring, verbatim:
    `(gamma ≤ r → r ≤ negamax d p) ∧ (r < gamma → negamax d p ≤ r)`.
  - **`bound_spec`: `bound` satisfies `BoundSpec` at every depth, position
    and window.** Proved by induction on depth; the loop is handled by
    `searchMoves_spec`, whose invariant is that the running `best` is itself
    a fail-soft-correct report of the running true maximum. The null-window
    step is the integer fact `b < 1 - gamma ↔ gamma ≤ -b`.
  - `#print axioms bound_spec` reports only `propext, Quot.sound`.

- **`Sunfish/Stalemate.lean`** — the stalemate-correction block
  (sunfish.py lines 422–473), modeled faithfully (king-capture
  normalization of lines 316–322, the `MATE_UPPER` sentinel invariant of
  lines 429–435, the `depth > 2` gate, the null-position in-check probe of
  lines 466–468). Proven sorry-free:
  - **`boundStale_spec`**: the corrected search satisfies the docstring
    against the draw-aware value `negamaxDraw`, under four hypotheses that
    modeling showed to be *individually necessary*: a band-`Bounded`
    evaluation, a correct probe (`CheckProbeOK`), an in-band window
    `-MATE_UPPER < gamma ≤ MATE_UPPER` (an interval closed under the
    null-window flip `gamma ↦ 1 - gamma`), and
    **`MateValuesAreKingCaptures`** — mate-band sentinel values always come
    from a real king capture, never from a shallow-horizon artifact.
  - **`boundStale_not_unconditional`**: a machine-checked 7-position
    counterexample (`Cex`) showing the last hypothesis is *not optional*:
    with bounded evals, a perfect probe and an in-band window, the
    depth-gated correction still returns a fail-low "upper bound" (−20)
    that the draw-aware value (0, a stalemate two plies down scored as a
    fabricated mate) exceeds. This is sunfish's own caveat at lines
    437–450 turned into a refutation.
  - **`negamaxDraw_depth_inconsistent`**: the `depth > 2` gate makes the
    draw-aware value itself depth-dependent (the same stalemate is `LOSS`
    at remaining depth 2 and `0` at depth 3) — the honest statement of the
    divergence between `negamaxDraw` and any depth-independent game value.
  - Supporting sorry-free lemmas: `negamaxDraw_bounded` (values stay in
    the `±MATE_UPPER` band), `boundStale_of_capture` (the sentinel
    invariant holds by construction), `searchMoves_ge_init`,
    `searchMoves_eq_init`, `cex_violates_hypothesis`.
  - The module comment records a killer-move exception found while
    modeling (a non-capture killer failing high would break the "always
    return `MATE_UPPER` if the king is capturable" requirement) — since
    upgraded to *provably impossible* by `Sunfish/Killer.lean`, see below.
  - **The QS val-filter and the exhaustion gate** (second half of the
    file; its comments reference master `bf72b43` — on `29c7887`
    (branch `d2-verify-pending`) the threshold and break live at
    sunfish.py lines 359 and 412–413, and the GATE ITSELF IS GONE from
    the code: the verify-on-suspicion landing replaced it with the
    legality scan, see the d2 block below).  The models above searched
    ALL moves, so "the loop ended at
    `LOSS`, hence every move was refuted" held by construction — the code
    discharged a hypothesis the model assumed.  Now closed: `QSGame` adds
    `pos.value`; the loop runs over `movesAbove (val_lower depth)` (the
    QS break as a filter, sort-order-equivalent like `boundFut`'s
    futility break); `negamaxQS` is the filtered draw-aware value (a
    single `(pos, depth)`-determined function — point-spec doctrine
    preserved); and the #136 gate `depth > 2 or all(pos.value(m) >=
    val_lower ...)` is modeled verbatim (`qsGateB`).  Proven sorry-free:
    - **`boundA1_spec` / `boundQS_spec`**: the filtered, gated (and
      A1-fixed, see below) search satisfies the docstring against
      `negamaxQS` under named hypotheses: `Bounded`,
      `KingCaptureValHigh` (king captures valued ≥ `MATE_LOWER`, so they
      pass every threshold — backed by `EvalBounds.kingCapture_val_above`),
      `MateValuesAreKingCapturesQS`, `CheckProbeOK`, in-band window, and
      (null variant only) `NullBetQS`.
    - **`boundA1_exhaustion` / `boundA1_exhaustion_captures` /
      `correction_trustworthy`** — the exhaustion lemma, the formal
      content of the #136 fix: if no legal move falls below the
      threshold, a filtered loop ending at the untouched `LOSS` sentinel
      certifies that EVERY legal move (full, unfiltered list) has the
      exact king-capture value — at ANY depth; with
      `MateValuesAreKingCapturesQS`, each is refuted by a real capture.
      `correction_trustworthy` covers *both* gate arms at once via
      `gate_implies_no_filtering`: under `ValFloor` (≤ 380) the
      `depth > 2` arm reduces to the `all(...)` arm.
    - **`depth_arm_redundant` / `tables_kill_filter_at_depth2`** —
      finding: the shipped tables' move-value floor is −192, and
      `val_lower 2 = −240` already sits below it, so `all(...)` is
      identically true at depth ≥ 2 and the `depth > 2` arm is a
      scan-skipping optimization, one ply more conservative than the
      tables require.
    - **`qsUngated_not_sound`** — machine-checked 4-position
      counterexample justifying the gate: with every `boundA1_spec`
      hypothesis satisfied, an UNGATED correction (fire on
      `best == -MATE_UPPER` alone) mislabels a filter-truncated
      non-stalemate as a draw (fail-high 0 against value `LOSS`);
      `cexQ_gated_ok` shows the gated search is exactly right there.
    - **`stalemate_fixed_all_depths`** — the #136 repair stated
      positively: moveless positions are corrected at EVERY depth ≥ 1
      (the `all(...)` arm is vacuously true), where `negamaxDraw` scored
      them `LOSS` at depth ≤ 2 (the `Qc4??` bug).
      `negamaxQS_depth_inconsistent` records that depth-inconsistency
      now stems from the depth-keyed filter itself (0/`LOSS`/0 at depths
      1/2/3 on the counterexample game).
    - **The A1 fix, modeled ahead of the code, SHIPPED at `0998739`,
      and since SUPERSEDED** by the verify-on-suspicion landing
      (`29c7887`, which removes `best_real` and the gate from the
      code): `boundA1` remains as the model of the HISTORICAL loop
      (`0998739..29c7887^`) — `best_real` accumulates real-move yields
      only, the sentinel test reads `best_real` (never the
      null-inclusive `best`), the gate is `best < gamma and
      best_real == -MATE_UPPER and (...)`, and mate-band null yields are
      suppressed so `NullBetQS` (the null bet, oracle form) need only be
      trusted below the mate band.  **`a1_unfixed_not_sound`** is the
      machine-checked A1 finding against the PRE-FIX loop shape
      (`boundA1Un`): a sound fail-low null yield masks the sentinel at a
      genuine stalemate and the returned upper bound is exceeded by the
      true value 0 — with every hypothesis, including the null bet,
      satisfied; `a1_fix_repairs` shows `boundA1` returns the exact 0 on
      the same inputs.  The SHIPPED loop is modeled by `boundD2`, next.
    - **The refuted-assumptions ledger** (final sections of the file).
      The fail-high dual of A1 — the correction is an override of exact
      terminal knowledge, and a pseudo-option scoring ABOVE the
      terminal value at a terminal node crosses the table — was first
      isolated as the named obligation `TerminalPseudoSafe`
      (`terminalPseudoSafe_not_free` + `cexT_crossing`: a machine-
      checked crossing on the A1-FIXED loop, with every other
      hypothesis satisfied).  Its two open instances were then REFUTED
      on real boards, and the ledger keeps all three refuted statements
      as countermodels that nothing consumes:
      `NullAtStalemateNonpositive` (witness
      `8/8/8/5k1p/3b1P1P/1p1P1P1P/pN1P1P2/K7 w`, score +175, the pass
      honestly consumed +11 at a genuine stalemate — pre-fix master
      stored `lower=11, upper=0`; mechanism: re-freezing protection
      fails at null-depth 1, where a re-frozen stalemate evaluates as
      stand-pat, not 0), `StandPatAtTerminal` (witness
      `6Rk/6QP/8/8/8/8/8/K7 b`, every pseudo-move a defended capture,
      pre-fix `lower=-1711, upper=-47923`; plus 100+ natural corner-
      mate hits from formal/scripts/standpat_terminal_search.py), and
      `KingCapturableReportsExact` (FALSE under general fail-soft
      windows — a king-capturable child may soundly cut off on
      stand-pat or a partial TT lower without reporting the mate band;
      machine-traced as the same move scoring -368 and -69290 across
      two probes, machine-checked as `cexR_two_windows`).  The
      REPLACEMENT theorem for the last is not that every search of
      such a node reports `MATE_UPPER` — it is that the dedicated
      legality probe does (`legalityProbeCorrect`); the named contrast
      is the root cause of the one still-open channel ("sentinel
      masking") and the next arc's target.
    - **The dedicated legality probe**: `qsProbe` models
      `bound(pos.move(m), MATE_UPPER, 0, root=True)` (lines 383, 464;
      unstored driver semantics per `rootProbe`, CanNull.lean, so no
      table entry can enter the definition of legality).
      **`legalityProbeCorrect`** — at window `MATE_UPPER` the probe is
      a complete decision procedure for "the move left the mover's
      king capturable": the easy direction is `KingCaptureValHigh` (+
      `val_lower_lt_ML`: captures top the order and outrank every QS
      threshold; in the engine also the futility bypass `else
      MATE_UPPER` of the mate-case futility yield), the hard direction is closed by the
      depth-0 pin (static sentinel origins — the model-level residue of
      the `MateValuesAreKingCapturesQS` machinery, cf.
      `killer_probe_sound`'s depth pinning).  `legalScan_iff_
      allIllegal` lifts it to the engine's `all(...)` scan;
      `qsProbe_failLow_legal` is the fail-low half that certifies
      stored moves.
    - **The reference search** — `boundD2` models `reference.py` of
      the kcx landing, the executable spec: the EAGER ENTRY SCAN
      (return the exact `MATE_UPPER` at any king-capturable node
      before table, repetition or loop) is the model's by-construction
      king-capture branch, the fold's `live` bit is the comparison
      `S ≤ n` (real fold vs null contribution), the null verifier
      `nullVerify` withdraws a positive uncertified pass to the fold
      identity at an oracle-confirmed terminal, and the correction
      `d2Fix` fires only on fail-low + `not live` + the oracle scan
      `allIllegalB`.  No `TerminalPseudoSafe`-style obligation, no
      `MateValuesAreKingCapturesQS`, no exhaustion-gate arithmetic:
      the correction's firing condition and the value's terminal
      branch are the SAME position-intrinsic predicate, so the old
      `hexh`/`hmask` alignment machinery vanishes.
    - **The two-layer spec** (decision: Thomas) — *the fold defines
      the semantics*.  Layer 1, **`bound_null_spec`**: the search
      brackets its OWN null-inclusive declared value function
      `nullValueD2` (the pass term is the fold's initial accumulator,
      admitted below the mate band, declined inside it — the
      fold-rule reading of the suppression; oracle-terminal nodes are
      the verified exact values; `(pos, depth)`-determined, killer-
      and window-free) — with NO null bet and NO pass-search
      hypothesis (the pass term is the model's own recursion, the
      dissolved `NullIsPassSearchD2`): premises are `Bounded`,
      `CheckProbeQuiet` (DISCHARGED for the shipped probe:
      `checkProbe_discharged`, via `legalityProbeCorrect` aimed at the
      rotated position plus the structural `RotateNegatesScore`),
      `KillerLegal` (itself a THEOREM given the store trace:
      `killerLegal_lifecycle`), the driver window range (what the
      bisection actually guarantees is proven in `Sunfish/Driver.lean`
      — including the carried-gamma finding), and the single
      chess-position statement `NoZugzwangInMateBand`, which provably
      cannot leave layer 1 while the suppression is report-keyed (the
      -900 vs -MATE_LOWER straddle is a definition-independent ENGINE
      crossing at mate-band-zugzwang positions; the file records why
      the mate-band-capped alternative definition only moves the
      obstruction to the fail-low side).  Layer 2,
      **`nullValue_eq_realValue_of_noZugzwang`**: under `NoZugzwang`
      ("pass-value ≤ best real move", stated once, the validity
      region of the approximation) the declared function collapses
      onto the real-move value `negamaxD2`.  The chess-facing
      **`boundD2_spec`** is the corollary of the two layers; table
      consistency (**`d2_no_crossing`**, `d2_terminal_stores`) needs
      layer 1 only — zugzwang can never cross the table.  Everything
      about the algorithm is unconditional; what remains assumed is
      chess (zugzwang), attached to the accuracy lemma — plus its
      mate-band fragment in layer 1, with the documented reason.
    - **Reference ≡ production (kcx)** — the production consumer
      (`kcx-verify` at `560799c`, lines 450–460) restores
      `KingCapturableReportsExact` WITHOUT the eager scan: a virtual
      (`None`) fail-high is validated before it may cut — a real king
      capture is SUBSTITUTED (the node reports `MATE_UPPER` through a
      real cutoff and `tp_move` stores the true capture: active
      preservation of `KillerAtKingCapturable`), a mate-band claim
      without a capture is the fold identity, and a positive claim at
      a verified terminal is folded to the identity (depth-gated: at
      depth 0 QS must not RETURN the reserved sentinel — the
      96-mismatch lesson).  Sub-mate futility yields are VIRTUAL (line
      417): the old yield-species caution was a live bug (crossed
      entry `Entry(lower=0, upper=-1054)`), now resolved in code.
      **`production_eq_reference`** proves the two consumers compute
      the SAME function at every driver-range window, under
      `CaptureFirst` — itself DISCHARGED from the sort spec
      (`captureFirst_of_sorted`: `MovesSortedByVal`, the one trusted
      primitive "Python's `sorted` sorts", plus `KingCaptureValHigh`
      and its converse `HighValIsKingCapture`) — and `KillerLegal` —
      itself a THEOREM over the `tp_move` lifecycle
      (`killerLegal_lifecycle`: all three store species, eviction,
      and cross-search persistence, machine-checked) — the machine
      twin of the build battery's reference == production over 9,600
      probes.  Transfers:
      **`kingCapturableReportsExact_restored`** (the invariant, now a
      theorem — `CexR` stays as the pre-kcx countermodel),
      `boundKCX_null_spec` / `boundKCX_spec` / **`kcx_no_crossing`**,
      `virtualCutoffValidated` (production never stores a positive
      lower bound at a verified terminal, killer-free),
      `nullIsPassSearch_of_production` (fidelity transfers through
      the equivalence), `kcx_repairs_cexT`, and **`HistoryLegal`** /
      `repetition_never_masks` — the input-validity hypothesis
      (fidelity-class, like `Bounded`) closing the one path that
      precedes the consumer: legal game histories never contain
      king-capturable positions.  Companions:
      **`storedMoveLegal`** (+ `storedMoveLegal_qs`, idealization-free
      at depth 0) — a real fail-high at an in-band gamma certifies its
      move legal, and legality is position-intrinsic so the
      certificate never goes stale — backing `KillerLegal`;
      **`negativeFailLowVerified`** / `d2Fix_unverified_passthrough` —
      the fail-low arm converts to a terminal value ONLY under the
      oracle's confirmation, and then exactly;
      **`positiveNullCutoffVerified`** / `virtualCutoffValidated` — a
      verified-terminal node never stores a positive lower bound, in
      either consumer; **`nullAtMateD2`** — the mate side stays a
      theorem (an enabled pass at an in-check node is exactly the fold
      identity); **`d2_repairs_cexT`** / `kcx_repairs_cexT` — both
      loops return the exact 0 at both windows on the countermodel
      that crossed the pre-verification design.
    - **The primed consumer (`boundKCX'`), modeled AHEAD of the code**
      (final section of the file) — the pending deletion of the veto
      arm's first disjunct (the anchor `score >= MATE_LOWER or`),
      which leaves the band-edge probe as the SOLE implementer of the
      mate-band decline at non-capturable nodes.  In the model the
      deletion is exactly one conjunct — `NCut`'s `rn < MATE_LOWER`
      guard; `NCut'` drops it — and
      **`production_prime_eq_production`** proves the primed consumer
      computes the SAME function as the shipped one at every position,
      depth and driver-range window: the deletion is
      semantics-preserving, and the declared `nullValueD2`, the
      reference `boundD2` and its `useD2` decline stay UNCHANGED (the
      decline remains the declared semantics; only its implementation
      moves into the probe).  Engine: **`bandReport_probe_failsLow`**
      (must-fail-low) — a mate-band virtual claim means the pass
      search failed low at its own window, so layer 1 pins the pass
      VALUE at or below `-MATE_LOWER` and the boundary probe cannot
      fail high; the band-edge arm then re-derives the very
      `-MATE_UPPER` the deleted disjunct hardcoded.  Depth coverage is
      structural (`nCut'_needs_depth`: the interception exists only
      above the null gate; futility yields are sub-gamma by
      construction, the stand-pat is depth 0).  Transfers, one rewrite
      each: `production_prime_eq_reference`, `boundKCX'_null_spec`,
      `kcx'_no_crossing`, `kingCapturableReportsExact_restored'`,
      `boundKCX'_spec`, `kcx'_repairs_cexT`; the terminal side is
      `virtualCutoffValidated'` (one case shorter), and the mate side
      (`nullAtMateD2`, `positiveNullCutoffVerified`) is
      reference-machinery, untouched.  Axiom note: the primed
      equivalence inherits `Classical.choice` (it consumes layer-1
      soundness itself, not just loop congruence) — the same set as
      `boundKCX_null_spec`.  Fidelity note, stated in the file: the
      model identifies the pass search and the boundary probe with one
      definitional recursion; in the engine the two calls may see
      different table states, and the gap is closed by the point-spec
      doctrine (both calls soundly bracket the one
      `(pos, depth)`-determined function, which is all must-fail-low
      consumes).
    - **The double-primed consumer (`boundKCX''`), modeled AHEAD of
      the code** (final section of the file) — the pending
      restructuring that deletes the ENTIRE remaining veto arm (the
      anchor `0 < score and not proof and all(`) and widens the
      post-loop correction gate from `best < gamma and not live` to
      `not live`, so the terminal override fires after the loop in
      BOTH fail directions.  **This is a SPEC CHANGE, not an
      implementation change**: `vetoArm_spec_change_witness` is the
      machine-checked separation — at an oracle-terminal node probed
      with `gamma = -5`, a `-3` pass claim returns loose from the
      shipped-shape consumer but the exact draw `0` from the
      restructured one (both sound; different functions).  The
      reference therefore moves in LOCKSTEP: `boundD2''` deletes the
      terminal-veto disjunct from the verifier (`nullVerify''` is
      band-edge-only) and normalizes the cut path at verified
      terminals; the eager scan and the mate-band decline stay; the
      DECLARED `nullValueD2` needs NO change (its terminal branch is
      already the exact value) and layer 2 is untouched.  Re-proven
      from scratch for the new pair: **`bound_null_spec''`** (layer 1)
      and **`production''_eq_reference''`** — and the premise list
      SHRINKS: the deleted arm's `not proof` was the only killer
      consultation in either consumer's verification, so `kill` leaves
      the reference's definition and **`KillerLegal` leaves every
      double-primed theorem** (it stays load-bearing for the killer
      YIELD, `storedMoveLegal`/Killer.lean).  The model never consumed
      the proof-skip semantically; the engine-side residue is cost
      (the widened scan runs at every surviving null cutoff,
      short-circuited by the first legal move).
      `positiveNullCutoffVerified` is SUBSUMED by the strictly
      stronger **`boundD2''_terminal_exact`**: an oracle-terminal
      non-capturable node returns EXACTLY the terminal value at every
      depth ≥ 1 and driver window, cut or no cut, killer-free
      (corollary `terminal_never_positive''`); the mate side stays
      free (`nullAtMateD2''` — an in-check pass reports the exact
      sentinel and can never fail high), and the in-check-terminal
      cut case of layer 1 no longer needs it at all.  The 96-mismatch
      lesson holds structurally: interception and correction live only
      in the `d + 1` equations, the depth-0 leaf is unchanged
      (`boundKCX''_qs_leaf` is `rfl`).  Transfers:
      `boundKCX''_null_spec`, `kcx''_no_crossing`,
      `kingCapturableReportsExact_restored''`, `boundD2''_spec` /
      `boundKCX''_spec` (via the unchanged layer 2),
      `kcx''_repairs_cexT`, `termFix2_spec_core` (the widened-gate
      core: at a verified terminal the override covers both directions
      by exactness; in the no-cut fold the old `best < gamma` conjunct
      was implied anyway).
    - Loop infrastructure: `searchMoves_eq_init_all` (loop exhaustion,
      the converse of `searchMoves_eq_init`) and `searchMoves_init_max`
      (the `max rn best_real` restructuring equals the code's single
      running `best`).

- **`Sunfish/Liveness.lean`** — the development's first LIVENESS
  theorem: **mate-in-k completeness**.  Everything above is safety
  (the search never over-claims); this file proves the engine also
  FINDS what is there.  `ForcedMate G k p` is the spec ("forced mate
  in ≤ k plies", king-capture vocabulary: a legal move reaching an
  oracle-terminal in-check position — `allIllegalB` ∧ `inCheckB`,
  exactly `terminalValue`'s mate arm — or a legal move, to a
  non-terminal position, all of whose LEGAL defender replies hand back
  a forced mate two plies shorter; an ILLEGAL reply is refuted by the
  sentinel branch, so the spec deliberately does not quantify over it,
  and "legal" on both sides is "does not leave the mover's own king
  capturable").  Proven sorry-free:
  - **`forcedMate_negamaxD2`** (the spine, real-move layer):
    `ForcedMate G k p` puts `negamaxD2` in the mate band
    (`MATE_LOWER ≤`) at EVERY depth `D ≥ k + 1` — one ply better than
    the `k + 2` first planned, because the checkmated leaf needs only
    remaining depth 1 for its terminal branch.  Premise: `ValFloor G
    192` alone (fidelity, tables) — the QS filter provably keeps every
    attacker move at remaining depth ≥ 2 (`mem_movesAbove_of_floor`,
    the liveness respend of `tables_kill_filter_at_depth2`'s
    arithmetic); the defender side needs no floor at all, since
    filtering a defender reply only shrinks the defender's fold.
  - **`forcedlyMated_negamaxD2`** — the mated-side dual
    (value ≤ `-MATE_LOWER` at `D ≥ k + 2`, at any node reached by a
    legal attacker move): a corollary of the spine, not a second
    induction.
  - **`forcedMate_complete`** — the transfer through
    `nullValue_eq_realValue_of_noZugzwang`: under `NoZugzwang` the
    DECLARED value `nullValueD2` — the function `bound_null_spec`
    brackets with no chess premise — is in the band too.
  - **`forcedMate_probe_failsHigh`** (+ `_kcx`, the production
    consumer) — the driver corollary: with a mate in reach of the
    depth, NO driver-range window at or below `MATE_LOWER` can fail
    low, so every bisection probe below the band fails high and the
    MTD-bi bracket is forced into the band: the engine REPORTS the
    mate.  (These two compose with `bound_null_spec` /
    `boundKCX_null_spec` and inherit their `Classical.choice`; the
    spine, dual and transfer are `propext`/`Quot.sound` only.)

- **`Sunfish/Classification.lean`** — milestone 3: **eventual
  classification**.  `eventual_classification` composes the whole
  liveness arc into ONE trichotomy for the game sunfish plays (chess
  without draw rules), at a legally-reached position: a forced mate
  puts the declared value in the band at every `D ≥ k + 1` (win), the
  mated dual at `D ≥ k + 2` (loss), and with no forced mate for either
  side at any horizon the value stays STRICTLY inside the band at
  every depth (neither) — the contrapositives of no-false-mates and
  its dual, floor-free.  Companions: `classification_exclusive` (the
  arms cannot coexist; spine-only, `propext`/`Quot.sound`),
  `eventual_mate_iff` / `eventual_mated_iff` (a forced mate exists at
  SOME horizon iff the value reaches the band at SOME depth), and
  `classification_visible` (+ `_kcx`): after the driver's 15-probe
  budget the converged bracket reports the classification — the
  certified end lands within `EVAL_ROUGHNESS` of the band edge in the
  win/loss arms, and in the neither arm BOTH ends stay strictly off
  the band edges (the slop-free direction, no-false-mates through the
  bracket invariant).  The premise ledger is the module comment: each
  fidelity/chess premise is named with the arm it pays for, alongside
  the two recorded (not implemented) discharge options — the
  depth-decaying guard for `NoZugzwang`, and the frontier-tail
  t-variant for `NoMaskedMobility` (Part B of the file, below).  The
  file also carries an honesty note: "draw" here means no-forced-mate
  in the ruleless game; FIDE draws are NOT detected as 0 (K+B vs K
  converges to the sub-band arm, not to 0).  Part B of the file is the
  frontier-tail t-variant (`negamaxD2t` / `nullValueD2t`) — the
  proven-but-not-shipped discharge of `NoMaskedMobility`; see the
  recorded-design-option block in the Liveness section below.  Axioms
  note: the entire value level of the file — both parts — is
  `propext`/`Quot.sound` only; en route, milestone 2C's no-false-mates
  theorems were freed of an accidental `Classical.choice` (an `omega`
  closing a non-arithmetic goal routes through
  `Classical.byContradiction`; replaced by `absurd`).

- **`Sunfish/Killer.lean`** — **KillerIsKingCapture**, proven sorry-free
  (`boundKill_spec`): threading `tp_move` through the search as state
  (position-keyed, store-on-fail-high-only, sunfish.py lines 373, 388–389,
  415–419), a single call starting from an invariant-satisfying table
  (i) preserves the invariant — *at a king-capturable position the stored
  entry, if any, is itself a king capture* — and (ii) returns exactly the
  `MATE_UPPER` sentinel at every king-capturable node (depth ≥ 1, in-band
  window), on the killer path *and* the loop path. So a killer cutoff can
  never under-report the sentinel: the exception is impossible given
  (a) in-band windows and (b) value ordering (king captures, value
  ≥ `MATE_LOWER`, sort strictly first — modeled by `orderedMoves`).
  Hypothesis (c), position-keying, is load-bearing: a ply-shared killer
  table would break the induction (noted in comments). Applied along any
  execution history from the empty table (`killerEmpty_OK`), the
  invariant holds forever. Empirical corroboration cited in comments:
  0 violations in 694,533 killer cutoffs over 1,270 real-game positions.

- **Retired mechanisms** *(no Lean file — git is the archive)*. Late
  Move Reductions shipped in two forms — gamma-adaptive *re-search* LMR
  (commits `58883ea..7f9f164`: late quiet moves probed at `depth - 2`, a
  reduced fail low yielded as-is, a reduced fail high re-searched at
  full depth) and *deterministic* LMR (`7f9f164..7fdd741`: reduction a
  function of (depth, index, value) only, one search at
  `depth - 1 - LMR`) — and were removed entirely at `7fdd741` after
  measurement: the honest re-search variant (min-semantics, identical
  cutoffs, sound bound propagation) is worth exactly **0.00 ± 34 ELO**
  over no LMR — the shipped variant's edge lived entirely in its
  unsound bound propagation (fail highs propagated as facts about a
  depth they did not search) — and deterministic LMR was net negative
  (its believed −16 price vs re-search was really ~−50; removing it
  measured +69 ± 40, +29 ± 33, +19 ± 45 across time controls). The
  formal apparatus was deleted with the mechanism: `Lmr.lean` (the
  machine-checked TT-crossing counterexample `lmr_tt_crossing`, its
  companion `bound_no_crossing`, and the deleted-even-earlier
  `Vhi`/`Vlo` "interval spec"), `LmrDet.lean` (`boundLmrDet_spec`,
  `boundLmrDet_no_crossing`, the `clamp_noop_*` no-op proofs) and
  `TableClamp.lean` (the `2c95ab0` clamp model) live in git history at
  `58883ea`, `7f9f164` and `7fdd741^` — none of it is needed to specify
  a search with no reductions. What survives is the doctrine they
  taught (see the guideline section below).

- **`Sunfish/CanNull.lean`** — the null-move/repetition search and its
  table, **flagless** as the code is since `eda66ee` (`can_null`
  removed): interior semantics are uniform — the null yield under the
  position-determined guard (lines 364–365, pass searched at
  `depth - 3` as an interior call, so consecutive null moves remain
  permitted — reproduced exactly), the repetition-0 unconditionally at
  `depth > 0` (line 341) — and the two driver probes (the search root,
  line 512, and IID, line 381) pass `root=True`: they skip the table in
  *both* directions and store nothing. Proven sorry-free:
  - **One value function**: `nullValue G hist d p` — no flag argument —
    is what every stored entry describes; the table is keyed
    `(pos, depth)` (reusing `Table` from Tricks.lean) and
    **`boundNullTT_spec`** proves the *point* spec plus preservation of
    `CTableOK`, unconditionally. No zugzwang hypothesis anywhere:
    self-consistency of search + table never depends on the bet.
  - **`ctableOK_empty`**: the empty table satisfies the invariant for
    *any* history — the fact that justifies sunfish clearing `tp_score`
    whenever `history` changes (the invariant is history-relative).
  - **The driver lemma `rootProbe_spec`**: `rootProbe` (= `bound` with
    `root=True`: the same move loop with no table access, no
    repetition-0, no null yield; children are ordinary interior
    searches) fail-soft brackets `rootValue` — the max over real moves
    of the children's interior `nullValue` — and preserves `CTableOK`
    (its only table effect is through its interior children).
    `rootValue` differs from `nullValue` exactly where a gate would
    have fired (`rootValue_eq_nullValue`), and the divergence is
    harmless because the driver never stores it. The IID call is the
    same `rootProbe` shape at `depth - 3` and stores nothing — its
    purpose, killer arrival via `tp_move` fail-highs, is
    `Killer.lean`'s territory, unchanged.
  - **The bridge (`nullValue_plain`), proven under `NullBetOK`**: with
    the bet hypothesis (some real move matches the reduced-depth pass;
    witness required, so guard-passing stalemates are excluded too) and
    an empty history, `nullValue` collapses to the null-free
    `plainValue` — composing with the point spec recovers the
    docstring. Zugzwang threatens only this bridge, never
    self-consistency.
  - Audit note that stays load-bearing: the move generator's *laziness*
    — a null cutoff means the IID recursion never runs, so the table
    state depends on the cutoff, and an eager model would mis-model
    `tp_score`. The cn-keyed machinery (`CTable` on
    `(depth, can_null, pos)` and the two-function layering it forced)
    was deleted with the flag — git has the last cn-keyed model at
    `eda66ee`.

- **`Sunfish/Tricks.lean`**, the proven parts:
  - `soften_null_window` — the mate-score softening lemma: for a window
    strictly above the mate band (`gamma > ML + 1`, `ML ≥ 0`), testing the
    softened negated child score against `gamma` is exactly the raw test
    `r ≤ -gamma - 1`. Softening (mate-distance bookkeeping) does not disturb
    null-window fail-high decisions.
  - `extended_value_not_key_independent` — a concrete counterexample showing
    the extended-search value is **not** a function of `(pos, depth)` alone
    (see below).
  - `mateEntry_deep_service` — `MateDepthMonotone` lifted to arbitrary
    deeper depths: exactly what serving a stored mate entry at a *deeper*
    query depth requires.
  - **`boundTT_spec` — the transposition-table invariant, proven** (the
    sorry discharged): every stored `(lower, upper)` brackets
    `negamax d p` — literally the comment at sunfish.py line 288
    (`# lower <= s(pos) <= upper`). Lookups (lines 335–336) are sound
    *because* every exit re-establishes the invariant; the proof is
    `bound_spec` plus invariant-threading through the state-passing loop
    (`searchMovesTT_spec`), with `tableOK_store`/`tablePart2_ok` for the
    store step and `negamax_bounded` discharging the fresh-entry default
    (the easily-missed `Bounded` side condition, without which
    `Entry(-MATE_UPPER, MATE_UPPER)` is not a valid bracket and the
    theorem is false). This is the point spec — and since `7fdd741`
    removed all reductions, `Bound.lean`'s loop *is* the shipped loop,
    so the point invariant is the whole story.
  - **`boundFut_spec` + `futilityOK_discharged` — `FutilityOK`
    discharged: from stated hypothesis to theorem.** `ValGame` records
    sunfish's *score identity* — `pos.move(move)` builds the child with
    `score = -(pos.score + pos.value(move))`, literal in the code and in
    the comment at lines 397–398 — as a structural property
    (model-faithful, not an assumption). With it, the futility yield is
    *exactly* the depth-0 child search result (`futilityOK_discharged`
    is an equality, not an inequality): the futility test
    `pos.score + val < gamma` is integer-equivalent to the child's
    stand-pat meeting its window, so the child fails high immediately
    and fail-soft returns precisely `-(pos.score + val)`.
    `boundFut_spec` now proves `BoundSpec` for the futility-augmented
    search **unconditionally** (single value function),
    with only `FutilityMateOK` (line 403's king-capture bypass) and the
    in-band window remaining. Fine print made explicit: the ∀-depth
    `FutilityOK` is *not* dischargeable (plain negamax has no stand-pat
    at `d ≥ 1`) — but the search only ever consumes the `d = 0`
    instance; the old statement over-required. **Contrast with the
    retired LMR**: futility's shortcut is a provable one-sided bound of
    the *same* value function (hence consistent, point spec); the
    historical re-search LMR's reduced value was incomparable to the
    full value (hence the TT crossing that ultimately ended it, and no
    honest weaker claim to retreat to).

- **`Sunfish/EvalBounds.lean`** — the numeric facts behind the named
  hypotheses, machine-checked from the transcribed piece-square tables
  (`decide`; the board-string → table link is the one unmodeled step,
  stated in the module comment): the `Bounded` discharge
  (`evalBound_lt_MATE_LOWER` — static evals never touch the mate band),
  the `MATE_LOWER` margin leak and its shipped `K − 13Q` repair
  (`kingGone_check_leaked`, `margin_covers`), the mop-up king table's
  spread (`kEndSpread_lt`), and — new with the QS-filter model — the
  **move-value floor**: `quietDropMax = 192` (the queen's worst table
  delta) with every other `pos.value` term nonnegative
  (`capture_terms_nonneg`, `promotion_terms_nonneg`,
  `castle_rook_deltas`), backing `ValFloor G 192`, and
  `kingCapture_val_above` (a king capture's value clears `MATE_LOWER`
  even after the worst mover drop), backing `KingCaptureValHigh`.

## The premise inventory (current, one place)

What the shipped search's correctness rests on today, sorted by kind.
Everything here is either proven, structural, a trusted primitive, or
explicitly open — there is nothing else.

**Layer 1 — the algorithm (`bound_null_spec`: the search brackets its
own null-inclusive declared value function).** Premises:

| Premise | Kind | Status |
|---|---|---|
| `Bounded` | fidelity (static evals lie in the score band) | `EvalBounds` |
| `KillerLegal` | invariant | **THEOREM** (`killerLegal_lifecycle`: the whole `tp_move` lifecycle — three store species, eviction, cross-search persistence) |
| wide-band window `(-MATE_UPPER, MATE_UPPER]` | driver fact | **DISCHARGED** (`driver_wide_is_now_the_range`: every probe, no clamp, no score assumption) |
| ~~`NoZugzwangInMateBand`~~ | — | **DISCHARGED** by the band-edge arm: a surviving sub-band virtual cutoff re-probes the pass at the boundary window, where both fail directions are decisive (`boundary_window_decisive`), so the search certifies what the premise asserted. **Layer 1 now carries NO chess statement.** Retained only as the record — the premise is FALSE in real chess (witness `8/6p1/6R1/k7/2K5/8/8/8 w`, verified against python-chess: legal, one Black reply, passing mates immediately, no forced White mate within three; three siblings found with it), which is exactly why it had to be verified rather than assumed |

Nothing else: the probe premise is gone entirely (the in-check test is
the board predicate `rotate().king_capture()` = `inCheckB`), the
pass-search premise dissolved into the recursion's definition, and the
band premise is now discharged by the band-edge probe.  **Layer 1 —
everything about the algorithm, including table non-crossing — is
therefore free of chess assumptions.**

**Layer 2 — the approximation's accuracy and completeness.**
`NoZugzwang` (chess: "pass-value ≤ best real move") is stated once and
assumed only in layer 2, where it now has TWO consumers: accuracy
(`nullValue_eq_realValue_of_noZugzwang` — the null-inclusive value
equals the real-move value) and completeness (the liveness subsection
below).  The table can never depend on it: `d2_no_crossing` /
`kcx_no_crossing` are layer-1 results.

**Liveness (mate-in-k completeness, `Sunfish/Liveness.lean`).**
`forcedMate_complete`: a forced mate in ≤ k plies (`ForcedMate`, the
spec in king-capture vocabulary) puts the declared value `nullValueD2`
in the mate band at every depth `D ≥ k + 1`, and hence no driver-range
probe at a window ≤ `MATE_LOWER` can fail low
(`forcedMate_probe_failsHigh` / `_kcx`).  Premises: the real-move
spine (`forcedMate_negamaxD2`) consumes only the fidelity-class
`ValFloor G 192`; the transfer consumes `NoZugzwang` — its second
consumer, both layer 2.  Layer 1 and table consistency still consume
no chess statement.  Recorded design option, NOT implemented: a
depth-decaying null guard (`abs(score) < 500 - 10*depth`) would switch
the pass off at large remaining depth and make completeness
unconditional (no `NoZugzwang`) at `D ≥ k + 52` — a code change,
deliberately not made; Thomas's decision was to give the existing
layer-2 assumption the extra exercise instead.

**Liveness milestone 2 — the `search()` package** (same file unless
noted).  *(A) Driver termination + convergence*: the MTD-bi inner loop
is modeled fuel-indexed (`driverLoop` over `Driver.lean`'s
`dstep`/`depthInit`) and termination is a theorem, not an assumption.
A fail-high raises `lower` to a score `≥ gamma`, and every midpoint
window is strictly inside its bracket, so the width shrinks by ≥ 1 per
probe (`dstep_strictly_narrows`); better, a midpoint probe at least
HALVES the bracket with no score hypothesis at all (`dstep_halves` —
fail-soft overshoot only overshrinks), so the budget is logarithmic:
one carried-window probe plus 14 halvings of the width-138579 reset
band — **15 probes per depth**, after which the loop is provably a
fixed point (`driver_probe_budget`).  On exit the bracket has
`upper - lower ≤ EVAL_ROUGHNESS` and contains the declared value
(`driver_depth_converges`, instantiated for both consumers as
`search_inner_loop_converges` / `_kcx` via `bound_null_spec` /
`boundKCX_null_spec` + `nullValueD2_bounded`).  One honest wrinkle,
recorded in the statement: the per-depth reset floor `1 - MATE_UPPER`
sits one above the value band's floor, so a root whose declared value
is the exact kingless sentinel `-MATE_UPPER` ends with `lower` one
above it — the conclusion's `max V (1 - MATE_UPPER)` says exactly
this; for every other root `lower ≤ V`.

*(B) Best-move soundness*: the docstring's `tp_move` clause is now a
theorem, in the STRONG form — attainment against the declared function
`nullValueD2`, not merely against the fold the search evaluated.  A
fail-high yield is the child probe's fail-low, so layer 1 at the child
certifies the negated declared child value at or above the yield:
`storedMove_attains` (store-site companion of `storedMoveLegal`, same
hypothesis shape, no chess premise).  At the node,
`boundD2_failHigh_attained`: when the real-move loop fails high (and
the null cut didn't fire), the returned report IS the cutting move's
own yield (`searchMoves_failHigh_exact` — the virtual accumulators sit
strictly below the window there), and that move is admitted, legal
(cited), and attains the report.  The substitution and mate-case
futility stores attain through the kingless sentinel
(`substitution_attains`); eviction stores nothing.

*(C) No false mates* — the converse of milestone 1, and the prize: a
mate-band declared value at a legally-reached root IS a forced mate
within the probed depth (`forcedMate_of_nullValueD2`), so the driver's
certified `MATE_LOWER ≤ lower` is never a lie (`mate_report_honest` /
`_kcx`, plus the mated-side dual `forcedlyMated_of_nullValueD2` /
`mated_report_honest`).  Two findings.  First, **`NoZugzwang` is not
needed**: the A1 suppression is baked into the declared function
(`nullTermD2_lt_ML`), so a band-value fold can never originate in the
pass term — no-false-mates holds for `nullValueD2` itself, not just
its zugzwang-free validity region.  Second, the **defender-side QS
filter is a genuine obstruction exactly at frontier depth**: the
value's defender fold ranges over `movesAbove`, real chess doesn't; at
defender remaining depth ≥ 2 the tables' floor admits everything
(`ValFloor G 192` vs `val_lower 2 = -240`), but at depth exactly 1 a
legal reply valued in `[-192, -100)` is invisible, and `CexF`
machine-checks the resulting falsity — all fidelity premises hold, the
depth-2 declared value is the full `MATE_UPPER`, NO forced mate exists
(the defender's ≥100cp escape to stalemate is filtered), and one more
ply computes the honest 0 (`cexF_false_mate_at_frontier`).  The
premise that closes the frontier is `NoMaskedMobility` — retired with
the rejected reduced scan, resurrected as a live layer-2 premise with
its first real consumer, and proven REQUIRED by the countermodel
(`cexF_masked_mobility`).  Spine premises: `ValFloor G 192` +
`EvalQuiet` (fidelity), `NoMaskedMobility` (chess, layer 2), root
legality (`hasKingCapture = false`, the `HistoryLegal` shape).

*(D) pst-swap soundness* (`Sunfish/TableSwap.lean`): why `search` may
retarget the evaluation between searches (the K_MID/K_END assignment,
run in BOTH directions every search per Thomas's review edit, so the
eval in force is a function of the current position alone, never of
module history).  `tp_score` bounds are EVAL-RELATIVE — the keyed
invariant `CTableOK` is game-indexed, and `tableEntries_eval_relative`
machine-checks that an exact entry for one evaluation violates the
invariant for the other — so the swap without the clear would be
unsound, and the per-search clear restores the invariant for the new
evaluation unconditionally (`ctableOK_empty`, any game, cited).
`tp_move` survives instead: `KillerInv` is position-intrinsic up to
the king-gone classification `eval ≤ -MATE_LOWER`, a material fact the
placement-only swap never moves (`SameKingClass`), under which the
whole lifecycle transfers (`killerInv_withEval`,
`killerLegal_lifecycle_pstSwap` — cross-search consumption of
old-eval `tp_move` entries is exactly this theorem).

**Liveness milestone 3 — eventual classification**
(`Sunfish/Classification.lean`).  The arc composed into one theorem:
`eventual_classification`, the trichotomy for the ruleless game at a
legally-reached position — win (band value at `D ≥ k + 1`,
`forcedMate_complete`), loss (the mated dual at `D ≥ k + 2`), neither
(strictly sub-band at EVERY depth — the inversions' contrapositives;
no `NoZugzwang` on this arm).  Premise ledger per arm in the module
comment; `classification_exclusive`, `eventual_mate_iff` /
`eventual_mated_iff`, and the driver corollaries
`classification_visible` (+`_kcx`) — the converged bracket reports the
classification, with `EVAL_ROUGHNESS` slop only on the certified side
and none on the no-false-mates side.  FIDE draws are out of scope by
construction ("neither" = no forced mate, not "score 0").
`NoMaskedMobility` is hereby a LIVE layer-2 premise with three
consumers (no-false-mates, its dual, and the trichotomy's honesty
arm); its recorded discharge option is the frontier-tail t-variant
(Part B of the same file, below).

**Recorded design option, now PROVEN (not shipped): the frontier-tail
variant** (`negamaxD2t` / `nullValueD2t`, Part B of
`Classification.lean`) — verify-on-suspicion applied to the QS filter,
the "other way" that avoids retuning `QS_A`.  Trigger (adjusted from
the first-draft "fold ≤ −MATE_LOWER" and RECORDED, with the reason):
**every ADMITTED move is illegal** — `NoMaskedMobility`'s own
antecedent as a computable, `(pos, depth)`-determined, gamma-free
scan; the fold-value trigger is NOT search-observable (the futility
species prices an illegal admitted move at its child stand-pat, which
can hold `best` above the band while every true admitted contribution
is sub-band), while the admitted-legality scan is species-blind, and
`trigger_shapes_agree_frontier` proves the two shapes equivalent at
the frontier under `EvalQuiet` (at depth ≥ 2 under `ValFloor` both are
inert).  When the trigger fires at a non-terminal node, the fold runs
over the FULL move list — the tail only ADDS defender options.
Proven, all sorry-free and `propext`/`Quot.sound` only:
**no-false-mates for the t-variant carries NO chess premise**
(`forcedMate_of_nullValueD2t` + dual: `ValFloor` + `EvalQuiet` +
root legality — `NoMaskedMobility` is discharged by construction, the
CexF channel closes); the completeness spine SURVIVES with the same
`ValFloor` premise and the same `k + 1` / `k + 2` bounds
(`forcedMate_negamaxD2t` — the attacker's witness is admitted so the
trigger provably never fires at attacker nodes); the two-layer
transfer survives (`nullValueD2t_eq_realValue_of_noZugzwangT`,
`forcedMate_completeT`); `eventual_classification_t` restates the
trichotomy with the honesty arm paid by fidelity alone; and **`CexF`
becomes a positive test** (`cexF_t_positive`: the honest 0 at depths 2
and 3 where the shipped value fabricated `MATE_UPPER`).  The engine
change this models (NOT made): at the correction gate, where the
`not live` scan already runs, when the scan finds legal moves but all
of them sit below `val_lower`, search those moves at `depth - 1` and
fold the yields into `best`/`live` — real yields, `storedMoveLegal`
applies, the `termFix` interaction is unchanged.  Cost: only at
not-live fail-low nodes.  The t-model sits alongside the shipped model
exactly as `boundKCX''` does, so the decision can be made with
theorems (and later an Elo screen) in hand; the search-side consumer
(`boundD2t`/`boundKCXt`) is future work on top of the double-primed
pair.

**The production ≡ reference transfer** additionally uses
`KingCaptureValHigh` (`EvalBounds`) and `CaptureFirst` — itself
**DISCHARGED** (`captureFirst_of_sorted`) from `MovesSortedByVal` +
`KingCaptureValHigh` + `HighValIsKingCapture`.

**Fidelity / structural facts** (true of the code by construction; the
drift guard pins the code regions they describe): `ScoreIdentity`
(`pos.move` builds the child with the negated summed score — what turns
`futility_iff_child_standpat` into a statement about stand-pats),
`MovesSortedByVal` (the one trusted primitive: Python's `sorted`
sorts), `HighValIsKingCapture` and `KingCaptureValHigh` (`EvalBounds`
margins), `EvalQuiet` (static evals stay below the mate band outside
the king-gone zone), `ValFloor` (table floor, used by the sentinel
argument), `KillerAtKingCapturable` (input to the killer fast path —
real-table caveat: entries written by other engine versions void it).

**Input validity**: `HistoryLegal` — legal game histories never contain
king-capturable positions (accepted per Thomas; closes the repetition
path that precedes the consumer).

**Genuinely open**, in priority order:

1. *`gen_moves` implements chess* — assigned to the leanpy /
   lean-surfaces track, not this model.  (The band-edge code arm has
   landed, so the model and the code agree again with no gap in either
   direction.)

**Scoped abstractions** — the complete list of what this model
deliberately does NOT cover, per the standing rule that the model must
otherwise always do what the code does.  Each is a layer boundary, not
a claim about the shipped code that is false:

| Abstraction | Scope |
|---|---|
| QS-as-eval at the leaf | depth 0 returns the QS value; the QS interior (its own stand-pat/capture recursion) is the leaf's fixpoint. The recursion only reaches it at non-futile children, where the leaf is exact; the depth-0 contract itself is `qsStrat_failHigh_of_capture` |
| Deadline / `Stop` | raises at node entry, before any store: an abort can leave a search unfinished, never a table entry unjustified |
| Eviction (`TABLE_SIZE`) | only forgets entries, which preserves every invariant here (and `KillerInv` is proven stable under it) |
| Move ordering | modeled through its load-bearing consequences only: `CaptureFirst` (discharged from the sort spec) and the sort-equivalence of the QS break and futility break with filters |
| Transposition table | `Sunfish/CanNull.lean`'s layer; this file's theorems are about the value function every entry describes |
| `Position` internals | `gen_moves`/`rotate`/`move`/`value` are axiomatized by their structural properties (`ScoreIdentity`, `EvalBounds` facts); the leanpy track owns "these implement chess" |

Retired premises, kept only as records of rejected designs:
`TerminalPseudoSafe`, `NullAtStalemateNonpositive`, `StandPatAtTerminal`
(all refuted), `KingCapturableReportsExact` (refuted for the pre-kcx
loop, restored by construction after it), `scanNewB` (the rejected
reduced scan), `NullBetQS` / `NullBetD2` (the null bet, superseded by
the two-layer split), `CheckProbeOK` (deleted).  `NoMaskedMobility` is
NOT on this list any more: retired with the reduced scan, it was
RESURRECTED by liveness milestone 2C as a live layer-2 premise — its
consumers are no-false-mates, the mated dual, and the trichotomy's
honesty arm (`Sunfish/Classification.lean`), and `CexF` proves it
required for the shipped value function.

## Zero sorries: named hypotheses instead

**`grep -rn sorry formal/Sunfish/*.lean` finds nothing outside prose.**
Every theorem about the shipped search is proven; where a claim is only
conditionally true, the condition is a *named hypothesis in the
statement* — the honest form — not a deferred proof:

- **`NullBetOK` / `nullValue_plain`** (`Sunfish/CanNull.lean`, proven) —
  the null-move bet, exactly as the code places it: some real move at the
  children's depth matches the pass at its reduced `depth - 3`; the
  witness requirement also excludes guard-passing stalemates. Under it
  (and an empty history) the search's value `nullValue` collapses to the
  null-free `plainValue`, composing with the unconditional
  `boundNullTT_spec` into the original docstring. Zugzwang is precisely
  `¬ NullBetOK` and threatens only this bridge, never the
  self-consistency of search + table. (`NullOK` in Tricks.lean remains
  as the same-depth core of the bet.)

- **`FutilityMateOK`** — the one hypothesis left on the futility yield:
  the `else MATE_UPPER` king-capture bypass at line 403 can fail *high*
  and asserts a fact about king captures, not about score arithmetic.
  (`FutilityOK` itself is **discharged** — see the proven inventory
  above.)

- **`MateDepthMonotone` / `MateDepthStable` / `KingGoneStable`** — the
  honest spec for an experimental variant that stores mate results under a
  depth-1000 sentinel key and serves them at any depth. Deeper service is
  justified by `MateDepthMonotone` (and `mateEntry_deep_service` proves
  the lift). *Shallower* service violates the depth-indexed `BoundSpec`
  outright (`negamaxDraw_depth_inconsistent` shows depth-dependence is
  real); it is chess-harmless only under `MateDepthStable` (mate-band
  membership independent of depth ≥ 1), which
  `mateDepthStable_of_kingGoneStable` (proven) derives from
  `KingGoneStable` (a captured king is permanent, on both sides of the
  sign alternation) plus "mate scores only come from real king captures".
  Such a variant should document itself as **weakening `BoundSpec`** to
  mate-band membership only.

- **`NullGuardBlocksAtCaptures` / `killer_probe_sound`**
  (`Sunfish/Killer.lean`) — the residual exception to the sentinel
  invariant: the null-move yield carries `None` (stores nothing, so
  `KillerOK` survives) but can end the loop below `MATE_UPPER`;
  sunfish guards it only by `abs(pos.score) < 500` (line 364, with its own
  comment at 351–363 conceding the guard is heuristic).
  `NullGuardBlocksAtCaptures` names the condition under which the guard
  closes the hole. `killer_probe_sound` (proven, both directions) shows
  the stalemate probe is a complete decision procedure for
  king-capturability *at the probe's depth* — the depth the engine
  actually runs it at. The statement is pinned there deliberately: at
  deeper depths the no-false-positives direction is genuinely unprovable
  without a sentinel-origins characterization (a "mated" killer fakes
  `-MATE_UPPER` one level down — the `boundStale_not_unconditional`
  artifact family), and the docstring documents why.

- **`MateValuesAreKingCapturesQS`** (`Sunfish/Stalemate.lean`) — the
  filtered form of `MateValuesAreKingCaptures`: `negamaxQS` mate-band
  sentinels are always backed by a real king capture.  Same necessity
  argument (the `boundStale_not_unconditional` artifact family), consumed
  by `boundA1_spec`'s `hmask` leg.  **The d2 spec (`boundD2_spec`)
  consumes NO sentinel-origins hypothesis**: the verified correction
  reads the oracle scan, never a score-shaped sentinel, so the
  `hexh`/`hmask` alignment this hypothesis existed for is gone.  It
  remains load-bearing for the historical `boundA1_spec` and for the
  killer file's caveats.

- **`KingCaptureValHigh`** (`Sunfish/Stalemate.lean`) — king captures are
  valued at or above `MATE_LOWER`, so they pass the QS val-filter at
  every depth (`val_lower < MATE_LOWER` is proven).  Concrete backing:
  `EvalBounds.kingCapture_val_above`.  Under d2 this hypothesis is
  spent where the engine actually spends the fact: in
  `legalityProbeCorrect`'s easy direction (the capture tops the probed
  child's order and outranks the depth-0 threshold), not in the spec.
  A PR changing `pos.value`'s capture term must re-check it.

- **`EvalQuiet`** (`Sunfish/Stalemate.lean`, d2 section) — static
  evaluations outside the king-gone zone stay below the mate band; the
  probe's stand-pat branch needs it (a stand-pat must not fake the
  `MATE_UPPER` classification).  Concrete backing:
  `EvalBounds.evalBound_lt_MATE_LOWER`.

- **`KillerLegal`** (`Sunfish/Stalemate.lean`) — away from
  king-capturable nodes, a `tp_move` entry is a legal move.  **Now a
  THEOREM** (`killerLegal_lifecycle`): the full lifecycle induction
  over the event trace — the fail-high store of a searched real winner
  (`storedMoveLegal`), the two king-capture stores (the kcx
  substitution, and the mate-case futility arm via
  `HighValIsKingCapture`), and FIFO eviction — preserves the
  position-intrinsic invariant `KillerInv` from the empty table, and
  the invariant mentions no search state, which is why it persists
  across searches and why exact-position keying is load-bearing.  A PR
  adding a store site must name its event species; one that ply-shares
  the table loses the induction.

- **`RotateNegatesScore`** (`Sunfish/Stalemate.lean`) — structural,
  like `ValGame.score_identity`: passing negates the static score,
  literal in `Position.rotate`.  Consumed by `checkProbe_discharged`.

- **`HighValIsKingCapture` / `MovesSortedByVal`**
  (`Sunfish/Stalemate.lean`) — the converse value fact (a mate-band
  `pos.value` IS a king capture; `EvalBounds` margins) and the one
  trusted primitive "Python's `sorted` sorts, descending by value".
  Together with `KingCaptureValHigh` they DISCHARGE `CaptureFirst`
  (`captureFirst_of_sorted`) and bind the futility store to its event
  species.

- *(dissolved)* `NullIsPassSearchD2` — the null yield is now the
  model's own pass recursion, part of `boundD2`/`boundKCX`'s
  DEFINITIONS; the fidelity residue is the mapping-table row for
  `-self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)` and
  the drift guard covers the line.

- **`ValFloor G B`** (`Sunfish/Stalemate.lean`) — every legal move's
  value is ≥ −B.  `B = 192` is backed by the tables
  (`EvalBounds.quietDropMax_eq` plus the nonnegativity of every additive
  `pos.value` term); `B ≤ 380` is what the historical `depth > 2` gate
  arm needed (`gate_implies_no_filtering`), `B ≤ 240` made that arm
  redundant (`depth_arm_redundant`).  **RETIRED from the correction
  argument by d2**: the legality scan covers QS-filtered moves too
  (filtering is irrelevant to terminality), so no exhaustion arithmetic
  guards the correction any more and `opt_ranges` tuning cannot break
  it.  The lemmas stay as the record of what the old gate rested on.

- **`NullBetQS`** (`Sunfish/Stalemate.lean`) — the whole-tree null-move
  bet in oracle form, as the historical `boundA1_spec` consumes it: a
  guard-passing, `depth > 2`, fail-HIGH null yield BELOW the mate band
  lower-bounds the position's `negamaxQS` value.  Fail-low yields need
  no hypothesis, and the A1 suppression makes mate-band yields dead
  code.  **Superseded for the shipped loop by the two-layer split**:
  no bet appears in any spec premise any more.  Layer 1 brackets the
  null-INCLUSIVE declared function (the option is definition, not
  assumption); zugzwang moved whole into layer 2's `NoZugzwang` — the
  validity region of the null-move approximation, stated once,
  attached to the accuracy lemma — and its only trace in layer 1 is
  the mate-band fragment `NoZugzwangInMateBand` (implied by
  `NoZugzwang`), which the report-keyed suppression provably requires.
  AT verified terminals nothing is assumed in any layer
  (`positiveNullCutoffVerified` / `virtualCutoffValidated`,
  `nullAtMateD2`).  The recursive (non-oracle) form of the bet is
  `NullBetOK` in `Sunfish/CanNull.lean`; zugzwang is precisely the
  failure of either form, and it threatens only value accuracy at
  non-terminal nodes, never the table's self-consistency.

- **`NoMaskedMobility`** (`Sunfish/Stalemate.lean`, kcx section) — "a
  position whose every depth-1-admitted move is illegal has no legal
  move at all."  Born as the premise the REJECTED threshold-reduced
  scan would have needed (`8843bb0` restored full coverage on the
  `reducedScan_needs_premise` countermodel), then RESURRECTED as a
  live layer-2 premise by liveness milestone 2C: it is what closes the
  defender-side QS filter at frontier depth, `CexF` machine-checks
  that no-false-mates is FALSE without it, and its consumers are
  `forcedMate_of_nullValueD2`, `forcedlyMated_of_nullValueD2`, the
  report-honesty corollaries and the trichotomy's honesty arm
  (`Sunfish/Classification.lean`).  Failure shape: all high-valued
  moves illegal while some legal move drops > 100cp of table value; no
  natural chess position is known, but table arithmetic does not
  exclude it.  Recorded discharge option, proof-first: the
  frontier-tail t-variant (Part B of `Classification.lean`), for which
  the premise's role is a theorem.

- **`NoZugzwang` / `NoZugzwangInMateBand`** (`Sunfish/Stalemate.lean`,
  two-layer section) — layer 2's validity region ("pass-value ≤ best
  real move") and its mate-band fragment ("if passing wins in the mate
  band, some real move does too" — you cannot be in zugzwang while
  delivering forced mate).  The fragment is the ONE chess statement in
  layer 1, and the file documents both why it cannot be removed (the
  suppression tests the pass REPORT for band membership; reports
  straddle the band across windows) and how far kcx discharges it
  (pass depths ≤ 2).

- **`CaptureFirst`** (`Sunfish/Stalemate.lean`, kcx section) — king
  captures head the sorted move list; what lets the production loop
  reproduce the reference's eager `MATE_UPPER` through an ordinary
  first-yield cutoff.  **DISCHARGED** (`captureFirst_of_sorted`) from
  `MovesSortedByVal` + `KingCaptureValHigh` + `HighValIsKingCapture`.
  A PR reordering the move sort must re-check the sort spec or
  `production_eq_reference` dies.

- **`HistoryLegal`** (`Sunfish/Stalemate.lean`, kcx section) — input
  validity, fidelity-class like `Bounded`: positions in the game
  history never have a capturable king (every reached position came
  from a legal move).  Closes the one path that precedes the
  production consumer: a king-capturable position inside `history`
  would answer with the repetition 0 and evade the restored invariant
  (`repetition_never_masks`); the reference dodges it via the eager
  entry scan.

- **`ExtKeyIndependent`** — *not* `sorry`d, but stated as the false claim a
  `(pos, depth)`-keyed table makes once search extensions depend on history
  (e.g. a recapture extension keyed on the last-capture square), and refuted
  by an explicit two-state counterexample. Consequence: such a TT key must
  include the last-capture square. Sunfish itself avoids the problem: its QS
  re-derives capture information from `pos` alone, and the one deviant
  semantics that remains — the driver probes' no-null, no-repetition view
  (`root=True`) — never touches the table at all (unstored in both
  directions since `eda66ee`), so `(pos, depth)` is a complete key with no
  flag.

## Model ↔ sunfish.py correspondence

| Lean | sunfish.py |
|---|---|
| `Game.moves`, `[] = lost` | `gen_moves`, king-capture convention (lines 316–322) |
| `Game.eval` | `pos.score` (QS collapsed into eval; lines 369–370) |
| `LOSS = -MATE_UPPER` | `best = -MATE_UPPER` (line 411) |
| `negamax` | the "true score `s*`" of the docstring (lines 301–304) |
| `searchMoves` | the `best` loop + cutoff (lines 411–420) — the loop matches `Bound.lean`'s `searchMoves` exactly: full-width, every child at `depth - 1`, no reductions (since `7fdd741`) |
| `tablePart2` (plain stores) | Table part 2 on master: `Entry(best, entry.upper)` / `Entry(entry.lower, best)` (lines 481–485) — no clamp needed under point specs |
| `-(bound d m (1 - gamma))` | `-self.bound(pos.move(move), 1 - gamma, depth - 1)` (line 408) |
| `BoundSpec` | the docstring (lines 301–304) |
| `NullGame.pass` | `pos.rotate(nullmove=True)` (line 365) |
| `Table`, `TableOK`, `tablePart2` | `tp_score`, `Entry`, `(pos, depth)`-keyed lookup + point store (lines 288–289, 324–336, 481–485) |
| `negamaxDraw`, `boundStale`, `staleFix` | king-capture normalization + stalemate correction (lines 316–322, 422–473) |
| `inCheckB`, `CheckProbeOK` | the null-position check probe `bound(flipped, MATE_UPPER, 0) == MATE_UPPER`; for the kcx layer the probe is DISCHARGED: `checkProbe` + `checkProbe_discharged` prove it correct wherever consumed (`CheckProbeQuiet`), via `legalityProbeCorrect` at the rotated position + `RotateNegatesScore` |
| `MateValuesAreKingCaptures` | the requirement of lines 429–435 and the caveat of lines 437–450 |
| `FutGame.val`, `boundFut` | `pos.value(move)` and the futility yield (lines 392–406) |
| `KTable`, `kstore`, `boundKill` | `tp_move`, killer try + store-on-cutoff (lines 373, 388–389, 415–419) |
| `orderedMoves` | the sort of line 392 (king captures, value ≥ `MATE_LOWER`, first) |
| `nullValue`, `boundNullTT`, `CTableOK` | the interior search (`root=False`): null move (364–365), repetition (341), `(pos, depth)`-keyed table (334–336, 481–485), IID as an unstored probe (375–381) |
| `rootProbe`, `rootValue` | the driver probes (`root=True`): the search root (line 512) and IID (line 381) — no lookup, no store, no repetition-0, no null yield |
| `nullGuard` | `abs(pos.score) < 500` (line 364), gamma-free |
| `QS`, `QS_A`, `val_lower` | the QS constants and threshold (lines 149–150, 359 on `29c7887`) |
| `movesAbove`, `QSGame.val` | the QS break `if val < val_lower: break` (lines 412–413) and `pos.value(move)`; the killer val-gate (line 406) keeps the killer inside the same set |
| `allAboveB`, `qsGateB` | RETIRED — the exhaustion gate is gone from the code; the verify-on-suspicion correction scans the FULL move list with the legality oracle instead (`allIllegalB`, lines 490–492 on `560799c`), so no "did the filter skip anything" arithmetic guards the correction.  Kept for the historical models |
| `negamaxQS`, `qsDrawFix` | the filtered draw-aware value the HISTORICAL gated search bracketed |
| `boundA1` (`best_real` = `S`, `nullMax`, `a1Fix`) | HISTORICAL: the A1-fixed loop as shipped `0998739..29c7887^`; `boundA1Un` is the pre-fix loop shape, `a1_unfixed_not_sound` its machine-checked hole; `cexT_crossing` the fail-high hole that retired it |
| `allIllegalB` | the legality scan `all(self.bound(pos.move(m), MATE_UPPER, 0, root=True) == MATE_UPPER for m in pos.gen_moves())` (correction, lines 490–492 on `560799c`; the interception's copy, lines 457–459) — over the FULL `gen_moves()` list |
| `qsProbe`, `legalityProbeCorrect` | the dedicated legality probe `bound(child, MATE_UPPER, 0, root=True)` (lines 383, 464): unstored driver semantics (`rootProbe`), stand-pat + filtered depth-0 loop; exact at `MATE_UPPER` only — `kingCapturableReportsExact_refuted` is the general-window countermodel |
| `nullVerify`, `useD2`, `nullPartD2` | the reference null verifier `if 0 < score and gamma <= score < MATE_LOWER and not killer and all(...)` and its fold-identity mate-band suppression (reference.py; production's consumer interception makes the same decisions — `nullArm_match`) |
| `boundD2` (`d2Fix`; `not live` = `S ≤ n`) | REFERENCE.PY (the kcx executable spec): eager entry scan = the by-construction capture branch, consumption fold `best, live`, verify-on-suspicion correction `if depth and best < gamma and not live and all(...)` (lines 490–499 on `560799c`) |
| `boundKCX` (`NCut`, `nFoldKCX`) | PRODUCTION (sunfish.py on `560799c`): no eager scan; the consumer interception `if move is None and score >= gamma: ...` (lines 452–460) — substitution / mate-band identity / verified-terminal identity (depth-gated) — plus the same correction; `production_eq_reference` is the bridge |
| `nullValueD2`, `nullTermD2` | the null-inclusive declared value function (layer 1's subject): the pass term as the fold's initial accumulator, band claims declined (the fold rule); terminal branch = the verified exact value |
| `NoZugzwang`, `NoZugzwangInMateBand` | layer 2's validity region and its band fragment (the one chess statement in layer 1) |
| `CaptureFirst` | king captures head the sorted move list (the sort of line 410, `EvalBounds` margins) |
| `HistoryLegal` | input validity: game-history positions are never king-capturable (repetition check, lines 352–353, precedes the consumer) |
| `checkProbe`, `CheckProbeQuiet` | the in-check probe as the code computes it — the legality oracle aimed at `pos.rotate(nullmove=True)`; `checkProbe_discharged` |
| the pass term in `boundD2`/`boundKCX` | `-self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)` (the null yield) — DEFINITIONAL correspondence, no hypothesis; covered by the drift guard |
| `KillTable`, `KillStore`, `KillerInv`, `killerLegal_lifecycle` | the `tp_move` lifecycle: fail-high stores, the substitution store, the mate-futility store, FIFO eviction, cross-search persistence |
| `MovesSortedByVal`, `HighValIsKingCapture`, `captureFirst_of_sorted` | `sorted(((pos.value(m), m) ...), reverse=True)` — the trusted sort primitive and the value-margin converse |
| `driverGamma`, `dstep`, `depthInit` (`Sunfish/Driver.lean`) | the MTD-bi bisection: `gamma = (lower + upper + 1) // 2`, per-depth bracket reset, CARRIED gamma; `driver_band_invariant`, `driver_wide_invariant`, `carried_gamma_escapes_band` |
| virtual futility yield | `yield (move, MATE_UPPER) if val >= MATE_LOWER else (None, pos.score + val)` (line 417) — sub-mate futility estimates are VIRTUAL; the mate case is a real king capture |
| `negamaxD2`, `terminalValue` | the `(pos, depth)`-determined value the shipped search brackets: exact terminal value at oracle-terminal nodes (line 472), plain filtered fold elsewhere |
| `KillerLegal`, `storedMoveLegal` | `tp_move` as a mobility certificate: exact-position key, store on real fail-high only (lines 362–366, 440–445) |
| `CorrectionTerminal`, `TerminalPseudoSafe`, `NullAtStalemateNonpositive`, `StandPatAtTerminal`, `KingCapturableReportsExact` | the refuted-assumptions ledger — countermodels only, consumed by nothing |
| `ValFloor`, `EvalBounds.quietDropMax` | the `pos.value` floor read off the tables — RETIRED from the correction argument with the gate; still documents the table facts |
| `KingCaptureValHigh` | king captures valued ≥ `MATE_LOWER` (the sort of line 410; the mate-case futility yield of line 417) — spent in `legalityProbeCorrect` and `CaptureFirst`'s backing |
| `EvalQuiet` | static evals stay below the mate band outside the king-gone zone (`EvalBounds.evalBound_lt_MATE_LOWER`) |

The historical rows — `boundLmr`/`red` (re-search LMR), `boundLmrDet`/
`negamaxDet` (deterministic LMR), `clampHigh`/`clampLow` (the `2c95ab0`
clamp) — were removed with their mechanisms and Lean files at `7fdd741`;
see "Retired mechanisms" above.

## Model fidelity

Audited against master at commit `9b1a7b4` (2026-08-05), re-audited
after the LMR removal at `7fdd741` (2026-08-06), after the
`can_null` removal at `eda66ee` (2026-08-07), and after the
verify-on-suspicion landing (2026-08-08, branch `d2-verify-pending` at
`29c7887`).  Line references in the d2 rows and bullets are to
`29c7887` (sunfish.py lines 298–516); older bullets keep their
`eda66ee` references — the d2 patch only shifts lines inside `bound()`.
The model tracks the code exactly except these explicitly listed
abstractions, each with its justification:

- **QS-as-eval at the `Bound.lean` layer**: depth 0 returns `eval`
  directly; QS's interior (stand-pat + capture recursion at clamped
  depth 0) is a fixpoint the abstract model treats as its evaluation.
  Its one load-bearing property — the stand-pat identity — is what
  `ValGame.score_identity` captures and `boundFut_spec` consumes.
- **Deadline/`Stop`** (lines 305–310): raises at node *entry*, before
  any store, so an abort can leave a search unfinished but never a
  table entry unjustified — aborts cannot corrupt `TableOK`/`CTableOK`.
- **Eviction** (`TABLE_SIZE`, lines 486–487 and the `tp_move` twin at
  418–419): only forgets entries, which trivially preserves every table
  invariant here.
- **`depth = max(depth, 0)`** (line 315): corresponds to the model's
  `Nat` depths with saturating subtraction — verified aligned.
- **Killer val-gate** (line 388) not modeled in `Killer.lean`; cannot
  affect `boundKill_spec` (king captures have `val ≥ MATE_LOWER`, far
  above every `val_lower`) — see the audit note there. The killer's
  in-loop duplicate is searched at the same `depth - 1` as the killer
  try itself, so it is idempotent under the fold max (the code's own
  comment at lines 386–387: "the tp will fix things for us"); the
  stalemate probe is an ordinary interior depth-0 call, table key
  `(flipped, 0)` (the repetition and null gates need `depth > 0` /
  `> 2`, so both are dead there).
- **Move ordering** (line 392) is modeled only through its load-bearing
  consequence: king captures sort first (`orderedMoves`).
- **QS-filter section** (Stalemate.lean, second half): audited against
  master at `bf72b43` (2026-08-07); its line references are to that
  commit.  The sorted `break` of lines 399–401 is modeled as the filter
  `movesAbove` (identical searched set under the sort order — the same
  abstraction `boundFut` uses for the futility break, and the killer
  val-gate of line 395 keeps the killer path inside the filtered set).
  The futility break of lines 407–413 is not re-modeled in this loop; it
  cannot disturb the exhaustion argument because its yield
  `pos.score + val > -MATE_UPPER` always raises `best` off the sentinel
  before breaking (the code's own comment at lines 464–466), and
  `boundFut` covers its bound-correctness separately.
- **A1 status**: HISTORICAL — the fix shipped at `0998739` and was
  superseded by the verify-on-suspicion landing, which removed
  `best_real` and the exhaustion gate from the code.  `boundA1` models
  the `0998739..29c7887^` loop; the pre-fix shape is kept as
  `boundA1Un` with its machine-checked stalemate-masking hole
  (`a1_unfixed_not_sound`), and `cexT_crossing` is the fail-high hole
  that retired the A1 design itself.
- **kcx status**: audited against `kcx-verify` at `560799c`
  (2026-08-08); `reference.py` of the kcx build is the executable spec
  and `boundD2` models it exactly (its eager entry scan is the model's
  king-capture branch; the invariant-restoring interception of
  production, lines 452–460, is modeled by `boundKCX`'s `NCut` /
  `nFoldKCX` on the null yield, with `production_eq_reference` the
  machine bridge — the engine-side twin was checked bound-for-bound
  over 9,600 probes).  Production's stand-pat interception at QS (the
  substitution arm is NOT depth-gated) is what backs the model's
  by-construction depth-0 capture branch — with kcx this is
  construction, not idealization.  Point by point: the consumption fold's
  `best, live` (lines 450 and 461–462, `if score > best: best, live = score,
  move is not None`) is modeled without an extra bit — the null
  contribution `n` folds first, so `live` is exactly `n < S` (a real
  yield strictly improved) and the code's `not live` is `d2Fix`'s
  `S ≤ n`; ties go to the earlier (virtual) yield in both.  The
  correction `if depth and best < gamma and not live and all(...)`
  (lines 490–492) is `d2Fix` gated on `allIllegalB`; the `if depth`
  exclusion is structural (`d2Fix` exists only at depth ≥ 1, and the
  model's depth 0 is the eval).  The reference null verifier is
  `nullVerify` (in production the same three decisions live in the
  consumer interception — `nullArm_match` proves them identical); the
  mate-band handling is fold-identity normalization, which the model
  represents as option disabling (`useD2`) — equivalent by the fold
  rule.  The scan runs over `pos.gen_moves()`, the FULL
  list, modeled as `G.moves p` — NOT `movesAbove` — which is what lets
  the exhaustion gate retire.  The legality probes and the IID probe
  are `root=True`: unstored in both directions (`rootProbe`,
  CanNull.lean), so the `(pos, depth)` key stays complete and no table
  entry enters the definition of legality.
- **Re-audit at the `kcx-review-fixes` merge** (`Searcher.bound`
  `f73e3217…` → `e7638485…`): a comment-and-local-name-only change (the
  interception's `king` → `proof` rename).  No semantics moved, and the
  strongest available evidence says so mechanically: the MINIFIED
  output and the `pack.sh` byte count are IDENTICAL before and after.
  The audit hash was refreshed in the same commit as the code change,
  per the standing rule; this line is the README half of that refresh.
  Two edits to the same block were rejected on measurement and are NOT
  in the model because they are not in the code: moving the killer
  lookup below the null/stand-pat yields (not behaviour-preserving —
  `tp_move` is position-keyed and QS below the null is not ply-limited,
  so the null subtree can transpose back to the same board and store at
  that key: 10 disagreements in 3.0M reads), and a walrus caching the
  terminal scan (the fold scan returned true 4 times in 43,006 scans).
- **The pass-term correspondence is definitional** (the dissolved
  `NullIsPassSearchD2`): the model's null yield IS its own recursive
  pass probe, exactly sunfish.py's
  `-self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)`.
  Nothing is assumed; what remains checkable is that the code line
  stays what it is — the drift guard pins it.
- **Model-code drift guard**: `formal/scripts/model_audit.py` (run by
  `tests/test_model_audit.py` in CI) hashes every audited region of
  sunfish.py — `Searcher.bound`, `Searcher.search`, `Position.rotate`/
  `move`/`value`/`gen_moves`/`king_capture`, and the constants — and
  fails on any change without a same-commit re-audit + hash refresh
  (`--update`).  **It also guards the citation class the hashes are
  structurally blind to**: a stale line NUMBER in a Lean comment is not
  code drift, so hashing cannot see it, and `Killer.lean` had rotted
  that way (citing 339/356-357/366 for code that had moved to
  391/422-423, alongside a stale `Bound.lean:20`).  Those citations are
  fixed, and the guard now (a) checks that every distinctive source
  ANCHOR the model cites by name still occurs in `sunfish.py`
  (`ANCHORS`), and (b) ratchets the number of raw line-number citations
  in `formal/Sunfish/*.lean` so it can fall but never rise.  Prefer
  anchors — function names and distinctive source text — over line
  numbers in new comments.
- **The golfed consumer status** (audited at `6ecb4af`): `live` is the
  STICKY two-way-evidence bit, modeled as the untouched-fold test
  `S = LOSS`.  The correction scan is FULL COVERAGE and both scan sites
  use the BOARD PREDICATE `pos.move(m).king_capture()` — definitionally
  the model's `allIllegalB` (and the in-check test is
  `rotate().king_capture()` = `inCheckB`); search probes remain only
  where a search VALUE is needed (the band-edge boundary probe, PR
  #162).  `scan_sites_unreachable_at_capturable`: neither scan site
  ever sees a capturable-opponent-king board.  REJECTED designs stay on
  record: `reducedScan_needs_premise` (the accepted disqualifier) and
  `scanNewB` — consumed by nothing shipped.  (`NoMaskedMobility`, once
  on that list, has since been resurrected as a live layer-2 premise —
  see its inventory entry.)  The
  killer fast path stands (`fastPath_decides`/`fastPath_skip_sound`).
- **The king-capture contract is STRATIFIED by depth** (`6ecb4af`
  deleted the depth-0 arm of the interception):
  **depth 0 — fail-high only** (`qsStrat_failHigh_of_capture`: the QS
  loop gives it for free, since either the stand-pat already meets the
  window or the capture does), **depth ≥ 1 — exact sentinel**
  (`boundKCX_capture_exact`), assembled as
  `kingCaptureContract_stratified`.  The weakened leaf costs the layers
  above nothing because of one arithmetic identity —
  **`futility_iff_child_standpat`: the parent's futility test IS the
  child's stand-pat test** (`pos.score + val < gamma` ⟺
  `-(pos.score + val) ≥ 1 - gamma`) — so a parent only SEARCHES
  children whose stand-pat cannot cut, and at those the QS loop runs
  past the stand-pat into the capture and reports the sentinel exactly
  (`qsStrat_exact_of_searched`, under the structural `ScoreIdentity`);
  hence the fold's two-way evidence survives
  (`searched_yield_two_way`).  The one futility-bypassing path, the
  killer yield, is a mobility certificate (`KillerLegal`).
  **CLOSED (the fold landed).** Futility is now a modeled yield
  SPECIES, not a documentary caution: `futileAt` is the engine's test
  read through the identity above, `searchedAt` is the set the loop
  actually searches, and `futTerm` folds the synthetic suffix estimates
  into the VIRTUAL accumulator.  Only searched real results reach `S`,
  so the correction's gate `S = LOSS` is exactly the code's `not live`
  at every depth — which is what the stratified leaf needed.  Two
  consequences: `termFix` now serves both models (the gates coincided,
  so `d2Fix`/`golfFix` merged), and the recursion only ever evaluates a
  depth-0 child at a NON-futile position, where the exact-sentinel leaf
  and `qsStrat` agree — so the model's leaf is faithful without
  weakening it.  `production_eq_reference` and
  `kingCapturableReportsExact_restored` therefore describe the SHIPPED
  consumer.  Two fidelity details the fold forced, both now right:
  a king capture is never futility-pruned (the code's `val >=
  MATE_LOWER` arm yields a real `MATE_UPPER` that cuts), and a futile
  child's declared contribution is covered by the virtual accumulator
  (`futile_contrib_le`), which is why the declared function needs no
  change and stays gamma-free.  `stratLeaf_needs_futility` is retained
  as the record of what the split fixes.
- **The driver-range finding**- **The driver-range finding**- **The driver-range finding** (`Sunfish/Driver.lean`): "MTD-bi only
  probes `(-MATE_LOWER, MATE_LOWER]`" — asserted by this README and
  the code comment — is TRUE for every window computed at the current
  depth while scores stay strictly in the band
  (`driver_band_invariant`) and FALSE for the carried first probe of a
  depth after a MATE-BAND score (`carried_gamma_escapes_band`,
  machine-checked: a forced-mate score carries a gamma above the band;
  a mated root parks it exactly at `-MATE_LOWER`).  What holds
  unconditionally is the wide invariant `(-MATE_UPPER, MATE_UPPER]`
  (`driver_wide_invariant`) — the range the pre-kcx wide-window
  theorems cover.  The kcx layered theorems are stated for the band;
  carried out-of-band first probes are NOT covered by them, and
  reference/production genuinely differ there (a mate-band fail-LOW
  pass report exists only at such windows).  A one-line clamp
  (`gamma = min(max(gamma, 1 - MATE_LOWER), MATE_LOWER)`) or a
  per-depth `gamma` reset would close the gap — CLOSED in `c72cf6d` by
  the clamp, then SUPERSEDED in `c79b39b`, which widened the bracket to
  the full band `[1 - MATE_UPPER, MATE_UPPER]` and dropped the clamp.
  Current state: `driver_wide_is_now_the_range` + `dstep_wide_sides` —
  both endpoints stay inside the wide band without any clamp, so every
  probe window (carried ones included) lies in
  `(1 - MATE_UPPER, MATE_UPPER]`, unconditionally.  The clamp lemmas
  and `carried_gamma_escapes_band` are historical.
  **Now closed — and note which side moved: the MODEL, not the code.**
  The shipped interception was already gated on `score >= gamma`, so a
  fail-LOW mate-band pass report has always been folded raw; it was the
  reference model's `useD2` that suppressed band-valued yields in both
  directions, which only coincided with the code inside the narrow
  band.  `useD2` is now gated on fail-high too (`gamma ≤ nv → nv <
  MATE_LOWER`), matching the code, and the layered specs
  (`bound_null_spec`, `boundD2_spec`, `d2_no_crossing`,
  `production_eq_reference`, `boundKCX_*`, `kcx_no_crossing`) are
  restated at the WIDE band `(-MATE_UPPER, MATE_UPPER]`.  Since
  `driver_wide_is_now_the_range` puts every driver window in
  `(1 - MATE_UPPER, MATE_UPPER]`, the in-band hypothesis is once again
  discharged for every probe, with no clamp and no assumption on
  scores.  Two proof steps had to learn the wide band: the disabled-
  option case now extracts BOTH facts from the re-gated `useD2` (a
  false gate means a fail-HIGH band report, and the integer flip pins
  the pass call as a fail-low at its own window), and the null part is
  bounded strictly below the sentinel by suppression on a fail-high and
  by the window on a fail-low.
- **Sentinel exactness is now construction, not idealization**: the
  model's king-capture branch (exact `MATE_UPPER` at every window and
  depth) is the reference's eager entry scan verbatim, and production
  earns it through the consumer interception
  (`kingCapturableReportsExact_restored`).  The pre-kcx gap — a
  king-capturable child soundly cutting off on its stand-pat or a
  partial table lower — is preserved as the REFUTED-for-the-old-loop
  `KingCapturableReportsExact` countermodel (`CexR`,
  `cexR_two_windows`): the sentinel-masking channel's root cause, now
  closed in code.  The killer yield at king-capturable nodes rests on
  `KillerAtKingCapturable` (Killer.lean), which the substitution arm
  now ACTIVELY preserves by storing the true capture; the table's
  entries at such nodes are exact for the same reason
  (KingCapturableTableExact — every return is the sentinel, so every
  stored bound is; the CanNull layer's invariant carries it).
- **Yield species (load-bearing in BOTH code and model)**: the engine's
  `moves()` yields are two truthy species (searched real results, and
  the mate-case futility yield — itself a real king capture that cuts)
  and three virtual ones (null, stand-pat, and SUB-MATE futility
  estimates).  The distinction was once a documentary caution here,
  then turned out to be a live engine bug (the crossed
  `Entry(lower=0, upper=-1054)`), fixed by making sub-mate futility
  yields virtual — and it is now modeled structurally as well:
  `futileAt` / `searchedAt` / `futTerm` keep the synthetic suffix
  estimates out of the real accumulator `S`, which is exactly what
  makes the correction's `S = LOSS` gate equal the code's `not live`.
  A PR adding a yield species must say which accumulator it feeds and
  what certifies any mobility it claims.
- **Ungated gate arms** (historical, `boundA1`): for in-band windows the
  model's `best < gamma ∧ best_real = LOSS` gate coincides with the
  code's bare `best == -MATE_UPPER` test (a `LOSS` loop result is
  automatically below an in-band window); out-of-band windows never
  reach the gate in the engine (the table's fresh-entry cutoffs answer
  them), which is why the spec carries the in-band hypothesis — as does
  `boundD2_spec`, for the same reason.

### Landing note (the #158 review, 2026-08-11)

The SHIPPED consumer is now the DOUBLE-PRIMED design: the terminal-veto
arm is deleted and the correction gate widened to `not live` in both
fail directions (`boundD2''`/`boundKCX''`, `bound_null_spec''`,
`production''_eq_reference''` — a verified SPEC CHANGE, tighter at
terminals: `vetoArm_spec_change_witness`), and the band fast-path
disjunct is deleted (`bandReport_probe_failsLow`).  The null validation
now lives IN THE GENERATOR at the yield site (veto by omission — the
fold rule made literal); this is a refactoring of the same function
(node-identical, 457,039 nodes at depth 9 from startpos) and the
mapping rows reading "consumer interception" should be read as the
generator's validated null yield.  Additions landed with the review:
the root-eviction guard on `tp_move` (the Qxc6 race — eviction skips
`self.root`; killerLegal_lifecycle covers it as a deletion choice) and
the commit-on-completed-depth rule in both driver loops (a mid-depth
fail-high is a candidate, promoted only when its depth's bracket
converges — closes the consumer-protocol gap that `storedMove_attains`
was too window-relative to exclude).

## Guideline for search-changing PRs

**A search-changing PR should identify which lemma it preserves or
weakens.** Concretely: does the change keep `bound_spec` (pure bound
logic)? Does it strengthen or newly rely on `NullOK` (zugzwang exposure)?
Does it preserve `TableOK` (is every store still a valid bracket, and is the
key still complete — cf. `ExtKeyIndependent`)? If the answer is "it weakens
X in positions Y", that is exactly the sentence the PR description needs.

The maintainer's design principle, distilled from the futility-vs-LMR
contrast and **enforced on master** — since `7fdd741` by the absence of
any reduction machinery at all: **"`gamma` may shape termination, and
may trigger shortcuts whose value provably one-side-bounds the same
function; `gamma` must never select between incomparable evaluations of
a move."** Futility passes (its shortcut equals the depth-0 search it
replaces — `futilityOK_discharged`); re-search LMR failed it, which is
exactly why its TT entries could contradict each other — and
measurement later showed its entire ELO edge *was* that failure (the
honest variant is worth exactly 0; the deterministic variant, which
passed the principle, was net-negative and removed too). The
doctrine, stated once for the whole repository: **there is no "interval
spec."** A claim of the form "the truth lies within a gap" is a
guarantee only if the gap is bounded, and no bound on search
instability is provable — one ply of depth can hide a mate, so the gap
is unbounded precisely in the positions that decide games. What such a
claim describes is a bug with a ledger. The only recognized contract is
the point spec: every stored bound describes one value function
determined by the transposition key.

The later files add further named conditions to check against:

- **`MateValuesAreKingCaptures`** — anything touching move ordering, the
  killer yield, king-capture scoring or the stalemate block must say
  whether `bound` still returns the exact `MATE_UPPER` sentinel whenever
  the king is capturable, and whether mate-band values can now be
  fabricated from shallow artifacts (`boundStale_not_unconditional` shows
  what goes wrong if so). Changes to MTD-bi's probe range must keep
  `gamma` inside `(-MATE_UPPER, MATE_UPPER]`, or the correction's
  soundness argument collapses at the window edge.
- **`KillerOK` (KillerIsKingCapture)** — the killer-cutoff exception to
  the sentinel invariant is *provably impossible* given in-band windows +
  value ordering (`boundKill_spec`); residual exception: the null-move
  path (`NullGuardBlocksAtCaptures`). A PR that re-keys the killer table
  (e.g. ply-shared killers), reorders king captures below anything, stores
  moves outside the fail-high cutoff, or widens the probe windows must say
  which leg of `boundKill_spec`'s induction it breaks.
- **`FutilityOK` / `FutilityMateOK`** — any change to `pos.value`, to the
  futility threshold or to the depth at which futility applies must
  re-justify that the static estimate still dominates the true child value
  (and that king captures still bypass it with the exact sentinel).
- **`MateDepthStable`** — a PR serving table entries across depths must
  state whether it serves only deeper (needs `MateDepthMonotone`) or also
  shallower (needs `MateDepthStable`, and should declare itself a
  weakening of `BoundSpec` to mate-band membership).
- **Table part 2 (the store at lines 475–487)** — a PR touching the
  store must preserve the point-spec `TableOK`: every stored bound a
  sound claim about the single key-determined value function. The
  historical clamp and its `IntervalTableOK` invariant are gone
  (deleted, not merely retired — see the doctrine note above); a change
  that cannot state a point spec is rejected, not clamped.
- **The verify-on-suspicion block (the legality oracle and both
  verifier arms)** — the shipped correction and null verifier rest on
  `legalityProbeCorrect`, `storedMoveLegal` and the two verified arms,
  so a PR touching any of the following must say which theorem it
  preserves:
  the PROBE (`bound(child, MATE_UPPER, 0, root=True)`) must stay at the
  window `MATE_UPPER` where nothing but a real king capture cuts off,
  and must stay UNSTORED in both directions (`rootProbe` — a stored
  probe would let a table entry enter the definition of legality);
  the MOVE ORDER must keep king captures first with mate-band values
  (`KingCaptureValHigh`, the futility `else MATE_UPPER` bypass
  included), or the probe's easy direction dies;
  `tp_move` must keep its exact-position key and store only proven
  moves — real fail-high winners (`storedMoveLegal`), the verified
  scan's find, or the substituted king capture — or `KillerLegal` and
  `KillerAtKingCapturable` die together;
  the CONSUMER INTERCEPTION must validate every virtual fail-high
  before it may cut, keep all three arms (substitution / mate-band
  identity / verified-terminal identity), keep the SUBSTITUTION arm
  depth-UNGATED (it is the channel-3 fix at QS) and the terminal-fold
  arm depth-GATED (folding a terminal stand-pat at depth 0 makes the
  node RETURN the reserved sentinel — 96 measured mismatches), and
  keep sub-mate futility yields VIRTUAL (`virtualCutoffValidated`,
  `production_eq_reference`, and the crossed-entry witness are what
  these buy);
  the CORRECTION must keep the oracle scan over the FULL
  `pos.gen_moves()` list and keep depth 0 excluded (`if depth and ...`
  — the `StandPatAtTerminal` refutation is what depth-0 exactness
  claims cost);
  the MOVE SORT must keep king captures first (`CaptureFirst`) or the
  production loop no longer reproduces the reference's eager sentinel;
  and NOTHING may reintroduce a score-shaped sentinel test on
  UNVALIDATED reports — `kingCapturableReportsExact_refuted` is the
  countermodel for trusting a fail-soft report as a legality fact;
  under kcx the report IS trustworthy again, exactly because the
  consumer validates it first (`kingCapturableReportsExact_restored`).
  A PR changing the spec layering must keep zugzwang OUT of the
  correctness layer: layer 1 (`bound_null_spec`) owns the algorithm
  unconditionally (one mate-band chess fragment, with its documented
  necessity), layer 2 (`NoZugzwang`) owns the approximation's
  accuracy, and the table can never depend on layer 2.
- **Adding a new flag/parameter to `bound()`** — the doctrine test it
  must pass, distilled from `can_null`'s life and death (in the key
  for years; removed at `eda66ee` once its only users stopped needing
  the table). Either the flag is **provably table-invisible**, like
  `root`: every flagged call skips the table in *both* directions and
  stores nothing, so each stored entry still describes the one
  `(pos, depth)`-determined value function — `rootProbe_spec` is the
  shape of the proof obligation, and `rootValue_eq_nullValue` the
  honest statement of where the flagged semantics diverge. Or the flag
  selects a **genuinely second value function** that reaches the
  table, and then it must join the transposition key and pay for the
  partitioned hit rate — with the value split stated explicitly, as
  the old two-function `CTable` layering did. The worked examples: the
  killer table (safe only because `tp_move` is position-keyed and
  stores only on fail-highs — `boundKill_spec`'s induction breaks for
  ply-shared killers); re-search LMR (a gamma-shaped second value that
  *could not* be keyed — the TT crossing); and the `can_ext` parity
  trick (an extension driven by position-derived data folds into the
  `depth` component of the key instead of adding a field — whereas a
  history-driven extension has no such fold and the key must genuinely
  grow, cf. `ExtKeyIndependent`).
- **Reductions (LMR)** — the engine ships none (`7fdd741`). A PR
  reintroducing one must (a) assign per-move depths as a function of
  position-derived data alone and claim only bounds on the resulting
  single value function (min-semantics `min(reduced, full)` bounds are
  also acceptable: key-determined by construction), and (b) beat the
  no-reduction baseline under `docs/TESTING.md` — a high bar, since the
  measured record is: honest re-search LMR exactly 0.00 ± 34 vs no LMR,
  deterministic LMR net-negative. Re-search LMR with full-value
  propagation is the canonical counterexample (`lmr_tt_crossing`, in
  git history at `58883ea..7fdd741^`) and is not mergeable regardless
  of measured strength: its edge is the bug.
- **The fold rule** (discovered by the tail-yield experiment) — *option
  removal can be represented by the fold identity*: yielding the
  max-fold's identity element `-MATE_UPPER` is exactly equivalent to
  omitting the yield.  The shipped mate-band suppression and the null
  verifier's withdrawal are both this kind (`yield None, score if
  score < MATE_LOWER else -MATE_UPPER`, and `score = -MATE_UPPER` on a
  verified terminal), which is why the model may disable the option
  (`useD2`) where the code yields the identity.  *Exact terminal
  knowledge that may LOWER the accumulated score is an override*, not a
  fold element, and must remain outside the fold: the stalemate/mate
  correction ASSIGNS `best = -MATE_LOWER if in_check else 0` after the
  loop — folding it would let any higher yield erase it.  This is why
  the correction can never move into the `moves()` generator.

**A fail-soft score is evidence about value, not evidence about
legality. Mobility is certified only by a legal fail-high move or by a
dedicated legality probe. When a bound would cross a terminal value
without such a certificate, verify before storing it.**

The refuted-assumptions ledger (Stalemate.lean, final sections) is the
empirical record behind that sentence: `NullAtStalemateNonpositive`
(false — the +175 ahead-stalemate), `StandPatAtTerminal` (false — the
corner mates), `KingCapturableReportsExact` (false — the -368/-69290
double report).  Each was an implicit "score implies legality" step;
each produced a real crossed table entry; each is now either verified
by the oracle or removed from the code path.

## Prior art

- Tobias Nipkow, *Alpha-Beta Pruning Verified*, invited talk, ITP 2024
  ([paper](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2024.1)),
  and the accompanying Isabelle/HOL AFP entry
  [Alpha_Beta_Pruning](https://isa-afp.org/entries/Alpha_Beta_Pruning.html)
  (2024), which verifies fail-hard and fail-soft alpha-beta over linear
  orders, distributive lattices and de-Morgan domains. The fail-soft
  correctness statement there is the two-sided bound that `BoundSpec`
  specializes to a null window; our `searchMoves_spec` invariant ("`best` is
  a fail-soft report of the fold so far") follows the same shape.
- *Formal Verification of Minimax Algorithms*
  ([arXiv:2509.20138](https://arxiv.org/abs/2509.20138)), which verifies
  minimax variants with alpha-beta and transposition tables in Dafny — the
  closest prior treatment of the table invariant (our `TableOK`).
- No existing Lean 4 formalization of alpha-beta/null-window search was
  found; this appears to be new ground for Lean, which is partly why the
  model is kept dependency-free.
