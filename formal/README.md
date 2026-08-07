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
    file; references to master `bf72b43`, sunfish.py lines 355, 399–401,
    471–472).  The models above searched ALL moves, so "the loop ended at
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
    - **The A1 fix, modeled ahead of the code**: `boundA1` follows the
      agreed `a1-fix` design — `best_real` accumulates real-move yields
      only, the sentinel test reads `best_real` (never the
      null-inclusive `best`), the gate becomes `best < gamma and
      best_real == -MATE_UPPER and (...)`, and mate-band null yields are
      suppressed so `NullBetQS` (the null bet, oracle form) need only be
      trusted below the mate band.  **`a1_unfixed_not_sound`** is the
      machine-checked A1 finding against the SHIPPED loop shape
      (`boundA1Un`): a sound fail-low null yield masks the sentinel at a
      genuine stalemate and the returned upper bound is exceeded by the
      true value 0 — with every hypothesis, including the null bet,
      satisfied; `a1_fix_repairs` shows `boundA1` returns the exact 0 on
      the same inputs.  At modeling time the `a1-fix` branch carried no
      code beyond master (verified: empty diff) — re-audit the model
      against the code when it lands.
    - Loop infrastructure: `searchMoves_eq_init_all` (loop exhaustion,
      the converse of `searchMoves_eq_init`) and `searchMoves_init_max`
      (the `max rn best_real` restructuring equals the code's single
      running `best`).

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
  by `boundA1_spec`'s `hmask` leg.

- **`KingCaptureValHigh`** (`Sunfish/Stalemate.lean`) — king captures are
  valued at or above `MATE_LOWER`, so they pass the QS val-filter at
  every depth (`val_lower < MATE_LOWER` is proven).  Concrete backing:
  `EvalBounds.kingCapture_val_above`.  A PR changing `pos.value`'s
  capture term must re-check it.

- **`ValFloor G B`** (`Sunfish/Stalemate.lean`) — every legal move's
  value is ≥ −B.  `B = 192` is backed by the tables
  (`EvalBounds.quietDropMax_eq` plus the nonnegativity of every additive
  `pos.value` term); `B ≤ 380` is what the `depth > 2` gate arm needs
  (`gate_implies_no_filtering`), `B ≤ 240` makes that arm redundant
  (`depth_arm_redundant`).  A PR retuning `QS`/`QS_A` (they are in
  `opt_ranges`!) or the tables must re-check this arithmetic: the depth
  arm needs `3·QS_A ≥ QS + floor` (shipped: 420 ≥ 40 + 192, 188 of
  slack), and a tuner can silently violate it.

- **`NullBetQS`** (`Sunfish/Stalemate.lean`) — the null-move bet in
  oracle form, as `boundA1_spec` consumes it: a guard-passing,
  `depth > 2`, fail-HIGH null yield BELOW the mate band lower-bounds the
  position's `negamaxQS` value.  Fail-low yields need no hypothesis, and
  the A1 suppression makes mate-band yields dead code — the bet is never
  trusted where mate scores could be fabricated.  The recursive
  (non-oracle) form of the bet is `NullBetOK` in `Sunfish/CanNull.lean`;
  zugzwang is precisely the failure of either.

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
| `inCheckB`, `CheckProbeOK` | the null-position check probe `bound(flipped, MATE_UPPER, 0) == MATE_UPPER` (lines 466–468) |
| `MateValuesAreKingCaptures` | the requirement of lines 429–435 and the caveat of lines 437–450 |
| `FutGame.val`, `boundFut` | `pos.value(move)` and the futility yield (lines 392–406) |
| `KTable`, `kstore`, `boundKill` | `tp_move`, killer try + store-on-cutoff (lines 373, 388–389, 415–419) |
| `orderedMoves` | the sort of line 392 (king captures, value ≥ `MATE_LOWER`, first) |
| `nullValue`, `boundNullTT`, `CTableOK` | the interior search (`root=False`): null move (364–365), repetition (341), `(pos, depth)`-keyed table (334–336, 481–485), IID as an unstored probe (375–381) |
| `rootProbe`, `rootValue` | the driver probes (`root=True`): the search root (line 512) and IID (line 381) — no lookup, no store, no repetition-0, no null yield |
| `nullGuard` | `abs(pos.score) < 500` (line 364), gamma-free |
| `QS`, `QS_A`, `val_lower` | the QS constants and threshold (`bf72b43` lines 149–150, 355) |
| `movesAbove`, `QSGame.val` | the QS break `if val < val_lower: break` (lines 399–401) and `pos.value(move)`; the killer val-gate keeps the killer inside the same set |
| `allAboveB`, `qsGateB` | the correction gate `all(pos.value(m) >= val_lower ...)` (the `depth > 2` arm was removed as a theorem-backed golf: wherever it held, the `all()` scan returns True under `ValFloor`) |
| `negamaxQS`, `qsDrawFix` | the filtered draw-aware value the gated search brackets |
| `boundA1` (`best_real` = `S`, `nullMax`, `a1Fix`) | the A1-fixed loop, SHIPPED on master since `0998739`; `boundA1Un` is the pre-fix loop shape, `a1_unfixed_not_sound` its machine-checked hole |
| `ValFloor`, `EvalBounds.quietDropMax` | the `pos.value` floor read off the tables |
| `KingCaptureValHigh` | king captures valued ≥ `MATE_LOWER` (the sort of line 392) |

The historical rows — `boundLmr`/`red` (re-search LMR), `boundLmrDet`/
`negamaxDet` (deterministic LMR), `clampHigh`/`clampLow` (the `2c95ab0`
clamp) — were removed with their mechanisms and Lean files at `7fdd741`;
see "Retired mechanisms" above.

## Model fidelity

Audited against master at commit `9b1a7b4` (2026-08-05), re-audited
after the LMR removal at `7fdd741` (2026-08-06) and after the
`can_null` removal at `eda66ee` (2026-08-07); line references
throughout this file are to `eda66ee` (sunfish.py lines 285–518). The
model tracks the code exactly except these explicitly listed
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
- **A1 status**: `boundA1` models the *agreed design* of branch
  `a1-fix` (best_real sentinel test, `best < gamma` in the gate,
  mate-band null suppression); at modeling time that branch carried no
  code beyond master (empty diff), so the model must be re-audited
  against the code when it lands.  Master's shipped loop shape is
  `boundA1Un`, and its stalemate-masking hole is machine-checked
  (`a1_unfixed_not_sound`) — the fidelity finding here is that the old
  "residual exception" note on the null yield (killer section above)
  understated the exposure: it covers stalemate masking, not just
  king-capturable nodes.
- **Ungated gate arms**: for in-band windows the model's
  `best < gamma ∧ best_real = LOSS` gate coincides with the code's bare
  `best == -MATE_UPPER` test (a `LOSS` loop result is automatically
  below an in-band window); out-of-band windows never reach the gate in
  the engine (the table's fresh-entry cutoffs answer them), which is why
  the spec carries the in-band hypothesis.

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
- **The QS gate (`qsGateB` / exhaustion)** — a PR touching `QS`, `QS_A`,
  `pos.value`, the move sort, or the stalemate gate must preserve the
  exhaustion argument: the gate must imply "no legal move was filtered"
  (`gate_implies_no_filtering`).  The shipped gate is the bare
  `all(...)` scan, which is self-justifying at any depth and under any
  tables; the historical `depth > 2` short-circuit rested on `ValFloor`
  arithmetic that `opt_ranges` tuning could silently break, which is
  why it was removed.  A PR touching the null yield must keep it out of
  the sentinel test (`best_real` — `a1_unfixed_not_sound` is the
  counterexample for feeding it `best`) and keep mate-band null yields
  suppressed (the fold-identity form `score if score < MATE_LOWER else
  -MATE_UPPER` IS the suppression), or strengthen `NullBetQS` into the
  mate band, which the doctrine forbids.
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
