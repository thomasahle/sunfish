# Formal verification of sunfish's search

A small Lean 4 formalization of the search algorithm in
[`sunfish.py`](../sunfish.py) — specifically of the contract stated in the
docstring of `Searcher.bound` (sunfish.py lines 286–290):

```
Let s* be the "true" score of the sub-tree we are searching.
The method returns r, where
if gamma >  s* then s* <= r < gamma  (A better upper bound)
if gamma <= s* then gamma <= r <= s* (A better lower bound)
```

This is the property the MTD-bi driver `Searcher.search` (lines 424–447)
relies on for its binary search: every call to `bound` must move one end of
the `lower <= score <= upper` bracket, whatever `gamma` it is probed with.

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
  sunfish.py line 379).

- **`Sunfish/Bound.lean`** — the core result.
  - `bound : Nat → Pos → Int → Int` is sunfish's search stripped to its
    logical skeleton: the fail-soft `best` loop with the early cutoff on
    `best >= gamma` (lines 378–388) over children searched with the flipped
    null window `1 - gamma` at `depth - 1` (line 376). No table, no null
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
  (sunfish.py lines 388–412), modeled faithfully (king-capture
  normalization of lines 298–303, the `MATE_UPPER` sentinel invariant of
  lines 398–401, the `depth > 2` gate, the null-position in-check probe of
  lines 408–412). Proven sorry-free:
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
    403–405 turned into a refutation.
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

- **`Sunfish/Killer.lean`** — **KillerIsKingCapture**, proven sorry-free
  (`boundKill_spec`): threading `tp_move` through the search as state
  (position-keyed, store-on-fail-high-only, sunfish.py lines 339, 356–357,
  382–387), a single call starting from an invariant-satisfying table
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

- **`Sunfish/Lmr.lean`** — Late Move Reductions (commit `58883ea`,
  sunfish.py lines 370, 386–397: late quiet moves probed at `depth - 2`,
  a reduced fail low yielded as-is, a reduced fail high re-searched at
  full depth). All proven sorry-free:
  - **`boundLmr_spec`**: the honest **interval spec**. No single value
    function can back both sides of the docstring any more; the sound
    statement is a *mutually recursive* pair `Vhi`/`Vlo` (`lmrVal`):
    fail highs are ≤ `Vhi`, fail lows are ≥ `Vlo`, and
    **`lmrVal_sandwich`** proves `Vlo ≤ negamax ≤ Vhi` pointwise.
    Two surprises from modeling the merged code (see the module comment):
    (1) the folklore fail-low target "reducible moves valued one ply
    shallower" (`negamaxShallow`) is *wrong* — the re-search fall-through
    yields a deep-value fact while the shallow value has just failed
    high, so the sound `Vlo` entry is the `min` of the two depths; and
    (2) fail highs are *not* sound against full `negamax` either — the
    re-search guard protects only the immediately reduced move, while a
    parent fail high inherits its children's fail-low (reduced-value)
    facts recursively, which is why `Vhi`'s children are `Vlo`-valued.
  - **`lmr_tt_crossing`** + **`bound_no_crossing`**: machine-checked
    3-position game where the same `(pos, depth)` produces a fail-high
    report (+10) strictly above a fail-low report (−50) — provably
    impossible for the unreduced `bound`. MTD-bi's
    `lower ≤ score ≤ upper` bracket is therefore conditional on the
    `Vhi − Vlo` gap staying within what `EVAL_ROUGHNESS` and re-probing
    absorb. Post-`2c95ab0`, the accurate statement is: crossing
    *reports* still occur (this theorem), but crossing *entries* can no
    longer be stored (`clamp_no_crossing`), and stored entries remain
    honest interval claims (`IntervalTableOK`) — see
    `Sunfish/TableClamp.lean`.
  - `RedRespectsCaptures` + comments confirm Killer and Stalemate are
    unaffected: king captures have `val ≥ MATE_LOWER > QS = 40` (never
    reducible), the killer is pre-loop (structurally never reduced),
    reduced fail lows are `< gamma` (never reach the `tp_move` store),
    reduced fail highs store only the full-depth result, and the
    `-MATE_UPPER` king-loss sentinel is depth-independent
    (`boundKill_kingGone`/`negamaxDraw_kingGone`), so the stalemate
    detection still sees exact sentinels from reduced searches.

- **`Sunfish/TableClamp.lean`** — the clamped store (commit `2c95ab0`,
  sunfish.py lines 435–443: fail-high stores
  `Entry(best, max(entry.upper, best))`, fail-low
  `Entry(min(entry.lower, best), best)`). All proven sorry-free:
  - **`IntervalTableOK`** — the honest post-LMR table invariant: every
    stored entry satisfies `lower ≤ Vhi(pos, depth)` and
    `Vlo(pos, depth) ≤ upper` (the `lmrVal` pair), the interval weakening
    of `TableOK` exactly parallel to how `boundLmr_spec` weakens
    `bound_spec`.
  - **`intervalTableOK_clampHigh` / `intervalTableOK_clampLow`** — the
    clamped store *preserves* the invariant, given only that the
    incoming bound is sound on its own side (exactly what
    `boundLmr_spec` returns). Key step: when a new lower `L` exceeds
    the stored upper `U`, the clamped entry `(L, max U L)` satisfies
    both sides — `L ≤ Vhi` from the new bound, `Vlo ≤ U ≤ max U L` from
    the old entry; symmetric on the other side.
  - **`clamp_no_crossing`** + **`NoCrossingTable`/`noCrossingTable_store`**
    — clamped entries satisfy `lower ≤ upper` by construction: with the
    clamp, the search-level crossing phenomenon remains
    (`lmr_tt_crossing`) but the table-level contradiction is gone.

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
    `negamax d p` — literally the comment at sunfish.py line 275
    (`# lower <= s(pos) <= upper`). Lookups (lines 309–310) are sound
    *because* every exit re-establishes the invariant; the proof is
    `bound_spec` plus invariant-threading through the state-passing loop
    (`searchMovesTT_spec`), with `tableOK_store`/`tablePart2_ok` for the
    store step and `negamax_bounded` discharging the fresh-entry default
    (the easily-missed `Bounded` side condition, without which
    `Entry(-MATE_UPPER, MATE_UPPER)` is not a valid bracket and the
    theorem is false). This is the **point-spec version, sound for the
    pre-LMR search model** (`Bound.lean`'s loop); under LMR the point
    invariant is unachievable and the honest story is
    `Sunfish/TableClamp.lean`, below.
  - **`boundFut_spec` + `futilityOK_discharged` — `FutilityOK`
    discharged: from stated hypothesis to theorem.** `ValGame` records
    sunfish's *score identity* — `pos.move(move)` builds the child with
    `score = -(pos.score + pos.value(move))`, literal in the code and in
    the comment at lines 365–367 — as a structural property
    (model-faithful, not an assumption). With it, the futility yield is
    *exactly* the depth-0 child search result (`futilityOK_discharged`
    is an equality, not an inequality): the futility test
    `pos.score + val < gamma` is integer-equivalent to the child's
    stand-pat meeting its window, so the child fails high immediately
    and fail-soft returns precisely `-(pos.score + val)`.
    `boundFut_spec` now proves `BoundSpec` for the futility-augmented
    search **unconditionally** (single value function, no interval),
    with only `FutilityMateOK` (line 371's king-capture bypass) and the
    in-band window remaining. Fine print made explicit: the ∀-depth
    `FutilityOK` is *not* dischargeable (plain negamax has no stand-pat
    at `d ≥ 1`) — but the search only ever consumes the `d = 0`
    instance; the old statement over-required. **Contrast with LMR**:
    futility's shortcut is a provable one-sided bound of the *same*
    value function (hence consistent, point spec); LMR's reduced value
    is incomparable to the full value (hence the `Vlo`/`Vhi` interval
    and the TT crossing).

## What is stated with `sorry`, and why

Each pruning trick is kept out of the proven model and instead given a
*named hypothesis* in `Sunfish/Tricks.lean`, because each one is only sound
conditionally — and the condition, not the induction, is the interesting
content:

- **`NullOK` / `boundNull_spec`** — null-move pruning (sunfish.py lines
  330–331) is correct *only under* `NullOK`: "in every non-terminal position
  some legal move scores at least the pass value." Zugzwang is precisely
  `¬ NullOK`; that is why sunfish guards the trick with
  `abs(pos.score) < 500` instead of trusting it everywhere. The proof is
  routine given `bound_spec`'s machinery and is `sorry`d; the deliverable is
  the named hypothesis. (Sunfish's extra `depth - 3` reduction would further
  need a depth-stability hypothesis, deliberately not modeled.)

- **`FutilityMateOK`** — the one hypothesis left on the futility yield:
  the `else MATE_UPPER` king-capture bypass at line 371 can fail *high*
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
  `mateDepthStable_of_kingGoneStable` (`sorry`d) derives from
  `KingGoneStable` (a captured king is permanent, on both sides of the
  sign alternation) plus "mate scores only come from real king captures".
  Such a variant should document itself as **weakening `BoundSpec`** to
  mate-band membership only.

- **`NullGuardBlocksAtCaptures` / `killer_probe_sound`**
  (`Sunfish/Killer.lean`) — the residual exception to the sentinel
  invariant: the null-move yield carries `None` (stores nothing, so
  `KillerOK` survives) but can end the loop below `MATE_UPPER`;
  sunfish guards it only by `abs(pos.score) < 500` (line 330, with its own
  FIXME at 323–329 conceding the guard is heuristic).
  `NullGuardBlocksAtCaptures` names the condition under which the guard
  closes the hole. `killer_probe_sound` states that the stalemate probe is
  a complete decision procedure for king-capturability (no null move at
  depth 0, stand-pat below the sentinel by `QuietEvalsInBand`, killer
  covered by `boundKill_spec`, capture-first order otherwise); the `mpr`
  direction is `boundKill_spec`, the no-false-positives direction is
  `sorry`d pending a `MateValuesAreKingCaptures`-style characterization of
  where `MATE_UPPER` reports originate.

- **`ExtKeyIndependent`** — *not* `sorry`d, but stated as the false claim a
  `(pos, depth)`-keyed table makes once search extensions depend on history
  (e.g. a recapture extension keyed on the last-capture square), and refuted
  by an explicit two-state counterexample. Consequence: such a TT key must
  include the last-capture square. Sunfish itself avoids the problem: its QS
  re-derives capture information from `pos` alone, and the one piece of
  history that *does* change values — `can_null` — is duly part of the key
  (line 308).

## Model ↔ sunfish.py correspondence

| Lean | sunfish.py |
|---|---|
| `Game.moves`, `[] = lost` | `gen_moves`, king-capture convention (lines 298–303) |
| `Game.eval` | `pos.score` (QS collapsed into eval; lines 335–336) |
| `LOSS = -MATE_UPPER` | `best = -MATE_UPPER` (line 379) |
| `negamax` | the "true score `s*`" of the docstring (lines 287–290) |
| `searchMoves` | the `best` loop + cutoff (lines 378–388; **NOTE**: since commit `58883ea` the real loop also carries LMR — `Bound.lean` models the loop *without* it, which is exact for early/loud moves; the reduction is modeled by `boundLmr`/`searchMovesIdx` in `Sunfish/Lmr.lean`) |
| `boundLmr`, `red` | the LMR block: `depth >= 3 and i_m >= 5 and val < QS`, reduced probe at `depth - 2`, re-search on fail high (lines 386–397) |
| `lmrVal` (`Vhi`/`Vlo`) | the interval that replaces the docstring's single `s*` under LMR |
| `-(bound d m (1 - gamma))` | `-self.bound(pos.move(move), 1 - gamma, depth - 1)` (line 376) |
| `BoundSpec` | the docstring (lines 287–290) |
| `NullGame.pass` | `pos.rotate(nullmove=True)` (line 331) |
| `Table`, `TableOK`, `tablePart2` | `tp_score`, `Entry`, lookup + point store (lines 275–276, 305–310, and the pre-`2c95ab0` Table part 2) |
| `clampHigh`, `clampLow`, `IntervalTableOK` | the clamped Table part 2 (`2c95ab0`, lines 435–443) and the interval invariant it maintains |
| `negamaxDraw`, `boundStale`, `staleFix` | king-capture normalization + stalemate correction (lines 298–303, 388–412) |
| `inCheckB`, `CheckProbeOK` | the null-position check probe `bound(flipped, MATE_UPPER, 0) == MATE_UPPER` (lines 409–411) |
| `MateValuesAreKingCaptures` | the requirement of lines 398–401 and the caveat of lines 403–405 |
| `FutGame.val`, `boundFut` | `pos.value(move)` and the futility yield (lines 360–374) |
| `KTable`, `kstore`, `boundKill` | `tp_move`, killer try + store-on-cutoff (lines 339, 356–357, 382–387) |
| `orderedMoves` | the sort of line 360 (king captures, value ≥ `MATE_LOWER`, first) |

Not modeled at all (they change *what* is computed, not whether bounds are
honest, or are performance-only): move ordering (line 360 — except for its
load-bearing consequence, the `MATE_UPPER` sentinel invariant, which
`boundStale` enforces by construction), killer/IID (lines 338–357; see the
killer-move caveat in `Sunfish/Stalemate.lean`), repetition/history
(lines 315–316), table eviction (lines 419–420 — eviction only forgets
entries, which trivially preserves `TableOK`).

## Guideline for search-changing PRs

**A search-changing PR should identify which lemma it preserves or
weakens.** Concretely: does the change keep `bound_spec` (pure bound
logic)? Does it strengthen or newly rely on `NullOK` (zugzwang exposure)?
Does it preserve `TableOK` (is every store still a valid bracket, and is the
key still complete — cf. `ExtKeyIndependent`)? If the answer is "it weakens
X in positions Y", that is exactly the sentence the PR description needs.

The maintainer's design principle, distilled from the futility-vs-LMR
contrast: **"`gamma` may shape termination, and may trigger shortcuts
whose value provably one-side-bounds the same function; `gamma` must
never select between incomparable evaluations of a move."** Futility
passes (its shortcut equals the depth-0 search it replaces —
`futilityOK_discharged`); LMR fails it (the reduced and full values are
incomparable), which is exactly why futility keeps the point `BoundSpec`
while LMR pays with the interval spec and crossable TT entries.

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
- **`IntervalTableOK`** — a PR touching Table part 2 (the store at lines
  435–443) must preserve `IntervalTableOK`: every stored lower must be a
  sound `Vhi` claim and every stored upper a sound `Vlo` claim, and the
  entry must stay non-crossing (`clamp_no_crossing`). Reverting the clamp
  reintroduces storable contradictions (`lmr_tt_crossing` is the
  witness); "fixing" it by discarding the old side instead of widening
  would break the preservation lemmas' use of the *old* entry's validity.
- **LMR (`boundLmr_spec`)** — Late Move Reductions weaken `BoundSpec`
  from point to interval: fail highs are sound against `Vhi`, fail lows
  against `Vlo`, with `Vlo ≤ negamax ≤ Vhi` (`lmrVal_sandwich`) — and
  provably *not* against full `negamax` on either side
  (`lmr_tt_crossing`). TT consistency is margin-conditional: entries can
  cross (`lower > upper`) by up to the `Vhi − Vlo` gap, which
  `EVAL_ROUGHNESS` + MTD-bi re-probing must absorb. Killer and Stalemate
  lemmas are preserved (`RedRespectsCaptures`: king captures are never
  reducible; the killer is pre-loop; reduced fail lows never store). A PR
  touching the reduction condition must keep captures un-reducible and
  keep yielding only *full-depth* results on fail high — dropping the
  re-search, or yielding a reduced fail high, breaks `boundLmr_spec`'s
  fail-high leg *and* the killer/stalemate sentinel arguments at once.

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
