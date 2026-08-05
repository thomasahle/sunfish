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
  - The module comment records a **potential gap in sunfish itself** found
    while modeling: the killer move is yielded before the sorted moves
    (lines 356–357), so a non-capture killer that fails high breaks the
    stated "always return `MATE_UPPER` if the king is capturable"
    requirement, which a parent's stalemate detection relies on.

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

- **`TableOK` / `boundTT_spec`** — the transposition-table invariant:
  every stored `(lower, upper)` brackets `negamax d p`. This is literally
  the comment at sunfish.py line 275 (`# lower <= s(pos) <= upper`); lookups
  (lines 309–310) are sound *because* every exit re-establishes it (lines
  415–418). The statement includes the easily-missed side condition
  `Bounded` (all evals in `[-MATE_UPPER, MATE_UPPER]`), without which the
  fresh entry `Entry(-MATE_UPPER, MATE_UPPER)` of line 308 is not a valid
  bracket and the theorem is false. Proof `sorry`d (it is `bound_spec` plus
  invariant-threading through the state-passing loop).

- **`FutilityOK` / `FutilityMateOK` / `boundFut_spec`** — the futility
  yield (sunfish.py lines 360–374): at `depth ≤ 1` a move with
  `pos.score + val < gamma` is answered by the *static estimate*
  `pos.score + val` instead of a child search. The estimate is always a
  fail-low report, so `BoundSpec` needs exactly
  `-(negamax d child) ≤ pos.score + val` — that is `FutilityOK` (the
  formal reading of the "opponent will just stand pat" comment at lines
  365–367; it fails precisely where stand-pat reasoning fails, e.g. the
  opponent in check). The `else MATE_UPPER` special case at line 371
  (king captures bypass the estimate and report the exact sentinel) can
  fail *high* and needs its own hypothesis, `FutilityMateOK`.
  `boundFut_spec` is stated for in-band windows and `sorry`d with a full
  sketch (the only missing machinery is a member-restricted variant of
  `searchMoves_spec`).

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
| `searchMoves` | the `best` loop + cutoff (lines 378–388) |
| `-(bound d m (1 - gamma))` | `-self.bound(pos.move(move), 1 - gamma, depth - 1)` (line 376) |
| `BoundSpec` | the docstring (lines 287–290) |
| `NullGame.pass` | `pos.rotate(nullmove=True)` (line 331) |
| `Table`, `TableOK` | `tp_score`, `Entry` (lines 275–276, 305–310, 414–420) |
| `negamaxDraw`, `boundStale`, `staleFix` | king-capture normalization + stalemate correction (lines 298–303, 388–412) |
| `inCheckB`, `CheckProbeOK` | the null-position check probe `bound(flipped, MATE_UPPER, 0) == MATE_UPPER` (lines 409–411) |
| `MateValuesAreKingCaptures` | the requirement of lines 398–401 and the caveat of lines 403–405 |
| `FutGame.val`, `boundFut` | `pos.value(move)` and the futility yield (lines 360–374) |

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

The stalemate/futility/mate-entry work adds three more named hypotheses to
check against:

- **`MateValuesAreKingCaptures`** — anything touching move ordering, the
  killer yield, king-capture scoring or the stalemate block must say
  whether `bound` still returns the exact `MATE_UPPER` sentinel whenever
  the king is capturable, and whether mate-band values can now be
  fabricated from shallow artifacts (`boundStale_not_unconditional` shows
  what goes wrong if so). Changes to MTD-bi's probe range must keep
  `gamma` inside `(-MATE_UPPER, MATE_UPPER]`, or the correction's
  soundness argument collapses at the window edge.
- **`FutilityOK` / `FutilityMateOK`** — any change to `pos.value`, to the
  futility threshold or to the depth at which futility applies must
  re-justify that the static estimate still dominates the true child value
  (and that king captures still bypass it with the exact sentinel).
- **`MateDepthStable`** — a PR serving table entries across depths must
  state whether it serves only deeper (needs `MateDepthMonotone`) or also
  shallower (needs `MateDepthStable`, and should declare itself a
  weakening of `BoundSpec` to mate-band membership).

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
