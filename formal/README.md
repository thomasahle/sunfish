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

- **`Sunfish/Tricks.lean`**, the proven parts:
  - `soften_null_window` — the mate-score softening lemma: for a window
    strictly above the mate band (`gamma > ML + 1`, `ML ≥ 0`), testing the
    softened negated child score against `gamma` is exactly the raw test
    `r ≤ -gamma - 1`. Softening (mate-distance bookkeeping) does not disturb
    null-window fail-high decisions.
  - `extended_value_not_key_independent` — a concrete counterexample showing
    the extended-search value is **not** a function of `(pos, depth)` alone
    (see below).

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

Not modeled at all (they change *what* is computed, not whether bounds are
honest, or are performance-only): move ordering (line 360), killer/IID
(lines 338–357), futility pruning (lines 362–374), repetition/history
(lines 315–316), the stalemate correction (lines 390–412), table eviction
(lines 419–420 — eviction only forgets entries, which trivially preserves
`TableOK`).

## Guideline for search-changing PRs

**A search-changing PR should identify which lemma it preserves or
weakens.** Concretely: does the change keep `bound_spec` (pure bound
logic)? Does it strengthen or newly rely on `NullOK` (zugzwang exposure)?
Does it preserve `TableOK` (is every store still a valid bracket, and is the
key still complete — cf. `ExtKeyIndependent`)? If the answer is "it weakens
X in positions Y", that is exactly the sentence the PR description needs.

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
