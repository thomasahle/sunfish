# Formal verification of Sunfish's search

This directory contains the Lean 4 model for the search contract stated by
`Searcher.bound` in [`sunfish.py`](../sunfish.py):

```text
if gamma >  s* then s* <= r < gamma
if gamma <= s* then gamma <= r <= s*
```

The MTD-bi driver relies on every call reporting a bound on one value function
determined by the position, depth, and fixed search parameters. `gamma` may
select a shortcut or stop the search, but it must not select the value being
bounded.

The development uses Lean's core library and `omega`; it has no Mathlib
dependency and no `sorry` declarations.

## Build and audit

```sh
cd formal
lake build
cd ..
python3 formal/scripts/model_audit.py
```

`model_audit.py` hashes the source regions modeled by Lean and checks named
anchors in `sunfish.py`. A search change must update both the applicable proof
and the audit only after the correspondence has been reviewed.

## Current null value

For an enabled null move, let `P` be the exact parent-side value of passing and
let

```text
C(pos) = pos.score + EVAL_ROUGHNESS
N(pos) = min(C(pos), P)
```

The Python search obtains a one-sided report `r` for `P` from the complementary
zero-window child probe and reports `min(C(pos), r)`.

`Sunfish/CappedNull.lean` proves the two local steps needed for this operation:

- `WindowReport.negate` transfers a child report at `1 - gamma` to the
  parent window at `gamma`.
- `WindowReport.cap` proves that `min(C, ·)` transports a valid report of `P`
  to a valid report of `min(C, P)`.
- `cappedNull_report` composes those two facts for the exact Python expression.

The proof is generic in `C`; it does not depend on chess or mate constants.

The production guard is:

```python
not root and depth > 2 and abs(pos.score) < 500 \
    and any(c in pos.board for c in "RBNQ")
```

With integer scores and `EVAL_ROUGHNESS = 15`, the score guard gives
`C(pos) <= 514 < MATE_LOWER`. Theorems
`guardedStaticCap_in_scoreBand` and
`guardedCappedNull_below_positiveMate` prove that an enabled null move cannot
claim a positive mate score. A catastrophically bad pass may retain a negative
mate value; no lower clamp is intended.

The remaining chess-strength premise is:

```text
min(C(pos), P) <= best legal real-move value
```

It concerns the quality of the null approximation, not the fail-soft report
transport. The non-pawn-piece guard excludes pawn-only zugzwangs; the score
guard and cap make null pruning more conservative in unbalanced positions.

## Terminal positions and legality evidence

The move fold maintains two independent facts:

```python
best, live = -MATE_UPPER, False
```

- `best` accumulates numeric reports from real and virtual candidates.
- `live` records that a searched real move was legal.

Null moves, stand pat, and non-mating futility estimates are numeric evidence
only. A searched move whose child report is above the illegal-move sentinel is
also legality evidence.

Mate and stalemate are a final classification, not another input to `max`.
When no legal move has been witnessed at positive depth, the post-fold scan
checks every generated move. If none is legal, it replaces the accumulated
numeric report with exact checkmate (`-MATE_LOWER`) or stalemate (`0`). This
placement is necessary because exact terminal classification may lower a
virtual cutoff.

At a terminal root, a fail-high has no move. Both UCI loops therefore stop
iterative deepening and emit `bestmove (none)` without dereferencing `move`.

## The QS frontier

When a positive-depth fold has no legal real witness, `Searcher.bound` scans
the full move set:

- no legal move gives exact mate or stalemate, overriding every numeric
  candidate;
- an admitted legal move proves that the ordinary filtered value is the right
  one;
- otherwise an unstored, null-free `qs_tail` probe reuses the ordinary move
  loop over only the filtered-out moves.

The trigger is determined by the position and fixed QS threshold, not by the
window or move table. With the default tables and thresholds, omitted legal
moves can occur only at depth 1; applying the rule at every depth also keeps
the repair aligned with runtime-tuned QS thresholds.
`WindowReport.max` validates the report join, and
`foldMax_filtered_tail_retry` identifies its fixed value with the full fold.
`forcedMate_of_nullValueD2t` needs `ValFloor` and `EvalQuiet`, but no
`NoMaskedMobility` chess assumption at the default frontier.

## Move-table contract

The current `tp_move` contract is deliberately narrow:

- every stored move is legal for its position;
- a searched real move that causes a fail-high at positive depth is written as
  that report's move witness;
- an interior cutoff caused by virtual evidence need not store a move;
- a nonterminal root fail-high supplies a real move witness;
- terminal roots supply no move.

The root entry is protected from FIFO eviction while its search is active.
King-capture substitution separately preserves the exact `MATE_UPPER` promise
at capturable nodes.

## Current source correspondence

| Python mechanism | Lean result or model |
|---|---|
| recursive zero-window move search | `Bound.bound_spec` |
| null child report negation | `WindowReport.negate` |
| `min(pos.score + EVAL_ROUGHNESS, pass_report)` | `cappedNull_report` |
| score guard keeps the cap below positive mate | score-band theorems in `CappedNull.lean` |
| full-width move fold and early cutoff | `Bound.searchMoves_spec` and the fold models in `Stalemate.lean` |
| sticky legality evidence and terminal override | terminal/finalizer results in `Stalemate.lean` |
| QS frontier-tail retry | `WindowReport.max`, `foldMax_filtered_tail_retry`, `forcedMate_of_nullValueD2t` |
| king-capture evaluation margins and ordering | `EvalBounds.lean` |
| legal killer lifecycle and eviction | `Killer.lean` |
| root versus interior null behavior | `CanNull.lean` |
| transposition-table interval updates | `TableSwap.lean` and table results in `Stalemate.lean` |
| MTD-bi bracket range and convergence | `Driver.lean` |

The model abstracts Python's board representation, move generation, sorting,
and table implementation. The audit pins the corresponding source regions;
tests and chess corpora validate those executable primitives.

## Module guide

- `GameTree.lean`: chess-free negamax game model.
- `Bound.lean`: core fail-soft search proof.
- `CappedNull.lean`: capped-null report transport and score-band facts.
- `Stalemate.lean`: selective-search fold, legality, and terminal finalizer.
- `EvalBounds.lean`: numeric bounds induced by the piece-square tables.
- `Killer.lean`: move-table legality and lifecycle.
- `CanNull.lean`: root/interior null and table-key conditions.
- `Driver.lean`: MTD-bi bracket invariants and convergence.
- `TableSwap.lean`: table-update properties.
- `Liveness.lean` and `Classification.lean`: mate visibility and search-result
  classification under their named fidelity premises.
- `Pruning.lean` and `Tricks.lean`: proof envelopes for selective-search
  techniques.

## Rules for search changes

A search-changing patch should state:

1. the value function after the change;
2. why every fail-high and fail-low still bounds that same function;
3. which evidence is numeric and which certifies legality;
4. every remaining chess-strength premise;
5. the deterministic floors, node screens, and game evidence used to price it.

Refresh the audit only after the relevant proof and source mapping agree. A
green hash check establishes correspondence, not playing strength; tournament
evidence remains a separate merge decision.
