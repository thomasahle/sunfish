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

## Mate distance

Checkmate is not one number.  The terminal correction assigns

```python
mate = -MATE_LOWER - min(depth, MATE_SPAN)
```

where `depth` is the search depth still UNSPENT when the mate was found and
`MATE_SPAN = MATE_UPPER - MATE_LOWER - 1 = 21366`.  Negated up the tree the
bonus survives unchanged, so at one fixed root depth `D` a forced mate `k`
plies away reports `MATE_LOWER + (D - k)`: faster mates score strictly
higher, and the losing side prefers the line that postpones the mate.  With
the previous flat `-MATE_LOWER` every mate tied, which is issue #11 (2014).

**Why distance from the horizon and not from the root.**  The value must
stay a function of `(pos, depth)` alone -- that is the invariant the whole
table rests on.  Unspent depth is such a function; distance from the root is
not, and would force store/probe adjustment on every table access.

**Why not a per-ply step on the score** (the other way to get distance from
the node, `score -= sign(score)` at each negation): it is UNSOUND with
sunfish's zero-window probe.  The step map is not injective at the band edge
(`up(MATE_LOWER) = up(MATE_LOWER - 1)`), so no single child window can
separate the child's fail-high from its fail-low, and the fail-soft point
spec breaks by one at `boundD2 child (1 - gamma) = -gamma` and at
`= 1 - gamma`.  Restoring it needs a gamma-dependent child window
(`1 - gamma - [MATE_LOWER <= gamma < MATE_UPPER] + [gamma <= -MATE_LOWER]`),
i.e. a change to the search contract itself.  `GameTree.lean` carries `up`
and its non-injectivity as the machine-checked record of that dead end.

Lean:

| fact | theorem |
|---|---|
| the terminal value stays in the band at every depth | `terminalValue_bounds` |
| it is exactly `-MATE_LOWER - depth` below the clamp | `terminalValue_exact` |
| deeper unspent depth is worse for the mated side | `terminalValue_anti` |
| a forced mate in `k` is worth `MATE_LOWER + (D - k)` | `forcedMate_negamaxD2` |
| the mated dual | `forcedlyMated_negamaxD2` |
| the old flat readings, as corollaries | `*_band` |

## Play-level liveness

`Liveness.lean` milestone 3.  Everything else in this directory is about
ONE search.  This is about the GAME: define the engine's own move choice,
iterate it, let the defender answer with anything legal.

```text
forcedMate_play_mates :
  MaximalChoice G guard d ch →
  ForcedMate G k p → 1 ≤ k → k + 1 ≤ d + 1 → (d : Int) ≤ 21366 →
  hasKingCapture G p = false →
  MatesWithin G ch k p
```

`MatesWithin G ch k p`: the attacker plays `ch`, the defender plays ANY
legal move, and a `Checkmated` position is reached within `k` plies.

The statement could not be FORMED before mate distance.  With every mate
worth the flat `-MATE_LOWER` the value function does not rank the mating
moves at all, so "the move the value picks" is any mating move whatsoever
and iterating it need not arrive.  What makes the proof go through is
exactly that a nearer mate is worth strictly more: the chosen move's value
is at least the spec witness's, hence (`forcedMate_of_value_dist`, the
converse refined to carry the distance) its mate is at least as near.

Honest about `MaximalChoice`: it says `ch p` maximises the declared value
among the admitted moves.  That idealises an exactly-converged bisection.
`search` stops at `upper - lower <= EVAL_ROUGHNESS`, so the shipped root can
settle for a move within 15 of the maximum, and mate distances differ by one
per ply -- the shipped driver therefore acts on this ordering only for gaps
wider than `EVAL_ROUGHNESS`.  Tie-breaking is free: the theorem holds for
every maximising choice.  Depth is fixed at `d + 1` for every move of the
play.

Premises: `ValFloor G 192` + `EvalQuiet` (fidelity, tables),
`NoMaskedMobility` (chess, layer 2 -- required, `CexF`), `NoZugzwang`
(chess, layer 2), root legality.  No new chess premise.  `#print axioms`:
`propext, Classical.choice, Quot.sound` (the choice comes from the classical
case split in `legal_of_allIllegalB_false`); the distance spine
`forcedMate_negamaxD2` itself stays choice-free at `propext, Quot.sound`.

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
| score guard keeps the cap below positive mate | `guardedStaticCap_in_scoreBand`, `guardedCappedNull_below_positiveMate` |
| full-width move fold and early cutoff | `Bound.searchMoves_spec` and the fold models in `Stalemate.lean` |
| sticky legality evidence and terminal override | terminal/finalizer results in `Stalemate.lean` |
| king-capture evaluation margins and ordering | `EvalBounds.lean` |
| `mate = -MATE_LOWER - min(depth, MATE_SPAN)` | `terminalValue`, `terminalValue_exact` |
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
