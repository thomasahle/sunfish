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
not root and 2 < depth < 6 and abs(pos.score) < 500 \
    and any(c in pos.board for c in "RBNQ")
```

The upper bound is new: the pass is a score candidate only *below* depth 6.
From depth 6 on it is a fuel oracle instead -- see the next section.

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
It is confined to `depth < 6`; above that the fuel oracle removes it.

## The deep-null fuel oracle

From depth 6 on the pass is not a score candidate at all. One probe at a
*fixed* target decides only how much depth the real moves spend:

```python
d = depth
if depth >= 6 and abs(pos.score) < 500 and any(c in pos.board for c in "RBNQ"):
    target = pos.score + NULL_MARGIN
    if -self.bound(pos.rotate(nullmove=True), 1 - target, depth - 3) >= target:
        d = depth - 1
```

The target depends on `(pos, depth)` alone -- `gamma` does not enter -- so the
window is position-determined, the probe is table-cacheable, and the resulting
"hot" bit is stable because a fail-soft report is side-exact at any fixed
window (`WindowReport.side_exact`, `hot_bit_determined`, `hot_bit_stable`).
Nominal `depth` still keys the tables and the QS admission; only the recursion
is shortened. So a deep null *cut* becomes a *reduction*: every real edge costs
1 or 2 plies (`fuel_edge_cost`), never infinity.

That is what buys the premise: a null cutoff gives every real move unbounded
pruning debt, and discharging it is exactly what `NoZugzwang` was for. A
bounded edge cost discharges it instead, so the classification theorem needs no
chess premise:

```text
eventual_classification_fuel :
  ValFloor G 192 -> EvalQuiet G -> (root legality) ->
    exists D0, forall D >= D0,  W / D / L read off the value, correctly
```

`NoZugzwang` appears nowhere in its premises, nor in those of
`eventual_classification_fuel_arms` (which pins the arming depths at
`D >= C*k + 4` for a win and `D >= C*k + C + 4` for a loss) or
`driver_sees_trichotomy_fuel`. The draw arm needs no depth floor at all: with
no forced mate for either side the value is strictly inside the band at *every*
depth. `Repetition.lean` adds the game-history rule (`repetition_not_lost`,
`draw_arm_strengthened`) on top.

`FuelTailBracketSpec` -- the layer-1 bracket for the composed search -- is
stated and flagged unproven; it is not used by the theorems above.

### The finiteness variant, without the frontier tail

`eventual_classification_fuel` is stated for `fuelValueD2t`, whose fold list
already contains the frontier tail -- PR #171's engine change. For the
*unpatched* fuel value `fuelValueD2` (`movesAbove` only), the same trichotomy
holds when the game itself is finite (`EventuallyFinite.lean`):

```text
eventual_classification_fuel_finite :
  ValFloor G 192 -> EndsWithin G N p -> (root legality) ->
    forall D >= C*N + C + 6,  W / D / L read off the value, correctly
```

`EndsWithin G N p` -- every legal play from `p` reaches a terminal within `N`
plies -- is true of adjudicated chess (50-move plus threefold under match
adjudication) and false of the ruleless modeled game. At `D >= C*N + 6` every
node the classification depends on is reached before the frontier, so the
masking sites are unreachable and `NoMaskedMobility`, the tail, and even
`EvalQuiet` all drop out. The bound is *effective* (`2N + 8` as shipped, no
classical `exists D0`), and the file's entire footprint is
`[propext, Quot.sound]`. The scope is eventual-only, by countermodel: `CexE`
(the infinite masked chain) violates the premise (`cexE_not_finite`), while
`CexD` satisfies it at budget 5 and still prices its drawable masked node in
the mated band at depth 1 (`cexD_fuel_M1`) before classifying it correctly
from depth 10 on (`cexD_M_eventually_classified`). Fixed-depth honesty below
the bound still requires `NoMaskedMobility` or the #171 tail.

## Mate distance

Checkmate is not one number.  The terminal correction assigns

```python
mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)
```

where `depth` is the search depth still UNSPENT when the mate was found.
Negated up the tree the bonus survives unchanged, so at one fixed root depth
`D` a forced mate `k` plies away reports
`MATE_LOWER + (D - k) * EVAL_ROUGHNESS`: faster mates score strictly higher,
and the losing side prefers the line that postpones the mate.  With the
previous flat `-MATE_LOWER` every mate tied, which is issue #11 (2014).

**Why a whole `EVAL_ROUGHNESS` per ply.**  MTD-bi stops bisecting at
`upper < lower + EVAL_ROUGHNESS`, so the driver's final window sits within 15
of the true value and any move within 15 of the maximum can take the last
cutoff.  At one point per ply the ordering would exist in the value function
and never reach the root.  Scaled, consecutive mate distances are a full
bracket apart.

**Band headroom, checked rather than assumed.**  The floor `1 - MATE_UPPER`
is the deepest mate value, `-69289`, exactly one point above the illegal-move
sentinel `-MATE_UPPER = -69290`; its negation `69289` is exactly one point
below the king-capture sentinel `MATE_UPPER`.  Writing the floor as
`-MATE_UPPER` instead would be off by one and unsafe: the value is negated on
the way up and negated again a ply later, so a node valued `-MATE_UPPER`
reaches its GRANDPARENT as exactly the sentinel, and `score > -MATE_UPPER` is
strict -- `live` would stay unset for a legal move and the terminal
correction could fire at a position that has legal moves.  Machine-checked:
the two spellings first differ at unspent depth 1425, and at that depth the
grandparent yield is `-69289` (live) versus `-69290` (not live).  The one
point is load-bearing in both directions and it is exact:

* `live |= move is not None and score > -MATE_UPPER` still separates "legal
  move into the deepest representable mate" (`-69289`, live) from "illegal
  move" (`-69290`, not live);
* `r = MATE_UPPER` at a king-capturable node stays unambiguous, since no mate
  value reaches it;
* the table's default `Entry(-MATE_UPPER, MATE_UPPER)` still contains every
  value, and the driver's reset `lower = 1 - MATE_UPPER` coincides with the
  deepest value -- the wrinkle `BracketOK`'s `max V (1 - MATE_UPPER)` already
  records;
* `pos.score <= -MATE_LOWER`, `pos.value(move) >= MATE_LOWER` and the null
  cap `pos.score + EVAL_ROUGHNESS <= 515` all read static quantities, which
  `EvalBounds` keeps strictly below `MATE_LOWER`; the gap to the nearest mate
  value is now `EVAL_ROUGHNESS` wider than before, so those margins only
  improve.

The clamp binds at unspent depth 1425; `search` iterates to 999, so it never
binds in play and is there only to make the band facts unconditional.

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
| a forced mate in `k` is worth `MATE_LOWER + (D - k) * EVAL_ROUGHNESS` | `forcedMate_negamaxD2` |
| the mated dual | `forcedlyMated_negamaxD2` |
| the old flat readings, as corollaries | `*_band` |

## Play-level liveness

`Liveness.lean` milestone 3.  Everything else in this directory is about
ONE search.  This is about the GAME: define the engine's own move choice,
iterate it, let the defender answer with anything legal.

```text
forcedMate_play_mates :
  MaximalChoice G guard d ch →
  ForcedMate G k p → 1 ≤ k → k + 1 ≤ d + 1 →
  (d : Int) * EVAL_ROUGHNESS ≤ 21366 →
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
settle for a move within 15 of the maximum.  That is exactly why one ply of
distance is worth a whole `EVAL_ROUGHNESS`: at one point per ply the shipped
driver could not act on the ordering at all.  Tie-breaking is free: the
theorem holds for every maximising choice.  Depth is fixed at `d + 1` for every move of the
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
| `min(pos.score + EVAL_ROUGHNESS, pass_report)` (depth < 6) | `cappedNull_report` |
| `target = pos.score + NULL_MARGIN` fuel probe (depth >= 6) | `hot_bit_determined`, `hot_bit_stable`, `fuel_edge_cost` |
| real-move recursion at the reduced `d - 1` | `fuelValueD2t`, `eventual_classification_fuel` |
| score guard keeps the cap below positive mate | `guardedStaticCap_in_scoreBand`, `guardedCappedNull_below_positiveMate` |
| full-width move fold and early cutoff | `Bound.searchMoves_spec` and the fold models in `Stalemate.lean` |
| sticky legality evidence and terminal override | terminal/finalizer results in `Stalemate.lean` |
| king-capture evaluation margins and ordering | `EvalBounds.lean` |
| `mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)` | `terminalValue`, `terminalValue_exact` |
| legal killer lifecycle and eviction | `Killer.lean` |
| root versus interior null behavior | `CanNull.lean` |
| transposition-table interval updates | `TableSwap.lean` and table results in `Stalemate.lean` |
| MTD-bi bracket range and convergence | `Driver.lean` |

The model abstracts Python's board representation, move generation, sorting,
and table implementation. The audit pins the corresponding source regions;
tests and chess corpora validate those executable primitives.

IID starts at depth 4 because quiescence cannot write `tp_move`.
`CanNull.lean` keeps a uniform recurrence at depth 3; its depth-zero root
transform is the identity, so the model and source are extensionally equal.

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
- `EventuallyWide.lean`: the fuel oracle -- bounded real-edge cost and the
  W/D/L trichotomy with no chess premise.
- `Repetition.lean`: the game-history draw rule on top of the fuel value.
- `EventuallyFinite.lean`: the finiteness variant -- the trichotomy for the
  untailed fuel value under `EndsWithin`, with an effective depth bound.

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
