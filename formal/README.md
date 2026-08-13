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

## Distance-to-mate optimality (`Shortest.lean`)

`forcedMate_play_mates` mates within *the `k` the spec handed it*.
`Shortest.lean` replaces that `k` with the LEAST one, which is the shortest-PV
half of the claim sunfish.py's constant block has made since 2014.

**Parity is the hinge, and it is not a chess fact.**  `ForcedMate`'s `mate`
constructor costs one ply and `step` costs two, and `step`'s reply quantifier
is non-vacuous (`hnt` names a legal reply through `legal_of_allIllegalB_false`),
so an EVEN budget always has a spare ply: `forcedMate_pred_of_even` hands it
back.  Hence the least distance is ODD (`leastMate_odd`) and two distinct
achievable distances are at least TWO plies apart (`leastMate_gap`).

That is what prices a ply at a whole `EVAL_ROUGHNESS` rather than two.  The
declared value's rung index is exactly `D - k` (`leastMate_value_block`: the
forward spine gives the lower edge, leastness gives the upper -- one rung
higher would exhibit a mate in `k - 1`).  Two plies is two rungs, so distinct
distances leave a full rung of clear air:

```text
leastMate_value_separation :
  EVAL_ROUGHNESS < nullValueD2 G guard D p - nullValueD2 G guard D q
```

STRICTLY more than `EVAL_ROUGHNESS`, and `search` stops at
`upper - lower <= EVAL_ROUGHNESS`, so no final bracket can hold two distinct
mate values.  **This retires an idealisation.**  `MaximalChoice` assumes an
exactly-converged bisection, which the shipped driver does not give;
`NearMaximalChoice` weakens it by exactly the driver's own stopping tolerance,
and `forcedMate_play_shortest_odd` proves the same conclusion under it.  Where
near-maximality loses a rung, parity refunds it: the slack budget is even, and
an even budget is never tight.

```text
leastMate_play_shortest :
  NearMaximalChoice G guard d ch →
  LeastMate G k p → k + 1 ≤ d + 1 →
  (d : Int) * EVAL_ROUGHNESS ≤ 21366 →
  hasKingCapture G p = false →
  MatesWithin G ch k p
```

So `EVAL_ROUGHNESS`-per-ply is the smallest step for which the theorem is
true, and it is true only because achievable distances have a fixed parity.
At one point per ply the gap is 2 against a tolerance of 15 and the shipped
driver can take the slower mate.

**The `3` in the null reduction is load-bearing for this.**  Parity survives
along every path because both depth steps are odd: a real move spends one ply
per negation, the null option spends three (`nullValueD2`'s `d + 1 - 3`).
Writing `d + 1 - 2` or `d + 1 - 4` would let one line reach a mate value of the
wrong rung parity, collapsing the separation gap to `EVAL_ROUGHNESS` -- exactly
the width the driver cannot resolve.  No proof here mentions the null term (the
parity lives in `ForcedMate`, whose `step` is two plies), but a change to that
constant is a change to this theorem.

| fact | theorem | axioms |
|---|---|---|
| no forced mate in zero plies | `not_forcedMate_zero` | `propext, Quot.sound` |
| an even budget is never tight | `forcedMate_pred_of_even` | `propext, Quot.sound` |
| the least distance is odd | `leastMate_odd` | `propext, Quot.sound` |
| distinct distances are two plies apart | `leastMate_gap` | `propext, Quot.sound` |
| the value's rung index is exactly `D - k` | `leastMate_value_block` | `+ Classical.choice` |
| a faster mate outscores a slower one by more than `EVAL_ROUGHNESS` | `leastMate_value_separation` | `+ Classical.choice` |
| the driver's own tolerance suffices | `forcedMate_play_shortest_odd` | `+ Classical.choice` |
| the engine attains the optimal distance | `leastMate_play_shortest` | `+ Classical.choice` |
| a mated node's own distance is even | `leastMated_odd_or_zero` | `propext, Quot.sound` |
| the engine's defence is distance-maximal | `defence_maximal_resistance` | `+ Classical.choice` |

The parity spine is choice-free, like the distance spine it extends; the
`Classical.choice` in the value results is inherited from
`legal_of_allIllegalB_false`, unchanged.

**The defender half.**  "The losing side drags the mate out as long as it can"
is the same ordering read the other way, and it needs no second move rule to
model.  At a lost node the reached positions are attacker-to-move and their
values are positive mate values `MATE_LOWER + (d - n) * EVAL_ROUGHNESS`, and
the engine's one rule MINIMISES the reached value -- so minimising the value is
maximising `n`.  There is no `NearMinimalChoice`: in negamax the defender's
rule is literally the attacker's rule, and the duality lives in the theorem.
`defence_maximal_resistance` proves the local step in distance form: no legal
reply leaves the attacker a nearer mate than the engine's own move does.
Parity refunds the tolerance in exactly the same place as on the attacker side
-- the block bounds plus one `EVAL_ROUGHNESS` of slack give `i <= j + 1`, and
both are odd.

### The global induction (`ResistsFor`)

`ResistsFor G ch n q` is the inductive dual of `MatesWithin G ch n p`: the two
quantifiers swapped and the bound turned around.  There the attacker plays `ch`
and mate ARRIVES within `n` plies against every legal defence; here the
DEFENDER plays `ch` and mate does NOT arrive before `n` plies against every
legal attack.  `MatesWithin.mate` is stated at every index `n + 1` -- once mate
has landed, any remaining budget is met -- and its mirror image is the pair
`zero` / `safe`: while mate has not landed, any budget of at most one ply is
met.  Resistance is downward closed (`resistsFor_anti`) where `matesWithin_mono`
is upward.

One constructor has no mirror image, and the asymmetry is the point.
`MatesWithin.step` carries `hnt` so a moveless defender cannot satisfy its reply
quantifier vacuously -- a stalemate is a draw, not a mate.  The same corner is a
WIN for resistance, so it appears here as the leaf `draw`: a defender with no
legal move who is not in check is never mated at all.  One guard, one leaf, one
reason -- a draw refutes the attacker's claim and establishes the defender's.

The recursion runs on a SPEC-form local step rather than the distance form:

```text
defence_resistance_step :
  NearMaximalChoice G guard d ch →
  allIllegalB G q = false → n % 2 = 1 → n + 1 ≤ d →
  ForcedMate G n (ch q) →
  ForcedlyMated G n q
```

If the engine's own defence lets the attacker mate in `n`, the position was
already mated within `n` against EVERY defence -- the engine gave nothing away.
This is where the driver's tolerance is paid for, and parity refunds it: an
alternative defence one rung worse is read by `forcedMate_of_value_dist` as a
mate in `n + 1`, and `n + 1` is even because `n` is odd, so `forcedMate_odd_le`
hands the ply straight back.  Same refund, same place, as the attacker half.

```text
leastMated_defence_resists :
  NearMaximalChoice G guard d ch →
  LeastMated G k q → k ≠ 0 → k + 1 ≤ d →
  (d : Int) * EVAL_ROUGHNESS ≤ 21366 →
  hasKingCapture G q = false →
  ResistsFor G ch (k + 1) q
```

`k` is the least attacker budget the position permits, so `k + 1` is the exact
distance to mate for the side to move (`leastMated_odd_or_zero` makes it even),
and the engine attains it: it survives as long as the position permits.  With
`leastMate_play_shortest` this is distance-to-mate optimality in both
directions, bundled as `dtm_optimal`.

Two honest notes.  The depth condition is `k + 1 ≤ d`, one ply stricter than
the attacker half's `k ≤ d`: the defender's claim is about what the position
does NOT permit, and refuting a faster mate needs the ply that would have shown
it.  And neither play predicate asserts that `ch` returns a legal move --
`MatesWithin` does not either; legality is supplied where the tree is built, by
`NearMaximalChoice` (`ch q ∈ movesAbove ... ⊆ G.moves q`).

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
- `Shortest.lean`: mate-distance parity, value separation, and shortest-mate
  play under the driver's own stopping tolerance.
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
