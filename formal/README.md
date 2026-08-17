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

If `C(pos) < gamma`, the Python search reports the cap immediately and omits
the child probe. `WindowReport.cap_failLow` proves that this is a valid upper
report for the same `N(pos)`. Otherwise it obtains a one-sided report `r` for
`P` from the complementary zero-window child probe and reports
`min(C(pos), r)`.

`Sunfish/CappedNull.lean` proves the two local steps needed for this operation:

- `WindowReport.negate` transfers a child report at `1 - gamma` to the
  parent window at `gamma`.
- `WindowReport.cap_failLow` permits the child to remain lazy when its fixed
  cap is already below `gamma`.
- `WindowReport.cap` proves that `min(C, ·)` transports a valid report of `P`
  to a valid report of `min(C, P)`.
- `cappedNull_report` composes those two facts for the exact Python expression.

The proof is generic in `C`; it does not depend on chess or mate constants.

The production guard is:

```python
(not root and 2 < depth < 6 and abs(pos.score) < 750
    and any(c in pos.board for c in "RBNQ"))
```

The pass is a score candidate only below depth 6. From depth 6 on it is a
fuel oracle instead -- see the next section.

The shallow candidate retains its original three-ply reduction. The deep
fuel probe has a different role and may use a more aggressive reduction;
sharing one reduction between them changes the declared shallow value.

`EvalBounds.lean` proves that every reachable both-kings static evaluation is
bounded by `EvalBounds.evalBound`. With `EVAL_ROUGHNESS = 15`, theorems
`staticCap_in_scoreBand` and `staticCappedNull_below_positiveMate` prove that
an enabled null move cannot claim a positive mate score. A catastrophically
bad pass may retain a negative mate value; no lower clamp is intended.

The remaining chess-strength premise is:

```text
min(C(pos), P) <= best legal real-move value
```

It concerns the quality of the null approximation, not the fail-soft report
transport. The score guard avoids applying either null mechanism in
statically unbalanced positions, while the non-pawn-piece guard excludes
pawn-only zugzwangs.

## The deep-null fuel oracle

From depth 6 on the pass is not a score candidate at all. One fixed target
shapes only how much depth the real moves spend:

```python
d = depth
guard = (depth >= 6 and abs(pos.score) < 750
    and any(c in pos.board for c in "RBNQ"))
if guard:
    nullpos = pos.rotate(nullmove=True)
    target = pos.score + NULL_MARGIN
    d -= -self.bound(nullpos, 1 - target, depth - 7) >= target

move_depth = d - 1 - (not root and guard and val < LMR)
```

The target depends on `(pos, depth)` alone -- `gamma` does not enter. Table
state may still change the numeric report. Stability therefore uses the normal
TT invariant: every reused interval reports on the same null-child value.
Given valid reports, side-exactness makes the hot classification stable under
different caller windows and table states (`hot_bit_stable`). A fixed target
alone would not repair an invalid or cross-semantics TT entry. Move reduction
uses only the static null-eligibility guard, so it needs no report theorem.

Nominal `depth` still keys the tables and QS admission; intrinsic move value
only selects the recursion depth. Every real edge spends one to three plies,
exactly matching the two code subtractions (`intrinsic_child_depth`,
`intrinsic_edge_cost`). Thus the killer can reorder a move but cannot change
its edge cost, and no MTD window changes the declared tree.

The driver root is deliberately exempt from intrinsic LMR. Root probes are
not stored in `tp_score`, so this does not create a second value for any TT
key; the model's `eligible` bit is false there. Interior nodes retain the
fixed edge-cost recurrence proved by `IntrinsicLMR.lean`.

That is what buys the premise: a null cutoff gives every real move unbounded
pruning debt, and discharging it is exactly what `NoZugzwang` was for. A
bounded edge cost discharges it instead. `EventuallyWide.lean` now quantifies
over an arbitrary move-dependent selector `spend(position, depth, child)`;
the previous null-only policy is the special case that ignores `child`.
`IntrinsicLMR.lean` instantiates this theorem with the fixed-target hot bit,
static eligibility guard, and intrinsic low-value bit. Thus the global theorem
applies directly with maximum edge cost three:

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
masking sites are unreachable and `NoMaskedMobility` and even `EvalQuiet`
drop out. The bound is *effective*: `2N + 8` for the
null-only selector and `3N + 9` for intrinsic LMR (no classical `exists D0`).
The file's entire footprint is
`[propext, Quot.sound]`. The scope is eventual-only, by countermodel: `CexE`
(the infinite masked chain) violates the premise (`cexE_not_finite`), while
`CexD` satisfies it at budget 5 and still prices its drawable masked node in
the mated band at depth 1 (`cexD_fuel_M1`) before classifying it correctly
from depth 10 on (`cexD_M_eventually_classified`). Fixed-depth honesty below
the bound still requires complete move admission.

## Positive-depth moves and shallow move caps

The Python producer admits only moves at or above `QS` at depth zero, but
admits every pseudo-legal real move at positive depth:

```text
producerMoves(depth, pos) =
  if depth = 0 then movesAbove(QS, pos) else moves(pos)
```

`producerMoves_zero` and `producerMoves_positive` state these two cases. The
positive-depth theorem is structural: it needs no move-value floor, window,
or move-table premise. In particular, no filtered legal evasion can fabricate
a mate at the depth-one frontier.

The producer also resolves an intrinsic mate-band move immediately as
`MATE_UPPER`. `producedScore_capture` proves the arithmetic branch, while
`producedScore_exact_capture` uses `HighValIsKingCapture` to show that the
branch is an actual king capture. Recursing into its kingless child would only
return `-MATE_UPPER`, so the normalization is exact.

At depths zero through three, every other admitted move passes through the
same static cap

```text
min(min(MATE_LOWER - 1,
        pos.score + pos.value(move) + (depth - 1) * QS_A),
    full child value)
```

This is a fixed function of the position, move, and depth. At depths zero and
one, natural subtraction makes the margin zero. The score identity then makes
the cap exactly the existing stand-pat futility report; this is
`shallowMoveCap_lowDepth` together with `futilityOK_discharged`. At depths two
and three, the cap defines the selective move value. If it is below `gamma`,
`cappedMove_failLow` proves that the cap itself is a valid fail-low report and
the child search is skipped. Otherwise `WindowReport.cap` transports the full
child report through `min`.

Only king captures bypass the cap. The cap is explicitly below the positive
mate band, so it cannot create a positive mate value;
`shallowMoveCap_below_positiveMate` and `cappedMove_positiveMate_only_from_full`
state that direction. For an eligible move, `pos.score + pos.value(move)` is
the negated static score of a both-kings child. `EvalBounds.lean` puts it
strictly above `-MATE_LOWER`, so the positive margin cannot cross downward and
no lower clamp is needed. `cappedMove_preserves_negativeMate` proves that a
genuine negative mate is retained exactly.

The cap deliberately changes ordinary shallow move values, including captures,
promotions, and checks. It can therefore delay a mate proof found exactly at
the selective frontier. It exists only at depths two and three, so deeper
iterations move every fixed proof above that frontier. Its strength is an
empirical question, separate from report correctness.

Each ordinary cap is evaluated independently. A sub-window cap avoids only
that move's child search; it does not terminate the producer. Thus ordering
changes proof work but neither move membership nor the declared value.

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

**An odd null reduction is load-bearing for this.**  Parity survives along
every path because both depth steps are odd: a real move spends one ply per
negation, and the current null probe spends seven. The generic model uses
three as the smallest nontrivial representative. An even reduction would let
one line reach a mate value of the
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
| faster mate gap exceeds `EVAL_ROUGHNESS` | `leastMate_value_separation` | `+ Classical.choice` |
| the driver's own tolerance suffices | `forcedMate_play_shortest_odd` | `+ Classical.choice` |
| the engine attains the optimal distance | `leastMate_play_shortest` | `+ Classical.choice` |
| a mated node's own distance is even | `leastMated_odd_or_zero` | `propext, Quot.sound` |
| the engine's defence is distance-maximal | `defence_maximal_resistance` | `+ Classical.choice` |
| resistance is downward closed | `resistsFor_anti` | `propext, Quot.sound` |
| a checkmated node resists for nothing | `not_resistsFor_of_checkmated` | `propext, Quot.sound` |
| the defence gives nothing away | `defence_resistance_step` | `+ Classical.choice` |
| the engine's defence is legal | `defence_move_legal` | `+ Classical.choice` |
| the defence lasts the whole distance | `defence_resists` | `+ Classical.choice` |
| the engine attains the optimal distance on the losing side | `leastMated_defence_resists` | `+ Classical.choice` |
| both directions at once | `dtm_optimal` | `+ Classical.choice` |

The parity spine is choice-free, like the distance spine it extends -- and so
is the new play predicate's monotonicity; the `Classical.choice` in the value
results is inherited from `legal_of_allIllegalB_false`, unchanged.  No new
chess premise: the defender half spends `ValFloor`, `EvalQuiet`,
`NoMaskedMobility` and `NoZugzwang` exactly as the attacker half does, and
`defence_move_legal` does not need `NoZugzwang` at all.

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

The carrier of the recursion is the negative statement `∀ i, ForcedlyMated G i q
→ N ≤ i + 1`, not a least distance, and deliberately so: it also covers the node
the attacker has already spoiled, where no forced mate exists and the quantifier
is vacuous.  The engine defends an escaped position as readily as a lost one and
the induction needs no case split for it.  Each attacker reply inherits a budget
two plies smaller, and the arithmetic closes with nothing to spare: the reply's
own mate budget `i` makes `ch q` a mate in `i + 2` (or in ONE, if the reply
mates at once), the local step turns that into `ForcedlyMated G j q` with `j`
odd and `j ≤ i + 2`, and the carrier at `q` gives `N ≤ j + 1 ≤ i + 3`.

Two honest notes.  The depth condition is `k + 1 ≤ d`, one ply stricter than
the attacker half's `k ≤ d`: the attacker only has to FIND the mate it plays,
while the defender has to outlast the faster mate that does not exist, and
refuting that one needs the ply that would have shown it.  And `ResistsFor`
does not assert that `ch q` is a legal move -- `MatesWithin` does not either --
so the fact is proved separately rather than assumed:

```text
defence_move_legal :
  NearMaximalChoice G guard d ch →
  allIllegalB G q = false → hasKingCapture G q = false →
  (∀ i, ForcedlyMated G i q → N ≤ i + 1) → 3 ≤ N → N ≤ d →
  hasKingCapture G (ch q) = false
```

An illegal defence hands over the king, which the model prices at the exact
`MATE_UPPER` sentinel -- the largest value there is -- and the engine minimises
the value it moves to, so every legal alternative would have to score within
`EVAL_ROUGHNESS` of the sentinel as well.  That is where the terminal clamp
becomes quantitative: `hspan` caps the horizon at `d ≤ 1424` plies, so
`forcedMate_of_value_dist` at `t = 1423` leaves room for a mate in ONE and
nothing longer, every legal move at `q` would be mate in one, and the carrier
says `q` is not mated in two.  Below three plies the question is empty: mate
cannot land before ply two whatever the engine plays.  The recursion hands each
child these same hypotheses, so the same theorem applies at every node of the
play whose remaining budget is at least three.

### Can the frontier premise be dropped? No — `CexD`

`NoMaskedMobility` is the model-side stand-in for the engine fix in PR #171
(search the QS-filtered evasions before declaring mate).  The natural hope is
that the distance machinery makes it unnecessary: a phantom mate is invented at
the frontier, where almost no depth is left, so surely it can only claim the
DEEPEST rungs, and a real mate in `k` with `k + 1 ≤ d` claims a higher one.

That hope is false, and one number says why.  **A masked node does not report a
shallow mate — it reports the sentinel.**  Its filtered fold has no admitted
legal move left, so nothing displaces the initial accumulator
`LOSS = -MATE_UPPER`, and the parent negates that to `MATE_UPPER`, which is not
a rung at all: it is strictly above every value the ladder can produce
(`mateFloor_lt_MATE_UPPER`).  A phantom outranks EVERY real mate, at every
distance and every depth, and no rung argument can separate what is off the
ladder.

`CexD` is that hope refuted at the exact hypotheses of
`leastMated_defence_resists`.  `Q` is genuinely lost in four plies; the correct
defence `D` leads to a real mate in three, the fast loss `B` to mate in one, and
one ply past the frontier of `D`'s line sits `M` — a defender node whose only
legal reply is filtered by the depth-1 threshold (`-150 < -100`) while an
illegal one survives it, the position class of the #171 report, one ply deeper.
`D` is therefore priced at `MATE_UPPER = 69290` instead of its true rung 47938,
the mate in one is priced honestly at 47968, and the engine prefers to be mated
in two.  Every premise but `NoMaskedMobility` holds — including the move rule in
its IDEALISED exact-argmax form, so the driver's tolerance is not what breaks it
— and the conclusion is false:

| fact | theorem | axioms |
|---|---|---|
| the masked node reports the sentinel | `cexD_M1` | `propext, Quot.sound` |
| ... which outranks the true mate at the root | `cexD_D4`, `cexD_B4` | `propext, Quot.sound` |
| the position really is lost in exactly four plies | `cexD_leastMated` | `propext, Quot.sound` |
| the engine is mated in two | `cexD_not_resists` | `propext, Quot.sound` |
| **the premise cannot be dropped** | `cexD_defence_needs_frontier` | `propext, Quot.sound` |

Two things this settles.  **Acyclicity does not help**: `CexD` is a finite tree
with no repetition in it, so no draw-by-repetition or well-founded-descent
argument touches the failure. **Depth does not help** when a selective frontier
travels with the search. The current producer removes that frontier outright:
at every positive depth it searches the complete move list, as stated by
`producerMoves_positive`. Positive-depth reports therefore target the complete
real-move fold without a masked-mobility premise.

## The eventual classification (`Eventual.lean`)

`eventual_classification` states the trichotomy at EVERY depth and
spends four premises.  This module asks how many of them survive
weakening the conclusion to "from some depth `D0` on" -- which is all
the driver's `range(1, 1000)` deepening loop ever needs.  Two answers,
one negative and one positive, and one premise that turns out never to
have been needed.

**`NearMaximalChoice` is free here, and the mate tempo is why.**  The
driver stops bisecting at `upper - lower <= EVAL_ROUGHNESS`, so the
shipped root may settle for a value within 15 of the best.
`Shortest.lean` pays for that one RUNG at a time and needs parity to
refund it.  At band granularity nothing needs refunding, and the
statement that makes this a theorem rather than an appeal to
magnitudes is:

```text
nullValueD2_offCorridor :
  0 ≤ B → EvalBand G.toNullGame.toGame B →
  ∀ d p, OffCorridor B (nullValueD2 G guard d p)
```

The declared value never enters the corridor between the static score
range and either band edge.  Four cases, no chess content: the two
sentinel branches, the terminal ladder (or the stalemate `0`), the
static eval, and closure of the corridor's complement under negation
and `max` -- which is all the fold does.  For the shipped tables that
corridor is `MATE_LOWER - evalBound = 47923 - 15437 = 32486` points
wide: `shipped_band_gap_wide` machine-checks that it exceeds **two
thousand** stopping tolerances.  So a 15-point slack cannot move a
value across a band edge, and the move the driver leaves in `tp_move`
is classified no worse than any admitted alternative:

```text
nearMaximal_band_exact :
  NearMaximalChoice G guard d ch → m ∈ movesAbove G (val_lower (d+1)) p →
  bandOf (nullValueD2 G guard d (ch p)) ≤ bandOf (nullValueD2 G guard d m)
```

with `nearMaximal_keeps_mate` ("a near-maximal choice never misses a
mate") and `driver_stop_band_stable` ("the converged bracket cannot
straddle a band edge") as the two readings that matter.

**The design fact both of these rest on.**  One whole `EVAL_ROUGHNESS`
of mate tempo per ply was a deliberate choice (#172), not a scaling
convenience, and it is load-bearing in two different places for two
different reasons.  At RUNG level it is the *smallest* step that works:
consecutive mate distances are two plies apart, two plies is two rungs,
and the resulting gap is strictly wider than the driver's tolerance --
at one point per ply the gap would be 2 against a tolerance of 15 and
the shipped driver could take the slower mate (`Shortest.lean`, where
parity does the refunding).  At BAND level the same constant is
irrelevant by three orders of magnitude, which is what lets
`NearMaximalChoice` be dropped from classification results entirely.
A change to the tempo would have to be re-argued in both places; an even null
reduction would have to be re-argued in the first.

**The frontier premise, however, survives the weakening.**  Masking is
genuinely local -- `val_lower 2 = -240` is below the tables' -192 move
value floor, so from remaining depth 2 up the filter is the identity
(`filter_identity_off_frontier`) -- which is why the eventual reading
looked promising: `CexF`'s phantom dissolves at depth 3.

What kills it is that a masked node does not report a shallow mate.  It
reports the OFF-LADDER sentinel (`maskedFrontier_value`): the filtered
fold has no admitted legal move, nothing displaces the initial `LOSS`
accumulator, and the parent negates it to `MATE_UPPER`.  The ladder
decays one rung per unspent ply; the sentinel decays not at all.  And
the frontier RENEWS the phantom at every horizon -- absorbing the old
one buys nothing when a new one arrives one ply deeper.

`CexE` is that argument machine-checked.  An infinite chain
`C 0 -> C 1 -> ...` in which every node is masked: each generates an
illegal move valued 0 (admitted at every depth) and the legal
continuation valued -150 (filtered at `val_lower 1 = -100`, admitted
from depth 2 on).  The root is a draw in the strongest sense -- no
forced mate at any budget for either side -- and its declared value is

```text
D:        0     1     2     3     4     5    ...
value:    0   -MU    +MU   -MU   +MU   -MU   ...
```

Odd depths report "I am mated", even depths report "I mate".  The value
never re-enters the band, so there is no `D0`: the classification is
not merely unsettled, it is WRONG in both directions, alternately,
forever.

Every fidelity premise holds, including `EvalBand` at every width
(every live static score in `CexE` is exactly 0), so this is not a
granularity failure and the section-2 machinery does not touch it.
`NoZugzwang` is vacuous -- the null option is off throughout, so the
pass has nothing to do with it.  What fails is exactly
`NoMaskedMobility` (`cexE_masked`).

Two escape routes are closed by construction.  **Acyclicity does not
help**: `cexE_acyclic` -- the only legal move from `C n` is `C (n+1)`,
so the chain index strictly increases and no position ever repeats.
**A read-time clamp does not help**: `cexE_clamp_no_help` -- the
phantom arrives as the exact sentinel, so clamping the root read at
`MATE_UPPER - 1` still leaves 69,289, well inside the band.

**What the weakening does buy.**  Both COMPLETENESS arms never needed
the frontier premise at all (`eventual_completeness_without_frontier`,
`ValFloor` + `NoZugzwang` only).  A phantom invents mates that are not
there; it cannot hide one that is.  `NoMaskedMobility` pays for exactly
one thing -- the honesty arm -- and that is now precise rather than
inherited from the shape of the proof.

**And the engine change still works.**  The frontier-tail variant of
Part B values `CexE`'s root at an honest `0` at every depth and every
chain position (`cexE_t_honest`), so `CexE` joins `CexF` as a positive
test for #171.  With `CexD` (fixed-depth play) and `CexE` (eventual
classification) both refuting the premise-free reading over the shipped
value function, the frontier tail is the only route left to retiring
`NoMaskedMobility`.

| fact | theorem | axioms |
|---|---|---|
| masking lives only at remaining depth 1 | `filter_identity_off_frontier` | `propext, Quot.sound` |
| a masked node reports the sentinel, not a rung | `maskedFrontier_value` | `propext, Quot.sound` |
| completeness needs no frontier premise | `eventual_completeness_without_frontier` | `propext, Quot.sound` |
| the declared value avoids the band corridor | `nullValueD2_offCorridor` | `+ Classical.choice` |
| the shipped corridor is 2000 tolerances wide | `shipped_band_gap_wide` | none |
| the tolerance cannot move a value across a band edge | `bandOf_eq_of_slack` | `propext, Quot.sound` |
| a near-maximal choice is band-exact | `nearMaximal_band_exact` | `+ Classical.choice` |
| ... and never misses a mate | `nearMaximal_keeps_mate` | `+ Classical.choice` |
| the converged bracket cannot straddle a band edge | `driver_stop_band_stable` | `propext, Quot.sound` |
| the drawn root's value oscillates between the extremes | `cexE_ladder` | `propext, Quot.sound` |
| ... with no forced mate for either side | `cexE_no_forcedMate`, `cexE_no_forcedlyMated` | `propext, Quot.sound` |
| ... and no repeated position anywhere | `cexE_acyclic` | `propext` |
| ... which the failing premise is `NoMaskedMobility` | `cexE_masked` | `propext` |
| a read-time clamp does not rescue it | `cexE_clamp_no_help` | `propext, Quot.sound` |
| **trichotomy needs the frontier premise** | `eventual_classification_needs_frontier` | `propext, Quot.sound` |
| the frontier tail classifies the same root correctly | `cexE_t_honest` | `propext, Quot.sound` |
| both directions at once | `eventual_classification_verdict` | `propext, Quot.sound` |

The countermodel and the verdict are choice-free; the `Classical.choice`
in the corridor results is the same by-case pattern
`eventual_classification` itself carries.  One new fidelity premise,
`EvalBand B` -- the two-sided form of the table bound `EvalQuiet` reads
one-sidedly, discharged for the shipped tables at
`B = EvalBounds.evalBound = 15437`.

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
King-capture substitution uses the position predicate directly, so its exact
`MATE_UPPER` promise does not depend on move-table contents.

## Current source correspondence

| Python mechanism | Lean result or model |
|---|---|
| recursive zero-window move search | `Bound.bound_spec` |
| null child report negation | `WindowReport.negate` |
| lazy `min(pos.score + EVAL_ROUGHNESS, pass_report)` (depth < 6) | `cap_failLow`, `cappedNull_report` |
| fixed-target fuel probe | `hot_bit_determined`, `hot_bit_stable`, `fuel_edge_cost` |
| static LMR eligibility and intrinsic move reduction | `intrinsic_edge_cost` |
| real-move recursion at the reduced `d - 1` | `fuelValueD2t`, `eventual_classification_fuel` |
| static cap below positive mate | `staticCap_in_scoreBand`, `staticCappedNull_below_positiveMate` |
| positive-depth complete producer | `producerMoves_positive` |
| exact king-capture producer report | `producedScore_exact_capture` |
| shallow move cap and lazy child evaluation | `shallowMoveCap_lowDepth`, `cappedMove_report` |
| cap mate-band properties | `shallowMoveCap_below_positiveMate`, `cappedMove_preserves_negativeMate` |
| filtered move fold and early cutoff | `Bound.searchMoves_spec` and the fold models in `Stalemate.lean` |
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

## Module guide

- `GameTree.lean`: chess-free negamax game model.
- `Bound.lean`: core fail-soft search proof.
- `CappedNull.lean`: capped-null report transport and score-band facts.
- `CappedMove.lean`: positive-depth move production and shallow move caps.
- `Stalemate.lean`: selective-search fold, legality, and terminal finalizer.
- `EvalBounds.lean`: numeric bounds induced by the piece-square tables.
- `Killer.lean`: move-table legality and lifecycle.
- `CanNull.lean`: root/interior null and table-key conditions.
- `Driver.lean`: MTD-bi bracket invariants and convergence.
- `TableSwap.lean`: table-update properties.
- `Liveness.lean` and `Classification.lean`: mate visibility and search-result
  classification under their named fidelity premises.
- `Eventual.lean`: what the "from some depth on" weakening of the
  trichotomy buys -- `NearMaximalChoice` retired at band granularity,
  completeness shown independent of the frontier premise, and `CexE`,
  the countermodel showing the frontier premise survives the weakening.
- `Shortest.lean`: mate-distance parity, value separation, distance-to-mate
  optimal play in both directions under the driver's own stopping tolerance, and
  the countermodel showing the frontier premise cannot be dropped from it.
- `Pruning.lean` and `Tricks.lean`: proof envelopes for selective-search
  techniques.
- `EventuallyWide.lean`: the fuel oracle -- bounded real-edge cost and the
  W/D/L trichotomy with no chess premise.
- `IntrinsicLMR.lean`: static LMR eligibility and the bounded, move-dependent
  edge cost used by intrinsic LMR.
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
