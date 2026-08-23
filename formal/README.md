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
guard = depth >= 6 and abs(pos.score) < 750 and any(c in pos.board for c in "RBNQ")
if guard:
    t = pos.score + NULL_MARGIN
    d -= int(-self.bound(pos.rotate(nullmove=True), 1 - t, depth - 7) >= t)

move_depth = depth - 1 - (guard and depth >= 6 and val < LMR) - int(nmr)
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
`[propext, Quot.sound]`. The scope is eventual-only for the FUEL reason (a
fuel-bounded value can be sub-horizon inaccurate), not for a masking reason:
since `c01915f` the once-masked node of `CexD` is priced honestly at depth 1
too (`cexD_fuel_M1 = 0`) and agrees with the eventual reading from depth 10
(`cexD_M_eventually_classified`). Complete move admission at positive depth is
what the code now provides, so fixed-depth honesty no longer waits on it.

## How much depth a forced mate costs (`MateDepth.lean`)

`eventual_classification_fuel_arms` arms a win at `D >= C*k + 4` and a loss at
`D >= C*k + C + 4` -- `3k + 4` and `3k + 7` as shipped. Neither constant is
sharp, and neither mechanism-checks against the shallow move cap.
`MateDepth.lean` replaces both, and certifies the replacements with a
countermodel rather than an argument.

**Where the slack was.** The old induction demands the real-only regime
(nominal depth `>= 6`) at *every* node of the mating line and charges a full
`C` for the edge into the mate. Two of those charges are not needed:

- an attacker node is safe at the admission floor. Its fold is a MAX, the
  mating child is admitted by `ValFloor` from remaining depth 2 on
  (`mem_movesAbove_of_floor`), and the sub-horizon pass candidate enters the
  same max -- `foldMax_le_of_mem` ignores the accumulator, so a pass can never
  pull the maximum down. Below the horizon the code also reduces nothing, so
  those edges cost exactly one ply. (`ValFloor` is what the *model's*
  `movesAbove (val_lower d)` needs, and since `c01915f` it buys admission from
  remaining depth 1 rather than 2: the model's `val_lower` is now the shipped
  two-valued admission and agrees with `producerMoves_positive`.)
- the checkmated leaf is classified by the depth-gated terminal correction at
  any depth `>= 1` (`fuelValueD2_checkmated`).

Only DEFENDER nodes need the horizon: their fold is bounded above, so the
sub-horizon pass -- the candidate the `not root and 2 < depth < 6 and ...`
guard admits -- can mask the mate. The mating line's last two plies are an
attacker node and the leaf, so the horizon has to be reached two plies before
the end, not at the end:

```text
forcedMate_fuelValueD2_sharp    :  D >= max 2 (C*(k-2) + 6)   -- 3k    shipped (was 3k+4)
forcedlyMated_fuelValueD2_sharp :  D >= max 6 (C*(k-1) + 6)   -- 3k+3  shipped (was 3k+7)
```

**The shallow cap costs one more ply, and `fuelValueD2` did not have it.**
The fuel value omits

```python
cap = MATE_UPPER if depth > 3 or val >= MATE_LOWER else pos.score + val + margin
```

and that cap puts every non-king-capture report strictly below `MATE_LOWER`
(`shallowMoveCap_below_positiveMate`, under the both-kings material
invariant `CapInBand` - the clamp that used to make this syntactic was
dead code and is gone): an attacker node at any nominal depth
zero through three cannot report a mate at all. This is the delay the section
above calls "a mate proof found exactly at the selective frontier", priced.
`capClamp` carries the same `depth <= 3` band as the shipped `cap`, and the
two ends of the band behave differently:

- at depths **two and three** the clamp is the selective cap, and it binds:
  the mate the attacker can see one ply below is replaced by
  `pos.score + val + (depth - 1) * QS_A`.
- at depths **zero and one** natural subtraction flattens the margin and the
  clamp is the old stand-pat futility estimate (`shallowMoveCap_lowDepth`).
  It is mate-neutral there, and not by luck: a fold weight can only reach the
  positive mate band through a child whose king is gone, and such a parent
  fires the node-level `hasKingCapture` branch before any fold is taken. So
  widening the band from `2 <= depth <= 3` to `depth <= 3` moves no value in
  this file.

The cap only ever LOWERS a report, so a defender node (bounded above) pays
nothing for it; the attacker's floor rises from 2 to 4 -- depths 2 and 3 are
blocked outright and depths 0 and 1 are below the fold floor either way.
`fuelValueD2C` is `fuelValueD2` with the clamp on every fold weight, and

```text
forcedMate_fuelValueD2C_sharp    : D >= max 4 (C*(k-1) + 4) (C*(k-2) + 6)  -- 3k+1
forcedlyMated_fuelValueD2C_sharp : D >= max 6 (C*k + 4) (C*k + 6 - C)      -- 3k+4
```

Premises unchanged throughout: `ValFloor G 192` and nothing else -- no
`NoZugzwang`, no mate-band agreement. Layer 1 for the fuel shape
(`FuelBracketSpec`) remains stated and unproven, so these are bounds on the
declared value, as the theorems they replace were.

**Sharpness.** `MDG` is a ten-position game inside the hypothesis class
(`sharp_valFloor : ValFloor MDG 192`) with a forced mate in 3 plies at `A1` and
in 5 at `A2`, and an edge spend of 2 (the hot bit plus the intrinsic-LMR bit,
`min (C-1) 2 = 2`) at every regime node -- a schedule the shipped code
realizes whenever the fuel probe fails high on a quiet move. One defender node
lands at nominal depth 5, inside the sub-horizon window, where the pass is
worth 0 and the mate is masked:

| certificate | statement |
| --- | --- |
| `sharp_mate3_at_8` | mate in 3 plies, value 0 at `D = 8 = 3*3 - 1` |
| `sharp_mate5_at_14` | mate in 5 plies, value 0 at `D = 14 = 3*5 - 1` |
| `sharp_mated3_at_11` | the dual escapes at `D = 11 = 3*3 + 2` |
| `sharp_cap_mate3_at_9` | with the cap, `D = 9 = 3*3` is still one ply short |

The pair at 8 and 14 is `2*C` apart, so no bound with a slope below `C` holds
either: the certificates pin the slope as well as the constants.

**The CI table.** `tools/quick_tests.sh` states the convention -- mate-in-`n`
moves is `k = 2n - 1` plies -- and currently spends `3k + 4`:

| suite | `k` | script today | proved, fuel model | proved, shipped (cap) |
| --- | --- | --- | --- | --- |
| `mate1.fen` | 1 | 7 | 2 (`ci_mate_in_1`) | 4 (`ci_code_mate_in_1`) |
| `mate2_eventual.fen` | 3 | 13 | 9 (`ci_mate_in_2`) | 10 (`ci_code_mate_in_2`) |
| `mate3_eventual.fen` | 5 | 19 | 15 (`ci_mate_in_3`) | 16 (`ci_code_mate_in_3`) |

The shipped column is the one a CI depth may be lowered to. The gap between
the columns is the shallow cap, and it is not academic: at depth 3 the suite's
mate-in-1 positions are all missed, and at depth 4 all eight are found. The
fuel-model column -- 2 / 9 / 15 -- costs the whole cap, on both branches of
the consumer, at the Elo the correction under the menu records; the cheap
route to it is refuted there, so those three depths are not reachable that
way.

**Menu instances.** The bound is generic in the edge-cost cap `C` and in the
sub-horizon guard, so the price of each mechanism is a corollary rather than a
new proof:

| engine variant | bound | theorem |
| --- | --- | --- |
| today (`C = 3`, cap, sub-horizon pass) | `3k + 1` | `forcedMate_fuelValueD2C_sharp` |
| one reduction bit (`C = 2`) | `2k + 2` | `forcedMate_fuelValueD2C_C2` |
| no reductions (`C = 1`) | `k + 4` | `forcedMate_fuelValueD2C_C1` |
| delete the sub-horizon pass ONLY | `3k + 1` -- unchanged | `code_mate_depth_bound_sharp_k3_guardOff` |
| delete the shallow cap ONLY -- on BOTH branches | `3k` | `forcedMate_fuelValueD2_sharp`, sharp per `sharp_mate3_at_8` |
| exempt only the SEARCHED report from the cap | `3k` for the declared value | **REFUTED on correctness** -- gamma-dependent, see below |
| delete both | `max 2 (C*k + 4 - 3*C)`, i.e. `3k - 5` | `forcedMate_fuelValueD2_noSubPass` |

The fourth row is the useful surprise: the cap and the sub-horizon pass mask
in *different* depth bands (2--3 for the capped attacker, 3--5 for the
defender's pass), and removing either one alone leaves the other binding. The
certificate is the same witness game with the guard off -- it still masks, at
the capped attacker node.

`defender_le_of_replies` is the step those last two share: a defender node
reports at or below `-MATE_LOWER` as soon as its fold carries no pass term --
either above the horizon, or with the guard off at every depth.

**Correction: the cheap instance of the `3k` row is REFUTED (2026-08-17).**
That row invites an obvious ten-byte reading -- keep the clamp, but exempt a
child report that came back at or above `MATE_LOWER` -- which was expected to
be Elo-flat, since it moves no node the search would otherwise visit. The
measurement lane built that arm, and it is unsound. Exempting only the
*searched* report makes `bound()` **gamma-dependent**: the consumer runs two
branches under one cap,

```python
if cap < gamma: move, score = None, cap
else: score = min(cap, -self.bound(pos.move(move), 1 - gamma, reduced_depth))
```

and the exemption lifts the clamp off the second while the first still claims
the move is worth at most its static estimate. Both write the same
`(pos, depth)` table key, so the entry holds both claims at once. The measured
witness is one key at depth 2 with `Entry(lower=47938, upper=1204)`:
`47938 = MATE_LOWER + EVAL_ROUGHNESS` is the exempted mate returning from a
child checkmated one ply down, `1204` is the same key's static estimate from
the fail-low branch, and `lower > upper` by 46,734. Twelve
`tests/test_terminal_bench.py` positions and a `tests/test_tt_consistency.py`
fortress fail, all with "ladder crossing". The mates themselves do arrive --
`mate1.fen` 0/8 to 6/8 at depths 2 and 3, with the 24-opening depth-8 node
battery byte-identical -- so what the exemption costs is exactly the table
invariant and nothing else.

**The error was not the transformer monotonicity.** `capClamp_le` is true and
stays true: the clamp only lowers, so dropping it on a searched report can
only raise the declared value, and `forcedMate_fuelValueD2_sharp` does give
`3k` for that value. The declared-value change is fine. What does not follow
is the code change, because the theorem bounds ONE `(pos, depth, gamma)`
report while the engine keeps ONE entry per `(pos, depth)` and both branches
of the cap write it -- `WindowReport` and table consistency are what break,
not the fold bound. Stated generally: **a cap may be dropped on a searched
report only if it is also dropped on the unsearched one**, and no static rule
can know that an unsearched child mates. Under a `(pos, depth)`-keyed table,
shallow futility and shallow mate detection are mutually exclusive.

Two sound routes survive, and neither is cheap. Dropping the cap on *both*
branches is the row above as written -- gamma-independent, because no second
branch is left to contradict -- and that engine is already priced at
**-60.41 +/- 26.61 Elo** over 488 games (SPRT H0), i.e. the ply is buyable at
about sixty Elo, which is Elo-inadmissible. A `(pos, depth, bound-type)`-aware
treatment, which would let the two branches disagree without lying to the
table, is unpriced and recorded here as a note only. Evidence:
`measure/search-features-ledger` at `0af3507`, arm `exp/mate-band-exempt`.

**Validity against the current search.** These bounds were first proved
against pre-#216 `bound()`. #215 and #218 moved the fuel probe and the
intrinsic-LMR bit out of the `moves()` generator, deleted the depth-one lazy
tail and the `depth <= 1` futility break, made admission unconditional at
positive depth, and widened the cap band to `depth <= 3`. The two mechanisms
the bounds spend are byte-identical after the move -- the probe is still one
ply at `depth - 7`, and the reduced move depth is still one intrinsic-LMR
bit and one fuel bit off `depth` - now spelled `depth - 1 - (guard and
depth >= 6 and val < LMR) - int(nmr)` with `guard = not root and calm`,
the same predicates - so `C = 3` holds -- and the
admission change cannot reach the proofs, which never fold at nominal depth
one. `MateDepth.lean`'s header carries the mechanism-by-mechanism audit. The
suites confirm it: first-success depths are unchanged across the refactor at
`mate1` 4, `mate2_eventual` 7, and `mate3_eventual` 15, with `mate1` still
missed at 3 -- the mate-in-1 corner is exactly tight at `3k + 1`.

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

**The selective fold agrees.** `CappedMove.lean` models this line as
`producerMoves`; the selective-search fold in `Stalemate.lean` models the same
line as `val_lower`, and until now it still carried the PRE-`c01915f` sloped
form `QS - depth * QS_A`. The two model files disagreed at depth 1 for moves
valued in `[-192, -100)` — conservative for the soundness theorems (the model
searched a subset of what the code searches) but load-bearing for everything
built ON the masking. `val_lower` is now the shipped two-valued admission

```text
val_lower(depth) = if depth = 0 then QS else -MATE_UPPER
```

with `val_lower_zero` and `val_lower_pos` naming the two arms. Its one
consequence, spent everywhere downstream, is `movesAbove_pos`: under ANY
move-value floor inside the band the filtered list at positive depth IS the
pseudo-legal list. The pre-`c01915f` threshold survives as `val_lower_pre`,
used only to state what the change bought (`admission_widened_at_frontier`:
the old admission edge was -100 with a -192 table floor below it; the new one
is -69290, which nothing in the band undercuts).

What that turns into, module by module:

| before `c01915f` | after, in the model |
|---|---|
| `NoMaskedMobility` — chess premise, required per `CexF`/`CexD`/`CexE` | THEOREM under `ValFloor` (`noMaskedMobility_of_valFloor`) |
| the #136 value gate (`depth > 2 or all(...)`) decided something | tautology at every positive depth (`gate_always_on`); the ORACLE gate `allIllegalB` is the one that still decides (`reducedScan_needs_premise`) |
| the sentinel after a filtered loop needed the gate to be trustworthy | trustworthy ungated (`correction_trustworthy_ungated`, `boundA1_exhaustion_ungated`) |
| `qsUngated_not_sound` — the ungated correction had a countermodel | no countermodel: `CexQ`'s node folds both moves and all three readings agree (`cexQ_ungated_repaired`) |
| `mem_movesAbove_of_floor` from remaining depth 2 | from remaining depth 1 |
| "every move is eventually admitted", unconditionally | admitted from depth 1, conditional on `ValFloor` — the one thing the change costs, and the floor was already a premise everywhere |

The producer also resolves an intrinsic mate-band move immediately as
`MATE_UPPER`. `producedScore_capture` proves the arithmetic branch, while
`producedScore_exact_capture` uses `HighValIsKingCapture` to show that the
branch is an actual king capture. Recursing into its kingless child would only
return `-MATE_UPPER`, so the normalization is exact.

At depths zero through three, every other admitted move passes through the
same static cap

```text
min(pos.score + pos.value(move) + (depth - 1) * QS_A,
    full child value)
```

This is a fixed function of the position, move, and depth. It carries no
mate-band clamp: king captures are peeled first, and `CapInBand` (the
both-kings material invariant, with its `piece[Q] >~ 2400` tuner caveat
stated at the definition) keeps the capped sum a third of the way to
`MATE_LOWER`, so the `min(MATE_LOWER - 1, ...)` ceiling the code used to
spell never bound - `EvalBounds`' headline is the concrete arithmetic. At depths zero and
one, natural subtraction makes the margin zero. The score identity then makes
the cap exactly the existing stand-pat futility report; this is
`shallowMoveCap_lowDepth` together with `futilityOK_discharged`. At depths two
and three, the cap defines the selective move value.

The implementation evaluates that same fixed fold lazily, and it never
computes the threshold. The producer yields `(value, move)` pairs - the sort
already paid for every `pos.value` call, so the consumer reuses them - and
the consumer caps each real move at its one scoring site. When the cap is
below `gamma` it folds the cap in place of a child search - a virtual
report, so the move leaves no legality witness either (`cappedMove_failLow`)
- and BREAKS: the stream is sorted and the cap monotone, so nothing after
the first settled move can cap higher. The break sits before the shared
`live` update and skips the cutoff block, so a settled move witnesses no
legality and stores no killer, exactly as the old suffix report did. No
tail is materialized and no count is kept.

The model still describes that as a partition, because it is one. Solving
`cap < gamma` for the intrinsic move value gives

```text
threshold = max(base,
    min(MATE_LOWER, gamma - pos.score - (depth - 1) * QS_A)),
```

and `shippedCap_iff_tail` proves the shipped predicate `cap < gamma` holds on
exactly the moves below that threshold - for EVERY window, now that the cap
is unclamped. (The old side condition `gamma <= MATE_LOWER - 1` marked where
the threshold's `min(MATE_LOWER, ...)` clamp and the cap's dropped
`min(MATE_LOWER - 1, ...)` clamp agreed; the `val >= MATE_LOWER` arm keeps
king captures out of the capped branch and carries the mate-band windows.)

`lazyMoveTail_cap_lt_gamma` proves every tail cap is below the window, and
`lazyMoveTail_report` proves the fold of those caps is a valid report for the
capped tail. The stop delivers exactly that report: the cap is monotone in
the intrinsic value (`shallowMoveCap_max`), so the first settled move of the
decreasing sort carries the maximum cap of the whole tail
(`foldMax_shallowMoveCap`, specialised as `lazyMoveTail_maxCap`), and
`WindowReport.max` absorbs any earlier settled-killer report.
`lazyMove_partition` proves that processing tail then prefix is exactly the
original producer fold, and `lazyMove_partition_prefixFirst` proves the same
for the order Python actually uses. `max` is commutative, so the order is
free, and `lazyMove_partition_emptyTail` covers the windows at which nothing
is settled. The partition depends on `gamma`; the declared capped value does
not.

The killer is the one move the break cannot see: it is yielded before the
sorted stream, so its cap says nothing about what follows. The producer
therefore admits it by its own ceiling - "admit it only if its own ceiling
still reaches gamma - the same number the consumer caps it at below", as
the code's comment puts it - spelled as the unclamped disjunction
`val >= MATE_LOWER or depth > 3 or pos.score + val + max(depth - 1, 0) *
QS_A >= gamma`, which IS the old threshold with its `min` unfolded
(`v >= min(a, b)` iff `v >= a` or `v >= b`), so the gate is exactly the
retired `val_lower` test. A killer that would settle is simply not yielded;
it still reaches the fold in its sorted position, inside the tail whose
report the break delivers anyway. A searched killer that failed low is
re-searched from the sorted stream against a table entry, as before. Above depth three the threshold equals `base`, the tail is
empty, and the cap disappears. For searched moves, `WindowReport.cap` still
transports the child report through `min`.

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
`NoMaskedMobility` (formerly chess, layer 2 -- required per `CexF`; since
`c01915f` a THEOREM under `ValFloor`, `noMaskedMobility_of_valFloor`),
`NoZugzwang` (chess, layer 2), root legality.  No new chess premise.  `#print axioms`:
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

**The null reduction is a FREE parameter — `NullRed.lean` retired the
"odd is load-bearing" claim that used to stand here.**  The parity in these
theorems lives entirely in `ForcedMate` (`mate` one ply, `step` two — game
structure, not search structure).  The null term never generates a rung: the
suppression keeps the pass term strictly sub-band at every reduction
(`nullTermDR_lt_ML`), the max-fold makes sub-band pass terms inert, and under
`NoZugzwang` the declared value equals the real-move value outright
(`nullValueR_eq_realValue_of_noZugzwangR`).  So `dtm_optimalR` holds for EVERY
reduction `R` — the whole `NULL_RED ∈ [3..10]` tuning range, odd and even
alike — with the shipped `3` recovered as an instance (`nullValueDR_three`,
`noZugzwangR_three`) and the even case witnessed (`dtm_optimal_R4`).  What
changes with `R` is exactly one thing: `NoZugzwangR R` evaluates the pass at
depth `d - (R - 1)`, so each reduction buys the same-shaped zugzwang premise
at its own pass horizon.  No engine change is needed at any `R`: the
`min(pos.score + EVAL_ROUGHNESS, ·)` cap already bars the positive mate band
(`EvalBounds`), and negative-band pass values cannot raise a max-fold.

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

`NullRed.lean` extends the chain to an ARBITRARY reduction — the rows the
retired paragraph said could not exist:

| fact | theorem | axioms |
|---|---|---|
| the shipped model is the `R = 3` instance | `nullValueDR_three` | `propext, Quot.sound` |
| the shipped premise is the `R = 3` premise | `noZugzwangR_three` | `propext, Quot.sound` |
| accuracy at every reduction | `nullValueR_eq_realValue_of_noZugzwangR` | `propext, Quot.sound` |
| the suppression at every reduction | `nullTermDR_lt_ML` | `propext, Quot.sound` |
| `NoZugzwang` is vacuous at guard-off | `noZugzwang_guardOff` | `propext, Quot.sound` |
| completeness at every reduction | `forcedMate_completeR` | `propext, Quot.sound` |
| no false mates at every reduction | `forcedMate_of_value_distR` | `+ Classical.choice` |
| block exactness at every reduction | `leastMate_value_blockR` | `+ Classical.choice` |
| separation at every reduction | `leastMate_value_separationR` | `+ Classical.choice` |
| shortest play at every reduction | `leastMate_play_shortestR` | `+ Classical.choice` |
| maximal resistance at every reduction | `defence_resistsR` | `+ Classical.choice` |
| both directions, every reduction | `dtm_optimalR` | `+ Classical.choice` |
| the even-reduction witness | `dtm_optimal_R4` | `+ Classical.choice` |

The parity spine is choice-free, like the distance spine it extends -- and so
is the new play predicate's monotonicity; the `Classical.choice` in the value
results is inherited from `legal_of_allIllegalB_false`, unchanged.  No new
chess premise: the defender half spends `ValFloor`, `EvalQuiet`,
`NoMaskedMobility` and `NoZugzwang` exactly as the attacker half does,
`defence_move_legal` does not need `NoZugzwang` at all, and
`NoMaskedMobility` is discharged from `ValFloor` since `c01915f`
(`noMaskedMobility_of_valFloor`), so only `NoZugzwang` is still bought.

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

### The frontier premise, and how `c01915f` dropped it — `CexD`

`NoMaskedMobility` was the model-side stand-in for the engine fix in PR #171
(search the QS-filtered evasions before declaring mate).  The natural hope was
that the distance machinery made it unnecessary: a phantom mate is invented at
the frontier, where almost no depth is left, so surely it can only claim the
DEEPEST rungs, and a real mate in `k` with `k + 1 ≤ d` claims a higher one.

That hope was false, and one number said why.  **A masked node does not report
a shallow mate — it reports the sentinel.**  Its filtered fold had no admitted
legal move left, so nothing displaced the initial accumulator
`LOSS = -MATE_UPPER`, and the parent negated that to `MATE_UPPER`, which is not
a rung at all: it is strictly above every value the ladder can produce
(`mateFloor_lt_MATE_UPPER`).  A phantom outranked EVERY real mate, at every
distance and every depth, and no rung argument could separate what is off the
ladder.

`CexD` was that hope refuted at the exact hypotheses of
`leastMated_defence_resists`.  `Q` is genuinely lost in four plies; the correct
defence `D` leads to a real mate in three, the fast loss `B` to mate in one, and
one ply past the frontier of `D`'s line sits `M` — a defender node whose only
legal reply dropped 150cp, below the pre-`c01915f` depth-1 threshold of -100,
while an illegal one survived it: the position class of the #171 report, one
ply deeper.  `D` was priced at `MATE_UPPER = 69290` instead of its true rung
47938, the mate in one was priced honestly at 47968, and the engine preferred
to be mated in two.  Every premise but `NoMaskedMobility` held — including the
move rule in its IDEALISED exact-argmax form.

**`c01915f` (#218) closed it in the code**, and the model now says so.  The
positive-depth admission is `-MATE_UPPER`, so `M` admits its escape, and the
same game re-measured inverts every line of the old table:

| fact | theorem | axioms |
|---|---|---|
| the once-masked node reports the honest draw | `cexD_M1` | `propext, Quot.sound` |
| ... and the true mate is priced ON the ladder, below the fast loss | `cexD_D4`, `cexD_B4` | `propext, Quot.sound` |
| the exact argmin at the root is now the correct defence | `cexD_maximal` | `propext, Quot.sound` |
| the position really is lost in exactly four plies | `cexD_leastMated` | `propext, Quot.sound` |
| the premise itself is a theorem | `cexD_unmasked` (`noMaskedMobility_of_valFloor`) | `propext, Quot.sound` |
| the engine resists for all four | `cexD_resists` | `+ Classical.choice` |
| **the premise no longer has to be dropped — it is free** | `cexD_defence_no_longer_needs_frontier` | `+ Classical.choice` |
| what the change bought, at the constants | `cexD_masking_was_arithmetic` | none |

`cexD_resists` is not a hand construction: with the premise discharged, every
hypothesis of `leastMated_defence_resists` holds for `CexD`, so the theorem the
game was built to refute applies to it and supplies the conclusion.

Two things the countermodel still settles about the OLD design.  **Acyclicity
did not help**: `CexD` is a finite tree with no repetition in it, so no
draw-by-repetition or well-founded-descent argument touched the failure.
**Depth did not help** when a selective frontier travels with the search — only
removing the frontier did, which is what `producerMoves_positive` states on the
code side and `movesAbove_pos` on the model side.

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
A change to the tempo would have to be re-argued in both places; the null
reduction needs no re-arguing at all (`NullRed.lean`: `dtm_optimalR` holds at
every reduction).

**The frontier premise survived the weakening, and was retired by the
code instead.**  Masking was genuinely local -- `val_lower_pre 2 =
-240` is below the tables' -192 move value floor, so from remaining
depth 2 up the filter was the identity -- which is why the eventual
reading looked promising: `CexF`'s phantom dissolved at depth 3.

What killed the weakening is that a masked node does not report a
shallow mate.  It reports the OFF-LADDER sentinel
(`maskedFrontier_value`): the filtered fold has no admitted legal move,
nothing displaces the initial `LOSS` accumulator, and the parent
negates it to `MATE_UPPER`.  The ladder decays one rung per unspent
ply; the sentinel decays not at all.  And the frontier RENEWED the
phantom at every horizon -- absorbing the old one bought nothing when a
new one arrived one ply deeper.

`CexE` is that argument machine-checked.  An infinite chain
`C 0 -> C 1 -> ...` in which every node was masked: each generates an
illegal move valued 0 (admitted at every depth) and the legal
continuation valued -150 (which `val_lower_pre 1 = -100` filtered,
admitting it only from depth 2 on).  The root is a draw in the
strongest sense -- no forced mate at any budget for either side -- and
its declared value USED TO BE

```text
D:        0     1     2     3     4     5    ...
value:    0   -MU    +MU   -MU   +MU   -MU   ...
```

Odd depths reported "I am mated", even depths "I mate"; the value never
re-entered the band, so there was no `D0` and the classification was
not merely unsettled but WRONG in both directions, alternately,
forever.  Every fidelity premise held, including `EvalBand` at every
width (every live static score in `CexE` is exactly 0), so it was not a
granularity failure; `NoZugzwang` is vacuous here.  What failed was
exactly `NoMaskedMobility`.  Acyclicity did not help (`cexE_acyclic`:
the chain index strictly increases, no position ever repeats) and a
read-time clamp did not help (the phantom arrived as the exact
sentinel, so clamping at `MATE_UPPER - 1` still left 69,289).

**`c01915f` (#218) retired the premise outright.**  The shipped
admission is `-MATE_UPPER` at every positive depth, so the depth-1
admitted set IS the move list, `NoMaskedMobility` follows from
`ValFloor` alone (`noMaskedMobility_of_valFloor`), and
`maskedFrontier_value`'s hypotheses became unsatisfiable
(`maskedFrontier_unreachable`).  The chain re-measured reports the
honest `0` at every depth and every chain position (`cexE_honest`) --
which is what the frontier-tail variant of Part B already computed for
it (`cexE_t_honest`), so the two now agree (`cexE_shipped_eq_t`) and
the trichotomy holds at EVERY depth with no frontier premise
(`eventual_classification_frontier_free`).

**What the weakening bought, and what is left of it.**  Both
COMPLETENESS arms never needed the frontier premise at all
(`eventual_completeness_without_frontier`, `ValFloor` + `NoZugzwang`
only).  `NoMaskedMobility` paid for exactly one thing -- the honesty
arm -- and that arm is now free as well.  The eventual weakening's
remaining content is `NearMaximalChoice` retired at band granularity;
the frontier question it was raised to answer is closed.

| fact | theorem | axioms |
|---|---|---|
| masking has no site at all above the frontier | `filter_identity_off_frontier`, `movesAbove_pos` | `propext, Quot.sound` |
| a masked node reports the sentinel, not a rung (true, unsatisfiable) | `maskedFrontier_value`, `maskedFrontier_unreachable` | `propext, Quot.sound` |
| completeness needs no frontier premise | `eventual_completeness_without_frontier` | `propext, Quot.sound` |
| the declared value avoids the band corridor | `nullValueD2_offCorridor` | `+ Classical.choice` |
| the shipped corridor is 2000 tolerances wide | `shipped_band_gap_wide` | none |
| the tolerance cannot move a value across a band edge | `bandOf_eq_of_slack` | `propext, Quot.sound` |
| a near-maximal choice is band-exact | `nearMaximal_band_exact` | `+ Classical.choice` |
| ... and never misses a mate | `nearMaximal_keeps_mate` | `+ Classical.choice` |
| the converged bracket cannot straddle a band edge | `driver_stop_band_stable` | `propext, Quot.sound` |
| the drawn root is valued 0 at every depth | `cexE_honest` | `propext, Quot.sound` |
| ... with no forced mate for either side | `cexE_no_forcedMate`, `cexE_no_forcedlyMated` | `propext, Quot.sound` |
| ... and no repeated position anywhere | `cexE_acyclic` | `propext` |
| ... and the once-failing premise now holds | `cexE_unmasked` | `propext, Quot.sound` |
| the shipped fold and the frontier tail agree here | `cexE_shipped_eq_t` | `propext, Quot.sound` |
| what the change bought, at the constants | `cexE_masking_was_arithmetic` (none), `admission_widened_at_frontier` | `propext, Quot.sound` |
| **the trichotomy needs NO frontier premise** | `eventual_classification_frontier_free` | `+ Classical.choice` |
| the honesty arm, isolated | `eventual_honesty_frontier_free` | `+ Classical.choice` |
| everything at once | `eventual_classification_verdict` | `+ Classical.choice` |

The re-measurement and the verdict are choice-free; the
`Classical.choice` in the corridor and premise-free results is the same
by-case pattern `eventual_classification` itself carries.  One new
fidelity premise, `EvalBand B` -- the two-sided form of the table bound
`EvalQuiet` reads one-sidedly, discharged for the shipped tables at
`B = EvalBounds.evalBound = 15437`.

## Terminal positions and legality evidence

The move fold maintains two independent facts:

```python
best, live = -MATE_UPPER, False
```

- `best` accumulates numeric reports from real and virtual candidates.
- `live` records that a searched real move was legal. The settled break sits
  BEFORE the shared `live` update, so a move answered by its cap witnesses
  no legality; and it skips the cutoff block, so it stores nothing.

Null moves, stand pat, and non-mating futility estimates are numeric evidence
only. A searched move whose child report is above the illegal-move sentinel is
also legality evidence.

Mate and stalemate are a final classification, not another input to `max`.
When no legal move has been witnessed at positive depth, the post-fold scan
checks every generated move. If none is legal, it replaces the accumulated
numeric report with exact checkmate (`max(1 - MATE_UPPER, -MATE_LOWER - depth *
EVAL_ROUGHNESS)`, the distance-carrying value) or stalemate (`0`). This
placement is necessary because exact terminal classification may lower a
virtual cutoff.

At a terminal root, a fail-high has no move. Both UCI loops therefore stop
iterative deepening and emit `bestmove (none)` without dereferencing `move`.

## The exact king-capture clauses cannot be weakened (`BandContract.lean`)

The docstring makes two EXACT promises about king capture: a kingless node
returns `-MATE_UPPER`, and at positive depth a king-capturable node returns
`MATE_UPPER`. Asked whether band membership (`r <= -MATE_LOWER`,
`MATE_LOWER <= r`) would do instead -- which would let the producer yield a
king capture's raw `pos.value` and drop the normalization -- the answer is no,
and the reason is short. Boundedness already gives `r <= MATE_UPPER`, so the
clause's content is exactly `MATE_UPPER <= r`, and the proposed band is the
half that was free (`yield_at_identity_iff_exact`).

Four consumers need the half the band drops:

| consumer | what breaks | result |
|---|---|---|
| the fail-soft bracket | the declared value at a capturable node IS `MATE_UPPER`, so a band report may fail low where no spec-valid report can | `capture_report_must_fail_high`, `band_report_can_fail_low` |
| the `live` bit | an illegal move's yield leaves the fold identity, so `not live` never holds at a mated or stalemated node | `bandLeaf_correction_misses`, `correction_fires_iff_exact` |
| `tp_move` legality | an illegal move can fail high and be stored as the score witness | `storedMoveLegal_band_refuted` |
| the fold identity | an exact illegal yield is neutral in the `max`; a band yield clamps a deep mate score | `sentinel_is_fold_identity`, `bandYield_clamps_mate_score` |

`CexB` is the smallest game separating the two contracts: one stalemated
position, one pseudo-move, every fidelity premise discharged, and the only
difference is what the king-capturable child reports. With the exact clause the
override assigns `0`; with a band report it assigns `-60000`.

Mate distance also lives in the band, and it is safe there, because every
consumer of the distance zone compares magnitudes while the two band edges are
compared for equality: `score > -MATE_UPPER` and `bound(child, MATE_UPPER, 0)
== MATE_UPPER` read them as tokens. Yielded scores are exactly `MATE_UPPER` or
below `MATE_LOWER`, never in the gap; searched scores stay strictly below the
token because the finalizer floors the child at `1 - MATE_UPPER`
(`searched_score_below_MU`), so reaching the token has king-capture provenance
(`MU_provenance`); and the seed entry `Entry(-MATE_UPPER, MATE_UPPER)` is
unreturnable at every window the driver uses
(`tt_sentinel_defaults_never_returned`).

The same file pins the DIRECTION of the mate value. `-MATE_LOWER - depth *
EVAL_ROUGHNESS` is antitone in unspent depth, so the mating side prefers the
faster mate; the alternative `-MATE_UPPER + depth * EVAL_ROUGHNESS` is monotone
and inverts both halves of `dtm_optimal` (`matedAlt_inverts_preference`), while
spending the sentinel margin down from the whole distance zone to one
`EVAL_ROUGHNESS` (`matedAlt_margin_is_one_step`).

### The reservation, at the recursion level

The two exact clauses are worth nothing unless `-MATE_UPPER` really is a
reserved token: `score > -MATE_UPPER` is the legality test one ply up, so a
LEGAL line must never produce it.  Two of the three ways it could are closed
locally — the depth-0 leaf (`qsLeaf_reserves_sentinel`: the stand-pat floor is
already above `-MATE_LOWER`) and the finalizer (`terminalValue_reserves_
sentinel`: the `1 - MATE_UPPER` floor is exactly one point of reservation).
The third is arrival from below, which is a statement about the recursion and
needs a dual induction:

* **reserved below** — a node whose own king is on the board returns strictly
  above `-MATE_UPPER` (`boundD2''_reserves_sentinel`);
* **reserved above** — a node that cannot capture the enemy king returns
  strictly below `MATE_UPPER` (`boundD2''_reserves_positive`).

Neither arm is available alone.  The lower one needs a searched legal child to
report below `MATE_UPPER`, so its negation lifts the accumulator off the
sentinel; the upper one needs every searched child to report above
`-MATE_UPPER`, so no negation reaches the positive token.  They are proven
together (`boundD2''_reserves_pair`, induction on depth) and combine into the
biconditional the code actually uses, `boundD2''_live_iff_legal`:

```text
-MATE_UPPER < bound(pos, gamma, depth)   ↔   pos still has its king
```

`boundKCX''_reserves_sentinel` carries both arms to the production consumer
through `production''_eq_reference''`.

**This is the theorem the stale `val_lower` blocked.**  The lower arm's live
case has to produce a legal move that reaches the real accumulator, and under
the pre-`c01915f` sloped admission a legal move could be missing from
`searchedAt` outright — filtered at remaining depth 1 with nothing else to
displace the accumulator, which is precisely `CexE`'s and `CexF`'s phantom.
With the shipped admission modeled, `movesAbove_pos` admits every legal move
at every positive depth, and the only remaining way to keep one out of the
REAL accumulator is futility — which pays for itself, because a futile move's
own stand-pat enters `futTerm` and a live child's stand-pat is below
`MATE_LOWER` (`EvalQuiet`).

Premises are fidelity only: a move-value floor inside the band (`ValFloor`,
tables -192) and `EvalQuiet`, plus the docstring's own window condition.  No
chess-side premise.

| fact | theorem | axioms |
|---|---|---|
| the QS leaf never returns the sentinel | `qsLeaf_reserves_sentinel` | `propext, Quot.sound` |
| the finalizer never returns it | `terminalValue_reserves_sentinel` | `propext, Quot.sound` |
| the recursion never returns it on a legal line | `boundD2''_reserves_sentinel` | `+ Classical.choice` |
| ... and never returns `MATE_UPPER` without a king capture | `boundD2''_reserves_positive` | `+ Classical.choice` |
| the legality test, both directions | `boundD2''_live_iff_legal` | `+ Classical.choice` |
| the production consumer inherits both | `boundKCX''_reserves_sentinel` | `+ Classical.choice` |

### The `bound()` docstring, clause by clause

A docstring is a model claim, so it is audited like one: `model_audit.py`
anchors the clauses below, and each has a theorem behind it. Statuses are
MATCHES (proved as written), AHEAD-OF-MODEL (proved for the modelled search,
with a named gap to the shipped one), and UNMODELED.

| docstring clause | status | where |
|---|---|---|
| `if gamma > s*` / `if gamma <= s*` bracket | MATCHES | `Bound.bound_spec`; the docstring's `s*`-split form and the proved `r`-split form are interderivable (`boundSpec_iff_docstring`), as is `WindowReport` (`windowReport_iff_boundSpec`) |
| `s*` is a function of `(pos, depth)` and fixed parameters alone | AHEAD-OF-MODEL | proved against the declared value `nullValueD2`, which is window-free by construction (`boundD2''_spec`). The FUEL-shaped value that models the shipped reduction is bracketed only by `FuelBracketSpec`, which is STATED and not proven (`EventuallyWide.lean`). The docstring is honest about what `s*` is; the open work is the mirror proof, not the wording |
| `1 - MATE_UPPER < gamma <= MATE_UPPER` | MATCHES | hypothesis of every spec; the range is closed under the null-window flip with one point to spare (`window_flip_preserves_range`) |
| the table may return a weaker but valid bound | MATCHES | the specs bracket, never equate; `TableSwap.lean`, `d2_no_crossing` |
| kingless: `r = -MATE_UPPER` | MATCHES | `boundD2''_kingGone` |
| depth >= 1, capturable: `r = MATE_UPPER` | MATCHES | `boundD2''_of_capture`; the depth-0 half is fail-high only, exactly as the `depth >= 1` gate says (`kingCaptureContract_stratified`) |
| no searched move can reach `MATE_UPPER`, so an exact `MATE_UPPER` proves a king capture | MATCHES (new) | `searched_score_below_MU`, `MU_provenance` |
| only a searched real move sets `live` | MATCHES | the three-species split (`searchedAt`, `futTerm`), `searched_yield_two_way`, and `termFix2`'s `S = LOSS` |
| mate/stalemate returns the exact `max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)` / `0` | MATCHES | `boundD2''_terminal_exact`, `terminalValue_exact` |
| the mate value carries the UNSPENT depth, so the winner takes the shortest line | MATCHES | `terminalValue_anti`, `leastMate_value_separation`, `dtm_optimal`; the formula-level direction and the cost of inverting it are `matedShipped_anti`, `matedAlt_inverts_preference` |
| every move in `tp_move` is legal | MATCHES | `storedMoveLegal`, `storedMoveLegal_qs`, `KillerLegal` |
| a nonterminal root fail-high leaves a real witness | MATCHES | `boundD2_failHigh_attained`, `storedMove_attains`, `substitution_attains` |

Two things the docstring deliberately does NOT claim, because the model does
not: that `s*` is the game-theoretic value (it is the value this search
declares, pruning included), and that a depth-0 capturable node reports the
sentinel (it only fails high).

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
| exact king-capture producer report | `producedScore_exact_capture` (and why the band restatement does not do, `BandContract.lean`) |
| shallow move cap and lazy child evaluation | `shallowMoveCap_lowDepth`, `cappedMove_report` |
| per-move lazy cap report | `cappedMove_failLow`, `shippedCap_iff_tail`, `lazyMoveTail_cap_lt_gamma`, `lazyMoveTail_report`, `lazyMove_partition` |
| unclamped cap stays below the band | `CapInBand` (both-kings invariant + tuner caveat), `shallowMoveCap_below_positiveMate`, `capClamp_eq_shipped` |
| monotone cap: the first settled report is the whole tail's | `shallowMoveCap_max`, `foldMax_shallowMoveCap`, `lazyMoveTail_maxCap` |
| prefix-first order and the empty tail | `lazyMove_partition_prefixFirst`, `lazyMove_partition_emptyTail` |
| cap mate-band properties | `shallowMoveCap_below_positiveMate`, `cappedMove_preserves_negativeMate` |
| filtered move fold and early cutoff | `Bound.searchMoves_spec` and the fold models in `Stalemate.lean` |
| sticky legality evidence and terminal override | terminal/finalizer results in `Stalemate.lean` |
| king-capture evaluation margins and ordering | `EvalBounds.lean` |
| `mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)` | `terminalValue`, `terminalValue_exact`, `terminalValue_reserves_sentinel` |
| `score > -MATE_UPPER` as the legality test | `boundD2''_live_iff_legal`, `boundKCX''_reserves_sentinel` |
| legal killer lifecycle and eviction | `Killer.lean` |
| root versus interior null behavior | `CanNull.lean` |
| transposition-table interval updates | `TableSwap.lean` and table results in `Stalemate.lean` |
| MTD-bi bracket range and convergence | `Driver.lean` |
| soft-clock termination only between completed MTD brackets | `softStop_requires_closed` |

The model abstracts Python's board representation, move generation, sorting,
and table implementation. The audit pins the corresponding source regions;
tests and chess corpora validate those executable primitives.

## Module guide

- `GameTree.lean`: chess-free negamax game model.
- `Bound.lean`: core fail-soft search proof.
- `CappedNull.lean`: capped-null report transport and score-band facts.
- `CappedMove.lean`: positive-depth production, shallow caps, and their exact lazy partition.
- `Stalemate.lean`: selective-search fold, legality, and terminal finalizer.
- `EvalBounds.lean`: numeric bounds induced by the piece-square tables.
- `BandContract.lean`: what the two exact king-capture clauses buy, the
  countermodel refuting their band weakening, and the band-cohabitation
  facts that keep mate distance and the king-capture tokens apart.
- `Killer.lean`: move-table legality and lifecycle.
- `CanNull.lean`: root/interior null and table-key conditions.
- `Driver.lean`: MTD-bi bracket invariants and convergence.
- `TableSwap.lean`: table-update properties.
- `Liveness.lean` and `Classification.lean`: mate visibility and search-result
  classification under their named fidelity premises.
- `Eventual.lean`: what the "from some depth on" weakening of the
  trichotomy buys -- `NearMaximalChoice` retired at band granularity,
  completeness shown independent of the frontier premise, `CexE` (the
  countermodel that showed the frontier premise survived the weakening,
  re-measured after `c01915f` and now honest at every depth), and the
  premise-free trichotomy `eventual_classification_frontier_free`.
- `Shortest.lean`: mate-distance parity, value separation, distance-to-mate
  optimal play in both directions under the driver's own stopping tolerance, and
  `CexD`, the countermodel that showed the frontier premise could not be
  dropped from it -- now the game on which `leastMated_defence_resists`
  itself supplies the resistance.
- `Pruning.lean` and `Tricks.lean`: proof envelopes for selective-search
  techniques.
- `EventuallyWide.lean`: the fuel oracle -- bounded real-edge cost and the
  W/D/L trichotomy with no chess premise.
- `IntrinsicLMR.lean`: static LMR eligibility and the bounded, move-dependent
  edge cost used by intrinsic LMR.
- `MateDepth.lean`: the sharp mate-depth accounting -- which mechanism costs
  which ply, the shipped shallow cap folded in, and the countermodel that
  pins the constants and the slope.
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
