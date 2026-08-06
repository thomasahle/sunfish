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

## The consistency decision (enforced on master)

As of commit `7f9f164` the maintainer's design principle is **enforced on
master, not grandfathered**: *"`gamma` may shape termination, and may
trigger shortcuts whose value provably one-side-bounds the same function;
`gamma` must never select between incomparable evaluations of a move."*
Every pruning decision in the shipped search is gamma-independent —
LMR is deterministic (`LMR = int(depth >= 4 and i_m >= 8 and val < 0)`,
searched once at `depth - 1 - LMR`) — so the engine has **point specs
end-to-end** on single value functions, transposition entries are
contradiction-free by construction (`boundLmrDet_no_crossing`), and the
store clamp became a provable no-op and was removed
(`clamp_noop_high`/`clamp_noop_low`). The price: the gamma-adaptive
re-search LMR measured ~16 ELO stronger (−16 ± 38 direct; both are large
wins over no LMR). That trade was made deliberately, and
`Sunfish/Lmr.lean` + `Sunfish/TableClamp.lean` remain as the formal
record of what the 16 ELO would cost in spec strength.

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

- **`Sunfish/LmrDet.lean`** — **deterministic LMR (commit `7f9f164`,
  current master)**: the reduction depends only on (depth, index, value),
  never on gamma; each move searched once at `depth - 1 - LMR`. All
  proven sorry-free:
  - **`boundLmrDet_spec`**: the point `BoundSpec` against the single
    value function `negamaxDet` (reducible moves valued one ply
    shallower, recursively) — unconditional; `bound_spec` with per-move
    depths, no mutual recursion.
  - **`boundLmrDet_no_crossing`**: fail-high reports never exceed
    fail-low reports at the same `(pos, depth)` — contradiction-free
    entries by construction, `bound_no_crossing` generalized.
  - **`clamp_noop_high` / `clamp_noop_low`**: under single-function
    bounds the `2c95ab0` clamp is a no-op — the formal justification for
    its removal. There is nothing to "reinstate" for a future
    gamma-dependent choice: such a choice is a bug (see the doctrine
    note below), not a configuration.

- **`Sunfish/Lmr.lean`** *(HISTORICAL: commits `58883ea..7f9f164`)* —
  re-search Late Move Reductions (commit `58883ea`,
  sunfish.py lines 370, 386–397: late quiet moves probed at `depth - 2`,
  a reduced fail low yielded as-is, a reduced fail high re-searched at
  full depth). Load-bearing content:
  - **The `Vlo`/`Vhi` "interval spec" was deleted.** It showed fail
    highs bound one function and fail lows another with the truth in
    between — but the gap between those functions admits no provable
    bound (one ply can hide a mate), and a guarantee relative to an
    uncontrollable gap guarantees nothing. Maintainer's verdict: never
    a spec, only a description of the bug. The file now keeps only what
    supports the doctrine — the definitions and the counterexample:
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
    *reports* still occur (this theorem); the `2c95ab0` clamp merely
    kept them from being *stored* (`clamp_no_crossing`) while the
    search went on consuming them — containment, not repair.
  - `RedRespectsCaptures` + comments confirm Killer and Stalemate are
    unaffected: king captures have `val ≥ MATE_LOWER > QS = 40` (never
    reducible), the killer is pre-loop (structurally never reduced),
    reduced fail lows are `< gamma` (never reach the `tp_move` store),
    reduced fail highs store only the full-depth result, and the
    `-MATE_UPPER` king-loss sentinel is depth-independent
    (`boundKill_kingGone`/`negamaxDraw_kingGone`), so the stalemate
    detection still sees exact sentinels from reduced searches.

- **`Sunfish/TableClamp.lean`** *(HISTORICAL: commits
  `2c95ab0..7f9f164`, removed with the re-search LMR)* — the clamped
  store (commit `2c95ab0`,
  sunfish.py lines 435–443: fail-high stores
  `Entry(best, max(entry.upper, best))`, fail-low
  `Entry(min(entry.lower, best), best)`). What remains:
  - **`clampHigh`/`clampLow`/`clamp_no_crossing`** — the record of what
    the clamp did (stored entries cannot cross), retained because
    `LmrDet`'s `clamp_noop_*` theorems prove it a no-op under a point
    spec. The `IntervalTableOK` invariant that once dressed the clamp
    up as a guarantee was deleted with the interval spec (an invariant
    relative to an unboundable gap certifies nothing).
  - **`clamp_no_crossing`** — clamped entries satisfy `lower ≤ upper`
    by construction: the clamp kept the table from *storing* the
    contradiction while the search went on consuming it. (The
    `intervalTableOK_*` preservation theorems that once accompanied
    this are deleted — see the doctrine note.)

- **`Sunfish/CanNull.lean`** — the `can_null` layering, modeled exactly
  as master uses it in all four roles: null-move gate (line 340, pass
  searched at `depth - 3` with `can_null=True` — sunfish permits
  consecutive null moves, reproduced exactly), repetition gate (line
  325, `history` a fixed per-search parameter), transposition key (line
  318, `CTable` keyed on `(depth, can_null, pos)`), and IID (line 355,
  `can_null=False` — the code, not the comment at 352–353, which is
  being fixed in PR #135). Proven sorry-free:
  - **`boundNullTT_spec` (Layer 1, unconditional)**: the
    null-and-repetition-augmented search brackets its own value
    function `nullValue` with a *point* spec, and the keyed table stays
    consistent (`CTableOK`). No zugzwang hypothesis anywhere:
    self-consistency of search + table is unconditional.
  - **`ctableOK_empty`**: the empty table satisfies the invariant for
    *any* history — the fact that justifies sunfish clearing `tp_score`
    whenever `history` changes (the invariant is history-relative).
  - **Layer 2 (`nullValue_plain`), proven under `NullBetOK`**: with the
    bet hypothesis (some real move matches the reduced-depth pass;
    witness required, so guard-passing stalemates are excluded too) and
    an empty history, `nullValue` collapses to the null-free
    `plainValue` — composing with layer 1 recovers the docstring.
    Zugzwang threatens only this bridge, never self-consistency. This
    completes (and replaces) the former `boundNull_spec` sorry.
  - Audit surprises recorded in the module comment: sunfish's
    `can_null` does **not** prevent consecutive null moves (the pass is
    searched with the default `True`); and the move generator's
    *laziness* is semantically load-bearing — a null cutoff means the
    IID recursion never runs, so the table state depends on the cutoff,
    and an eager model would mis-model `tp_score`.

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
    search **unconditionally** (single value function),
    with only `FutilityMateOK` (line 371's king-capture bypass) and the
    in-band window remaining. Fine print made explicit: the ∀-depth
    `FutilityOK` is *not* dischargeable (plain negamax has no stand-pat
    at `d ≥ 1`) — but the search only ever consumes the `d = 0`
    instance; the old statement over-required. **Contrast with LMR**:
    futility's shortcut is a provable one-sided bound of the *same*
    value function (hence consistent, point spec); LMR's reduced value
    is incomparable to the full value (hence the TT crossing, and no
    honest weaker claim to retreat to).

## Zero sorries: named hypotheses instead

**`grep -rn sorry formal/Sunfish/*.lean` finds nothing outside prose.**
Every theorem about the shipped search is proven; where a claim is only
conditionally true, the condition is a *named hypothesis in the
statement* — the honest form — not a deferred proof:

- **`NullBetOK` / `nullValue_plain`** (`Sunfish/CanNull.lean`, proven) —
  the null-move bet, exactly as the code places it: some real move at the
  children's depth matches the pass at its reduced `depth - 3`; the
  witness requirement also excludes guard-passing stalemates. Under it
  (and an empty history) the layer-1 value `nullValue` collapses to the
  null-free `plainValue`, composing with the unconditional
  `boundNullTT_spec` into the original docstring. Zugzwang is precisely
  `¬ NullBetOK` and threatens only this bridge, never layer-1
  self-consistency. (`NullOK` in Tricks.lean remains as the same-depth
  core of the bet.)

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
  `mateDepthStable_of_kingGoneStable` (proven) derives from
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
  closes the hole. `killer_probe_sound` (proven, both directions) shows
  the stalemate probe is a complete decision procedure for
  king-capturability *at the probe's depth* — the depth the engine
  actually runs it at. The statement is pinned there deliberately: at
  deeper depths the no-false-positives direction is genuinely unprovable
  without a sentinel-origins characterization (a "mated" killer fakes
  `-MATE_UPPER` one level down — the `boundStale_not_unconditional`
  artifact family), and the docstring documents why.

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
| `boundLmrDet`, `negamaxDet` | the deterministic LMR block on master: `LMR = int(depth >= 4 and i_m >= 8 and val < 0)`, one search at `depth - 1 - LMR` (`7f9f164`) |
| `tablePart2` (plain stores) | Table part 2 on master: `Entry(best, entry.upper)` / `Entry(entry.lower, best)` — no clamp needed under point specs |
| `boundLmr`, `red` *(historical)* | the re-search LMR block of `58883ea..7f9f164`: `depth >= 3 and i_m >= 5 and val < QS`, reduced probe at `depth - 2`, re-search on fail high |
| `-(bound d m (1 - gamma))` | `-self.bound(pos.move(move), 1 - gamma, depth - 1)` (line 376) |
| `BoundSpec` | the docstring (lines 287–290) |
| `NullGame.pass` | `pos.rotate(nullmove=True)` (line 331) |
| `Table`, `TableOK`, `tablePart2` | `tp_score`, `Entry`, lookup + point store (lines 275–276, 305–310, and the pre-`2c95ab0` Table part 2) |
| `clampHigh`, `clampLow` *(historical)* | the clamped Table part 2 of `2c95ab0..7f9f164` (kept only for the `clamp_noop_*` no-op proofs) |
| `negamaxDraw`, `boundStale`, `staleFix` | king-capture normalization + stalemate correction (lines 298–303, 388–412) |
| `inCheckB`, `CheckProbeOK` | the null-position check probe `bound(flipped, MATE_UPPER, 0) == MATE_UPPER` (lines 409–411) |
| `MateValuesAreKingCaptures` | the requirement of lines 398–401 and the caveat of lines 403–405 |
| `FutGame.val`, `boundFut` | `pos.value(move)` and the futility yield (lines 360–374) |
| `KTable`, `kstore`, `boundKill` | `tp_move`, killer try + store-on-cutoff (lines 339, 356–357, 382–387) |
| `orderedMoves` | the sort of line 360 (king captures, value ≥ `MATE_LOWER`, first) |

| `nullValue`, `boundNullTT`, `CTable` | the can_null-aware search: null move (340–341), repetition (325), keyed table (318), IID (355) |
| `nullGuard` | `abs(pos.score) < 500` (line 340), gamma-free |

## Model fidelity

Audited against master at commit `9b1a7b4` (2026-08-05), sunfish.py lines
286–448. The model tracks the code exactly except these explicitly listed
abstractions, each with its justification:

- **QS-as-eval at the `Bound.lean` layer**: depth 0 returns `eval`
  directly; QS's interior (stand-pat + capture recursion at clamped
  depth 0) is a fixpoint the abstract model treats as its evaluation.
  Its one load-bearing property — the stand-pat identity — is what
  `ValGame.score_identity` captures and `boundFut_spec` consumes.
- **Deadline/`Stop`** (lines 297–301): raises at node *entry*, before
  any store, so an abort can leave a search unfinished but never a
  table entry unjustified — aborts cannot corrupt `TableOK`/`CTableOK`.
- **Eviction** (`TABLE_SIZE`, lines 445–446 and the `tp_move` twin):
  only forgets entries, which trivially preserves every table invariant
  here.
- **`depth = max(depth, 0)`** (line 306): corresponds to the model's
  `Nat` depths with saturating subtraction — verified aligned.
- **Killer val-gate** (line 366) not modeled in `Killer.lean`; cannot
  affect `boundKill_spec` (king captures have `val ≥ MATE_LOWER`, far
  above every `val_lower`) — see the audit note there. The killer's
  in-loop duplicate follows normal LMR rules (see `Lmr.lean`'s refined
  claim and `foldMax_dup`); the stalemate probe's table key is
  `(flipped, 0, True)` (both `can_null` gates are dead at depth 0).
- **Move ordering** (line 360) is modeled only through its load-bearing
  consequence: king captures sort first (`orderedMoves`), and
  `FutilityLmrDisjoint` (futility at depth ≤ 1 vs LMR at depth ≥ 3 —
  never co-occur, which is why their models compose from separate
  files).

## Guideline for search-changing PRs

**A search-changing PR should identify which lemma it preserves or
weakens.** Concretely: does the change keep `bound_spec` (pure bound
logic)? Does it strengthen or newly rely on `NullOK` (zugzwang exposure)?
Does it preserve `TableOK` (is every store still a valid bracket, and is the
key still complete — cf. `ExtKeyIndependent`)? If the answer is "it weakens
X in positions Y", that is exactly the sentence the PR description needs.

The maintainer's design principle, distilled from the futility-vs-LMR
contrast and **now enforced on master** (commit `7f9f164`): **"`gamma`
may shape termination, and may trigger shortcuts whose value provably
one-side-bounds the same function; `gamma` must never select between
incomparable evaluations of a move."** Futility passes (its shortcut
equals the depth-0 search it replaces — `futilityOK_discharged`);
deterministic LMR passes (the reduction is gamma-free, `boundLmrDet_spec`
is a point spec); re-search LMR failed it, which is exactly why its TT
entries could contradict each other — and why it was retired. The
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
- **Table part 2 (the store at lines 435–443)** — a PR touching the
  store must preserve the point-spec `TableOK`: every stored bound a
  sound claim about the single key-determined value function. The
  historical clamp and its `IntervalTableOK` invariant are gone
  (deleted, not merely retired — see the doctrine note above); a change
  that cannot state a point spec is rejected, not clamped.
- **LMR** — any reduction scheme must assign per-move depths as a
  function of position-derived data alone and claim only bounds on the
  resulting single value function (deterministic LMR:
  `boundLmrDet_spec`; min-semantics LMR: the conjunction
  `min(reduced, full)`, key-determined by construction). Re-search LMR
  with full-value propagation is the canonical counterexample
  (`lmr_tt_crossing`) and is not mergeable

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
