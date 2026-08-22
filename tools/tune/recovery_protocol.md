# Five-tuner recovery protocol

Status: refrozen on 2026-08-23. This protocol compares GP, Chess Tuning Tools
(CTT), RBFOpt, SPSA, and CLOP by held-out Elo recovered per training game.
The exact machine-readable definition is
[`recovery_benchmark.json`](recovery_benchmark.json). Earlier pilot runs use
the superseded space and are not admissible in this comparison.

## Artifact pins

- Engine SHA-256:
  `c5039b4a0f88a2e11cf18ad13cd3f7d2c206b0a6901249d67d0b43ec02c95478`
  Built from commit `c01915f2349849598e617d24149b74d2fc65ef2a` on x86-64 with
  `/usr/bin/gcc`, SHA-256
  `f679a0ba1bddf27acd9523a1df45909b8e681f1f84f2d0f1cc87f5e115a6ec26`,
  version `gcc (GCC) 11.5.0 20240719 (Red Hat 11.5.0-14)`, via
  `make -C tools/ctwin CC=/usr/bin/gcc CFLAGS='-O3 -march=native -Wall -Wextra'`.
- Tables SHA-256:
  `d09fc445930a0e4babbab1aa176847a77923d3bf00c0b6fb0c46b0a90d14bbf0`
- fastchess SHA-256:
  `3d0a8f3c8b837a96366ac838f3ddf6fe87b813abb4ad2852284c0ce409132566`
  (alpha 1.8.2, build `f618e34`)
- Training-book SHA-256:
  `2c17253fc1282d7f41410d64e0621d801bb992c3bfba6ecd2372b06e7c5eae5f`
- Held-out-book SHA-256:
  `3f499996ff0b674a04f85f2634811d102dd53b5115841e8f11d18e1f550ba2ca`

The manifest also pins every local runner by SHA-256, the project lockfile,
CTT commit `a88f94725ac074ba4df635e263ffa98a502cee08`, RBFOpt commit
`458402c6f0c1d57f93ef2caaba461a51054e00ea`, and the official CLOP 0.0.9
archive. Both container recipes pin their Python base image. Record the built
image ID in each run's metadata; all three starts for that method must use the
same image ID.

The training book has 1,991 unique positions. The held-out UHO book has
241,670 positions, all unique after canonicalizing to the first four FEN
fields, and has no position in common with the training book.

The recorded campaign used held-out lines 1--1,000 sequentially. An audit
also reproduced all 197 positions selected by three recorded random-seed
runs with fastchess's exact `mt19937_64` Fisher--Yates shuffle and mapped the
recorded telemetry FEN. None occurs in either frozen validation slice.

## Corrected parameter space and starts

The refreeze removes `FUT_MAX`. Values 0--3 were behaviorally dead because
the active ordinary-move cap through `FUT_CAP_DEPTH` already covered the same
nodes. It adds `NULL_CAP_MARGIN` as an independent axis instead of silently
coupling it to `EVAL_ROUGHNESS`.

The three measured starts remain behavior-identical. Start 5 uses an explicit
null cap of 21, and starts 15 and 23 use 26, equal to each start's old
`EVAL_ROUGHNESS`. Their old `FUT_MAX` settings are dropped. The reference cap
is explicitly 15. The manifest pins each option vector and centered-space
hash.

This was checked with one fresh frozen-engine process per option map. The
process receives every start option, one `ucinewgame`, and then training-book
lines 1--100 in order at `go depth 8`, without resets between positions. The
SHA-256 of all `info` and `done` lines was identical before and after:

- start 5: `4f2ece41d53624bd2a1acc0f0f1e9ac9d1a6a25abcd4c294ac1c443af3996bda`
- start 15: `e90b22c4f19a07041b3448fe3d71cb7a0a14999df02e9cca9eac8f30c266127a`
- start 23: `676d0d21de510653ef239e343fa73519ed9430df363da371023415b5873c7176`

Reproduce both the static manifest audit and those searches with:

```sh
python3 tools/tune/verify_recovery.py \
  --engine ENGINE --tables TABLES --training-book TRAINING_BOOK
```

The command first rejects artifacts whose hashes do not match the manifest.
Add `--output-spaces DIRECTORY` to materialize the three exact centered JSON
spaces used by every method.

## Fair-training prerequisites

Do not use held-out results until every recommendation is frozen. All methods
must use the pinned engine, tables, parameter space, degraded starts, training
book, opening schedule, game budget, and feasibility gate. Apply a gate to
every method or to none. Method-specific comparison semantics, such as SPSA's
symmetric perturbations, remain part of the frozen method definition.

The x-axis is completed games, not optimizer iterations or requested points.
At each checkpoint, freeze the incumbent after the last complete observation
that does not exceed the budget and retain its actual `trained_games` value.
The starts are `5`, `15`, and `23`; checkpoints are `0`, `100`, `200`, `400`,
`700`, and `1000` games.

Reject a combined recommendation set if its engine, parameter-space,
training-opening schedule, or gate fingerprints differ. In particular, fresh
validation cannot repair training-opening contamination or a method-only
gate in an earlier pilot.

Each method receives exactly 500 complete color-swapped observations, or
1,000 games, from the same request-index schedule. Observation `i` uses
training line `i + 1`; completion order never changes that assignment. The
fixed-opponent methods use the explicit reference vector in the manifest.
SPSA retains its defining symmetric plus-versus-minus comparison.

The method definitions are frozen, not merely their names:

- Logistic GP uses UCB, a 24-point initial design, the 0.5-to-0.2 exploration
  schedule with half-life 40, no duels, a 128-point sparse basis, one pair per
  observation, and ten in-flight pairs.
- CTT uses MES, a 24-point initial design, one pair per observation, and 500
  total observations including that design. Its point-selection loop is
  sequential; the two games in its current pair may run concurrently.
- RBFOpt uses its pinned MSRSM/genetic policy with noisy and accurate sample
  sizes both set to one pair. Point selection is sequential.
- SPSA uses one paired perturbation per step and 500 steps. The stochastic
  approximation is intrinsically sequential.
- CLOP uses ten processors, `Replications 2`, `DrawElo 65`, `H 3`, and all
  correlations. Consecutive seeds form one color-swapped observation.

Every numerical and Boolean setting, including otherwise implicit defaults,
is in the manifest. Ten in-flight games or pairs are allowed where the method
can select them without changing its algorithm. Report wall time, CPU time,
and maximum observed engine processes separately from game efficiency. CTT's
`concurrency = 2` runs only the two colors of its one current pair; unlike GP
or CLOP, its local optimizer cannot choose ten independent points in advance.

## Exact restart semantics

A scheduler interruption must not change an optimizer's observations,
proposal clock, opening assignment, or budget:

- Logistic GP journals each accepted proposal group, all its reservations,
  and the acquisition clock in one event before starting games. Complete
  identity-bound pair logs are reused if observation journaling was
  interrupted. Its absolute 500-observation target makes a clean replay a
  validated no-op instead of starting another tranche.
- CTT durably reserves an opening before starting fastchess. It advances the
  reservation only after the corresponding CTT data iteration exists. A
  complete identity-bound log is reused, and persisted data supersedes a
  lagging model pickle.
- RBFOpt reuses complete identity-bound evaluations. Its model pickle,
  including the black-box game state, is authoritative and is atomically
  replaced before the human-readable JSON checkpoint.
- SPSA reuses the exact completed step before applying and saving its update.
- CLOP's official data-file replay resubmits the same seed and parameters; an
  identity-bound cache returns that seed's completed result without a second
  game.

Run the same command against the same state and log directories after an
interruption. Never delete or transplant only one part of a study. Explicit
engine failures remain fatal and are never retried into a favorable result.
An incomplete color-swapped observation is never fed to an optimizer; any
physical games lost in an infrastructure interruption must be reported in
the separate elapsed/CPU accounting.

## Deduplicated recommendation set

Expand sparse option maps over the pinned engine defaults, normalize numeric
types, sort the keys, and identify a configuration as:

```text
sha256(canonical_json(effective_options))[:16]
```

Keep all 90 `(method, start, checkpoint)` records as aliases, but play each
exact configuration once. If `C` is the set of identifiers, the number of
matches is `U = |C|`. The five methods share each of the three checkpoint-zero
starts, so:

```text
U <= 3 + 5 * 3 * 5 = 78
```

Further exact collisions reduce `U`. Each configuration stores its effective
options; each alias stores method, start, checkpoint, actual training games,
and the SHA-256 of its source recommendation artifact.

## Primary validation

Use held-out-book start `220001` for 400 sequential opening pairs, covering
lines 220001--220400. The slice SHA-256 is:

```text
ea3b33a266182072685abc4b81685d2e2e7e7e223cd17ede4ddb4076cd201cc4
```

The reserved 1,000-pair superset, lines 220001--221000, has SHA-256:

```text
1d5d81f89008c876e58bcc8f533594db5d953c6dc3e55173713aae3fb0342206
```

Run at `3+0.1` with a fixed master baseline, `-games 2 -repeat`, sequential
openings, one match per worker, and ten simultaneous matches. Use the same
adjudication rules for every configuration. A resumed match starts at
`220001 + completed_pairs`, preserving book-line alignment.

At most 78 configurations require 62,400 primary games. Save every normalized
opening-pair score in `{0, 0.25, 0.5, 0.75, 1}`; the pair, not either game, is
the sampling unit.

## Inference and selection

Align all methods by start, checkpoint, and book-line index. Compute each
recommendation's Elo against master and its Elo recovered from that degraded
start. Compare methods directly rather than comparing overlapping marginal
error bars.

For a method contrast, resample opening indices within each start, applying
the same sampled indices to both methods and the shared starting
configuration. In each of 100,000 stratified bootstrap replicates, recompute
logistic Elo, recovery, the mean over three starts, and the method difference.
Report the paired mean-score difference and percentile 95% interval.

The primary metric is mean held-out Elo recovered at 1,000 training games.
Advance its empirical leader and runner-up, plus any other method whose paired
95% difference from the runner-up includes zero. Normalized trapezoidal area
under the 0--1,000-game recovery curve is secondary and cannot override a
primary statistical tie.

## Independent confirmation

Validate the three degraded starts and every finalist's three 1,000-game
recommendations on held-out start `230001` for 1,000 pairs, covering lines
230001--231000. This independent slice has SHA-256:

```text
a423b3fcd39e2cf4f7277629d94950a07f15aa53674c4d77c510e09c7bc04921
```

With two finalists, confirmation has at most nine unique configurations, or
18,000 games. Each extra finalist adds at most 6,000 games. Declare a unique
winner only if its confirmation recovery is positive and its paired method
difference is positive against every other finalist under Holm-corrected
one-sided `alpha = 0.05`. Otherwise report a tie.

## Validity gate

Every match must retain the exact artifact and protocol hashes and the exact
number of complete pair scores. Any engine crash, illegal move, disconnect,
nonresponsive engine, stall, or time loss invalidates that configuration's
result; recovery by the tournament manager does not turn it into valid data.
A scheduler or host interruption is recoverable only under the exact
identity-bound rules above.
