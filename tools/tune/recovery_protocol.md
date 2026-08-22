# Five-tuner recovery protocol

Status: frozen on 2026-08-22. This protocol compares GP, Chess Tuning Tools
(CTT), RBFOpt, SPSA, and CLOP by held-out Elo recovered per training game.

## Artifact pins

- Engine SHA-256:
  `c5039b4a0f88a2e11cf18ad13cd3f7d2c206b0a6901249d67d0b43ec02c95478`
- Tables SHA-256:
  `d09fc445930a0e4babbab1aa176847a77923d3bf00c0b6fb0c46b0a90d14bbf0`
- fastchess SHA-256:
  `3d0a8f3c8b837a96366ac838f3ddf6fe87b813abb4ad2852284c0ce409132566`
  (alpha 1.8.2, build `f618e34`)
- Training-book SHA-256:
  `2c17253fc1282d7f41410d64e0621d801bb992c3bfba6ecd2372b06e7c5eae5f`
- Held-out-book SHA-256:
  `3f499996ff0b674a04f85f2634811d102dd53b5115841e8f11d18e1f550ba2ca`

The training book has 1,991 unique positions. The held-out UHO book has
241,670 positions, all unique after canonicalizing to the first four FEN
fields, and has no position in common with the training book.

The recorded campaign used held-out lines 1--1,000 sequentially. An audit
also reproduced all 197 positions selected by three recorded random-seed
runs with fastchess's exact `mt19937_64` Fisher--Yates shuffle and mapped the
recorded telemetry FEN. None occurs in either frozen validation slice.

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
number of complete pair scores. Any crash, illegal move, disconnect,
nonresponsive engine, stall, or time loss invalidates that configuration's
result; recovery by the tournament manager does not turn it into valid data.
