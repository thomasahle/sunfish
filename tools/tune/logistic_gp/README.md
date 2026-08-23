# Adaptive game-result tuning

`adaptive_gp.py` tunes any UCI engine's options directly from win/draw/loss
results produced by fastchess. It uses a logistic Gaussian process suited to
noisy binary evidence instead of treating Elo estimates as noise-free scalar
measurements.

Each free lane runs a color-swapped opening pair. As soon as the pair finishes,
the posterior is updated and that lane receives another configuration. The
allocator mixes a permanent maximum-variance exploration arm with GP-UCB
exploitation, so one unlucky pair cannot permanently discard a region. It can
also play supported candidates against one another while retaining games
against the fixed baseline as an absolute anchor.

## Files

- `adaptive_gp.py` is the asynchronous fastchess scheduler and acquisition
  loop.
- `logistic_gp.py` contains mixed parameter spaces and exact/sparse logistic-GP
  inference.
- `report_gp.py` reports the posterior and coordinate slices from a saved run.
- `all_parameters.json` is Sunfish's joint search/evaluation example.
- `global_search_parameters.json` is Sunfish's search-only global space.
- `sunfish_gate.py` is Sunfish's deterministic mate-safety gate example.
- `test_logistic_gp.py` covers the model, scheduler, restart journal, UCI option
  validation, exploration policy, and Sunfish examples.

The optimizer requires Python, NumPy, fastchess, a UCI engine, and an EPD
opening file. It validates every option against the engine's `uci` response
before spending games.

## Parameter spaces

JSON parameters may be `integer`, `real`, `discrete`, `categorical`, or
`boolean`. Integer spaces accept `min`, `max`, and `step`; real spaces accept a
linear or logarithmic transform and a finite `count`; the other types take an
explicit `values` list. Conditional rules can reset inactive parameters to
their defaults, preventing the model from treating equivalent policies as
different points.

Large Cartesian products are represented by a deterministic space-filling
design plus required default, one-axis, and local-interaction points. Coordinate
refinement lets acquisition search between those initial design points.
`--full-axis-design` makes coverage literal: before model allocation, every
value of every parameter is played with every other parameter at its default,
along with explicitly required structural combinations.

## Sunfish example

From the repository root, after building `tools/ctwin/sunfish_c`:

```sh
python3 tools/tune/logistic_gp/adaptive_gp.py \
  --fastchess /path/to/fastchess \
  --engine tools/ctwin/sunfish_c \
  --engine-args tools/ctwin/tables_classic.txt \
  --baseline-options default \
  --space tools/tune/logistic_gp/all_parameters.json \
  --openings /path/to/openings.epd --cycle-openings \
  --gate "python3 tools/tune/logistic_gp/sunfish_gate.py --horizon-only \
    --node-factor 2 --node-book /path/to/openings.epd" \
  --gate-design --gate-workers 20 --full-axis-design \
  --slots 10 --queue-batches 30 --refill-batches 10 \
  --pairs 1 --initial-design 256 --inducing 128 --update-batches 8 \
  --explore-start .5 --explore-floor .2 --duel-fraction .3 \
  --wall-time 3d --batches 1000000
```

One pair per observation is intentional: robustness comes from the posterior,
not from hiding noisy results inside large batches. The larger pending FIFO
only keeps game lanes occupied while acquisition runs; fantasy variance keeps
its look-ahead choices diverse. Opening reuse is balanced by a deterministic
shuffle per epoch; final confirmation still needs an independent book.
Ten slots run twenty engine processes. A pairwise-only state pooled from SPSA
can be imported with `--seed-state`; both sides of every duel then count as
observed axis coverage, while games against the fixed default supply the new
study's absolute Elo anchor.

`--baseline-panel panel.json` can replace one fixed baseline with a stationary
integer-weighted mixture. Each seeded block shuffles one copy of every weighted
slot, so restarts reproduce the schedule and every complete block has exact
weights without a fixed opponent phase. For example, weights `2, 1, 1` give
50% master, 25% Stockfish, and 25% an independent peer. Use `--duel-fraction 0`
when every global-search observation should be anchored to that panel. The JSON is a list of objects
with `name`, `engine`, `weight`, and optional `args` and `options`; the special
`"options": "default"` applies the parameter-space default to that member.
Optional `source`, `revision`, and `license` strings record provenance, while
`identity_files` names lock manifests or source files that must remain byte
identical when a study resumes. GP and SPSA use the same one-based opening
sequence and seed, so their shuffled weighted blocks are identical.

A finite inducing basis leaves residual variance that its GP features cannot
learn, even after replaying one configuration many times. The exploration arm
conditions that residual on the exact number of completed pairs, using the
binomial logit's Fisher-information bound. This prevents sparse approximation
error from masquerading as permanently useful uncertainty; GP-UCB itself still
uses the unmodified posterior.

`--baseline-options default` pins the configured default point to zero. The
Sunfish space covers every live numeric search choice except time, memory, and
historical flavor switches. Its domains preserve the mate-band invariants, and
the gate rejects policies without a finite eventual-mate horizon before they
consume games.

`TABLE_SIZE` and `DELAY` are deliberately absent: the former buys strength with
memory, while the latter changes the clock budget. The global-search objective
holds both resource budgets fixed and records those exclusions in the space.

The mate gate derives each suite depth from the candidate's maximum real-edge
cost, move-cap horizon, and null-candidate horizon. `--horizon-only` cheaply
rejects policies without a finite bound during a broad optimization; omit it
to run the executable mate suites on finalists. Scoring-null and fuel-null
horizons are independent, so either mechanism can be disabled without making
the other unbounded.
The optional node ceiling rejects policies whose fixed-depth tree exceeds a
generous multiple of the default engine on a small deterministic sample. It is
a feasibility constraint, not a surrogate objective: accepted policies are
still ranked only by games. The same cached gate protocol is supported by the
RBFOpt adapter.

Results append to a JSONL journal and periodically compact into the state file.
Runs are restartable, preserve their exploration clock, and finish reserved
color pairs before a wall-time stop. Accepted configurations are durable before
games start, and each pair result is durable as soon as the scheduler collects
it. A restart therefore replays only pairs without a saved result, including
inside a multi-pair posterior update. `report_gp.py` can inspect the saved
posterior without resuming the tournament.

Every reusable log is bound to a fresh state generation, the frozen study,
both configurations, and its opening. Parallel gate groups commit all clocks
and reservations together before starting any of their games.

`--batches N` is one durable tranche: after a crash, the same command finishes
exactly N updates in total rather than N more. A clean exit closes the tranche,
so a later invocation starts a fresh N-update tranche.
