# Tuning tools

This directory contains the reusable parts of Sunfish's parameter-tuning
experiments. The game-result tools work with any UCI engine; the included
parameter spaces and correctness gate are Sunfish examples.

## Optimizers

| Directory | Method | Intended use |
| --- | --- | --- |
| [`logistic_gp/`](logistic_gp/) | Logistic Gaussian process | Global search with explicit exploration |
| [`spsa/`](spsa/) | Fishtest-style SPSA | Local refinement in larger spaces |
| [`chess_tuning_tools/`](chess_tuning_tools/) | Chess Tuning Tools adapter | Chess-specific Bayesian optimization |
| [`rbfopt/`](rbfopt/) | RBFOpt adapter | Global radial-basis surrogate search |
| [`clop/`](clop/) | Official CLOP adapter | Local quadratic optimization |
| [`texel/`](texel/) | Texel fitting | Static evaluation tables, not search parameters |

The logistic-GP runner also provides a maximin space-filling policy for a
non-adaptive control. Every game-result method consumes the same JSON
parameter-space format, including integer, real, logarithmic, discrete,
categorical, and Boolean parameters.

## Shared workflow

- `recovery_starts.py` creates reproducible, deliberately weakened starting
  configurations.
- `screen_starts.py` measures those starts against a fixed UCI baseline.
- `recommend.py` extracts comparable incumbent configurations at fixed game
  budgets. It can also pool pairwise-only SPSA studies, or materialize declared
  parameter-space points without an optimizer; `required --off-only` produces a
  validation-ready default-plus-mechanism-off artifact.
- `freeze_recommendations.py` normalizes and audits a complete method/start/
  checkpoint recommendation grid before held-out games are consulted.
- `audit_consensus.py` independently recomputes the frozen support/L1/SHA
  consensus and its eventual-correctness parameters before validation.
- `validate.py` measures recommendations on an independent opening set.
- `plot_recovery.py` produces held-out Elo-versus-training-games curves and a
  paired method-comparison CSV at the primary checkpoint.
- `recovery_decision.py` selects tied primary finalists and performs the
  independent, familywise-corrected decision among those screened finalists.
- `pentanomial.py` parses color-swapped opening pairs and estimates their
  score and uncertainty without treating the two games as independent.
- `gating.py` caches deterministic feasibility checks before games are spent.
- `locking.py` protects resumable studies from concurrent writers.
- `uci_wrapper.py` exposes a UCI command with arguments as one executable.

The global GP, SPSA, and Chess Tuning Tools adapter accept seeded,
integer-weighted opponent panels. Panel-anchored SPSA compares both perturbations with the same opponent
on the same color-swapped opening before applying the gradient; one update is
therefore two opening pairs and four games. Five such lanes use 20 engines.
Both runners shuffle the same weighted block for each group of opening numbers,
which preserves exact weights without a fixed opponent phase. The seed, panel
helper, provenance, and extra source-lock files are included in resumable study
identities, along with executable hashes and UCI options.

The CTT adapter binds every persisted iteration to one deterministic panel
member and checks the panel hash before starting a game. Its generated initial
point file contains the default plus every distinct point listed under
`required` in the shared parameter space. This guarantees that declared
mechanism-off boundaries are measured instead of merely being available to the
surrogate by chance.

`freeze_panel.py` builds the C twin twice on the tournament host, rejects a
non-reproducible binary, copies the pinned opponents and licenses, and writes
an operational `panel.json` plus a relocatable `manifest.json`. Verify an
existing artifact with `freeze_panel.py --verify path/to/manifest.json`.
`global_search_panel.lock.json` records the frozen 2:1:1 campaign artifact;
the binary itself remains outside Git. Stockfish's release, exact upstream
commit, executable, and license are separate locked identities.

The freeze completes calibration before writing its final manifest. It gives
each non-master opponent 50 common color-swapped openings against frozen
master at 3+0.1, rejects engine failures, incomplete results, and saturated
opponents, then freezes the result summary and raw-log hashes. Host-specific
commands and paths from the temporary result file are discarded. Thus nominal
engine ratings and a merely pending calibration are never accepted as panel
evidence.

Calibration proves reliability and non-saturation; it is not subtracted as an
Elo offset. The target is expected weighted-panel score, including matchup
effects. Each normalized pair score is in `[0, 1]`, so one Bernoulli-equivalent
observation after randomized opponent assignment is conservative.

`search_parameters.json` is a small generic search-tuning example.
`logistic_gp/all_parameters.json` includes evaluation experiments, while
`logistic_gp/global_search_parameters.json` is the frozen search-only space.
The latter records every excluded C-twin option and why it is not a current
Python numeric search parameter. The Sunfish-specific gate rejects
configurations that violate the search's fuel and eventual-widening
invariants before they enter a tournament.

Run each program with `--help` for its command line. Game-result studies need
fastchess, a color-swapped opening book, and engines whose tunable values are
exposed as UCI options. The Dockerfiles pin the external dependencies needed
by Chess Tuning Tools and RBFOpt; CLOP itself is installed separately.

## Comparing methods fairly

Count the x-axis in games, not optimizer iterations. Validate checkpoint
recommendations on openings that were never used for training, and never use
an optimizer's internal score as comparable Elo. One opening played twice
with colors reversed has five possible candidate scores—0, 0.5, 1, 1.5, or
2—so the pair is naturally pentanomial.

Keep the shipping defaults as the permanent zero-Elo incumbent. Losing points
are useful optimizer observations, but never tuning results: if no independently
validated recommendation is non-losing, ship the defaults and report zero gain.
For a strength tune, require the recommendation to pass the pre-registered
0-versus-positive SPRT against the defaults; rejection or an inconclusive maximum
keeps the defaults. Quote gain only from a separate fixed-N match.

Use one frozen training-opening schedule for every method and apply any
feasibility gate to every method or to none. `recommend.py` emits only reached
checkpoints; CLOP recommendations use the recorded result seeds, rather than
completion-order row parity, and require a complete two-game replication. The
game runners reject logs that report a disconnect,
nonresponsive engine, illegal move, stall, crash, or forfeit, even when
fastchess recovery returns success.

Games measure sample efficiency, not total compute. If gates, optimizer cost,
or concurrency differ, report elapsed or CPU time separately. Freeze methods
before consulting validation results, reuse one starting-position match, and
confirm the selected winner on a fresh opening set. Pair scores are pooled over
the equally weighted degraded starts before converting the mixture to Elo. The
paired comparison uses one shared opening-index draw across every start; its default is
100,000 deterministic replicates at the 1,000-game checkpoint. It rejects
inputs that do not share the complete validation protocol and starting match.

The frozen five-method comparison is specified in
[`recovery_protocol.md`](recovery_protocol.md), with exact settings and hashes
in [`recovery_benchmark.json`](recovery_benchmark.json). Run
`python3 tools/tune/verify_recovery.py` before starting a campaign; supplying
the frozen engine, tables, and training book also reproduces the corrected
start-vector trace checks. Pass `--method-root` to additionally verify a
checkout of the manifest's frozen optimizer-source commit; live tuner code is
allowed to evolve without rewriting historical hashes.

The integration tests exercise parameter translation, resumable state,
pentanomial parsing, feasibility gates, checkpoint extraction, and the shared
validation pipeline:

```sh
python3 -m unittest tools.tune.logistic_gp.test_logistic_gp \
  tests.test_tuning_tools
```

## Sunfish search campaign, August 2026

The production campaign treated the checked-in defaults as a permanent
zero-Elo incumbent. Losing configurations remained optimizer observations;
they were never promoted merely because they were smaller or faster. Search
changes had to pass a positive SPRT, and a separate fixed-size match supplied
the quoted Elo magnitude.

The production search-only space held piece values, PST values, time management,
and memory size fixed. The compact five-method recovery benchmark used a
deliberately small 12-control projection. A separate full-space follow-up
covered all 20 numeric search controls exposed by the C twin, including depth
thresholds, null-search reductions, LMR, shallow move caps, and their explicit
off-boundaries. The selected production configuration changed only five values:

| Parameter | Old | Selected |
| --- | ---: | ---: |
| `QS` | 40 | 36 |
| `QS_A` | 140 | 180 |
| `LMR` | 75 | 70 |
| `LMR_MIN_DEPTH` | 6 | 7 |
| `FUT_CAP_DEPTH` | 3 | 4 |

The combination was tested as one policy; these are not five independent Elo
claims. It added no executable line. Its direct paired results against the old
defaults were:

| Evidence | Result |
| --- | --- |
| Pentanomial SPRT `[0, +10]` | H1 after 1,314 games; `+23.83 +/- 15.61` Elo |
| Independent fixed 1,000 games | `+32.41 +/- 18.52` Elo; LOS 99.97% |
| Frozen 2:1:1 opponent panel | `+21.84` Elo; 90% interval `[+11.11, +30.22]` |

The panel combined old Sunfish, a calibrated Stockfish, and ChessIdle. Both
external-opponent point estimates were non-negative. There were no crashes,
illegal moves, stalls, disconnects, or time losses.

The parameter change also passed the full Lean build and the source/model
audit. The resulting uniform eventual-search bounds are `D >= 3k + 2` for a
forced mate in `k` plies and `D >= 3k + 5` for the forced-loss dual; a concrete
mate-in-three witness proves the first bound sharp for this recurrence. Python
and the C twin agreed on 5,220 bound reports and 1,042 generated move lists.
The deterministic gates were WAC 170/300 at depth 8, Bratko-Kopec 11/24 at
depth 8, and Lichess regressions 451/1,736 at depth 8.

Explicit deletion tests did not reveal a free simplification. Disabling
shallow null and compensating with earlier fuel reduction was inconclusive at
`-3.47 +/- 13.31` Elo; compensating with earlier LMR was rejected at
`-17.78 +/- 16.99` Elo. Other mechanism-off variants expanded the fixed-depth
tree substantially. Under the five-Elo-per-line rule, none earned deletion.

## Comparing the five game-result tuners

The final optimizer comparison asks how much strength a method recovers from
three deliberately degraded configurations after 0, 100, 200, and 400
training games. Logistic GP, CTT/MES, RBFOpt, SPSA, and CLOP receive identical
engine builds, parameter bounds, starts, training openings, and physical-game
budgets. The reference engine is stationary. Each checkpoint recommendation
is frozen before any held-out result is read.

The training score is never plotted as Elo. Recommendations play the same 50
unseen color-swapped opening pairs at each start. Pair scores are pooled over
the three equally weighted starts before applying the logistic Elo transform.
Method contrasts resample the same opening index across all starts, preserving
the pairing. Finalists use another disjoint 50-pair slice, followed by a
predeclared 100-pair look only if necessary. The confirmation uses exact
sign-flip tests with Holm correction.

This compact benchmark replaced two tempting but invalid shortcuts. A first
one-start pilot was far too noisy to separate methods, and a later 5,981-game
run had unequal method/start cells and no held-out matches. Optimizer posterior
scores from those games are useful diagnostics, but they are not comparable
Elo and do not enter the final ranking.

The current follow-up searches the complete 20-control space from the merged
incumbent. It uses the C twin at 3+0.1 and three independent CTT/MES seeds. The
finite candidate sets are encoded as compact integer coordinates so sentinel
values such as “mechanism off” remain searchable without giving the surrogate
huge meaningless numeric intervals; the engine wrapper decodes those
coordinates before forwarding UCI options. A raw full-range trial and the
compact RBFOpt comparison were discarded after their runners failed, before
any held-out result was read. No new engine setting is claimed until a fresh
opening screen and an independent SPRT both clear the incumbent.
