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
  budgets. It can also pool pairwise-only SPSA studies.
- `freeze_recommendations.py` normalizes and audits a complete method/start/
  checkpoint recommendation grid before held-out games are consulted.
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

Both the global GP and SPSA accept seeded, integer-weighted opponent
panels. Panel-anchored SPSA compares both perturbations with the same opponent
on the same color-swapped opening before applying the gradient; one update is
therefore two opening pairs and four games. Five such lanes use 20 engines.
Both runners shuffle the same weighted block for each group of opening numbers,
which preserves exact weights without a fixed opponent phase. The seed, panel
helper, provenance, and extra source-lock files are included in resumable study
identities, along with executable hashes and UCI options.

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
start-vector trace checks.

The integration tests exercise parameter translation, resumable state,
pentanomial parsing, feasibility gates, checkpoint extraction, and the shared
validation pipeline:

```sh
python3 -m unittest tools.tune.logistic_gp.test_logistic_gp \
  tests.test_tuning_tools
```
