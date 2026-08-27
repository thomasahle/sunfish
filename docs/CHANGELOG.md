# Changelog

## sunfish 2026.2

A small strength follow-up containing the final search-tuning result that
landed immediately after the 2026.1 tag.

- The shallow scoring-null child is searched one ply less. The exact change
  passed a `[0, +10]` pentanomial SPRT at `+13.24 +/- 10.29` Elo over 3,046
  games, with an independent 1,000-game match measuring `+18.4` Elo.
- The fail-soft and eventual-mate proofs, C twin, compressed engine, and tuning
  defaults were updated with the executable. No executable line was added.

## sunfish 2026.1

A strength and production update following the first 2026 release.

- Five existing shallow-search parameters were tuned jointly without adding
  engine code. The change passed a `[0, +10]` pentanomial SPRT at
  `+23.83 +/- 15.61` Elo and scored `+32.41 +/- 18.52` Elo in an independent
  1,000-game match.
- The search now uses fixed intrinsic reductions, monotone shallow move caps,
  and a deep null probe as a position-dependent fuel signal. The corresponding
  fail-soft and eventual-mate models were updated with the executable.
- Time management now allocates a shared clock pool across moves, with
  production-tested handling for increments, pondering, overhead, and hard
  deadlines.
- The C twin, game-result tuning suite, and 1,736-position Lichess blunder
  corpus make search changes faster to measure and harder to regress.
- The packed NNUE sibling, its training and verification tools, and both
  production Lichess-bot bundles are included in the repository.

## sunfish 2026

The first PyPI release. Three years of work since sunfish 2023, in four
stories: the search was formally verified and got *simpler* for it, two
families of production bugs were found and fixed, the engine became a
proper package, and a strong NNUE sibling grew up next to it.

### The engine (138 clean lines, ~3.3KB packed)

- **The search's promises are now theorems.** The `bound()` docstring
  states the full contract — the fail-soft bracket, the exact promises
  (king-gone, capturable-node sentinel, verified mate/stalemate values,
  and the narrower real-cutoff `tp_move` witness) — and every clause maps to
  a machine-checked theorem in `formal/` (Lean 4, zero sorries, no Mathlib).
  The correctness layer carries **no chess assumptions**;
  zugzwang can cost accuracy, never table consistency.
- **Null moves are capped at static evaluation plus one score bucket**:
  `min(E + 15, pass)` is monotone, so one child probe preserves the
  fail-soft contract and the old mate-boundary verification probe
  disappears. The existing material and evaluation guards remain the
  zugzwang defenses; the cap is a deliberate, regression-tested search-value
  change, with its report transformer and score-band invariant proved in Lean.
- **Mate/stalemate detection is verify-on-suspicion**: when no searched
  move proves legal, every generated move is checked with a board
  predicate, and verified terminals return their exact values
  (0 / −MATE_LOWER) at every depth ≥ 1 — in both fail directions.
  Trusting search scores as legality evidence is provably unsound;
  three historical "score implies legality" assumptions were refuted on
  real boards and their fixes are regression-tested.
- **Late Move Reductions were removed** after measurement showed the
  honest variant is worth exactly 0.00 ± 34 Elo — the historical
  variant's entire edge was its unsound bound propagation (the
  transposition-table crossing is machine-checked). The move loop is
  full-width again, and the docstring is provable as written.
- **`MATE_LOWER` margin repair**: the king-value margin now covers the
  largest army a kingless side can field (`K − 13Q`); the old margin
  provably leaked.
- The MTD-bi driver brackets the full window band, converges in a
  proven ≤ 15 probes per depth, and only commits a best move from a
  completed depth (see bug fixes below).

### Production bug fixes (found by auditing the lichess bot's games)

- **The killer-eviction race**: `tp_move`'s FIFO eviction could age out
  the *current search root* mid-search once the table churned past
  capacity; a deep fail-low probe then stored whatever capture sorted
  first, and a timeout played it. Three queen/piece giveaways in 145
  production games. Fixed: eviction never removes the search root.
- **The dive-window family**: a mid-depth stop could answer from a
  fail-high obtained at an absurd probe window (including a ponderhit
  variant). Fixed: both UCI loops promote a candidate move only when
  its depth's bracket converges — with the one sound exception proven
  and kept (the first probe of the next depth, which runs at the
  converged bracket's midpoint).
- **`position fen` handling** (#156): an installed engine that cannot
  find its real UCI interface now says so and stops, rather than
  silently playing on with a reduced one.
- **Long-TC time overruns**: deadlines are enforced inside the search
  (checked every 2048 nodes), not just between iterations.
- **Finished-position UCI handling**: checkmate and stalemate roots return
  `bestmove (none)` cleanly in both interfaces instead of dereferencing the
  deliberately absent root move.

### Packaging and interfaces

- **`pip install sunfish && sunfish`** starts the terminal play
  interface; **`sunfish-uci`** is the engine for GUIs and tournament
  managers. Published to PyPI via trusted publishing; releases are
  tag-driven and smoke-test the wheel end-to-end, including on Windows.
- `sunfish_ui/` ships in the wheel: the full UCI interface (pondering,
  Hash, spec-complete `go` parsing, FEN positions) and the fancy
  terminal board.
- Repository reorganized: `docs/`, `formal/`, `sunfish_ui/`, `tests/`,
  `tools/` — with build scripts, deploy configs, and test data filed
  under their owners.

### Verification and testing infrastructure

- `formal/`: the Lean development — the core fail-soft contract, the
  layered null-move specs, the killer-table lifecycle theorem, liveness
  results (mate-in-k completeness, driver convergence, no false mates,
  and the win/loss/no-mate classification theorem) — plus a CI drift
  guard that hashes every audited code region and ratchets stale
  citations. A paper draft lives in `formal/paper/`.
- `docs/TESTING.md`: the tournament methodology, each rule with the
  confident wrong number that created it. Decision-grade matches run at
  30+1 minimum with a book that covers the round count.
- The test suite grew the terminal bench (148 Stockfish-validated
  mate/stalemate positions), transposition-consistency witnesses, and
  deterministic regressions for both production bug families.

### The NNUE sibling (in review, `nnue_4k/`)

A packed big-integer NNUE engine — the whole accumulator and evaluation
head live in one Python int (SWAR lanes, modular horizontal sums, king
buckets) with training and verification toolchains. Currently measured
around +200 Elo over classic at 30+1, aimed at the TCEC 4k division and
a lichess bot of its own. Lands in the next release.

## sunfish 2023

The classic 131-line engine: MTD-bi search, piece-square tables,
null-move pruning, quiescence via the value filter, futility pruning,
and the king-capture convention. See git history.
