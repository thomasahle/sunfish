# Working on sunfish

This repository contains **two chess engines with different goals**. Almost
every design argument here is settled by asking which one you are touching.

## sunfish (classic) — `sunfish.py`

> Simple, elegant, and verifiably correct.

- Readability is a feature, not a nicety. The engine is ~140 lines with the
  comments stripped, and the README quotes that number; CI checks it.
- Its search contract is machine-checked. `formal/` holds a Lean 4 development
  (no Mathlib, zero `sorry`) proving the fail-soft bracket, terminal exactness,
  and the move-table contract, plus liveness results. A CI audit hashes the
  modelled source regions, so **the model and the code land together** — a
  divergence is a blocker, not a footnote.
- The invariant everything rests on: the value at a transposition-table key is
  a function of `(pos, depth)` and fixed search parameters alone. `gamma`,
  killer state, and incidental table state may change *how* the search runs,
  never *what value it denotes*.
- Consequently classic **cannot take** most of the modern search family — late
  move reductions, late move pruning, reverse futility, history heuristics,
  correction history — because they make the searched value depend on mutable
  ordering state or on `gamma`. That is a deliberate trade, not an oversight.
- Line and byte deltas are costs. A named constant used once, a helper called
  from one place, a second way to do an existing thing — all have to earn their
  space.

## sunfish-nnue (packed) — `nnue_4k/sunfish_nnue.py`

> At most 4096 bytes when packed, and as strong as possible for a Python engine.

> "Anything goes. The goal is to win!"

- The entry is **one file of at most 4096 bytes, and the evaluation data counts
  toward it** (TCEC 4k rules; see `nnue_4k/README.md`). Nothing is exempt — not
  weights, not tables.
- Elegance is explicitly **not** a constraint. The shipped artifact is machine
  generated and nobody reads it. Bytes and Elo are the only currency.
- Search instability is **licensed**. LMR, LMP, RFP, history, correction
  history, dropping the depth component of the table key — all fair game. What
  is required in exchange is that the driver survive it: bracket-crossing
  stops, a hard probe cap that trips loudly, never committing a move from an
  unconverged depth, and a regression that searches with a deliberately lying
  `bound()` and still terminates.
- Divergence from classic is fine where it pays: a mutable board, a different
  table layout, a different move representation.
- Tricks from rival 4k engines (ice4, 4ku, c4ke) are fair game as **ideas**.
  They are GPLv3 and so is sunfish, so there is no licence obstacle, but we
  reimplement independently and attribute rather than transplanting code.
- What does **not** relax: correctness gates. The ten fixed-depth floors, the
  verify battery, shapecheck, and the exact node-identity bench still apply. An
  engine that is fast, small and wrong is worth nothing.

## Both

- **Measurement decides.** Fixed-depth floors are a regression net, not
  evidence of strength; playing strength comes from games. See
  `docs/TESTING.md` for the tournament methodology — C 3+0.1 for a
  node-identical classic change, Python 30+1 otherwise, and a book that covers
  the round count.
- **Never hide an error.** No `except: pass`, no silent fallback, no degraded
  mode that keeps running quietly. An engine that cannot find its interface
  says so and stops.
- Measurement verdicts are appended to `nnue_4k/MEASUREMENTS.md`, not left in
  commit messages.
