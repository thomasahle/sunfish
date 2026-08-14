# nnue_4k/train — the PyTorch training pipeline

Config-driven training for the packed big-int net family: tiny
ternary/quantized nets with PST-shaped features (ps768, king buckets,
bilinear folds, rff sketches) and — via `packed_layers.py` — multi-layer
nets built from the engine's own big-int operations with exact backprop.

## Design principle: everything end-to-end

**Every structural constraint the artifact imposes lives INSIDE the
training graph.** Lane semantics, quantization, sparsity and rate,
codebooks, factorization, the constraint modules — the net is trained
through the thing it will actually be, never fitted first and squeezed
after. (Thomas, 2026-08-14: "Agreed. We should train everything
end-to-end.")

This is a measured position, not an aesthetic one. The bake-off priced
the same two structures post hoc and both lost: codebook-over-blocks
**cb8 +141 B** and low-rank+residual **lr_svd +326 B** on l1-trained
ternary nets — a net trained for a dense trit stream has no reason to
repeat blocks or to be low-rank. Same mechanism, three walls up: rounding
a float fit after training optimises the wrong function
(quant-error compounding), and the kbbil collapse says a constraint the
loss cannot see is free in val and ruinous in play.

A new net family must state **what, if anything, remains post-hoc and
why**. Today's answer for this pipeline: nothing in layer 1 — grid
(`constraints.ternary_ste`), sparsity (`l1`), payload rate
(`rate_penalty`), codebook and low-rank factorization
(`structures.py`) are all in the graph; the *gain schedule* (shift and
`g_k`) is still chosen at export from the trained `|v|`, and the base
(flat material) is still a fixed constant — both named as the next
end-to-end candidates in `../TRAINQUEUE.md`.

Two standing truths, both paid for in the ledger (`../MEASUREMENTS.md`):

1. **Val does not gate landing — play does.** Early-kill exists only for
   obviously-broken runs (non-finite loss, worse than the do-nothing
   anchor after warmup). Never tighten it into a quality gate.
2. **satpen is default-ON** (0.03 @ 480 cp). The engine's hard clip passes
   zero gradient outside the band; without the penalty, saturation is free
   in val and ruinous in play (the kbbil collapse).

## Run an experiment

```
python3 train.py my_experiment.yaml            # config: see config.py dataclasses
python3 train.py CFG.yaml --resume             # continue from ckpt.pt
python3 train.py --repro-arm1 CACHE.pkl        # the pinned REPLNET v1 arm-1 recipe
```

Every run directory (`runs/<name>-<confighash>/`) contains:

- `PROVENANCE.json` — git sha+dirty, seed, torch/numpy/python versions,
  sha256 of every data file read, canonical config hash, val-set sha,
  argv, start time. Written before the first batch; an aborted run still
  says what it was.
- `config.yaml` — the full resolved config (rerunnable as-is).
- `metrics.jsonl` — one record per epoch (train/val/mae/sat/best).
- `ckpt.pt` — resumable state incl. optimizer, scheduler and BOTH RNG
  streams; `--resume` refuses a checkpoint whose config hash differs.
- `best.pickle` (+ `.payload` for ternary) — best-by-val export.
- `certificate.json` — the field-budget certificate (ml2 archs only).

Determinism: `torch.manual_seed(seed)` + one `random.Random(seed)` driving
split and epoch shuffles, in `train_packed.py`'s exact call order — a
`legacy-perm` run's val set is byte-identical to the historical trainer's
(that is what makes `--repro-arm1` a comparison and not a vibe).

## Data

`data.py` loads any of: the lichess `.jsonl.zst` dump (train_packed's
parser + a FEN hash per position), legacy `--cache` pickles (READ-ONLY, no
FENs → legacy-perm split only), labeled `.npz` FEN sets
(distill160k-style), or this pipeline's own `.npz` cache (one parse serves
every extractor: ps768 + kb4/kb8/kb16 codes + fen hashes). `.binpack` is a
flag-gated scaffold (`binpack.py`) — reader-only by design, after
nnue-pytorch; not implemented until a corpus exists to validate against.

**Splits are keyed on the position, not the row number** (house rule):
val iff `sha256(split_seed + fen) % val_mod == 0`. `legacy-perm` (+
`valn`) reproduces the historical permutation splits where FENs are gone.

## Add a net family

1. Features: if it is an index transform over ps768, add it to
   `features.Extractor`; if it needs new per-position data, extend
   `data.extract` + the cache format (bump `CACHE_VERSION`).
2. Model: add a module in `model.py` (or compose `packed_layers` /
   `structures`), and register it in `build_model`. Antisymmetry must hold
   BY CONSTRUCTION — `constraints.check_antisymmetry` on a probe batch is
   the cheap proof. State what remains post-hoc (see the design principle).
3. If it is multi-layer / uses products: write its certifier in
   `field_budget.py` first. Uncertifiable configs refuse to train.
4. Export: extend `export.export_model`. Float-only until the val loss
   earns the packed build (the house rule for every extension so far);
   ternary/packed exports must extend `verify_export.py` and pass
   bit-exact before anything plays.
5. Tests: per-layer bit-exactness vs the python big-int ops joins
   `test_packed_layers.py`; run it on every pipeline change.

## The packed layer family (`packed_layers.py`)

The engine's big-int trick as torch modules with exact semantics:
`LaneConv` (one big-int multiply = linear lane convolution; the mod
`2^(Fm)-1` fold = circular — the fix for the recorded rank-1 read-out
trap), `SwarClamp` (crelu with per-lane caps), `HSum` (modular lane sum),
`ShiftRenorm` (signed trunc shift — the depth enabler). Forward runs in
float64 where every certified value is an integer < 2^53, so float64
arithmetic IS integer arithmetic; `test_packed_layers.py` holds every
layer to bit-exact equality with actual python big-int evaluation.
Gradients are the true polynomial gradients everywhere except two
documented STE points (trunc shift; optional clamp pass-through).

`field_budget.py` certifies, per layer, BEFORE training: no-carry (field
width), no-borrow (offset lanes), the modular-hsum precondition, and
float64 exactness — by exact interval arithmetic, with margins. It also
quantifies depth: with renorm-to-12-bits between layers, 16+ conv layers
certify at F=32. The three recorded walls (carry coupling, field-budget
collapse, quant-error compounding) each map to a named check.

## Trained structure (`structures.py`)

Two weight parametrizations for layer 1, both certified before training
and both priced by their own bake-off arm (never a composed estimate):

- **`arch: cb`** — product quantization. Blocks of `cb_block` consecutive
  features are codewords from a learned ternary book of `cb_k` entries.
  STE: hard argmin forward; backward is identity to the shadow weight
  (`ternary_ste`'s own rule) and softmax-weighted, `p = softmax(-d /
  (cb_temp·D))`, to the book — the distances are frozen under `no_grad`,
  so both routes are exact gradient identities that
  `test_packed_layers.py` checks as identities. Exports book + index
  stream; arm `trained_cb`.
- **`arch: lowrank`** — `W = clip(U@V + R, ±lr_wmax)`, `U`/`V` ternary,
  `R` the ordinary ternary residual. `V` is zero-initialised, so epoch 0
  is **exactly** the plain net (`torch.equal`) while `V` still receives
  gradient and can wake up — the ml2 u2-silent precedent. Exports U, V
  and the residual; arm `trained_lr`, which additionally drops residual
  entries the clip makes invisible.

**The grid is ternary for both**, and `field_budget.max_certified_grid()`
is why: a weight of `c` lane-gain units puts `32·c·gmax + 45` into an
offset lane bounded by `2^14`, so the ceiling is **c = 5** (c = 6 REFUSES
at 17133 > 16383). Free-int entries are legal up to 5 but cost
log2(11)/log2(3) = **2.19× per stored element**, cut the no-borrow margin
from 13490 to 2098 lane units, and leave the shipped codec's
representable set — a net with no b81 payload has no denominator for its
own bake-off table. Ternary is the cheaper certified form and it is
directly executable: the decoded stream *is* the trit stream the entry
already consumes.

## Export + pricing

```
python3 export.py runs/<name>/ --price
```

Ternary: writes the payload, splices it into `../replnet_proto.py`, runs
the `verify_export.py` triangle (payload decode == trainer quantization;
entry == integer reference == torch mirror, BIT-EXACT on 200 fens x 3
views + a 60-ply walk, plus the replnet_check invariants), then
`tools/build/pack.sh` — the printed byte count is pack.sh's own
measurement, never composed arithmetic. Gate-ladder commands are STAGED
(printed), never launched: gates and screens are coordinator-dispatched.

## Queue runner

```
python3 queue_runner.py            # consumes queue/*.yaml serially, forever
```

Implements the always-training rule for TRAININGS ONLY (it never launches
matches/screens/gates). Box-aware: `nice -n 19`, threads/workers capped at
8, atomic run lock, and a forfeit tripwire — if "loses on time" counts in
the configured pgn globs rise while training, the run is SIGSTOPped and
`PAUSED_REPORT.txt` written; deleting the report resumes (operator
decision, never automatic).

## Pipeline validation record (2026-08-14)

Reproduction of REPLNET v1 arm 1 (`--repro-arm1`, box, legacy cache, seed
0, tolerance pre-stated |dval| <= 0.0002, zeros ±5): see the migration
note in `../MEASUREMENTS.md` for the measured numbers and the export
bit-exactness verdict.
