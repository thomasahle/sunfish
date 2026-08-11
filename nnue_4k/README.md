# nnue_4k — the packed big-int NNUE sunfish

A sunfish variant whose evaluation is classic's exact piece-square score
plus a trained neural residual — with the entire accumulator *and* the
evaluation head living in **one Python integer**, so a wide net costs a
handful of big-int operations per node instead of a Python-level loop.
The engine still packs to a few kilobytes (see *The 4k build*).

## Architecture

```
score(pos) = pst(pos) + clip(nn(pos), ±600)
```

- **pst** is classic sunfish's piece-square score, verbatim and exactly
  incremental: `value(move)` stays an exact PST delta, so move ordering,
  the QS stand-pat gate and futility pruning behave exactly as classic.
- **nn** is a 768→N→1 residual, `Σ v_k (crelu(a_k(us)) − crelu(a_k(them)))`,
  with one output weight serving both perspectives — the evaluation is
  **exactly antisymmetric**, bit for bit, which the transposition table's
  single-value-function invariant wants.

The packed representation (`packed/pnet.py` documents every invariant):

- 16-bit lanes, offset-binary, a borrow-guard bit per lane; the output
  weight is folded into the first-layer rows (`G_k = C·|v_k|`), so the
  head needs **no multiplies** — clipped ReLU is a SWAR clamp with
  per-lane ceilings, and the horizontal sum is one modular reduction
  (2^16 ≡ 1 mod 2^16−1), made *provably* exact by `sum(G) ≤ 65534`,
  asserted at build time.
- Quantization is certified: `excursion_bound()` proves every lane stays
  inside the guard over all *legal* piece placements, and
  `packed/verify.py` proves lane integrity, incremental == from-scratch,
  engine == reference, and exact antisymmetry over thousands of
  random-game positions. Nothing plays a game before that battery is
  green.
- **King buckets** (`--kb`): first-layer rows conditioned on each side's
  own king (8 buckets: file pairs × back-two-ranks/advanced). A king
  move across a bucket boundary rebuilds the accumulator (~32 adds,
  rare); everything else stays incremental.
- **Extensions** (bilinear lanes with group convolutions, a narrow odd
  tail, a material-phase output scale) are implemented end-to-end and
  verified, but currently *not* deployed: see the lesson below.

## The 4k build

```
tools/build/pack.sh nnue_4k/sunfish_nnue.py out.packed
```

pyminify + xz + a 116-byte self-extracting header. Current state: the
extended engine packs to 4944 bytes against the 4096-byte budget —
the extension machinery, go-loop hardening and the pickle-free net
loader are 848 bytes over, tracked as an open golf debt in the ledger. The v1 engine (92c4746) packed to
3952 bytes; nets are external (`SF_NET`) and not part of the budget.

## Net format (`.sfnn`)

One JSON header line (all scalar fields — no code execution on load;
pickles never ship), then base64 tokens, one per big int (minimal
two's-complement little-endian), in canonical order: `base`, `gp`,
`ts`, then piece rows (`PNBRQKpnbrqk` × 120; kb nets store rowsW
buckets then rowsB). The engine's loader is ~15 lines; conversion is
bit-identical (round-trip asserted in the ledger).

## Training quickstart

```
python packed/train_packed.py \
  --data lichess_db_eval.jsonl.zst \
  --N 128 --kb 8 --epochs 14 --batch 16384 --limit 30000000 \
  --losspow 2.6 --factor 1 --cache quiet30M_kb8.pkl --out net.pickle
python packed/build_ext.py net.pickle net.sfnn        # quantize + certify
python packed/verify.py sunfish_nnue.py net.sfnn    # prove it
python packed/shapecheck.py sunfish_nnue.py net.sfnn  # search-friendliness
```

Data is the lichess Stockfish-eval dump, quiet-filtered; training is
sigmoid-space loss (exponent 2.6) with factorized virtual features.
Float exports (`--kb`/`--nb`/`--phase` etc.) stay pickles internally;
everything distributed is `.sfnn`.

## Lichess deployment

`lichess/` holds the full bundle (setup.sh with frozen-tag + net-sha256
pins, config.yml, systemd unit): the bot runs this engine through
`sunfish_ui/uci.py` (pondering, FEN, Hash) on an ARM instance, with
the verify battery as a hard install-time gate. See `lichess/README.md`.
