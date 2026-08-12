# Packed NNUE measurement ledger

Every experiment this lane runs — verdicts, negatives, prices, and the
reasoning that follows from them. **Newest first.** Append a dated entry
for each measurement; never rewrite an old one (corrections get their own
entry that says what changed).

An entry carries the numbers with their error bars, the game/position
counts behind them, what the result means, and what happens next. Negatives
are recorded with the same care as wins — most of the value in this file is
knowing what was already tried and priced.

Entries dated 2026-08-09 through 2026-08-12 were backfilled from the commit
messages that served as the ledger before this file existed (`git log
--grep="Measurement record"`); those commits remain in history unchanged.

## Two targets, different currencies

Since 2026-08-12 this lane serves **two separate goals** and entries say which:

- **The lichess bot** — no size limit. Keeps the large net (256kb8@100M, 14.9 MB).
  Everything about strength, speed and search quality serves this.
- **The 4k entry** — ONE file ≤ 4096 bytes, *including the evaluation data*. This
  is a compression problem with a strength objective, and almost none of the
  large-net work transfers.

## Index

| Date | Experiment | Verdict |
|---|---|---|
| 2026-08-12 | **4k design space priced (weights RAW, not xz)** | Ternary base-3 packing + factorisation beat the width-5 baseline **5-50×** in parameters at 1920 B; width is ~free in speed at this scale |
| 2026-08-12 | **4k budget re-derived: the net counts** | Real artifact = **541,781 B** (engine 4488 + net 537,152), not "3798, 298 under". Packing mechanism recovered and verified running |
| 2026-08-12 | **Field study: ice4 + 4ku eval packing** | ice4's ENTIRE eval = **333 characters**; both engines factorise PST into rank+file. Our 768×128 is 98,304 values |
| 2026-08-12 | Historical 1207-byte net decoded | **Trained rank-6 factorisation**: 816 int8 → 4608 PST values, exact by construction |
| 2026-08-12 | **CORRECTION: the bottleneck is `nn_cp`, not the board** | The "85% board" claim was an inference error (marginal ≠ total). Measured: net 8.1µs vs board 2.9µs of a 14.6µs move — mutable board is worth ~+15 Elo, not +71…+110 |
| 2026-08-12 | Hot-path profile (superseded in part) | ~85%-board claim WRONG — see the correction entry above |
| 2026-08-12 | **GOAL-LINE VERDICT: +187.0 ± 49.7 vs classic @60+1** | **272 games, zero time losses. Target +400 NOT met — but against a classic that gained ~+130 during the campaign** |
| 2026-08-12 | `_ext` integerization: scoped and priced | DONT BUILD (SWAR tail 5.2-10.3µs vs 3.8µs now) — but a dead-code third removed: rehab800 0.643 → 0.742× kb8, +21 Elo |
| 2026-08-12 | **Quality-term hunt restarted: labels + 3 new families** | IN FLIGHT — 28-pair fixed-node label RR running locally; metric C shows the first honest signal (churn ranks kbbil worst, w256 best) |
| 2026-08-12 | **H2 paired form (the honest successor)** | **FAILS validation — sign flips across labeled pairs; H2 is closed, quality is fixed-node games only** |
| 2026-08-12 | H2 optimism bias, controls | DEAD in simple form — every net is an optimist on its own losses (kb8 +105 worst) |
| 2026-08-12 | krff gates (256×kb8×rff64) | PASS all — val 0.00729, shape 0.53%, **nps 0.991× — rff is free at width** |
| 2026-08-12 | History heuristic removal | REMOVED — sound history measures 1.01 node ratio; the −49% was the bug |
| 2026-08-12 | Timeval (shared driver) | **+91.1 ± 50.7 @60+1**, +45.9 ± 46.8 @30+1, zero time losses |
| 2026-08-12 | KCX port screen | −15.7 ± 34.9 over 200g — holds, no regression |
| 2026-08-12 | History futility-break bug | FIXED — −449 Elo was a search-soundness bug, not a regression |
| 2026-08-12 | 256ng flagship | Best val ever (0.00678) but 0.553× speed → model says unwinnable, NO GAMES |
| 2026-08-12 | Ext tax profile | Float `_mlp` tail = 47% of `_ext`; rff lanes are the affordable nonlinearity |
| 2026-08-12 | Capped-null decision match | −10.4 ± 23.3 over 300g — statistically flat, no Elo case either way |
| 2026-08-11 | **Why nets lose (root cause)** | **SPEED IS ELO: ~100 Elo/doubling; speed-only predicts both collapses within 1.3 Elo** |
| 2026-08-11 | 200M + satpen | val 0.00740 FAILS gate, but shape 0.27% — 10× cleaner than the incumbent |
| 2026-08-11 | Tuning frontier under cp-loss | Agreement axis LIED; QS=40/ER=10 and QS=80/ER=10 are the true dominators |
| 2026-08-11 | Search constants, offline sweep | Classic-era defaults dominated; QS_A is a dead axis |
| 2026-08-11 | kb16r composition | val 0.00740 PASS — best 128-wide ext number; flagship launched |
| 2026-08-11 | Compensation oversampling | FAILS informatively — the class is representation-limited, not data-limited |
| 2026-08-11 | rff pre-play gates | PASS all — shape 0.40%, the best of any gated net |
| 2026-08-11 | Phase-sketch / RFF | val 0.00765 (−3.9%) — Thomas's multiplicative idea works in unitary form |
| 2026-08-11 | Width screen | **256kb8@100M +52.5 ± 43.6 over kb8@128** — width converts in play |
| 2026-08-11 | King-capacity solos | kb16 pays (−1.5%); bilinear m=8 ties out — fold stays m=4 |
| 2026-08-11 | 200M val record | val 0.00717 but shapecheck FAIL 2.73% — the clamp is filling up |
| 2026-08-11 | rehab800 | val 0.00753, shape clean — the wider band buys the val back |
| 2026-08-11 | Material-base attribution | mat costs 0.0016 val vs its true twin; lane closed at this scale |
| 2026-08-11 | Material-base A/B | FAILS gate (0.00812 vs 0.00800) — honest negative |
| 2026-08-11 | Rehabilitation (rehab600) | val 0.00760, saturation 4.93% → 0.00% — pathology eliminated |
| 2026-08-11 | TCEC-4k field ladder | v2 21.5% vs molly (−225 ± 65); classic 10.5% (−372 ± 91); zero time losses |
| 2026-08-11 | Decision RR + kbbil collapse | v2/kb4 ≈ +200 over classic; **kbbil best-val net collapses −118** |
| 2026-08-10 | Extension generation | Every prototype passed its gate; the odd tail is what makes bilinear pay |
| 2026-08-10 | v2 + kb4 training | kb4 takes the val gate (0.00825 vs 0.00875) |
| 2026-08-09 | Bilinear head pricing | Affordable at cropped width; the obvious read-out is rank-1 (fold mod 2^16m−1) |
| 2026-08-09 | Multiply-and-split | DECLINED on price before loss was reached |
| 2026-08-09 | Width sweep + k=3 activation | Width 128 chosen; 3-segment activation declined (16% node time for 0.5% loss) |
| 2026-08-09 | Packed convolution | CLOSED — layer-2 cascade costs 2-40 nodes per node |

---

## 2026-08-12 — The 4k design space, priced: weights are RAW, and width is nearly free

Two premise corrections applied before pricing. **(1) The weight blob is appended
RAW, not compressed** — the historical packer xz's the engine *source* and
concatenates the model untouched, so for the blob what matters is bit-packing and
parameter count, never entropy; only the engine source benefits from
compressibility and from ice4's lzma parameter search. **(2) numpy is permitted
by the TCEC rules**, so arbitrary load-time expansion is free in bytes. Local
note, not a rules matter: our pypy3 has no numpy on either the laptop or the
bench box, so any numpy-using build needs it installed for pypy or must fall
back to CPython. Measured, that fallback costs **83552 nps (pypy3) vs 39424 nps
(CPython) = 2.12×, about −110 Elo** — painful but survivable, and much less than
the order of magnitude I expected, because this engine is big-integer heavy and
CPython's bigint operations are C-level too. Load-time-only numpy is still the
safe form: import it, expand into the packed rows, never touch it in search.

### Width is nearly free at 4k scale — my earlier arithmetic was wrong

`nn_cp` costs 6.43 µs at width 256 (512 lanes), i.e. **0.025 µs per lane**, while
board mechanics run ~20 µs per node. So at the widths a 4k net can afford:

| width | nn_cp | node | vs width 5 | speed Elo |
|---|---|---|---|---|
| 5 | 0.13 µs | 20.1 µs | 1.000× | 0 |
| 25 | 0.63 | 20.6 | 0.976× | −3 |
| 50 | 1.26 | 21.3 | 0.947× | −8 |
| 100 | 2.51 | 22.5 | 0.894× | −16 |
| 256 | 6.43 | 26.4 | 0.761× | −40 |

I previously priced width-64 at "−368 Elo" by scaling the *whole node* with
width. That was wrong: only the eval scales, and it is a small share at these
sizes. **Correct conclusion: below width ~50 the byte budget is the binding
constraint, not speed** — so the design should spend everything on parameters
per byte and stop worrying about width.

### The ten ideas, priced in RAW bytes at Thomas's 1920-byte budget

Baseline: 384 features (6 pieces × 64 squares) × 5 hidden × int8 = 1920 B.

| scheme | width at 1920 B | free params | note |
|---|---|---|---|
| dense int8 | 5 | 1,920 | **the baseline** |
| dense int4 (2/byte) | 10 | 3,840 | trivial to implement |
| mirror-folded dense int8 (32 files) | 10 | 3,840 | pure symmetry win |
| factorised rank-12 int8 | 16 | 1,920 | rank-limited |
| rank+file (4ku style) | 20 | 7,680 | no full 64-sq table |
| **dense ternary, base-3 packed** | **25** | **9,600** | 5 values/byte, 3⁵=243<256 |
| shared 8-basis int8 | 29 | 1,904 | 8 spatial bases + coeffs |
| DCT top-10 per table | 32 | 1,920 | smooth-PST prior |
| factorised rank-6 int8 | 42 | 1,896 | the historical scheme |
| **mirror + ternary** | **50** | **19,200** | symmetry × packing, stacks |
| factorised rank-6 ternary | 256 | 9,600 | rank 6 caps real capacity |

Two honest caveats on that table. The factorised rows buy *width* but not
independent capacity — a rank-6 scheme constrains the 384×W matrix to six
spatial patterns however wide it gets, so "width 256" there is not comparable to
a dense width 256; its value is extra hidden units with independent clamps, not
extra spatial resolution. And the "free params" column counts stored numbers, not
expressiveness.

The clear winners are the ones that reduce **bits per weight** and exploit
**symmetry**, because both raise parameters *and* keep full rank: ternary base-3
packing gives 5 values/byte deterministically (no compressor in the loop), and
folding the board about the king's file halves the table. Together they are
**19,200 parameters at width 50 in the same 1920 bytes — 10× the baseline's
parameters** at a measured −8 Elo speed cost.

### Seeded random projection: tested first, and it does not dominate

The cheap decisive test — how well can a *fixed random* basis represent the
trained embedding, against the optimal learned basis of the same width (SVD,
fraction of unexplained variance):

| width | learned | random | ratio |
|---|---|---|---|
| 4 | 0.405 | 0.994 | 2.5× |
| 16 | 0.257 | 0.980 | 3.8× |
| 64 | 0.084 | 0.917 | 10.9× |
| 256 | 0.000 | 0.670 | — |

A random basis needs roughly **100× the width** for comparable fidelity: random
at 256 is still worse than learned at 4. The input space is sparse
piece-square indicators, and Johnson-Lindenstrauss preserves distances, not the
specific structure a learned basis captures. It stores zero basis bytes, and
width is cheap, so it is not *absurd* — but at equal width it is far worse than
learned, and its byte advantage is beaten outright by ternary+mirror, which is
full-rank and learned.

Where seeded random features **have** already earned their place in this project
is as an *addition* rather than a replacement: the rff work (random Fourier
features, a seeded random projection with a cosine read-out) produced the largest
single-feature val gain we ever measured (−3.9%) and krff runs at 0.991× speed.
That is the correct role for idea (a) — free extra nonlinear width on top of a
learned core, not a substitute for it.

### What to build

1. **Ternary base-3 packing + king-file mirror**, dense and learned, trained with
   a ternary-aware scheme (straight-through estimator, per-row scale). Target
   width 40-50 at ~1900 B. This is the option that beats the width-5 baseline by
   10× in parameters with a −8 Elo speed cost and no rank ceiling.
2. **Distillation from the big net as teacher** (idea h) — orthogonal, stacks
   with any representation, and the 14.9 MB net is a far better target than raw
   labels for a model this small.
3. **rff lanes on top** if bytes remain, since they are free-width and already
   validated here.

Deferred with reasons: compression-aware training (moot — the blob is raw);
low-rank/tensor decompositions (rank ceiling, and SVD of a dense net measured
badly); DCT and shared-basis (dominated by ternary+mirror on parameters, worth
revisiting only if training shows the smooth prior helps); feature hashing and
codebooks (real but second-order once bits-per-weight is already minimal).

Still to measure before building: joint xz of engine+weights versus the split
scheme, which the historical packer chose without recording a comparison.

## 2026-08-12 — The 4k budget, re-derived: the net counts, and the mechanism already existed

**Premise correction (Thomas, via b267a19): the net counts toward the 4096
bytes.** The README's claim that nets are external and unbudgeted was, in his
words, cope. Under TCEC 4k the entry is one file ≤ 4096 bytes and the evaluation
data is part of it — which is precisely why the division is hard.

### What the artifact actually weighs

Built with the recovered two-part packer (below), current engine + our
*smallest* net:

| part | bytes |
|---|---|
| self-extracting head | 141 |
| engine, minified + xz | 4488 (3724 with the ext machinery stripped, as `pack.sh` does) |
| net (net128v2, the smallest we have) | 537,152 |
| **total** | **541,781** |

Against 4096 that is **132× over** with the smallest net and **1830× over** with
the shipped 7.5 MB kb8 net. "3798 bytes, 298 under budget" measured the engine
only — and that engine cannot even evaluate on its own, since the piece-square
tables now live in the net file. The tables-in-net migration therefore saved
**nothing**: it moved ~600 counted bytes from one counted place to another.

Working budget with today's engine: 4096 − 141 (head) − 3724 (engine) = **231
bytes for all evaluation data.** The engine has to shrink by ~1000-1500 bytes
before a net of useful size can exist.

For reference, **classic packs to 3234 bytes including its piece-square tables**
— the engine branded 4k is the one that does not fit; the one not branded 4k
does.

### The packing mechanism, recovered and re-verified

`build/pack_nnue.sh` @ `0c0a33a` (verified present) xz's the minified engine,
appends the net **raw**, and splits them in a self-extracting head:

    tail -c +130 "$0" | head -c 2672 | xz -d > $T    # engine
    tail -c 1207 "$0" > $M                            # net
    pypy3 -u $T $M

Its committed artifact `build/sunfish_nnue.sh` is **4008 bytes = 129 head + 2672
engine + 1207 net** — verified by `git ls-tree`. I rebuilt the same shape
against today's toolchain (`SF_NET=$M` instead of argv) and **it builds and
runs**: `uci` → `uciok` → `readyok` → `bestmove g1f3`. So the mechanism is not
speculative; only the sizes are wrong.

### What 1207 bytes bought (the existence proof)

`models/color2.pickle` decoded: `{"ars": 6 int8 arrays, "scale": float}`, 1140
bytes of payload. The engine at that commit expands them:

    pst = np.einsum("ocp,sc->pso", nn[1].reshape(L1,6,6), nn[0].reshape(64,6))

- `nn[0]` = 64×6 square embedding (384 int8)
- `nn[1]` = 12×6×6 piece mixer (432 int8)
- product = 6 pieces × 64 squares × 12 outputs = **4608 values from 816 bytes**

then a small MLP (24→21→14→1) on the accumulated features. This is a **trained
rank-6 factorisation**, and it is exact by construction — the model *is* the
factorisation, so there is no approximation error. That distinction matters:
SVD-ing our current dense 768×128 matrix to rank 8 costs 2077 bytes for a 0.52
mean relative error, while training the factorised form directly costs 816 bytes
and no error at all. **Approximating a big net is the wrong move; training a
small structured one is the right one.**

### The frontier, measured on real trained weights

Bit-packed then xz -9e (pb=0), from the kb8 float export's plain 768-feature
slice:

| width | 8-bit | 4-bit | 2-bit |
|---|---|---|---|
| 128 | 94461 | 45692 | 14980 |
| 32 | 23803 | 11600 | 3967 |
| 16 | 11921 | 5813 | **2043** |
| 8 | 6050 | 2971 | **1092** |
| 4 | 3046 | 1498 | 558 |

Feature-count reduction at 8 hidden, 4-bit: 768 features 2971 B → 384 features
1486 B → 192 (piece × 32, file-mirrored) **754 B**. Low-rank SVD of the dense
matrix is dominated everywhere (rank 8 = 2077 B at 0.52 error, worse than simply
training 768×8 at 2-bit for 1092 B).

So in a realistic ~1400-byte net budget the dense options are 768×8 @2-bit
(1092 B) or 384×16 @2-bit (~1000 B) — while the *factorised* option buys a
768×12-equivalent table for ~816 B, and that is before quantising the factors
below int8.

### Field study: how ice4 and 4ku fit an eval in a few hundred bytes

**ice4** (MinusKelvin, Rust/C++ hybrid; read from source). Its **entire
evaluation is one string literal**:

    #define DATA_STRING L"7QM862- :G<851&\";CLIG;-&AMVWPA<.MUwfb]I:&!E[P>..."
    #define EG_OFFSET 166
    int get_data(int i) { return data[i] + 0x10000 * data[i+EG_OFFSET] - S(32,32); }

**333 characters total = 166 midgame + 167 endgame parameters**, one character
per parameter, biased by 32 into printable range, midgame and endgame in two
halves of the same string, combined into a packed `S(mg,eg)` int. Zero syntax
overhead — compare a Python list literal at ~4 bytes per value before
compression. Their PSTs are assembled procedurally in `init_tables()` from
rank/file components rather than stored as 64-square tables.

**4ku** (kz04px): same packing idea, `S(mg,eg) = (eg<<16) + mg`, and the PST is
explicitly **decomposed into `pst_rank[48]` + `pst_file[48]`** — 96 values
instead of 6×64 = 384. Terms: material[6], the rank/file PSTs, mobilities[5],
king_attacks[4], passers[4], pawn protection/threat/doubled, bishop pair.
Bitboards throughout, with the usual modern search (aspiration windows, NMP with
a static-eval-scaled reduction, LMR, both futility directions, TT with
upper/lower/exact flags).

**The unifying trick both use, and the historical sunfish net also used:
factorise the piece-square table.** Rank+file decomposition (4ku, ice4) is
literally rank-1-plus-rank-1; the old sunfish NNUE used a learned rank-6 latent.
None of them stores a full 6×64 table, let alone 768×128.

One free byte win adopted from ice4's `compress.sh`, which brute-forces 1350
lzma parameter combinations and keeps the best: swept `lc`/`mf`/`nice` over our
minified engine and found **4 bytes** (`lc=3,mf=hc4,nice=64` vs our fixed
`preset=9e,pb=0`). Small because `pb=0` was already the dominant choice, but
free and it applies to classic too.

### The research problem, priced

Target: strongest eval in ~1200-1600 bytes, with the engine cut to ~2400-2700.
The bigint accumulator is retained — factors expand into the packed rows **at
load time**, so the artifact ships ~1 KB of factors while the search still gets
the fast packed accumulator. That is the synthesis: 2026-era bigint speed with
2023-era packing discipline.

| option | bytes | predicted strength | note |
|---|---|---|---|
| trained rank-6 factorisation, 768→12 (historical shape) | ~816 | the floor to beat — that artifact was weak | exact, proven to fit |
| rank-8/12 factorisation, 768→16-24, int8 factors | ~1100-1600 | best candidate: more capacity than the floor, still fits | needs a trainer change |
| dense 768×8 @ 2-bit | 1092 | fewer effective params than the factorised form at equal bytes | no trainer change |
| dense 384×16 @ 2-bit (colour-shared rows) | ~1000 | as above with feature sharing | cheap to try |
| rank+file PST, 4ku style, hand-tuned | ~200-400 | proven strong in this division, but abandons the NNUE thesis | fallback |
| SVD of the current dense net | 2077 @ rank 8 | dominated — 0.52 rel error | **rejected, measured** |

Engine-side, the ~1000-1500 bytes must come from: the sfnn loader (JSON + base64
is expensive; raw int8 appended after the payload needs only `int.from_bytes`),
the tables (regenerated from factors instead of stored), and a feature-by-feature
re-examination now that the budget is real — the KCX port cost +62 bytes and the
history heuristic already came out for being worthless.

**Nothing here changes the lichess bot**, which has no size limit and correctly
keeps the 14.9 MB net; the `nn_cp` and search findings continue to serve it.

## 2026-08-12 — CORRECTION: the bottleneck is `nn_cp`, and the mutable board is a ~+15 item

**The entry below this one is wrong and I am correcting it before anyone builds
on it.** It claimed ~85% of a node is board mechanics and ~15% is the network,
and priced a mutable board at +71…+110 Elo. The inference was bad: I measured
that widening 128→256 moves `move()` by only 3.05 µs and read that as "the
network costs 3 µs". That is the **marginal** cost of doubling the width, not
the **total** cost of the network. A 128-wide net still pays a full output
layer.

Measured directly (pypy, 40 middlegame positions, w256 — the play king),
component by component inside a 14.56 µs `move()`:

| component | µs | kind |
|---|---|---|
| **`nn_cp` (packed head: SWAR clamp + 2 modular hsums)** | **6.43** | network |
| accumulator delta (4 big-int row adds) | 1.68 | network |
| `board[::-1].swapcase()` (the always-white rotate) | 1.88 | board |
| `Position(...)` namedtuple construction | 0.64 | board |
| `put` splice ×3 | ~0.34 | board |
| `value(move)` | 0.045 | board |
| `hash(pos)` — the TT key | 0.409 | TT |
| …of which `hash(acc)` alone | 0.504* | TT |
| `hash(board)` alone | 0.129 | TT |
| `eq(pos, child)` | 0.332 | TT |

\* measured separately, so it exceeds the whole-tuple figure — namedtuple
hashing short-circuits on the fields it reaches first; the point stands that the
accumulator is the expensive field in the key.

**So the network is ~8.1 µs (≈55% of `move()`) and board mechanics are ~2.9 µs
(≈14%).** Against a full node of roughly `move` + `gen_moves` + `value` ≈ 23 µs,
a perfect mutable board removes at most the 2.86 µs of rotate + namedtuple +
splices, and gives some of it back as make/unmake bookkeeping. That is ~10-12%,
i.e. **+15 Elo, not +71…+110**.

The honest consequence: **the mutable board was approved on the strength of a
number I got wrong.** It is still positive, and "anything goes" still licenses
it, but it is now a high-effort/high-risk item worth about as much as the
one-line `_ext` fix that already landed (+21), and less than the search-constant
RR now running (+30…+68). It should not be the next thing built.

**The real target is `nn_cp` at 6.43 µs** — 28% of a node, paid on every single
position created. It is ~22 sequential big-integer operations on 8192-bit ints
(AND, shift, ×`ONES`, OR, subtract, two `% M16` reductions). `ONES` is
`2^15 − 1`, a *small* constant, so these are linear-time big×small operations,
not the n^1.7 multiplies that killed multiply-and-split — the cost is the
op *count* at that width. Two concrete leads, both cheap to test and both
bit-exactness-checkable: (1) fuse the two `% M16` reductions into one by
differencing the blocks first and recovering the sign from the residue, since
|lane-sum difference| < M16; (2) drop one mask construction by re-deriving the
cap mask from the relu mask. Each big-int op removed is ~0.3 µs ≈ +4 Elo.

Also worth its own small experiment: the TT key hashes the accumulator, which is
**derived** from the board and therefore redundant for identity — `hash(pos)`
0.409 µs against `hash(board)` 0.129 µs, on every probe and store.

### Archaeology of the previous attempt (verified, not repeated)

`0622039` / `86141a6` on `nnue-mutable-board` (2026-08-05) rewrote
`Position.move`/`rotate` as `@contextmanager`s. The interface was right; the
body was the problem, and it is worth quoting because it inverts the whole
point:

    orig_board = self.board
    orig_wf = self.wf.copy(); orig_bf = self.bf.copy()
    board_list = list(self.board)
    wf = self.wf.copy(); bf = self.bf.copy()

A full board copy **plus four feature-vector copies per node**, restored from
the snapshot afterwards — strictly *more* allocation than the immutable version
it replaced, with contextlib generator overhead on top. It bought the syntax and
none of the speed. `dc6c554` ("Fix crash on black-to-move FEN positions — rotate
is a context manager") shows the shape also leaked into callers that assumed
value semantics.

If it is ever built, the design constraints are now known: true make/unmake by
inverse operation (touch only the 2-4 squares moved, subtract exactly the deltas
added, no copies on the hot path); hand-written `__enter__`/`__exit__` rather
than contextlib, measured against an explicit make/unmake variant as the speed
ceiling; every caller holding a `Position` across a move enumerated first (the
driver's `hist` list above all), because make/unmake ends value semantics; and
**perft before the node-identity bench**, since a mutable board that mutates
wrong fails perft instantly. Note also that the TT keys on `Position` — with
mutation, that key must become an explicit incremental hash, which is a second
substantial piece of work the +15 has to pay for.

## 2026-08-12 — Hot-path profile (the "85% board" claim here is WRONG — see the correction above)

The `_ext` audit found a third of that path was dead work nobody had timed. The
**main** path — the one every net pays, including the play king — had never had
the same treatment. Measured on pypy over 40 real middlegame positions
(28.6 moves/position mean), for three nets spanning the accumulator size range:

| component | v2 (128, B=1) | kb8 (128, B=8) | w256 (256, B=8) |
|---|---|---|---|
| `Position.move` (one move) | 11.51 µs | 12.85 µs | **14.56 µs** |
| `gen_moves` (whole list) | 6.55 | 6.95 | 6.85 |
| `value` (all moves) | 1.93 | 1.97 | 1.84 |
| `score` (attribute read) | — | — | 0.13 |
| `rotate` | — | — | 1.91 |

**The accumulator is not the cost.** Widening 128→256 with the same bucket
scheme moves `move()` by only 3.05 µs, so at 256 width the NNUE update is ~21%
of `move()` and at 128 it is ~10%; the *packed head read* is 0.13 µs, i.e.
free. Everything else — ~11.5 µs of `move()`, all 6.9 µs of `gen_moves`, all
1.9 µs of `value` — is board-string splicing, namedtuple construction and
castling/ep bookkeeping, identical for every net.

Rough per-node arithmetic: **~85% of the engine's time is Python board
manipulation and ~15% is the neural network.** Years of this lane's effort have
gone into making the 15% cheaper (SWAR clamps, folded weights, fused loaders)
while the 85% went unmeasured.

*(Superseded: the ~85%/15% split below rests on the marginal-vs-total error
corrected in the entry above. Measured properly, the network is ~55% of `move()`
and board mechanics ~14%, so the figures in this paragraph are wrong by roughly
5×. Kept for the record.)*

This reframes the remaining Elo gap. At the measured ~100 Elo per speed
doubling: halving board cost is ≈1.64× overall → **+71 Elo**; a 3× board
speedup is **+110 Elo**. Nothing in the eval column can offer that — and it
would apply to classic identically, which is either a shared win or a reason
the relative number moves less than the absolute one (both engines share
`gen_moves`/`move`; the packed side pays the extra accumulator, so speeding the
shared part helps *classic slightly more* in relative terms). That caveat is
exactly why it needs measuring rather than assuming.

Not proposed blindly: a `nnue-mutable-board` branch exists in this repo's
history, so the idea has been touched before; the numbers above are the first
time its value has been quantified. Costing and design belong in their own
entry before any code moves.

## 2026-08-12 — GOAL-LINE VERDICT: +187.0 ± 49.7 over classic at 60+1

The +400 campaign's scoreboard match: the play king (256kb8@100M) on the current
engine vs current-master classic, 60+1, both sides on the same `sunfish_ui`
driver so the time-formula gain cancels and this measures engine+eval only.

**Final: +187.01 ± 49.65 Elo (nElo +201.80 ± 43.96), 272 games, zero time
losses.** Stopped early at 272/400 by coordinator decision: the estimate had
converged upward (157 → 179 → 187) while the interval tightened (±89 → ±59 →
±50), so the remaining 139 games would have sharpened a settled conclusion while
five queued experiments waited.

**The target is +400 and we measure +187. Both of these are true, and the second
number is measured against a moving baseline.** During this campaign classic
absorbed the killer depth gate (+42 there), the capped-null work, and the same
new time formula this lane validated (+91 at 60+1, +46 at 30+1) — so today's
+187 is over a materially stronger classic than the one the target was set
against. Against the classic of the goal's origin the same engine would measure
substantially higher; against today's classic it measures +187. The gap is
closing from both ends, which is good for the engine and inconvenient for the
scoreboard.

Where the missing ~213 can and cannot come from, on this lane's own evidence:

**Dead or mined out.** Width converted once (+52.5) and will not again at this
size — 512 would cost more speed than its val could repay. Material base
(−0.0016 val), compensation oversampling (representation-limited, not
data-limited), dense L2 heads, the k=3 activation, multiply-and-split, packed
convolution, and the history heuristic are all closed with numbers. The val
ladder itself has flattened: the record net (256ng, 0.00678) is speed-blocked
and benched.

**Live, with predicted Elo and cost:**

| # | Item | Predicted | Cost | Status |
|---|---|---|---|---|
| 1 | Search constants (QS/ER) | **+30…+68** — offline says −22%…−37% nodes at equal cp-loss, which at 100 Elo/doubling is a 1.3-1.6× effective speedup | zero, already built | RR running now |
| 2 | **Board representation** | **+15** (corrected; the +50…+110 was the marginal-vs-total error) | high: semantic port, exactness ladder, shared with classic and with the formal model | proposal, needs its own costing |
| 3 | krff play screen | 0…+20 — val 0.00729 ≈ w256's 0.00731, but at 0.991× speed instead of a tax | ~2h box | queued |
| 4 | `_ext` dead-code fix | **+21 to every bilinear-family net** (0.643 → 0.742×) | done | landed 4810a5a |
| 5 | King-safety features | +20…+40 if it converts — the diagnosed weakness (compensation-class loss runs ~5× overall) that no arch change has yet addressed; an incrementally-maintainable pawn-shield/king-ring plane is the cheapest form | trainer + engine + one training run | proposal |
| 6 | Validated quality metric | indirect — makes every future candidate triageable without a 200-game screen | in flight | labels accumulating |

The honest shape of it: no single remaining item is worth +213. Items 1 and 2
are the only ones with three-digit potential, and both are *speed*, not eval —
which is consistent with everything this lane has measured since the speed model
landed.

## 2026-08-12 — Goal-line 60+1, interim read (superseded)

Recorded at 240/400 games: +187.0 ± 49.7, zero time losses. Superseded by the
final verdict entry above, which it agrees with to the decimal — the last 32
games moved nothing, which is itself the evidence that stopping early cost
no information.

## 2026-08-12 — `_ext` integerization: scoped, priced, and mostly declined

The root-cause analysis named this the ext family's unlock: the extension nets
are speed-blocked, not quality-blocked, so making `_ext` cheap would let their
measured eval advantage compete. Scoped against the profile (`_ext` = 8.0µs of a
38.6µs move: float `_mlp` tail 3.8µs, `cnt` scan 2.7µs, bigint extract 0.98µs,
m² conv 0.56µs).

**1. A third of it was dead code — fixed.** `cnt` was computed *unconditionally*
at the top of `_ext`, but it is only used inside `if PHASE_S:`. rehab800 has
**zero** phase buckets, so every evaluation spent 2.7µs — 34% of `_ext` — on a
value it then discarded. Now computed only when phase exists, and via one
C-level `bd.count(".")` (64 squares minus empties) instead of a 120-step
generator. Verified **bit-identical on 1500 positions for both rehab800
(phase-less) and kbbil (phase-8, exercising the live path)**; 14 tests green,
verify battery green (18208 positions), artifact unchanged at 3798 bytes (the
ext path is minifier-hidden, so this is free).

**2. Folding the tail into a big-int multiply: priced and declined.** The
project's signature trick would put each of the 16 weight rows at its own lane
offset and get all 16 dot products from one multiply. That needs 16 rows × 9
inputs at stride 18 = 288 lanes = 4608 bits at 16-bit lanes (6912 at 24-bit),
and the measured multiply-width curve (0.123µs@512b → 13.5µs@8192b, ≈n^1.7) puts
that at **5.2µs (16-bit) to 10.3µs (24-bit) against the 3.8µs the Python loop
costs today**. Slower. This is the same wall that closed multiply-and-split and
the packed convolution: big-int multiplies only pay when the lane count is small.

**3. "Integerizing" per se is not the lever.** Under pypy the tail's cost is
288 multiply-adds and 32 `tanh` calls — loop iterations, not float boxing. Int
arithmetic would execute the same number of iterations.

**Verdict: do not build the integerization.** The one free win is landed; the
rest of the tail is irreducible in this language, and the measured answer to
"the ext family is speed-blocked" already exists in the other direction —
**krff runs at 0.991× because rff replaces bilinear+tail entirely**. The
family's future is rff-shaped, not tail-optimized.

What the cheap fix actually buys, measured (same interleaved probe, before and
after):

| | rehab800 nps | ratio vs kb8 | implied Elo hurdle |
|---|---|---|---|
| before | 51317 | 0.643 | −65 |
| after | **56050** | **0.742** | **−44** |

**+21 Elo for deleting a line of dead work.** Note the arithmetic predicted only
0.696 from the 2.7µs microbench; the measured 0.742 beats it, because dropping a
120-step generator per evaluation relieves allocation and JIT pressure beyond
the isolated cost of the loop itself — a reminder that microbench components
under-count what removing them is worth.

Still a −44 hurdle, so the fix helps and does not rescue the bilinear family;
only rff does. (Cross-machine note: the before-ratio measured **0.643 on the
laptop against 0.647 on the bench box** — independent confirmation that these
speed ratios are machine-independent, which is what makes the local fixed-node
labels valid.)

## 2026-08-12 — The quality term, restarted: why four metrics failed, and the label problem

Thomas rejected "quality = fixed-node games only" and named the root cause the
four dead metrics share. It is worth stating exactly, because it is the design
rule for everything that follows:

> **Elo depends on eval error only through the decisions it changes.** Error far
> from a decision boundary is free; error between two near-equal moves flips the
> choice. Every metric so far averaged error uniformly over a position set,
> diluting the signal that matters with error that does not — and sampled the
> wrong distributions (dump positions, own-loss positions, frozen FENs) instead
> of positions where the engine's choice is actually close.

There is a second, independent problem: **we were validating against six
pairwise labels.** Six cannot separate a real correlation from luck, which means
a good metric could already have been rejected wrongly. Labels come first.

### Workstream 1 — more labels (running)

Fixed-node results are machine-independent, so the labels do not need the bench
box (busy with Thomas's pr171 match). Running locally on the Mac with fastchess
built there: **8 nets, round-robin, 28 pairs, 20000 nodes/move, 60 games/pair =
1680 games**, openings_2k, `-recover`, concurrency 3 and niced (it is Thomas's
working laptop). The roster is v2, kb4, kb8, kbbil, rehab800, w256, msp, krff;
the pre-registered 256ng-vs-w256 test is chained behind it.

This required `go nodes N` in the driver (landed e500a9a) and turned up an
infrastructure bug worth recording: two net files had arrived **truncated** from
an interrupted copy — w256 at 5.4MB against its real 14.9MB — and one was
already in a running match. Caught by a size check against the source before it
burned an overnight run; every net is now byte-size verified and load-verified
before use. *Any pgn produced between those two events would have been silently
garbage.*

### Pre-registered predictions (the speed model bets on this RR)

Fixed-node labels have the speed term **zero by construction**, so the speed
model makes a falsifiable prediction: fixed-node ΔElo should equal the timed
ΔElo minus 102·log2(nps ratio). Written down before the games finish:

| pair | predicted fixed-node ΔElo |
|---|---|
| kb8 vs kb4 | +70.6 |
| w256 vs kb8 | +77.1 |
| kb4 vs v2 | −27.8 |
| rehab800 vs kb8 | **−1.2** (the whole −70.4 was speed) |
| kbbil vs kb8 | **−1.0** (the whole −83.2 was speed) |

If rehab800 and kbbil come out near zero at fixed nodes, the speed model is
confirmed a second independent way and the ext family's eval is vindicated. If
they come out clearly negative, the speed-only model was over-crediting speed
and the quality term is bigger than believed. Either outcome is informative.

### Workstream 2A — metric family C: search cooperation (measured)

The mechanism H4 always needed: an eval that is accurate but *jumpy between
siblings* makes MTD-bi re-probe, so the same depth costs more probes and the
engine is effectively slower even at equal nps. Measured at equal depth 5 over
60 real-game positions:

| net | nodes@d5 | probes/depth | PV flips | sibling sd |
|---|---|---|---|---|
| v2 | 1350259 | 6.22 | 0.283 | 0.557 |
| kb4 | 982170 | 6.24 | 0.290 | 0.572 |
| kb8 | 1340319 | 6.28 | 0.287 | 0.641 |
| kbbil | 1092014 | **6.44** | **0.330** | 0.628 |
| rehab800 | 961613 | 6.34 | 0.290 | 0.667 |
| w256 | 1216432 | **6.22** | **0.273** | 0.684 |
| msp | 1060932 | 6.29 | 0.290 | 0.608 |
| krff | 1126190 | 6.24 | **0.273** | 0.628 |

The churn columns order the two extremes correctly on the first try: kbbil (the
−83 collapse) has the most re-probing and the least stable PV; w256 (the +52.5
play king) has the least of both, with krff tied at the top on flips. Note
**sibling sd does NOT track play** (w256 has the highest raw jumpiness while
playing best) — so the useful quantity is what the *search* does with the eval,
not the eval's raw variance. That distinction is the whole content of "with a
mechanism".

Preliminary validation against the six speed-adjusted quality labels (timed
ΔElo minus the speed term): probes/depth LOO RMS 46.5, PV flips 48.7, sibling sd
51.5, quiet val 55.4 — all against a null of 45.7, i.e. **none clears the bar on
six labels**, though probes (+0.60) and flips (+0.66) have the right sign by
Spearman where val is *negative* (−0.26). This is exactly the resolution problem
Thomas identified; the verdict waits for the 28-pair set.

### Workstream 2B — decision-margin-weighted regret (measured)

SF multipv=2 at depth 12 over 400 real-game positions; engine choices at fixed
depth 4; cp-loss restricted to positions where the oracle's top two moves are
within the margin. Sensitivity sweep, so "restricting helps" is testable:

| net | ≤15cp | ≤30cp | ≤60cp | ≤120cp | all |
|---|---|---|---|---|---|
| v2 | 24.7 | 28.8 | 35.5 | 42.6 | 46.7 |
| kb4 | 23.4 | 27.5 | 32.6 | 39.6 | 44.3 |
| kb8 | **21.3** | **25.1** | 31.7 | 39.7 | 43.8 |
| kbbil | 25.3 | 29.3 | 37.6 | 44.3 | 47.3 |
| rehab800 | 25.0 | 31.0 | 36.2 | 41.7 | 44.5 |
| w256 | 21.4 | 28.4 | 32.0 | 37.4 | **38.7** |
| msp | **19.1** | **23.5** | **27.9** | **35.7** | 39.4 |
| krff | 23.1 | 26.8 | 33.4 | 38.8 | 43.1 |
| *n positions* | 173 | 225 | 289 | 340 | 400 |

The restricted columns put kb8/w256/msp at the top and kbbil at the bottom,
which is the play order — but note the sweep does **not** yet show restriction
helping: on the six-label preliminary, Spearman is +0.20 at ≤15cp against +0.43
unrestricted. On six labels that comparison is not worth much either way; it is
recorded so the 15-pair rerun can confirm or kill the margin hypothesis itself.

**A useful tension to settle:** family B says rehab800 (25.0) and kbbil (25.3)
are clearly *worse in quality* than kb8 (21.3), while the speed-adjusted labels
say both are within ~1 Elo of kb8 once speed is removed. Both cannot be right.
The fixed-node RR is the referee: if rehab/kbbil land near zero, B is measuring
something that does not reach play; if they land clearly negative, the
speed-only model has been over-crediting speed and the quality term is real and
large. This is the most informative single thing the running matches will decide.

### Workstream 2A — outcome calibration: confounded as computed

| net | K_own | logloss@K_own | Brier | logloss@K_shared(233) | n |
|---|---|---|---|---|---|
| v2 | 169 | **0.52944** | 0.13058 | 0.53791 | 4000 |
| kb4 | 246 | 0.59143 | 0.10982 | 0.59168 | 4000 |
| kb8 | 258 | **0.60855** | 0.15378 | 0.60928 | 4000 |
| kbbil | 258 | 0.58094 | 0.12312 | 0.58179 | 4000 |
| rehab800 | 223 | 0.56777 | 0.12942 | 0.56791 | 4000 |
| w256 | 254 | 0.59557 | 0.14857 | 0.59613 | 4000 |

This ranks v2 **best** and kb8 **worst** — anti-correlated with play — and the
reason is a confound in the *position sets*, not in the idea: each net's games
come from a different screen, so the opponent differs. v2's pgn is against
classic (a far weaker opponent, so outcomes are lopsided and easy to predict);
kb8's is against w256 (a near-equal opponent, so outcomes are near coin-flips
and logloss is necessarily high). Calibration measured on head-to-head screens
mostly measures **opponent parity**.

The fix is already running rather than hypothetical: the local RR has a uniform
opponent mix by construction, so A must be recomputed on its pgn before it is
judged. No verdict on A until then.

### Status and honest sizing note

All three families are computed; none is validated. On the six speed-adjusted
labels every family — and quiet val as control — fails to beat the null RMS of
45.7 (probes/depth 46.5, B-unrestricted 47.1, B≤15cp 48.1, PV flips 48.7,
sibling sd 51.5, val 55.4). That is the resolution problem, not a set of
verdicts.

Sizing reality, recorded because it constrains everything: at 20000 nodes/move
these engines take ~90s per game on the laptop, so the original 8-net/28-pair
plan needed ~42 hours — too much for Thomas's working machine. Re-scoped to the
**six nets that already have timed labels** (15 pairs, still 2.5× the old label
count), which doubles the games-per-pair rate and lets the same run test the
speed model directly. fastchess cycles pairs evenly, so partial results stay
balanced and labels can be harvested at any moment; the run simply accumulates
precision until stopped. H3 remains unstarted.

## 2026-08-12 — H2 paired form: fails validation, and H2 is closed

The successor to the dead unpaired form: `bias_A − bias_B` on identical
positions, candidate vs incumbent, on the candidate's own-game positions, with
the reverse pairing as the tautology control.

First, a structural simplification worth recording — in the paired form the
oracle **cancels exactly**:

    bias_A − bias_B = (score_A − SF) − (score_B − SF) = score_A − score_B

No Stockfish is needed at all, and since both nets carry the same pst base the
difference isolates the net residual. The metric is free to compute.

Results (head-to-head games only; `D_X` = mean(score_X − score_other) over
positions from games X lost, X to move):

| pair | D_candidate | D_incumbent | asymmetry | measured play ΔElo |
|---|---|---|---|---|
| rehab800 vs kb8 | +86.9 ± 1.8 (n=3688) | +119.8 ± 2.6 (n=2208) | **−32.9** | −70.4 |
| kbbil vs kb8 | +94.8 ± 2.0 (n=3846) | +125.0 ± 2.7 (n=2222) | **−30.2** | −83.2 |
| kb8 vs 256kb8@100M | +122.9 ± 1.7 (n=3771) | +68.4 ± 2.0 (n=2637) | **+54.6** | −52.5 |

**Neutral control** (same net pairs, mean score difference over the 1500 frozen
non-selected shapecheck FENs): rehab −3.2 ± 2.9, kbbil −3.9 ± 3.3, w256 −4.0 ±
2.6 cp. All zero within noise. So the +68…+125 numbers above are **entirely a
selection effect**, not a per-net eval offset — the selection effect is real and
enormous, and it points the same way for both members of every pair, which is
the tautology in quantified form.

**The asymmetry fails to rank play.** In the first two pairs the worse net has
the *smaller* own-loss optimism (asymmetry negative); in the third the worse net
has the *larger* (asymmetry positive). The sign flips while the label's sign
does not — 2-of-3 with opposite thirds is not a predictor at n=3.

The cleanest statement of the failure: `D_kb8` is **+119.8 / +125.0 / +122.9**
against three different opponents — essentially constant — while kb8's play
result against those same opponents varies from clearly winning (vs rehab, vs
kbbil) to clearly losing (vs w256). A quantity that stays fixed while the label
moves carries no information about the label.

**Consequence, stated plainly: H2 is closed.** Both forms of the offline
optimism metric are dead, joining timed cp-loss, agreement, and quiet val on the
list of offline proxies that do not measure play quality. Quality is measured by
**fixed-node games (H1)** and nothing else this lane has built. H3 (the
loss-function change) is NOT sketched: it was conditional on H2 validating, and
designing a training objective around a signal that failed its own validation is
exactly the mistake this ledger exists to prevent. If a quality-side training
lever is wanted, it must be derived from fixed-node game outcomes — which is
what the H1 battery will produce.

Cost of the whole H2 arc: ~2 hours of box time, no games. That is the argument
for cheap offline instruments even when they fail — this one cost less than a
single 200-game screen and removed a whole class of hypotheses.

## 2026-08-12 — H2 optimism bias: the simple form dies on its controls

On-policy signed bias (net's eval minus SF depth-12, mover POV, on positions
from the net's **own** games, 143-150 positions per cell):

| net | own-loss games | own-win games |
|---|---|---|
| kbbil | +41.5 ± 18.8 | −95.4 |
| rehab800 | +37.5 ± 14.4 | −102.2 |
| kb8 (control) | **+105.2 ± 15.1** | −86.1 |
| 256kb8@100M (control) | +60.9 ± 15.6 | −91.6 |

Every net is an optimist in the games it lost and a pessimist in the games it
won — and the play-BEST net shows the LARGEST optimism on its losses. Mean
bias on own-loss positions cannot rank nets: it is close to tautological (the
games you lose are the games you misjudged, whoever you are) plus
opponent-conditioning (kb8's losses came against the stronger w256).

What actually discriminated remains the **paired** design: two nets evaluating
the SAME positions — on rehab's lost positions rehab read +51 where kb8 read
−35, an 86cp relative gap on identical inputs. Next form: `bias_A − bias_B`
paired per position, candidate vs incumbent, on the candidate's own-game
positions, with the reverse pairing as control. H3 (loss-function change)
waits for a validated H2 target rather than chasing a tautology.

## 2026-08-12 — krff: rff is free at width

TRAIN256KRFF (256 × kb8 × rff64 × satpen @100M — the first *model-designed*
training, all-integer nonlinearity, no bilinear/tail/phase) cleared its whole
pre-registered gate ladder: val **0.00729** (gate < 0.00731), shapecheck
**0.53%** (incumbent 2.53%, near the cliff), and the decisive number —
**nps 0.991× of w256**.

The rff angle lanes cost 0.892× on a 128-wide net and are *free* at 256: the
fixed rff work amortizes against the wider base update. The
affordable-nonlinearity thesis lands — this is the ext research line's first
candidate with no speed tax at all. With speed parity, fixed-node ΔElo equals
timed ΔElo, so its screen rides the H1 protocol (200g @20k nodes vs w256).

## 2026-08-12 — The history heuristic is removed: sound, it measures worthless

Follow-up to the soundness fix below. With the frontier order restored, the
history table's true contribution: **node ratio 1.01** at completed depth 7
over 30 real-game positions, score gaps median +0 (p10 −2, p90 +3), 2/30 move
choices differ. The −49%/−50% fixed-depth node reduction that justified
landing it was the unsound futility break discarding real work, not ordering
skill.

Caveat: measured at depths 6-7; no evidence either way at depth 9+. But a
heuristic that cannot show value at the depths we actually reach does not keep
bytes. Removal took the artifact to **3798 bytes** (298 under budget).

The litmus finding stands unrevised: `value()` ordering has real headroom (SF
best at median rank 8). Exploiting it needs a mechanism whose soundness is
argued at *every consumer of iteration order* — the killer/tp precedent
covered re-admission; the futility break was the consumer nobody re-checked.

## 2026-08-12 — Time formula +91 at 60+1; KCX holds

**Timeval** (TESTING.md rule 5 multi-TC; new `wtime/12 + 0.9*inc` vs old
`t/40 + inc`, same kb8 net both sides, 160 games per leg, **zero time losses**
at both TCs):

- 60+1: **+91.06 ± 50.74** (62.8%)
- 30+1: **+45.87 ± 46.84** (56.6%)

The production audit's clock-bleed diagnosis (2.9s spent of a 35s clock, 57%
of rating bleed as depth-ceiling drift) converts to the cheapest Elo this
project has found, and the gain grows with the clock — the signature of a
formula that was structurally underspending. *Shared-engine note: this lives
in `sunfish_ui/uci.py`, so both bots ride it.*

**KCX screen** (new certified search vs old, 200g @30+1): −15.65 ± 34.94 —
within noise, no play regression, and the correctness properties (verified
null cutoffs, terminality) were the point. Goal60's −30 auto-hold not tripped.

Chain note: goal60 aborted once at classic's preflight (classic imports the
shared `sunfish_ui` driver; its wrapper lacked PYTHONPATH — fixed, preflight
green as black) and was relaunched with both gates read correctly.

## 2026-08-12 — The history order key broke the futility break's soundness (−449 Elo)

The hist screen (200g @30+1) returned **−449.35 ± 93.62** — not a regression,
a broken search. Forensics: identical depth and time per move but 931-vs-494
own-eval collapses, and an A/B at equal completed depth showed the history
build's scores inflated **one-sided** (median +38, p90 +136, p10 +0; sound
reordering gives a symmetric near-zero gap).

Root cause: the `depth <= 1` futility branch yields an estimate and then
BREAKS, justified by "we have ordered the moves by value" — but the order key
had become `v + hh`. An early low-val/high-credit move triggered the break and
discarded later moves with higher static value that were not futile (some
above gamma). The node failed low, the parent negamax inflated, and the
optimism compounded to the root.

Fix: frontier nodes sort by static value alone; interior nodes keep history
ordering. Validation: the same A/B reads median +0 / p10 −4 / p90 +1, zero
gaps > 100, 1/30 move disagreements, node ratio 1.03; 14 tests green; verify
battery green (18208 positions, worst lane excursion 5686 < 15480 bound).
Production bot redeployed and verified online.

## 2026-08-12 — 256ng: best val ever trained, held out of the arena by the gate

The flagship composition (256 × kb16 × bilinear+tail × phase × rff64 × satpen
@100M) finished at val **0.00678** — past the old best 0.00717 and the
incumbent's 0.00731 — with shapecheck PASS 1.93% and a clean pack (B=16 nb=32
phase=8 rff=64, excursion 9733).

Pre-registration then ran as written: **nps probe before any games**. Result
**35587 nps = 0.553× kb8 / 0.659× w256** — the predicted kbbil-class tax,
measured. Model pricing vs the incumbent: speed −61 ± 23, quality upside ~+45
± 20 (scaling the only calibrated val→play conversion, kb8→w256). Net −16 ±
~30. Not winnable → **no games**. The strongest eval this project has produced
rides the bench because it pays 0.55× for features whose float tail cannot be
golfed viable.

Probe-drift note: kb8 measured 64298/60912/58129 nps across runs (±5% box
drift); only within-run ratios feed the model.

## 2026-08-12 — The ext tax, profiled

Where rehab800's 35% goes (pypy microbench, 5118-bit accumulator): `_ext`
costs **8.0 µs of a 38.6 µs `pos.move()`**. Inside it:

| component | µs | share of `_ext` |
|---|---|---|
| `_mlp` float tail | 3.8 | 47% |
| `cnt` board scan (feeds PHASE_S) | 2.7 | 34% |
| bigint field extraction | 0.98 | 12% |
| m² conv loop | 0.56 | 7% |

pypy JITs the bigint and conv work fine — the big-int-shift hypothesis was
tested and dropped. Consequence: even a 2×-golfed `_ext` leaves the
bilinear+tail family ~−40 Elo under the speed model; the float tail cannot be
golfed into viability. The alternative was already measured: rff angle lanes
at 0.892× (−17 Elo hurdle), all-integer, no float path. That became the krff
training above.

## 2026-08-12 — Capped-null decision match: no Elo case either way

cap (`min(score+ER, pass)`) vs base, 300 games @30+1, zero time losses:
**−10.4 ± 23.3** (nElo −17.7 ± 39.3), 48.5%, Ptnml [14, 10, 102, 19, 5].
Statistically flat — the edit neither gains nor measurably costs play
strength. If the case for it is correctness/simplicity, this is consistent
with no-regression; if the case was Elo, there is none at 300-game resolution.

## 2026-08-11 — Why nets lose: speed is Elo, and the ledger has the exchange rate

The rehab800 screen returned −70.4 ± 23 vs kb8. Five hypotheses went in.

**The root cause is speed, and speed alone.** rehab800 runs at 0.647× kb8's
nps under pypy (interleaved ×3; the old "~5%/node" ext-latency figure measured
the wrong runtime). Both sides spent 1.25s/move in the screen; rehab reached
7.13 mean depth vs kb8's 7.55 — and log2(0.647)/log2(EBF 2.7) = −0.44 ply. The
depth gap IS the speed gap. With speed removed, rehab **wins every quality
axis**: fixed-depth-4 SF cp-loss 26.7 vs 31.5 (reproduced 27.4), child-ranking
Kendall tau 0.071 vs 0.055, depth-5 tree 2% smaller.

Closed side-findings: distribution shift REFUTED (both nets pass shapecheck on
the 3688 positions of rehab's own lost games); scale-equivalent constants
REFUTED (QS×1.27/ER×1.27 all worse at fixed depth: 28.2-29.6cp vs baseline
27.4). Descriptive but non-actionable: rehab's eval steps are 1.27× larger,
and on positions it went on to lose it reads +51cp where kb8 reads −35.

**The model** (6 direct 200g pairwise labels, WLS, leave-one-out): ΔElo =
a·log2(nps ratio) + b·(timed cp-loss diff) gives a = 109 ± 35 Elo per speed
doubling, b = 7.4 ± 5.1, χ²/dof 3.9. The honest split: the **speed-only**
model (a = 102 ± 38) predicts both ext-family collapses to within 1.3 Elo —
kbbil−kb8 −82.2 predicted vs −83.2 measured; rehab−kb8 −69.1 vs −70.4. Two
400-game verdicts reproduced by a ten-minute nps probe.

The quality axis FAILS validation: timed 1.2s cp-loss on 200 dump positions
puts v2 and kbbil at kb8 parity (paired SE ~3cp) where play has them −48 and
−83; quality-only χ²/dof 10.9. Mechanism, not noise: engines live in positions
they *steer into*; neutral-position per-move quality cannot see optimism
walking games into lost structures.

Measured speed ladder (pypy, kb8 = 1): kb4 1.026, v2 0.967, rff 0.892, w256
0.846, kb16r 0.629, rehab800 0.625, kbbil 0.572.

**The gate this replaces val-only qualification with:** (1) nps ratio → speed
Elo at ~100/doubling, a hurdle paid before any quality case; >0.5 doublings of
tax exceeds every quality gain this family has converted (max +67) and is an
engineering problem, not a screening candidate. (2) shapecheck veto + val for
equal-speed ordering. (3) Play screens decide quality-side gains.

*Follow-up (2026-08-12): Thomas's critique — speed-only is degenerate outside
the family, since a null eval would be predicted to win. The fix under
construction is fixed-node matches, where ΔElo is quality in Elo units and the
formula becomes ΔElo_fixednode + 102·log2(nps ratio) with no fitted b.*

## 2026-08-11 — 200M + satpen: fails its val gate with the cleanest shape a 256 net has shown

TRAIN256KB8200MSP final: val **0.00740** vs gate < 0.00731 — the saturation
penalty at 200M eats the entire data-scale gain (raw 200M reached 0.00717 and
shape-FAILED at 2.73%). But shapecheck **0.27%**, ten times cleaner than the
incumbent's 2.53%, which sits a hair under the 2.6% cliff the kbbil collapse
calibrated. Same pure-int kb8 architecture, no speed tax (B=8 N=256 shift=3,
sum_G 42501, excursion 11710).

Not discarded on val: it became a prediction target for the speed model, then
the one candidate whose question is purely quality/shape (screen staged vs
w256 at measured speed parity, 51188 vs 51543 nps).

## 2026-08-11 — The cp-loss axis flips the tuning frontier

The 19-config frontier re-scored under Stockfish-17.1-at-depth-12 centipawn
loss (576 unique evaluations, capped mean at 300 with the blunder tail
separate). Full table — quality, blunders, cost, and the discredited
agreement column:

| config | cp-loss | bl>300 | nodes@d5 | Δ | agree@d4 |
|---|---|---|---|---|---|
| QS=40 QS_A=140 ER=15 (default) | 36.0 | 2 | 68219 | — | 37.0% |
| QS=0 ER=15 | 32.5 | 2 | 96985 | +42% | 41.0% |
| QS=20 ER=15 | 36.9 | 3 | 66200 | −3% | 37.5% |
| QS=80 ER=15 | 36.9 | 3 | 51922 | −24% | 38.5% |
| QS=140 ER=15 | 41.4 | 3 | 39920 | −41% | 37.0% |
| QS=219 ER=15 | 92.6 | 34 | 42391 | −38% | 28.5% |
| QS_A=60 / 100 / 180 / 240 / 300 | 35.0-36.7 | 2 | ~68219 | ~0% | 36.5-37.5% |
| ER=5 | 34.5 | 3 | 63725 | −7% | 37.0% |
| **ER=10** | **33.1** | 2 | 53183 | **−22%** | 38.0% |
| ER=25 | 36.3 | 3 | 44462 | −35% | 36.5% |
| ER=40 | 36.7 | 4 | 46063 | −32% | 37.0% |
| **QS=80 ER=10** | **33.9** | 2 | 42723 | **−37%** | 39.5% |
| QS=110 ER=10 | 35.0 | **1** | 43638 | −36% | 39.5% |
| QS=140 ER=10 | 38.5 | 3 | 50445 | −26% | 40.0% |
| QS=0 ER=10 | 38.0 | 4 | 95044 | +39% | 40.5% |

Readings: (1) QS_A is a dead axis — five settings, identical node counts. (2)
ER=10 beats both neighbors on cp-loss AND nodes. (3) The axes do not factor —
QS=0 is the best cell under ER=15 and the worst non-cliff cell under ER=10, so
single-axis tuning at fixed ER would mislead. (4) The agreement column is
retained to document the mirage: it ranks QS=0/ER=10 near the top where
cp-loss ranks it last — agreement was rewarding coin flips between equal
moves. (5) QS=219 is the cliff: half the node savings of QS=140/ER=10 at 17×
the blunders.

*Caveat before porting to classic: measured on the PACKED engine. The classic
engine's eval scale and ordering differ, so these are candidates there, not
conclusions — and classic tuning is Thomas's own experiment.*

## 2026-08-11 — Search constants: the classic-era defaults are stale

The first-stage offline Pareto sweep (deterministic bench nodes@d5 ×
SF-best-move agreement@d4, 200 dump-oracle positions) showed the defaults
dominated: QS=40/ER=15 at 68219 nodes / 37.0% vs QS=80/ER=10 at 42723 / 39.5%.
Method note: this two-stage design (free deterministic frontier mapping, then
one shared tournament over the frontier) replaced an infeasible grid of 7-hour
A/Bs. The agreement axis was later discredited — see the cp-loss entry above,
which is why the frontier plays rather than the axes.

## 2026-08-11 — kb16r qualifies; the flagship launches

net128kb16r (kb16 × the rehab800 recipe): val **0.00740** — the best 128-wide
ext-family number, past its 0.00753 gate and past unrehabbed kbbil — verify
3378 positions through the 256-combo kb16 tables, shapecheck 1.93% PASS.

With every component validated separately (kb16, rff, the rehab recipe, width,
data scale, satpen), the flagship composition was earned and launched:
TRAIN256NG, all of it at once at N=256 over 100M positions.

Also this pass: the gate chain caught an ext-constants scoping regression
(BTAIL orphaned by the rff block insertion; bilinear+tail nets crashed at load
while the tested paths passed) — fixed, and the regression ladder's synthetic
net is now EVERY-feature composed so the hole cannot reopen.

## 2026-08-11 — Compensation oversampling fails informatively

TRAIN128COMP (compboost 8×: 456841 positions to 11.1% effective share, ctrl
recipe otherwise): overall val **0.00911** vs ctrl 0.00796 (+14.4%, an order of
magnitude past the ~1% budget) while the class metric moved only 0.04959 →
0.04568 (−7.9%) across 14 epochs.

Reading: a 7.3× exposure boost buying under 8% on the target class while
wrecking the average means the compensation class is not oversampling-limited
— the net cannot EXPRESS king-attack compensation at this feature set, and
repetition does not create representation. The dataset-paper diversity lever
is parked; the king-safety FEATURE direction (rff phase lanes, kb16
conditioning, the bilinear tower) is the confirmed lever. Standing
quantification of the blind spot: class loss runs ~5× the overall loss.

## 2026-08-11 — rff clears every pre-play gate

net128rff (kb8 + phase-capped + satpen + 64 phase-sketch lanes, val 0.00765):
packed build B=8 shift=4 rff=64 (excursion 8812), verify green (3378 positions
through the 32-bit angle fields), shapecheck **0.40% PASS** — the best shape
number of any gated net, p99 565 with real headroom.

## 2026-08-11 — The phase-sketch passes: multiplicative features in unitary form

TRAIN128RFF **0.00765** vs the ctrl gate 0.00796: −3.9% relative, the largest
single-feature val gain since kb8 itself, from 64 phase lanes with a cos
read-out (random Fourier features = all-order piece interactions; the unitary
reduction of Thomas's tensor-sketch idea). For scale: kb16 bought −1.5%, the
entire bilinear+tail stack −5.4% — one idea at half the bilinear stack's yield
with a fraction of its machinery.

Packed form priced at design time: angle lanes wrap mod 2^15, dev read-out is
per-lane cos in the ext path. (The final packed design uses 32-bit angle
fields with plain adds and zero extra ops; the wrap-AND variant was abandoned
because per-op guard clears corrupt transient inter-lane borrows.)

## 2026-08-11 — Width converts: 256kb8@100M is the new play king

The width screen (200 games @30+1, same engine both sides, openings_2k):
256kb8@100M beats kb8@128 by **+52.5 ± 43.6** (nElo +59.2). Width plus data
pays in play, not just val — the v1-era "widths tie" verdict is overturned
under the v2 stack. New freeze candidate for the lichess bot.

## 2026-08-11 — King-capacity solos: kb16 pays, m8 ties out

TRAIN128KB16 **0.00788** vs kb8 0.00800: depth-of-advance conditioning passes
at −1.5% relative (half the kb4→kb8 step — diminishing returns, but the bucket
ladder is not done). TRAIN128M8 **0.00754** vs rehab800 0.00753: doubling the
bilinear fold groups is a statistical tie — at nb=32/30M the m=4 convolution
already extracts what the group structure offers; the fold stays m=4.

## 2026-08-11 — The 200M val record fails the shape gate

TRAIN256KB8200M: val **0.00717**, the best number the deployable family has
produced (30M 0.00741 → 100M 0.00731 → 200M 0.00717 on the pinned split — data
keeps paying). And it does not ship: shapecheck **2.73% > 2.6%**, p99 pegged at
the clamp, build shift down to 2.

The family-wide trend is monotone — kb8@128 1.87%, 256@100M 2.53%, 256@200M
2.73% — better training sharpens the residual into the ±600 band until it pegs:
**the kbbil lesson generalizes to pure-int nets, arriving gradually instead of
catastrophically.** This is shapecheck doing exactly what it was built for:
catching the pathology BEFORE 200 games get spent discovering it.

Disposition: satpen graduates from ext-family rehabilitation to **default for
every future net**.

## 2026-08-11 — rehab800: the wider band buys the val back

TRAIN128REHAB800 (rehab recipe at clampcp 800, satthresh 640): val **0.00753**
— beats rehab600's 0.00760 and sits 0.00003 from the collapsed kbbil's 0.00750,
with clip-saturation 0.00% through training. Gates green: shift 6 (excursion
10236), verify 3378 positions, shapecheck 1.80% PASS.

Instrument caveat: shapecheck counts ≥599cp residuals, which for an 800-clamp
net includes legitimate band use, not pegging — p99 sits at 646 against its own
800 ceiling; pegging-at-own-clamp is ~0%. A clamp-relative shapecheck v2 is
owed.

## 2026-08-11 — Material-base attribution closed

TRAIN128CTRL (the pst-base twin of the mat runs) lands at val 0.00796.
Attribution is clean: mat800 0.00812 − ctrl 0.00796 = **the material-base
decomposition costs 0.0016 val** at 128/30M with everything else held equal.
The pst positional prior stays; the mat lane is closed at this scale. Side
finding: ctrl 0.00796 vs plain kb8 0.00800 — phase+satpen are val-neutral on
pure-int kb8 nets.

4k budget finding from the same pass: classic master packs to 3296 bytes — it
golfed ~650 since the packed engine forked, and none of it reached the packed
engine's shared regions. Porting classic-current is the identified route to the
4096 claim, and the KCX portion is a semantic port with the full verification
ladder, not text golf.

## 2026-08-11 — Material-base fails its val gate

TRAIN128MAT600 0.00815, TRAIN128MAT800 0.00812 against the 0.00800 gate — the
honest negative: at N=128/30M the net does not recover the pst prior's value.
The clip A/B says the range barely matters under satpen. Confound noted at the
time (no pst twin), which the ctrl run above then resolved.

## 2026-08-11 — The rehabilitation works on every gate short of play

TRAIN128REHAB (kbbil architecture + satpen 0.03 @480cp, phasecap 2.0): val
**0.00760** — better than every pure-int net including kb8 — and the eval shape
is transformed: shapecheck **0.00%** over the frozen 1500-position set (kbbil:
4.93%), p99 502. Training held 0.00% clip-saturation from epoch 1: the penalty
binds immediately and costs almost no val.

The 0.0001 val price for eliminating the pathology says the saturation capacity
kbbil spent 0.74% of training positions on was nearly worthless even on quiet
data — free capacity in the loss, ruinous in play.

## 2026-08-11 — The TCEC-4k field ladder

600 games @30+1 (100 per pairing, moves-based tcec_book), zero recoveries:

| pairing | score | Elo |
|---|---|---|
| packed128v2 vs molly | 21.5% (9W 25D 66L) | −225 ± 65 |
| packed128v2 vs 4kc | 0.0% (0W 0D 100L) | shutout |
| packed128v2 vs STRO4K | 1.5% (1W 1D 98L) | ~−727 |
| classic vs molly | 10.5% (3W 15D 82L) | −372 ± 91 |
| classic vs 4kc | 0.0% | shutout |
| classic vs STRO4K | 0.5% | ~−920 |

Loss taxonomy over all 600 games: **zero time losses anywhere** — the forfeit
class is absent at this TC. Losses are overwhelmingly middlegame outclassing
(median loss length 31-39 moves); endgame-conversion losses are rare (≤6 per
pairing past move 60). Reading: the field engines search much deeper; eval
quality alone moved the molly number by ~+150, but the remaining ~225 to molly
parity — and the ~450+ gap to 4kc/STRO4K — is depth.

## 2026-08-11 — Decision RR final; the kbbil collapse diagnosed

**Decision RR** (600/600 @30+1): field Elo v2 +99.95 ± 32.2, kb4 +84.1 ± 33.9,
classic −199.1 ± 37.4. Pairwise: v2 beats classic +193 (142W 17D 41L), kb4
beats classic +205 (148W 10D 42L), kb4 vs v2 −19 (kb4's val edge did NOT
convert; within noise).

**Generation RR**: kb8 beats kb4 pairwise +96 — kb8's val edge DID convert.
**kbbil (best val 0.00750) COLLAPSED: −118 pairwise vs both.**

Diagnosis, three measurements: (1) NOT node starvation — pgn depth mean 7.03 vs
7.43, zero low-depth moves after move 10. (2) NOT eval latency — fixed-depth
bench 31.3 vs 29.2-31.9 µs/node (~5%; *this measurement was later shown wrong
for pypy — see the 2026-08-11 root-cause entry, where the real ratio is
0.572×*). But kbbil searches +27% more nodes for the same depth: the ext eval's
SHAPE inflates the tree. (3) Heavy tails on 1559 real game positions: |residual|
p99 598 vs 477, saturation at the ±600 clip **1.0% vs 0.1%** — ten times as many
pegged evals, poisoning QS leaves.

Conclusion: **quiet-position val does not measure search-friendliness.** This is
the finding that produced shapecheck, satpen, and eventually the speed model.

## 2026-08-10 — The extension generation: every prototype passed its gate

All on the 30M cache, identical val split, N=128. Baselines: v2 0.00875, kb4
0.00825.

| net | val | what it is |
|---|---|---|
| net128kb8 | 0.00800 | 8 own-king buckets, file pairs × back/advanced |
| net128bilt | 0.00795 | 32 bilinear lanes m=4 + odd tail 16, NO buckets |
| net128bil | 0.00841 | bilinear lanes, linear read-out only |
| net128phase | 0.00833 | 8 material buckets scaling the residual |
| net128phase1 | 0.00836 | CONTROL: single global scale |

Readings: (1) The odd tail is what makes the bilinear lanes pay — without it
they trail kb8; with it they beat kb8 with no buckets at all. (2) The phase-8
gain is almost entirely its GLOBAL scale (the phase1 control lands within 0.4%),
so per-bucket variation adds ~nothing on val. (3) The extensions stack.

## 2026-08-10 — v2 and kb4 trained; kb4 takes the val gate

30M quiet positions, 14 epochs, losspow 2.6, factorizer on, best-by-val export.
Val anchors on this split: zero 0.02131, pst 0.01533.

| net | val | MAE |
|---|---|---|
| v1 net128 (2M quick distill) | ~0.0106 | 122 cp |
| v1 net256 | ~0.0095 | 116 cp |
| net128v2 (full-scale) | 0.00875 | 117 cp |
| net128kb4 (+4 king buckets) | 0.00825 | 113 cp |

Full-scale training alone buys more val than the 64→256 width sweep spanned;
king buckets add a further −5.7% relative on top at the same width.

## 2026-08-09 — The bilinear head is affordable; its obvious read-out is rank-1

One big-int multiply on pypy, by width: 512 bits 0.123 µs, 1024 0.327, 2048
1.282, 4096 4.218, 8192 13.545 (≈ n^1.7). So a complete candidate head at
cropped width costs 0.508 / 0.988 / 2.565 µs at 512 / 1024 / 2048 — about a
quarter of the existing head at 1024 bits, ~4% of a node. Affordable precisely
because the multiply count is 1-4, not n².

But the **obvious read-out throws all of it away**: the head sums lanes with
2^16 ≡ 1 (mod 2^16−1), and applied to a product that identity gives exactly
(Σa)(Σb) — a rank-1 form. Every genuine second-order term cancels. The fix is
to fold modulo 2^(16m)−1, so lane k lands in group k mod m and the residue
carries m distinct bilinear features. Verified against explicit scalar
convolution, m = 2 and 4, 300 random lane vectors.

## 2026-08-09 — Multiply-and-split: priced and declined

Every structural claim behind the proposal checks out (verified exactly, 20k
random trials): A·B == (H << S) + L; the intermediate crop is load-bearing; odd
B makes the low half a bijection; the cross-mix is invertible.

Priced at engine width (N=256, 8192 bits) against a 22.3 µs node: whole-int
multiply + split + crossmix 9.45 µs (of which the multiply alone 8.76, the
clever part 0.53); lane-safe width 60.5 µs (2.7 nodes per node); per-lane form
103 µs (4.6 nodes per node). **Declined on price before the loss question was
reached.** Three reasons: the clever part is free and the expensive part is the
already-closed packed convolution; a big-int multiply is a convolution, not a
Hadamard product, so the cheap form does not compute what the fixed-point
reading describes; and there is no incremental escape (P_new = P + d·B is still
a full-width multiply).

## 2026-08-09 — Width sweep, and the 3-segment activation declined

Val loss (50k held-out quiet positions) and speed (min-of-3, fixed depth 5):

| net | val | MAE | nodes(d5) | µs/node | nps |
|---|---|---|---|---|---|
| classic pst alone | 0.01483 | 148 cp | 137,767 | 18.3 | 54.6k |
| packed N=64 | 0.01125 | 128 cp | 96,760 | 21.3 | 47.0k |
| packed N=128 | 0.01025 | 122 cp | 97,361 | 19.7 | 50.8k |
| packed N=256 | 0.00924 | 116 cp | 107,585 | 22.3 | 44.8k |
| packed N=512 | 0.00832 | 109 cp | 90,296 | 28.8 | 34.7k |

Every packed width reaches depth 5 in fewer nodes than classic. At depth 6 the
picture inverts, so the time-to-depth tax is depth- and position-dependent
rather than uniform — which is why only the clock decides.

The k=3 convex piecewise-linear activation: N=256 k=1 val 0.00924 at 22.3
µs/node vs k=3 val 0.00919 at 25.8. **0.5% of loss for 16% of node time —
declined.** The likeliest reason it does not transfer from Stockfish: classic's
pst already carries the linear structure exactly, so the net has only a bounded
residual to shape and is not activation-limited at these widths. Worth
re-asking if the pst part is ever replaced by a learned one.

## 2026-08-09 — Packed convolution: closed

A convolution does use every coefficient of the Kronecker product, so the whole
filter bank comes out of one multiply, and both layers matched a scalar
reference. But the cascade is fatal in a search (12 input channels, 3×3 kernels,
8×8 board, per node on pypy):

| filters F | 16 | 32 | 64 |
|---|---|---|---|
| layer1 scratch | 119 µs | 249 µs | 622 µs |
| layer1 delta | 3.0 µs | 5.5 µs | 16.3 µs |
| layer2 cascade | 50 µs | 235 µs | 761 µs |

Layer-1 incremental already costs as much as the entire packed NNUE head (3.5
µs at width 256, evaluating a far more expressive net), and layer 2 costs two to
forty NODES per node. Structural: a one-lane input change moves a whole
neighbourhood of layer-1 outputs, once per filter, so the layer-2 delta is F
wide-operand multiplies and grows with depth. Recorded, closed.
