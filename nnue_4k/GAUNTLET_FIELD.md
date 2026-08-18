# The gauntlet field

Thomas, 2026-08-18: *"When you test nnue rating, you shouldn't just test
against classic, but against a gauntlet of tcec 4k and python engines."*

This file is the manifest for that field: what every opponent is, where it came
from, what licence it carries, how it was built, what it is worth, and what its
UCI actually implements. Measurements live in `MEASUREMENTS.md`; this file is
the apparatus.

**Every opponent here is run locally and never redistributed.** Several are
GPL-3 and one (molly) carries **no licence at all**; run-only use is unaffected
by either, but the licence is recorded for each so that a future decision to
ship, mirror or publish anything cannot be taken in ignorance.

---

## 1. Rules of engagement — learned the hard way, not assumed

These follow from a per-engine probe of every `go` form, one fresh process per
form (`gauntlet-20260818/probe/`), and they are binding on every match this
field plays.

1. **Drive with `go wtime A btime B winc C binc D`, always, including the
   increments at zero.** It is the only form the whole field implements, and it
   is the only form the [TCEC 4k rules](https://wiki.chessdom.org/TCEC_4k_Rules)
   require an entrant to parse. STRO4K and we4k *scan past end-of-line* if the
   increments are missing and then eat the following command.
2. **Never send `position fen`.** Replay `position startpos moves ...`. The
   book must therefore be **PGN**, not EPD. (An EPD book against an engine that
   ignores FEN silently starts it from the initial position — the structural
   void that killed meter 4's EPD cell.)
3. **Never handicap a 4k engine by `nodes=` or `depth=`.** Not one of them
   implements either. The failures are *silent or fatal*, not loud: positional
   parsers read `go depth 5` as "5 milliseconds", molly panics, M4sseur and
   STRO4K hang. **The clock is the only handicap knob in this field.**
4. **Build from source at one thread; never use the `-tcec` release binaries.**
   Those request 256–512 threads and up to 256 GiB of hash (ice4, c4ke, STRO4K,
   M4sseur), and moonfish grabs every core through `sysconf`.
5. **The zero-illegal gate applies to opponents.** An opponent that emits an
   illegal move voids the games it played, so every engine passes
   `legality_gate_clock.py` (100 positions, 40 of them in check with ≤2 legal
   replies) before it is allowed into a field.

### UCI dialect, measured 2026-08-18/19 on the bench box

| engine | `wtime/btime` | `movetime` | `depth` | `nodes` | `infinite`+`stop` | `position fen` |
|---|---|---|---|---|---|---|
| 4ku | OK | hang | hang | hang | hang | OK |
| ice4 | OK | hang | hang | hang | OK | OK |
| c4ke | OK | hang | hang | hang | OK | OK |
| 4k.c | OK | **OK** | hang | hang | OK | (see note) |
| M4sseur | OK | hang | hang | hang | hang | OK |
| STRO4K-1t | OK | hang | hang | hang | hang | **FAIL** |
| molly | OK | **dies** | **dies** | **dies** | **dies** | OK |
| pygone | OK | hang | hang | hang | hang | **FAIL** |
| sungorus 1.4 | OK | hang | OK | hang | OK | **dies** |
| bbc11 | OK | OK | OK | hang | OK | OK |
| classic / entry | OK | OK | OK | OK | OK | OK |
| Stockfish 15 | OK | OK | **OK** | **OK** | OK | OK |

*Note.* `c4k/BUILD.txt` (2026-08-07) states verbatim: *"Books must be
moves-based ("position startpos moves ..."): none of these engines parse
"position fen ..."."* That is **wrong for 4ku, ice4, c4ke, M4sseur and molly** —
each returned a move that is legal only in the probe FEN
`7k/8/8/8/8/8/8/K6R w - - 0 1` and illegal from the start position. Only
STRO4K-1t and pygone genuinely fail. The operational conclusion (use a PGN
book) survives, for a different reason than the one recorded.

---

## 2. The field

### 2a. Our own arms

| name | source | pin | bytes | sha256 |
|---|---|---|---|---|
| **entry** | `nnue_4k/pst_entry.py`, `tools/build/pack.sh` | nnue-4k `aa54a5a` | 3440 | `21d55236280dd8d6c63dc790e7e7a9e7cac2a7ce0f5cc4802927dd7efba46e99` |
| **classic** | `sunfish.py`, `tools/build/pack.sh` | master `e670434` | 3358 | `5b9baf2036f74afd71e3df14be92c1c84220871acaadfc62dd96bfb568f932de` |

Both run under `pypy3.11-v7.3.20`; the packed artifacts exec
`$(command -v pypy3 || echo python3)`, so the wrappers put pypy on `PATH`.

### 2b. TCEC 4k engines

All seven were built on the bench box on 2026-08-07/08-10 (`c4k/BUILD.txt`);
provenance re-verified 2026-08-18, **licences added here** — the original
record carried none.

| engine | author | lang | source | pin | licence | build |
|---|---|---|---|---|---|---|
| **ice4** | MinusKelvin + Analog Hors | Rust→C++ | github.com/MinusKelvin/ice4 | `fabe3b1` (v6.1 + TCEC-crash fix) | **GPL-3.0** | `g++ -DOPENBENCH -O3 -pthread src/main.cpp` |
| **4ku** | kz04px et al. | C++ | github.com/kz04px/4ku | `917a087` | **MIT** (2021) | `g++ -std=c++17 -DNDEBUG -O3 -march=native -pthread` |
| **c4ke** | citrus610 + cj5716 | C++23 | github.com/citrus610/c4ke | `22e318f` = tag `v3.0` | **MIT** (2025) | `g++ -std=c++23 -DNDEBUG -O3 -march=native -pthread` |
| **4k.c** | G. Masaitis | C23 | github.com/GediminasMasaitis/4k-dot-c | `1894f0e` | **MIT** (2024) | `make EXE=bin/4kc` |
| **M4sseur** | M. Guntermann | C++20 | github.com/Diazepawn/M4sseur | `470fa53` | **MIT** (2023) | `g++ -xc++ -std=c++20 -O3 -march=native -pthread` |
| **STRO4K** | ONE_RANDOM_HUMAN | Rust + x86 asm | github.com/ONE-RANDOM-HUMAN/STRO4K | `d4d5532` (branch `version_4.0`) | **GPL-3.0** | `./build4k bin/STRO4K-1t 1 32 --avx512` — output **is** the 4008 B artifact |
| **molly** | latuernich | Rust | codeberg.org/latuernich/molly | `e317a0c7b8` (the TCEC commit) | ⚠ **NONE** — no LICENSE, no README; all rights reserved by default | native `rustc -O` |
| **pygone** | scs-ben | Python | github.com/scs-ben/pygone | `cbaebee` (2026-06-18) | **GPL-3.0** | `dist/pygone`, **4090 B**, sha256 `62346a10e2b0e13b…`; self-extracting xz → `pypy3` |
| **pygone2-11b142** | scs-ben | Python | same repo, `historical/` | the confirmed **4kVIII-era entrant** | **GPL-3.0** | **4093 B**, sha256 `f44c111d821c5bff…` |

**pygone is the one true peer**: the only other Python program ever entered in
a TCEC 4k season, and — see §3 — the only 4k engine TCEC rated at one thread.

### 2c. The Stockfish node ladder — the tunable opponent

The 4k field cannot be dialled. Stockfish can: it parses `go nodes N`
properly, and **TCEC publishes a node-limited Stockfish 15 ladder on the same
Bayeselo scale as the 4k engines** (`tcec-chess.com/bayeselo.txt`), having run
it against them in the S26 *"Old vs 4K Top Bonus"* event. So the rungs are not
an invention of ours — they are the 4k field's own yardstick.

| name | source | pin | licence | build | sha256 |
|---|---|---|---|---|---|
| **Stockfish 15** | github.com/official-stockfish/Stockfish | tag `sf_15`, `e6e324eb28fd49c1fc44b3b65784f85a773ec61c` | **GPL-3.0** | `make build ARCH=x86-64-avx2` | `8d98fae296d51ae94b66fef2ab96d2306a248b0dfd84073506fe3f202d56e344` |

Rungs, all at `Threads=1 Hash=16`, driven by fastchess `nodes=N tc=6000+0`
(the large clock is a wall-clock safety net, never the budget):

| rung | per-move nodes | realised median / p90 | Blass SF16.1 reference |
|---|---|---|---|
| `sf512` | 512 | **512 / 513** | ≈**1700** |
| `sf1024` | 1024 | (measured in run) | ≈**2050** |
| `sf2048` | 2048 | **2049 / 2051** | ≈**2292** |

**Fixed `go nodes`, NOT the `nodestime` option — and the reason is measured.**
`nodestime` was tried first, because it keeps Stockfish's own time manager and
was the recommendation this lane received. It does not survive contact with the
harness: `nodestime` makes the *engine* account elapsed time in nodes, but the
*harness* still enforces the wall clock, and a nominal clock small enough to
imply a ~512-node budget is also small enough for per-move process overhead to
drain it. Realised median spend collapsed to **20 and 43 nodes a move**
(`sfcheck.pgn`) against a single-position probe's 530 and 1323. Fixed `go
nodes` holds the budget exactly, and is additionally **the quantity the
published anchors are measured on** — Blass's SF16.1 fixed-node league and
Sopel97's slope are both per-move node limits.

Version note to keep attached: Blass's absolute anchors are **SF16.1**; we run
**SF15**, which is weaker per node, so those figures **overstate** our rungs by
an unmeasured amount. SF15 is chosen deliberately anyway — it is the version
TCEC's own published node ladder uses, so the rungs cross-link to the 4k
field's yardstick as well as to Blass's.

Three properties make this the best instrument in the field: it is **exactly
tunable** to any band; being node-limited it is **hardware- and load-
independent**, unlike every clock-driven opponent here; and it is
**reproducible in five years**, because a node budget is not a wall clock.

⚠ **Expect a spiky profile.** Node-limited modern Stockfish is tactically
*above* its nominal rating and positionally *below* it. That is why real rated
engines sit beside it in the field rather than instead of it: if the entry's
shape is unusually soft or hard against search-limited play, it shows up as the
SF rungs disagreeing with the CCRL anchors.

### 2c-bis. CCRL-listed anchors, played at FULL strength

Every published absolute scale is opponent-pool dependent — the Stockfish
developers' own skill-level anchoring admits **±100 against CCRL** and
documents a **33% Elo-scale compression** in a closed round-robin, and two
careful large-N studies of the same nominal setting disagree by **~500**. So
the ladder is anchored *locally*, with real rated engines playing unhandicapped.

| engine | source | pin | licence | build | CCRL Blitz |
|---|---|---|---|---|---|
| **Sungorus 1.4** | github.com/rofl0r/sungorus | `0af8dd0b` | none stated | `gcc -O3 -DHAVE_POPCNT -march=native -msse2 -flto *.c` | **2241 ± 16** (1280 games) |
| **BBC 1.1** | github.com/maksimKorzh/bbc | `75544dff`, `src/old_versions/bbc_1.1.c` | **GPL-3.0** | `gcc -O3 -march=native bbc_1.1.c -lm` | **2019 ± 17** (1243 games) |

Dialect: **sungorus** answers the clock and `go depth`, hangs on `movetime` and
`nodes`, and **dies on `position fen`** — harmless here because the book is
moves-based, but recorded. **bbc11** answers clock, `movetime`, `depth` and FEN,
and hangs on `nodes`.

⚠ **TC caveat on both numbers.** CCRL Blitz is 2′+1″; we play **30+1**, about
two doublings faster. Every CCRL figure quoted here therefore overstates what
that engine will show in this venue — by a similar amount across engines, but
not identically.

### 2d. Python engines outside the 4k class

Kept for the Python-field question, which is not the same as the 4k question.
The purity bar must always travel with the ranking.

| engine | source | pin | licence | deps | bar |
|---|---|---|---|---|---|
| **d-house** | our fork of the upstream engine (Berserk-derived) | `51bf4c6` (= upstream + our is_legal geometry and double-check/promotion gates) | **GPL-3.0** | **stdlib only** | the only rival at our own purity bar |
| **numbfish** | dimdano/numbfish | `97fcbbc` | **GPL-3.0** | numpy + TFLite C++ runtime | eval head is not Python |
| **neurofish** | eapenkuruvilla/neurofish | `7afe4e8` | ⚠ **NONE** | torch, pybind11 C++ movegen, Cython kernels | ~380 of its Elo is native code, by its own ablation |

d-house is ported to the bench box (`gauntlet-20260818/src/d-house`, runs under
pypy3, `OnlineSyzygy` defaults **false** so no network probing). numbfish and
neurofish remain laptop-only: both need native wheels that do not exist for the
box's toolchain without real work, and both fail the purity bar anyway.

### 2e. Surveyed and NOT adopted — with the reason

| candidate | why not |
|---|---|
| **moonfish** (zamfofex, C89, 3941 B, ~1500–1750) | obtainable from `moonfish.neocities.org/moonfish.sh` despite the sourcehut outage, but its licence is **contested** (site says 0BSD, GNU Guix packages it AGPL3+); band already covered by `sf1k` |
| **we4k** (ONE_RANDOM_HUMAN, GPL-3.0) | repo **archived**; superseded by STRO4K, whose band we already have |
| **micro-Max / umax / fairy-max** | **xboard/CECP only, no UCI, no FEN**; micro-Max additionally has **no licence**. fairy-max is the salvageable one (CCRL 40/2 1891) but needs a protocol adapter |
| **Toledo Nanochess / atomchess** | **all rights reserved** despite common claims to the contrary; Nanochess is CCRL Blitz 1019 (xboard) |
| **iota** (DanielWhite94) | 2487 B with a real UCI loop, but **CCRL Blitz 882** — far below the band — and **no licence** |
| **4kbomb** (connortynan) | **no licence** |
| **4kbengine** (b-paul) | GPL-3.0 but dead since 2020 |
| **Nalwald** | **CC BY-NC-SA 4.0** — not open source, and not 4k-sized |
| BootChess, LeanChess, ChesSkelet, 1K ZX Chess | human-interactive or boot-sector; structurally undriveable |
| **Cicada** | never emits `bestmove` under `go movetime`/`go nodes` — would hang the gauntlet |
| **Walleye** | no depth/nodes/movetime at all |
| MinimalChess, Rustic Alpha, Blunder, Zagreus, Shallow Blue, Bit-Genie | good permissive ladders spanning 900–2440 with full go-option support — **held in reserve**; the SF-node ladder covers the same band with a published TCEC cross-link, which these lack |

---

## 3. What the field is worth — three scales, kept separate

Never pool these. They are different instruments.

### TCEC Bayeselo (`tcec-chess.com/bayeselo.txt`)

```
c4ke        3270 ±31        Stockfish_15_1M   3258
LESS_STRO4K 3190 ±96        Stockfish_15_300k 3137
ice4        3175 ±33        Stockfish_15_100k 2977
4k.c        3066 ±30        Stockfish_15_30k  2749
4ku         3016 ±46        Stockfish_15_10k  2498
STRO4K      2968 ±44        Stockfish_15_3k   2124
molly       2356 ±259       Stockfish_15_1k   1726
pygone      1677 ±132
```

⚠ **The 4k engines ran at 256–512 threads in TCEC; pygone ran at `Threads=1`.**
So of the 4k rows **only pygone's 1677 transfers to a single-threaded
gauntlet**. For the top engines use CCRL single-CPU instead: **ice4 v5 3021
(40/15)**, **4ku 5.1 3030 / 3043 blitz**, **STRO4K 5.0 2965**. molly's declared
101 threads looks like TCEC bookkeeping — its source has no threading at all.

### TCEC 4k placement history, and our own past in it

Seven editions have been played (4kI–4kVII, seasons 23–30; there was **no 4k
event in S25**); **4kVIII is upcoming in S31**. **sunfish entered the first
two**: 4kI, **4.0/48, last of five, rated 2193**; 4kII, **5.0/60, last of six,
rated 1903**. pygone scored **0/48 in both S29 and S30**. Winners: ice4 (4kI,
4kIV, 4kV, 4kVI), 4ku (4kII, 4kIII), **c4ke (4kVII)**.

### Our own head-to-head record (bench box, 30+1 unless noted)

| pairing | date | n | result |
|---|---|---|---|
| classic vs **4k.c** | 2026-08-11 | 100 | **0.0%** (0-100-0) |
| classic vs **STRO4K-1t** | 2026-08-11 | 100 | **0.5%**, −919.54 |
| classic vs **molly** | 2026-08-11 | 100 | **10.5%**, −372.25 ± 90.90 |
| packed128v2 vs **molly** | 2026-08-11 | 100 | **21.5%**, −224.97 ± 65.30 |
| pesto2g32 vs **ice4 / 4ku / c4ke** (60+1) | 2026-08 | 180 | **0-180** |
| shipD vs **pygone2-11b142** | 2026-08-08 | 100 | 95-2-3, **≈ +576** |

Both arms have moved a long way since (meter 3 **+200.24 ± 38.35**, meter 4
**+108.17 ± 24.64** over a classic that itself gained **+96.19 ± 33.81**), so
these are cross-links and history, not current placements.

---

## 4. Why the strong 4k engines are anchors and not opponents

Arithmetic, stated before any game was played so that a shutout cannot later be
reported as a surprise. The top of the field is ~900 Elo above us on our own
data. The field's Elo-per-clock-doubling is ~50–70, so a clock handicap needs
on the order of **thirteen halvings** to close that — about **1 ms a move**,
where the measurement becomes a study of scheduler jitter and time forfeits
rather than of chess. And the clock is the *only* knob (§1.3).

So: **ice4 / c4ke / 4ku / 4k.c / STRO4K enter at full strength, at low game
share, as ceiling anchors** — they fix the scale and keep the field honest —
and the informative share goes to the near-strength rows. Whether a clock
handicap reaches the band at all is an empirical question, measured by the
`HCAL` screen rather than assumed.

---

## 5. Reproducing the apparatus

Everything lives in `~/sunfish-bench/gauntlet-20260818/` on the bench box:

```
bin/          artifacts (entry.packed, classic.packed, pygone2-11b142, stockfish15)
src/          clones (pygone, Stockfish15, d-house)
w_*.sh        wrappers -- put pypy3 on PATH, nothing else
probe/        one UCI-dialect probe per engine (uciprobe.py)
gate/         one legality-gate report per engine (legality_gate_clock.py)
logs/         build and run logs
```

Book: `~/sunfish-bench/c4k/tcec_book.pgn` — 330 unique five-move lines taken
from real TCEC 4k games, moves-based (see §1.2).

Match discipline: concurrency 8, `nice 5`, `-recover` always, census the box by
**parentage** before launching, yield to the owner's work, and never touch a
process this lane did not start.

---

## 6. Gate results — 15 of 15 PASS

`legality_gate_clock.py`, 100 positions per engine (40 FORCED: in check with
≤2 legal replies), one fresh process per position, **0 no-move and 0 illegal**
for every engine in the table:

entry · classic · pygone · pygone2-11b142 · molly · 4ku · ice4 · c4ke · 4k.c ·
M4sseur · STRO4K-1t · sungorus · bbc11 · stockfish15 · d-house

## 7. Known defects in field members — recorded, not hidden

- **pygone2-11b142 cannot manage a 30+1 clock.** It lost **14 of its first 14
  decided games on time** in the HCAL screen, at the full clock, unhandicapped.
  Its row measures a time manager, not a strength, and it is excluded from the
  round-robin. pygone HEAD (`cbaebee`) has forfeited nothing.
- **sungorus 1.4 exits on `position fen`.** Only reachable via an EPD book,
  which this field never uses.
- **molly dies on `go movetime`, `go depth`, `go nodes` and `stop`.** Drive it
  with the clock form and nothing else.

## 8. sha256 of every binary that plays — the provenance closes here

Recorded 2026-08-19 from the bench box. A pin without a hash is a claim about a
repository, not about the thing that played the games.

| binary | sha256 |
|---|---|
| `c4k/bin/4ku` | `07cde8f18b1d821d083db58f55d34366aee079d615fc8ecdc1094f7b75dde1b4` |
| `c4k/bin/ice4` | `cc04e6ca9470277de0b12197c075779c63266da0afc1728f28af753787050132` |
| `c4k/bin/c4ke` | `1066c885fdb1ed3d8f42c69d5fae4174384569c98e06c451cb8840a87e8955dd` |
| `c4k/bin/4kc` | `48491b063b17ed789153acba9c80b57eb92f1b6fec384b8613e0ecac2ba4e52e` |
| `c4k/bin/M4sseur` | `8b2dc7c9667f454517f094bfcddf64a97dcbae11944f0c0cf5225453dbc814fb` |
| `c4k/bin/STRO4K-1t` | `9275ce969eab1af208a83e1bfe9789d9b8266faeaa3fc086ae9f8156a59a5ca6` |
| `c4k/bin/molly` | `0c4f835f8f52c57d8978177aa3f548b6015b2676b3bd0ed7c58a4a4f0b4b5816` |
| `gauntlet-20260818/bin/bbc11` | `5ceedcfd59744a67533abc2f5abd0ad1ec63793a61dc31a6a3a2505b58de6fd5` |
| `gauntlet-20260818/src/sungorus/sungorus` | `9a617b1272ea6abc0d24c3f6ffa4b2170f8bb56f9d2b4a53ec07fd49b8910715` |
| `gauntlet-20260818/bin/stockfish15` | `8d98fae296d51ae94b66fef2ab96d2306a248b0dfd84073506fe3f202d56e344` |
| `gauntlet-20260818/bin/entry.packed` | `21d55236280dd8d6c63dc790e7e7a9e7cac2a7ce0f5cc4802927dd7efba46e99` |
| `gauntlet-20260818/bin/classic.packed` | `5b9baf2036f74afd71e3df14be92c1c84220871acaadfc62dd96bfb568f932de` |
| `gauntlet-20260818/bin/pygone2-11b142` | `f44c111d821c5bffb9117154dcffbfb0e83ab41b35e174b084a9bb5000ee1768` |
| `gauntlet-20260818/src/pygone/dist/pygone` | `62346a10e2b0e13b86a4f600ae985d7c28b74d9f0dfb1a66b168e0f57820e692` |

## 9. Time-management check at the field's real TC

An opponent that misjudges 30+1 forfeits and voids its own row — which is
exactly how `pygone2-11b142` was caught. So every clocked entrant had its
first-move spend measured at the real clock (`go wtime 30000 btime 30000 winc
1000 binc 1000`), **twice in the same process**, because fastchess reuses the
process across games and a `ucinewgame` that does not reset is its own defect.

| engine | first move, game 1 | game 2 (same process) |
|---|---|---|
| pygone HEAD | 1.20 s | 1.26 s |
| 4ku | 1.49 s | 1.49 s |
| sungorus 1.4 | 1.71 s | 1.71 s |
| bbc11 | 2.01 s | 2.01 s |
| molly | 2.73 s | 2.99 s |

All five budget 1.2–3.0 s out of 30 and reset cleanly. No forfeit risk in the
round-robin from any of them.
