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

## Index

| Date | Experiment | Verdict |
|---|---|---|
| 2026-08-12 | Mate distance (issue #11) | Floors unchanged, score separation real, **practical reach capped by EVAL_ROUGHNESS**; 30+1 match QUEUED |
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

## 2026-08-12 — Mate distance: the value function separates, the driver mostly cannot use it

Issue #11 (2014, "Tempo"): every checkmate scored the flat `-MATE_LOWER`, so
"a mate in 6 is considered the same as a mate in 1". The terminal correction
now deposits the depth still unspent when the mate was found,
`mate = -MATE_LOWER - min(depth, MATE_SPAN)`, which negation carries home as
`MATE_LOWER + (depth - plies)`.

**Value level — works exactly as designed.** On
`8/3Q4/8/8/8/3R4/5K1k/8 w` (three mating moves, eight moves that mate in
three) at depth 6:

| | mate in 1 | mate in 3 |
|---|---|---|
| master | 47923 | 47923 |
| matedist | 47928 | 47924 |

Master cannot tell them apart at all; matedist separates them by 4, which is
exactly the two-move difference doubled. Pinned by
`tests/test_regressions.py::TestMateDistance` (score separation, the exact
`MATE_LOWER + depth - 1` for a mate in 1, the exact `-MATE_LOWER - depth` at
a checkmated node).

**Fixed-depth floors — bit-for-bit identical to master** on every suite:
mate1 8/8, mate2 20/20, mate3 5/5, mate4 5/10, stalemate0 4/4, stalemate1
3/4, stalemate2 18/130, WAC 94/300 @d3, bratko 5/24, 3fold 2/8. 271 tests
pass. Packed 3250 B (master 3234 B, limit 4096).

**Conversion probe — no measurable effect, and the reason is instructive.**
60 random won endgames (KQK / KRK / KRRK), fixed depth 5, cap 40 plies,
defender held fixed at frozen master: 29/60 converted for BOTH trees, mean
10.52 plies for both, **zero positions differed**. A second probe over 40
forced-mate-in-3 positions at depth 8 (attack and defend directions):
**every single playout identical**.

Two reasons, both worth recording:

1. **Iterative deepening already sorts by distance.** A mate in 1 first
   appears at depth 2, where it is the only move in the band; it is stored
   as the killer and re-tried first at every deeper iteration, cutting
   immediately. IID (`bound(pos, gamma, depth-3, root=True)`) does the same
   inside a single probe. Sunfish was already picking the shortest mate it
   had *found*, without being able to *score* it.
2. **`EVAL_ROUGHNESS = 15` blurs the new information at the root.** MTD-bi
   stops when `upper - lower <= EVAL_ROUGHNESS`, so the last window sits
   within 15 of the true value and any move within 15 of the maximum can
   take the cutoff. Mate distances differ by 1 per ply, so the driver cannot
   separate mating lines less than ~15 plies apart. The fix is sound and the
   value function is strictly better ordered, but the *driver* only sees it
   for large distance gaps.

That second point is the honest ceiling on this change as shipped, and the
obvious follow-up: make the bisection's stopping rule mate-aware (keep
bisecting while the bracket is inside the mate band) so the root can act on
the distance it now has. Not done here — it is a driver change with its own
Elo question.

**Rejected alternative, with a proof-level reason.** The other way to get
distance-from-node is a per-ply step on the score
(`score -= sign(score)` at each negation, mate in k = `MATE_UPPER - 2k`).
It is UNSOUND with sunfish's zero-window probe: the step map is not
injective at the band edge (`up(MATE_LOWER) = up(MATE_LOWER - 1)`), so no
single child window separates the child's fail-high from its fail-low, and
the fail-soft point spec breaks by one at both
`boundD2 child (1-gamma) = -gamma` and `= 1 - gamma`. Restoring it needs a
gamma-dependent child window
(`1 - gamma - [MATE_LOWER <= gamma < MATE_UPPER] + [gamma <= -MATE_LOWER]`),
i.e. a change to the search contract. It was implemented, the hole was found
in the Lean transport, and it was reverted; `formal/Sunfish/GameTree.lean`
keeps `up` and its machine-checked non-injectivity as the record.

**Elo: QUEUED, not measured.** 300 games at 30+1, openings_2k.epd, srand
20260812, `-recover`, concurrency 6, waiting on `WIDENING_RR.txt` plus a
20-minute fastchess-quiet window (`~/sunfish-bench/matedist/`). Given the
probes above, the prior is "flat"; the match is there to rule out a
regression, not to find a win.

**Related negative, already on record:** the other half of issue #11, the
tempo term itself, was measured and rejected — T-eval −8.1 ± 32.6, T-null
−115.2 ± 43.7.

---

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
