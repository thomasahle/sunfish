# Why the entry loses: a loss taxonomy

Mined 2026-08-14 from existing PGNs only — no engine was run. Rules were pre-registered in
`tools/analysis/loss_rules.md` (committed before counting); `tools/analysis/loss_mining.py`
reproduces five hand-labeled control games before it will print a number, and its output is
byte-for-byte deterministic. Swing positions for the follow-up probe are in
`tools/analysis/swing_positions.epd` (1383 positions, tagged).

## Corpora used (ledger)

| corpus | source | games | used as |
|---|---|---|---|
| pyleague | laptop `~/repos/opponents/out/pyleague_20260813_123259/games.pgn`, snapshot 2026-08-14 ~00:54, ladder mid-run (append-only) | 411 (50 undecided incl. in-progress) | ENTRY vs field, timed 300+0; also classic vs same field |
| c1screen / c2screen / d1screen | the bench box `~/sunfish-bench/eval-c1-20260813/` | 651 / 405 / 462 | entry (`base`) fixed-node losses; `d1` = known-bad eval calibration |
| elo-noiid | the bench box `~/sunfish-bench/elo-noiid-20260813/match.pgn` | 1000 | CONTROL: classic-vs-classic (delete-IID) at 30+1 |

Name mapping (from `Engine*Name` tags): `sunfish 2026-packed` = the entry
(`sunfish4k` in pyleague, `base`/`c1`/`c2`/`d1`/`aa` in eval-c1); `sunfish 2026` = classic
(`classic`, `master-*`). Every `elo-*-20260813` and `*-gauntlet-20260812` bench-box corpus
was checked: all are classic-vs-classic. Fetched but not tabulated (all classic-side
material, ledgered with game counts): elo-fresh-king-score (600), elo-masked-cap (600),
threat-cap-match (1000), search-gauntlet (1000), lmr-gauntlet (1000), elo-171-full-tail
(303), elo-173-exact (168), elo-adaptive-null-d7 (287), elo-exact-list (114),
elo-frontier-lmr (107), elo-intrinsic-tune (104), elo-lmp30-latch (189), elo-lmp30-valid
(138), elo-rfp80 (51), elo-rfp80-valid (102), aactl/aactl2 A-A controls (120+120).
VOID_ dirs exist (elo-noiid/VOID_conc4, elo-exact-list/VOID_{cotenancy,background_distill,
mixed_candidate_heads}, root-stability/VOID_cotenancy, widening/VOID_process_leak); their
contents were not used for any conclusion.

## Headline result

**In the live python-league snapshot the entry's loss profile differs from classic's in
SHAPE, not just rate — and the differences point at three specific deficits: endgame eval,
mate-proneness, and depth collapse in sharp middlegames.** Meanwhile the known-bad-eval
calibration (d1) shows what "uniformly worse eval" looks like: same profile shape as base,
just ~1.7x the loss rate. The entry's shape shift is therefore a targeted signal, not
generic weakness.

Snapshot scores, decisive games (300+0): entry 19W-130L vs the pool, classic 29W-112L.
Head-to-head entry vs classic: 11W-22L-9D (~-90 Elo nominal, ±~90 — small n, but flatly
inconsistent with the +100-130 measured at 30+1 on the bench box; see H4). Median recorded
depths: d-house 15, entry 10, classic 9, numbfish 6, neurofish 5 — the two engines that
out-depth nothing (neurofish d5, numbfish d6) still score 67W vs the entry's 7W against
them. In this pool eval quality dominates depth.

## Taxonomy: entry losses, pyleague timed 300+0 (n=130)

| phase | class | n | d:LOW | d:NORM | d:HIGH | optimism | med.move |
|---|---|---|---|---|---|---|---|
| OPENING | SELF-DETECTED | 1 | 0 | 1 | 0 | 126 | 14 |
| MIDDLEGAME | SELF-DETECTED | 94 | 34 | 60 | 0 | 407 | 32 |
| MIDDLEGAME | CREEPING | 6 | 2 | 3 | 1 | 260 | 30.5 |
| ENDGAME | SELF-DETECTED | 28 | 6 | 12 | 10 | 382 | 48 |
| ENDGAME | CREEPING | 1 | 1 | 0 | 0 | 87 | 59 |

terminations: ADJUD 87, MATE 43. By opponent (losses / self-det / creep / open-mid-end):
d-house 41/39/2/0-32-9, numbfish 35/33/2/1-24-10, neurofish 32/32/0/0-28-4,
classic 22/19/3/0-16-6.

## Classic's losses in the SAME league (n=112) — the profile difference

| phase | class | n | d:LOW | d:NORM | d:HIGH | optimism | med.move |
|---|---|---|---|---|---|---|---|
| OPENING | SELF-DETECTED | 15 | 3 | 9 | 2 | 301 | 14 |
| MIDDLEGAME | SELF-DETECTED | 86 | 6 | 77 | 3 | 458 | 26.5 |
| MIDDLEGAME | CREEPING | 1 | 0 | 1 | 0 | 224 | 23 |
| ENDGAME | SELF-DETECTED | 10 | 1 | 4 | 5 | 499 | 44.5 |

terminations: ADJUD 89, MATE 23.

Same conditions, same opponents, so the contrasts are clean:

| signal | entry | classic (same league) | classic control (30+1, n=386) |
|---|---|---|---|
| ENDGAME loss share | **22.3%** (29/130) | 8.9% | 12.2% |
| OPENING loss share | 0.8% | **13.4%** | 7.3% |
| got MATED | **33%** | 21% | 9.6% |
| mid-game swings preceded by BELOW-median depth (d:LOW) | **36%** (34/94) | 7% (6/86) | 14% (42/304) |
| median middlegame swing move | 32 | 26.5 | 28 |

The entry fixed classic's opening problem (distilled PSTs: 15 opening losses -> 1) and
pushed collapse later (move 32 vs 26.5) — then gives it all back after the queens come off
and in king attacks. Entry endgame swings happen at NORMAL/HIGH depth (10 of 28 at
above-median depth): not clock trouble — it searches deep and still steps wrong, an eval
knowledge gap. Hand-checked exemplars: C2 (B-vs-N endgame, npm 6, collapses from -1.0 at
depth 12) and C3 (bishop endgame vs classic, 41.h3?? allows c3 at depth 11).

## Calibration: what a known-bad eval looks like (d1screen, fixed nodes, n=247 vs base n=146)

| signal | d1 losses | base losses (same corpus) |
|---|---|---|
| loss rate | 247 | 146 |
| MIDDLEGAME self-det share | 82% | 76% |
| ENDGAME share | 5.7% | 7.5% |
| OPENING share | 11.3% | 12.3% |
| mean OPTIMISM before swing | ~252 | ~251 |

A uniformly bad eval loses MORE everywhere with the SAME shape — it usually dies in the
middlegame before an endgame can happen, and its PGN-visible optimism is indistinguishable
from base's. Two honest consequences: (1) the OPTIMISM proxy failed as an eval-wrongness
discriminator — in practice it tracks the opponent's depth advantage (entry optimism vs
d-house 438 / neurofish 505 / classic 193), so it is reported but carries no ranking
weight; (2) the entry's SHAPE differences vs classic (endgame x2.5, mate x1.6, d:LOW x5)
are exactly what the uniform-bad-eval control does NOT produce, which is why the
hypotheses below are phase/mechanism-specific.

Entry fixed-node profiles for reference: c1screen base losses n=232, c2screen n=130 —
same middlegame-dominant shape; endgame share 8.2% and 3.8% at fixed nodes vs 22.3% timed,
consistent with the endgame deficit being partly eval (present everywhere) and amplified
under timed sudden-death play.

## Ranked hypotheses (screenable; nothing armed tonight)

**H1 — Tapered endgame terms: passed-pawn advance + king activity.**
Observation: endgame loss share 22.3% vs classic's 8.9% under identical conditions; 27% of
head-to-head losses to classic are endgames; swings at NORMAL/HIGH depth (not time);
both hand-checked endgame losses were lost from near-equality by pawn-race/activity
blindness (41.h3?? c3!, 39.Nd5? a-pawn runs). Mechanism: distilled PSTs are
middlegame-weighted; nothing scores a runner or an active king, so at npm<=12 the entry
drifts into lost races it evaluates as -0.5. Bringing the endgame share to classic's level
converts ~13 of 130 losses; on this pool that is roughly +40-60 Elo. Screen: byte-cheap
tapered terms (rank-scaled passed-pawn bonus; king-centralization already tapers via kend —
extend weight), fixed-node A/B exactly like the c1/c2 screens, then one 30+1 validation.

**H2 — King safety / mate-proneness.**
Observation: 33% of entry losses end in checkmate vs classic's 21% (same league) and 9.6%
(control); C1 shows d-house walking a queen+knight in from a -1 eval with the entry's
depth steady at 11 — it never priced the attack. Mechanism: pure-PST eval has no king-ring
pressure term and qsearch does not extend checks, so attacks assemble outside the horizon.
Screen: cheapest-first — (a) check evasions/extensions in qsearch (search-only, few bytes),
(b) pawn-shield/king-ring attacker count term. Fixed-node screen, mate-rate and Elo both
gated. Expected +20-40 Elo on this pool.

**H3 — Depth crater in sharp middlegames.**
Observation: 36% of the entry's middlegame swings are preceded by a below-median-depth
window vs 7% for classic under the same clock — a 5x shape difference the control does not
show (14%). Mechanism candidates PGN cannot separate: (a) node explosion in sharp
positions (qsearch/no delta pruning) starving the iteration, or (b) the time manager
committing at a shallow completed depth exactly when tactics decide. First step is free:
mine the existing fastchess logs (time-per-move and depth-per-move curves around the 34
flagged games — data already on disk, no engine needed). Then screen delta pruning /
qsearch caps at fixed time. Recovering ~1 effective ply in sharp positions is worth
+30-60 Elo by the usual ply-value heuristic at these depths.

**H4 — Sudden-death time management (benchmark-critical).**
Observation: the entry trails classic 11-22-9 at 300+0 in this snapshot despite measuring
+100-130 at 30+1, while out-depthing it (median 10 vs 9); its per-move time curve drains
25s -> 1s with no increment to lean on. The python-league benchmark IS 300+0, so if the
30+1 edge is real, TM is leaking most of it on the metric that matters for the 4k goal.
Screen: TM-constant sweep (fraction-of-remaining, min-iteration guard) at 300+0 vs
classic, ~200 games. Zero eval bytes. Worth up to the entire missing head-to-head gap
(~100+ Elo on the ladder metric) if the 30+1 measurement is the true strength.

Ranking logic: H1 and H2 attack the two largest shape anomalies and survive the d1
calibration test; H3 is the largest ratio (5x) but needs the free log-mining step to pick
its mechanism; H4 is the cheapest possible Elo on the benchmark metric but conditional on
the head-to-head deficit replicating (n is small). Byte budget note: entry is 3350 B of
4096 — ~740 B of headroom, so H1(a-term)+H2(a) are simultaneously affordable; H4 costs
bytes only if constants change.

## What PGN mining cannot decide, and the follow-up probe (spec only — NOT launched)

PGN data cannot separate "eval was wrong" from "search was too shallow": both appear as
SELF-DETECTED swings, and the OPTIMISM proxy demonstrably failed (see calibration). The
disambiguation probe, for a later bench-box slot:

- Input: `tools/analysis/swing_positions.epd` — every swing/anchor position with corpus,
  loser, class, phase, played move, pre/post evals in the `c0` tag. Filter
  `pyleague#* sunfish4k` (130) for the entry; `d1screen#* d1` (247) and a 130-position
  sample of `elo-noiid` classic swings as known-bad-eval / known-search-bounded anchors.
- Per position: run the losing engine offline at fixed depths 1..(in-game depth+2);
  record bestmove+score each depth. Label EVAL-WRONG if the played (losing) move is still
  chosen at in-game depth+2 (deeper search does not fix it — the eval likes the wrong
  plan), SEARCH-BOUNDED if the move flips at depth <= in-game+2 (record the needed +k
  ply). Depth-1 score of the post-swing position gives the static mis-score directly.
- Gate: the d1 sample must come out majority EVAL-WRONG and the classic control majority
  SEARCH-BOUNDED, else the probe design is rejected — same controls-first discipline as
  this report.
- Budget: ~500 positions x ~8 fixed-depth probes, pure Python, well under one box slot,
  nice -n 15. To be queued only when Thomas's match queue is idle; nothing armed by this
  report.
