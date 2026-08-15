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

## H3 follow-up: explosion vs TM-allocation from the existing logs (2026-08-14)

Rules pre-registered in `tools/analysis/h3_rules.md` (committed before computing);
`tools/analysis/h3_log_mining.py` implements them, gated on an instrument-sanity control.
Telemetry audit first (ledgered in the rules file): the entry emits no node counts and
flushes all UCI output in one burst, so only per-move wall time, exact `go wtime`, and
final depth are recoverable for it; classic's full nodes/nps lines serve as a machine-load
instrument; the bench-box fixed-node and 30+1 logs carry no usable telemetry (warning-echo
/ driver-stdout only) — the planned fixed-node nodes-per-depth test is impossible from
existing logs.

**Formal verdict: LOGS-INSUFFICIENT — by two independent routes.** (1) The pre-registered
co-tenancy guard tripped: 74/164 (45%, limit 33%) of collapse-window moves fall in
wall-clock buckets where classic's median nps is < 0.8x its global median — the ladder
runs concurrent games, and machine load is a live confound for exactly the depth/time
statistics that separate (a) from (b). (2) Independently, the selection-matched control
(SM) barely exists — only 2 of 38 non-loss games have a d:LOW pseudo-window at all — so
the frozen (a)/(b) criteria had no valid comparison set and neither fired. The rules set
no minimum SM size; that is a pre-registration defect, recorded here rather than patched
silently.

What the run did establish (numbers from the single pre-registered compute):

- **P0**: the /12 policy is confirmed live on the ladder — 97.2% of 4158 matched entry
  moves have F = t/(wtime/12) inside [0.75, 1.10], median 0.919. The artifact's own
  source ships a /40 sudden-death fix in a minifier-hide block (dead in the artifact);
  the ladder is playing the known-bad branch its comments warn about.
- **R0 (PGN-only, and the SM emptiness restates it)**: d:LOW windows are collapse-
  SPECIFIC, not policy-wide — pseudo-windows at move 32 in non-loss games are d:LOW at
  7.4% (2/27; 3/29 at move 28, 0/22 at move 36) vs 36.2% before middlegame collapses.
  A pure "the TM curve craters every game equally" version of (b) is REJECTED.
- **Stop-mode descriptives (confounded, labeled as such)**: window moves slam the hard
  deadline — median F 0.985 in W vs 0.936 baseline / 0.913 in the unselected control;
  hard-abort rate 54% vs 39% control. Median think in windows is 4.7 s vs 7.4 s baseline:
  by the collapse region the /12 front-load has drained the clock, and what is left is
  being spent to the hard limit mid-iteration. Leans (a)-in-the-small (iterations not
  fitting) sitting on top of a (b) allocation slope — but the load confound means this
  is descriptive, not a verdict.

**Consequence for H3's screen design.** The load confound and the R0 result reorder the
queue: (1) FIRST screen is the TM change, because it is verdict-independent — every
reading (a, b, or load) is improved by more late-game slack, the /40 line already exists
in-source (zero new bytes for the dev/ladder build; H4 wants the same screen), and it is
the only candidate that cannot be invalidated by the confound. (2) The qsearch/pruning
screen is DEFERRED until the offline probe (above) is extended with one cheap block:
re-search the 34 windows' positions at fixed depth offline and record nodes-to-complete-
depth-k vs baseline positions — clockless and loadless, it separates bushy-tree from
machine-noise directly and costs a few minutes of the same box slot. (3) The coordinator's
full-harvest H4 confirmation should record the ladder's fastchess concurrency setting;
if a future ladder can run concurrency 1, the co-tenancy guard becomes unnecessary.

## Appendix: P0 fix — sudden-death TM ships in the artifact (2026-08-14, pre-registered)

The P0 divergence is closed at the build level: the budget line is now ONE TC-conditional
line, outside minifier-hide, in `nnue_4k/sunfish_nnue.py` and therefore in the generated
entry and the packed artifact —
`think = min(wtime / (12 if winc else 40) + 0.9 * winc, wtime / 2 - 1000)`.
The hide block was DELIBERATE (commit 478c9f4, 2026-08-12: "TCEC is 1800+3 ... the branch
would be dead code"), not a build defect — but its premise ("the artifact never sees
winc == 0") is exactly what this report's P0 falsified, so the pattern is eliminated
rather than patched. `tests/test_time_budget.py` now asserts the line is unique, outside
any hide block, and byte-identical to the old /12 policy for every winc > 0.

Bytes (verified through `pack.sh`/`check_entry.sh` only): pre-fix baseline 3445 B
(reproduced tonight from the committed entry; the "3350 B" above was not reproducible
through pack.sh and is superseded), fixed entry **3451 B** — **+6 packed bytes, 645
spare** of 4096. Artifact-alone smoke (empty dir, SF_NET and PYTHONPATH unset): `uci` →
`uciok`, legal bestmove at both `winc 100` and `winc 0`.

Pre-registered validation, to run when the laptop frees (NOTHING armed tonight):

- **(a) Gates first**, on the packed artifact: legality gate 100/100 (seed 20260813
  sample, 0 no-move / 0 illegal), mate gate 8/8 (parity bar — no position lost vs the
  pre-fix entry), and the first-yield window under the driver. Expected gamma-seed
  interaction: NONE — the fix touches only the budget line, and the first-yield metric
  counts nodes of the depth-1 gamma=0 probe on the seed-fixed position sample, which
  reads no clock; any first-yield drift therefore means the build changed something
  besides TM and FAILS the fix.
- **(b) Confirmation match** (shared-tournament methodology, one round-robin, classic as
  anchor): entry_tmfix vs entry (pre-fix shipped artifact) vs classic at **300+0**, on
  the laptop AFTER the ladder harvest, ≥200 games per pairing, ladder opening book,
  fastchess concurrency recorded (1 if feasible). KEEP bars, exact numbers fixed now:
  1. **Increment non-inferiority**: established analytically — the winc > 0 budget is
     byte-identical to the old policy and test-asserted, so no 30+1 match is required.
     If one is run anyway, the 95% CI of (tmfix − entry) at 30+1 must not lie entirely
     below **−20 Elo**.
  2. **Sudden-death improvement (required)**: direct (tmfix − entry) at 300+0 point
     estimate **≥ +40 Elo** with 95% CI lower bound **> 0**, and anchored
     (tmfix − classic) − (entry − classic) **≥ +40 Elo** in the same tournament.
  3. Revert rule: the fix is reverted only if (tmfix − entry) at 300+0 has its 95% CI
     entirely below 0; failing bar 2 without that keeps the fix un-shipped for ladder
     claims and sends H4 back for a TM-constant sweep.
- **(c)** The ladder's fastchess concurrency setting MUST be recorded at harvest — the
  H3 rerun's co-tenancy guard needs it, and concurrency 1 would retire the guard.

Staged gate commands (run from repo root when the laptop frees; artifact built fresh via
`bash tools/build/check_entry.sh` first):

```sh
bash tools/build/pack.sh nnue_4k/pst_entry.py /tmp/entry_tmfix.packed
nice -n 15 python3 tools/build/legality_gate.py /tmp/entry_tmfix.packed 300 \
    --nodes=20000 --first-yield=2048        # bar: GATE PASSED, 0 no-move / 0 illegal
nice -n 15 python3 "$ARENA/mate_gate.py" /tmp/entry_tmfix.packed tests/files/mate1.fen 4
                                            # bar: 8/8 (mate_gate.py rides with the
                                            # screen arena; suite is the 8-position
                                            # tests/files/mate1.fen)
```

## Appendix: a SECOND one-path fix — the polling holdback never reached the clock branch (2026-08-15, found by play)

Same defect class as the P0 above, opposite direction: not "the fix is in the source but
not the artifact", but **"the fix is in the artifact and guards only one of the two paths
that reach it."**

`nnue_4k/pst_entry.py`, builtin loop:

```
think = times.get("movetime", think) / 1000
if "movetime" in times: think -= max(think * .05, .03)
```

The 5%-minimum-30 ms holdback exists because 425 local games once forfeited to it, and
its comment says so. But it subtracts only when a `movetime` key is present. **Under a
real clock (`wtime`/`winc`) nothing is held back**, while the hard limit is
`think = min(5*soft, A/2)` and `searcher.deadline` is polled every 2048 nodes — so the
search returns at `think + epsilon` and a zero-margin arbiter flags it.

Observed: the +400 progress meter's stage 2 (60+1, PACKED artifacts, first timed match
this artifact has ever played on its own loop) took **2 time forfeits in 41 games, both
the entry, overruns 100 ms and 101 ms**, against **zero** for classic. Classic is
protected structurally rather than by a holdback: its loop breaks at `think * 0.8`,
keeping 20% of its only limit in reserve, whereas the entry's soft break at `soft` can be
followed by a hard limit **5x larger** with no reserve at all.

Why it hid: every previous timed result in this project ran the SOURCES through
`sunfish_ui`, which computes its own `think` and enforces its own deadline. The builtin
loop's clock path had never been played under a real arbiter.

**Rate not established** — two events at concurrency 4 cannot separate artifact overrun
from scheduler contention, and ~100 ms is that scale. The MECHANISM is established from
the source; the frequency needs a concurrency-1 replication and a `timemargin` sweep,
both registered in MEASUREMENTS.md and neither run. No fix is made here: changing the
budget line of a shipped artifact needs its own registration, byte price and gates.
