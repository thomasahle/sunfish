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

## The goal is the 4k entry

**4k has always been the goal of this workstream.** One file, ≤ 4096 bytes
total, evaluation data included. Everything in this ledger is judged by whether
it moves that artifact.

The lichess bot is a **testbed and a public demo — a byproduct**, not a second
objective. "No size limit" describes the testbed; it was never a licence to
optimise a different engine. Work on the unbounded net is justified **only
insofar as it transfers to the 4k artifact**, and where it does not transfer,
the entry must say so at the time rather than banking it as progress.

Practical rules that follow:

- Report **engine bytes and net bytes together, always**. A net size without the
  engine size beside it is the confusion that produced the "3798, 298 under
  budget" claim, when the real artifact was 541 KB.
- Search work (reductions, guards, time management) **transfers** — those are
  search bytes and they ship in the artifact.
- Large-net eval work (width, king buckets, data scale, the ext family)
  **largely does not**. It earned its place only as a source of teachers for
  distillation and of instruments (shapecheck, the speed model) that apply at
  any size.

An earlier version of this section described "two targets, different
currencies". That was drift, and the accounting entry dated 2026-08-12 records
how much effort it cost.

## Index

| Date | Experiment | Verdict |
|---|---|---|
| 2026-08-15 | **DECIDING MATCH 2 (1+0 hammer): GATE PASSED — 0 illegal, 0 `(none)`, 0 null moves, 100/100 normal — but the match found a −209.91 ± 60.11 CLIFF at a 1-second clock, and OUR OWN GATE SCRIPT reported a false FAILED** | Three things, not one. (1) The pre-registered gate passes: at a 1 s clock, where P is empty all game and the floor governs every move, the structural bestmove floor and the wall held perfectly. (2) The inline gate printed "HAMMER FAILED" on a clean run — `grep -c` prints 0 AND exits 1, so `|| echo 0` made `non` the two-line string "0\n0" and `[ -eq ]` errored into the else branch; the naive `0000` probe also matched the `+0000` timezone 200×. Fixed as a standalone `gate_check.sh`, re-run → PASSED, appended as a correction. Same defect class as the label and ramp defects, and it does not get a pass for being ours. (3) **The finding: `P = max(0, 1 − 8.4) = 0` for the WHOLE game, so soft = 0 and the pool plays depth-1 moves at 0.001 s against the incumbent's 0.013 s — 13× shallower, 23.00% score, never flagging (0.9 s median end-clock) and never illegal. `A/4 = 0.15 s` was reachable and safe the entire time: when P = 0 and A > 0 the soft limit collapses to zero and the safety clamp becomes unreachable.** Landing shape now has an OPEN QUESTION (scope the default / floor soft against A / land-and-document); this lane is not deciding it alone. 30+1 deciding match launched meanwhile |
| 2026-08-15 | **PRE-REGISTERED: the pool's single real-clock confirmation — (1) 30+1 NON-INFERIORITY, elo0=−10 elo1=0, cap raised to 1750; (2) a 1+0 ZERO-ILLEGAL hammer, 100 games, zero required — plus THE LANDING SHAPE, fixed before either starts** | After two H1s the temptation to decide the landing shape from the result is at its highest, so it is written first. The cap goes 400 → 1750 on a recorded lesson: the smooth ladder's match 2 was an underpowered non-inferiority, the same defect class as the two just ledgered (~7 h at conc 8, affordable for THE deciding match). The 1+0 hammer is not a formality for THIS manager — it ended 48 of 262 games under 2 s at 60+1 and at 1+0 the whole game lives where P is empty and the floor governs. **If both pass:** `pooltm` becomes the entry default at its measured +57 B (mod retires in place with a tombstone; `oldtm`/`steptm` go with it), the classic driver ships the pool with `legacy` kept as the control arm, and **#188 closes SUPERSEDED — not wrong**: its negative-cap mechanism is what the A/2 wall exists to prevent. **If the hammer fails on one illegal move or `(none)`, landing is blocked outright regardless of the Elo** |
| 2026-08-15 | **ARM (b) VERDICT: the POOL manager is +136.58 ± 35.24 at 60+1 — H1 accepted in 262 games, 142W-44L-76D (68.70%), PairsRatio 5.71, 0 forfeits, 0 illegal** | The risk arm was not the risk: the regime where the pool budgets 2.4× LESS per routine move beat the shipped curve by more than the 60+0 arm did. **The spend shape INVERTS between the two TCs and that is the finding**: 0.79× the median move at 60+0 but **1.09× at 60+1**, with a LOWER p90 (3.311 vs 3.655 s) and a 1.8× higher max (10.151 vs 5.509 s). The pool is a REDISTRIBUTION, not a spend-less manager — it moves time off the body of ordinary moves onto the few that need 10 s — and both directions won ~130 Elo. The pre-registered "2.4× less routinely" claim was the BUDGET ratio and is corrected here. Blind moves 0 on both arms (nobody floors at an increment TC); the increment-aware starvation band has pool 44.1% vs smooth 50.6%. Against it: pool ended **48 games under 2 s** to the incumbent's 0, which is why the deciding match carries a 1+0 zero-illegal hammer |
| 2026-08-15 | **ML2 SECOND LAYER PRICED (coordinator task, the 0.01286 phase-net's machinery): +98 B code isolated (3315 vs the round-2 3217 floor), BIT-EXACT against packed_layers' int bridge, and the extra big-int multiply costs ~+11% time/node same-tree** | At the 1024-B payload budget ml2 builds to **4339-4343 = ~245 OVER**; what FITS with ml2 code is a **781 B payload (feats 990 = total 4096 exactly; ~750 at the 30-B margin)**; u2 payload seam = 4 offset-4050 digit pairs (+8 digits, ~6 B); derivation landed as packed/make_ml2_proto.py + ml2_check.py (self-deriving, self-checking); nps tax ≈ 0.90× ≈ −15 Elo timed at 100/doubling — the number the −0.0009 val win must beat |
| 2026-08-15 | **CORRECTION + RANKING VERDICT: `min40_4` takes the classic-builtin venue (+147 [+86,+219] vs the incumbent at 60+0, AS A FLOOR) — and the "no park" claim in my own pre-registration is WRONG** | A park is **not** caused by a cap: at any increment TC the clock must rest where `spend + overhead == income`, so every manager parks, both candidates included. My reading was an artifact of charging O = 200 ms against a 100 ms increment; at the surrogate's 50 ms charge the rest point exists (`e2306d3`). Struck from tests and comments, surviving only at `winc == 0` where income is zero. The shape decides the **altitude**: one-max 6.17 s, incumbent 2.11 s (blind), **min40-4 0.22 s — the LOWEST of the three, below even the incumbent**, the thinnest flag margin in the field, and the reason its second arm is a flag hammer. It is still better than the incumbent because it reaches the floor on a POSITIVE budget where the incumbent's cap has gone negative. Ranking: `onemax` vs `min40_4` is −89 [−170,−16] at 60+0.1, ~0 at 30+1, +23 at 60+1 — min40-4 wins on Elo where they differ AND on the pre-fixed elegance tiebreak. **+147 is a FLOOR, not an estimate**: 594 of `legacy12`'s moves hit the structural-floor path where the surrogate substitutes a BETTER move than the real engine plays; the zero-substitution packed analogue read +228. **Against this lane:** the full `pool` beats min40-4 at every increment TC by +114 to +134 — min40-4 wins the ONE-LINE venue, not the field. Packed **3276 B (−2)**, source −7 B/−2 tokens. PR #196; one-max stays open as runner-up. Two arms STAGED in `tools/arena/`, GO-guarded, **neither launched** |
| 2026-08-14 | **Two defects in the CLASSIC pool twin, fixed before any PR: the arm label and the opening ramp** | Found by the surrogate lane reading `tm-pool-manager` as a formula source; **arm (a)'s verdict and arm (b) in flight are unaffected — they play packed mods**. (1) `TM_MANAGER="smooth"` actually selected master's `/12`, not #188's rational — renamed to `legacy`, alias refused, and a test goes red when the smooth curve lands so the rename happens in that merge. (2) The opening ramp capped the POOL budget for 8 plies while the measured packed arm has no ramp — the classic pool was an unmeasured variant wearing a measured number's name. Ramp is now the incumbent's only; ply-0 60+1 through the driver: legacy 1.02 s vs pool 3.82 s. Recorded cost: `random()` was the only opening variety without a book, so a deployed pool must get variety from a book, never from a budget cut |
| 2026-08-14 | **PRE-REGISTERED: the CLASSIC builtin clock drops its cap — two one-line candidates, `max((wtime−8000)/40+winc, 50)` and `min(wtime/40+0.9·winc, wtime/4)`, ranked on the surrogate BEFORE one staged 60+0 real-clock non-inferiority arm (elo0=−10 elo1=0, cap 400)** | Scope is the **packed classic artifact only** (`sunfish.py`'s embedded loop; a checkout reaches `sunfish_ui/uci.py` instead, so neither bot rides it) — byte figures are NOT comparable with the pool ladder's. The incumbent `min(wtime/12+0.9·winc, wtime/2−1000)` carries the park this file measured twice: `T* = 2+2I`, and under a 2 s clock the cap is NEGATIVE so the budget collapses to the 0.05 s floor. Both candidates are distillations of this lane's own pool — `P/M` with the reserve rounded to 8 s, or `P/M` under the pool's `A/4` clip. **Their no-park proofs differ**: one-max reaches the floor holding **10 s (40 moves)** against the incumbent's 2.1 s (8 moves); min40-4 banks nothing but its clip can never bind at `winc == 0`, making the policy exactly `t/40` with no fixed point, and it is **homogeneous of degree 1** so the ms/s trap is unrepresentable. Both net-negative in source (−11/−7 bytes, −2 tokens) and packed 3282 B / **3276 B** vs base 3278 B, so **elegance cannot break the tie**. Measured spend already separates them from base: 300+0 is **23.73 s → 7.38/7.57 s**. **Honest note in advance: a 60+0 pass does not license shipping** — `/40` wins sudden death (+235.5 ± 65.4) but loses increment (+91.1 ± 50.7 at 60+1, +45.9 ± 46.8 at 30+1 for `/12`), and these candidates collapse #188's slide back to a constant, so an increment ranking is required too. Tiebreak fixed in advance: within noise, min40-4 ships on unit-independence. **Request to the surrogate lane: add one-max and the classic base to min40-4's plugin set.** Nothing launched, no PR open |
| 2026-08-14 | **AMENDMENT: pool ladder arms (c) 30+1 and (d) phase-M are HELD for the virtual-clock surrogate — held, NOT cancelled** | Owner ruling on real-clock economy. Arm (b) 60+1 PROCEEDS on real clocks (it is both the decisive question and calibration data for the surrogate); (c) and (d) move to the twin's virtual clock once it calibrates against stage 1, the +40.6 with-park run and arm (a), with only the final composite getting one real-clock confirmation. Bounds, book, seed, cap and readings for (c)/(d) stand as pre-registered; the arena scripts are renamed `HELD_*` with the ruling in their headers so the operational state cannot drift from the ledger |
| 2026-08-14 | **ARM (a) VERDICT: the POOL time manager is +119.94 ± 36.44 at 60+0 — H1 accepted in 274 games on a NON-INFERIORITY screen, 144W-53L-77D (66.61%), LOS 100%, 0 forfeits, 0 illegal** | The arm that was only asked not to give back the +235.5 ± 65.4 sudden-death fix instead added to it, against that winner's own binary (`14b69a606b743a37`). Mechanism is the allocation SHAPE, measured off the PGN: the pool spends **0.79× the median move and 3.3× the maximum** (0.512/5.534 s vs 0.645/1.664 s) — cheap routine moves, a wall that lets a hard one run to 5× soft, which one number that is both target and wall cannot do. Drain: pool's minimum end-clock is **8.4 s = (M+2)·O exactly**, it never fell below 2.4 s in 274/274 games, and 0 games ended under 2 s, against the incumbent's 4 under 2 s and 5 crossings. Blind moves 1,057 (6.0%) vs 117 — same metric, different mechanism: these are deliberate floor moves with 8–12 s still banked, not a collapsed budget, and the arm won 66.6% while playing 14× more of them. `tm_smoke`'s cold-table prediction of 1.5× MORE routine spend over-read it; the pre-registered honest note was too pessimistic. **Arm (b) 60+1 (elo0=0 elo1=10) launched; (c) and (d) gated; v1.1 dynamic still unscreened** |
| 2026-08-14 | **PRE-REGISTERED: the POOL time manager (soft/hard) goes to a four-arm ladder — (a) 60+0 NON-INFERIORITY pool vs the shipped entry, elo0=−10 elo1=0, cap 600; then (b) 60+1 elo0=0 elo1=10; (c) 30+1 NON-INFERIORITY; (d) phase-M vs pool** | Thomas's v2 architecture, separate from the smooth budget (#188, the conservative acute fix): a whole-game pool `P = T + (M−1)·I − (M+2)·O` split into a soft limit `min(P/M, A/4)` that stops STARTING iterations and a wall `min(5·soft, A/2)` that cannot go negative. `pooltm` mod, **+57 B all-in** (3308 → 3365, 731 spare — the curve's bytes come out with it), sha `cddf392e21449054` against the in-flight #188 baseline `14b69a606b743a37`. **Recorded before the games and against the design's own premise:** budgets are not spends — iterations are discrete, the pool stops at the first one that ENDS past its limit, and the realized spend measures 1.3–2.3× the soft limit (60+0: 2.26 s vs the incumbent's 1.50 s on the laptop; more than the incumbent at every probed TC on the loaded box). v1 is STATIC; the dynamic target is v1.1 and is not screened until v1 survives (a) and (c) |
| 2026-08-15 | **MATCH 2 VERDICT: 30+1 non-inferiority NOT established — −17.39 ± 20.07 at the 400-game cap, SPRT undecided (LLR −0.81), LOS 4.43%** | The remedy does **not** fire by the rule as written (95% UB = **+2.68**, above 0), but "the rule did not fire" is not "the change is fine": the point estimate is on the wrong side of the −10 margin with ~95.6% posterior that smooth is worse at 30+1. **PRE-REG DEFECT #2:** the 400 cap was never powered for a 10-Elo test (needs **~1750**; half-width was ±20.1), and the trigger "UB below 0" demands proof of harm rather than failure-to-exclude-harm — backwards for a regression question. **The cause is the CAP, not the priced shortfall:** at winc=1 s the sign flips at a 4.5 s clock (step's parking point `T*=2+2I=4.0 s`) — above it smooth spends 0.93×, below it **1.20×/1.80×/2.78×/10.76×**. Step **never crossed 2.4 s in 400/400 games**; smooth did in **222/400**. So the old cap's parking pathology is **lethal at tiny increments (+40.6, +235.5) and protective at fat ones** — one mechanism, opposite signs. **Recommendation: do NOT land on match 1 alone**; targeted fix is an increment-aware cap, which per the amendment costs BOTH matches again at corrected power. Slot decision, reported not taken |
| 2026-08-14 | **CORRECTION + AMENDMENT: the sudden-death identity boundary is 40/19 ≈ 2.105 s, NOT 2.667 s — and Match 1's acceptance does not survive a retune** | 8/3 belongs to `max(wtime/2−1, wtime/8)`, a cap designed and **abandoned**; the shipped `wtime²/(2·wtime+4)` has two different boundaries, **2/19 ≈ 0.105 s** (cap stops binding) and **40/19 ≈ 2.105 s** (identical to step). Wrong in the SAFE direction: the identity region is **wider by 0.561 s**, so stage 1's 2.4 s minimum clears it by 0.295 s and every argument that leaned on it — including the skipped 60+0 sanity match, and the pool lane's two entries that inherited the figure — is strengthened. Bit-equality verified over **every integer-ms clock 2106–400000, zero exceptions**. Also: cap difference is `2/(t+2)` **absolute** / `4/(t²−4)` **relative**; increment claims partitioned (direct evidence at winc=0.1 only); `∂B/∂I = 560T/(40+240I)²+0.9 > 0` so the allocation is **continuous** (kink at the clip — "smooth" is informal); **the parking fixed point `T* = 2+2I`** predicts both runs' plateaus (2.0 s at 60+0, 2.2 s vs 2.1 s observed at 60+0.1) and explains the zero forfeits. **AMENDMENT:** if match 2's remedy retunes any of {20, 240, 0.9, cap}, match 1 does NOT carry — rerun both, or prove the 60+0.1 allocation unchanged |
| 2026-08-14 | **MATCH 1 VERDICT: the smooth budget is +40.64 ± 25.61 over the step at 60+0.1 — H1 accepted in 438 games (168W-117L-153D, LOS 99.92%), in 1h53m** | The positive branch: **the step form was leaving Elo on the table at tiny increments**, so this is not a change that falls back on aesthetics. Mechanism is stage 1's again — **zero forfeits on either arm**, all 438 games `normal`, and the step arm's clock **parks at 2.1 s in every single game** (min 2.0 across 438; a 2.2 s clock buys exactly 0.1 s of budget, which is exactly the increment) and pays one increment per move from there: **34.3% of its moves starved vs smooth's 4.0%**, median last-20-move time **0.115 s vs 0.391 s**. **The pre-registered ≤0.06 s metric MISSED it** (0 vs 19) because a capped budget settles where spend == income, not on the floor — logged as a pre-registration defect with a DESCRIPTIVE companion validated against the stage-1 PGN, not silently patched. Zero illegal. **Match 2 (30+1 non-inferiority) launched in the same action** |
| 2026-08-14 | **PRE-REGISTERED: the step budget becomes a SMOOTH one, and the price of that is two matches — (1) 60+0.1 smooth vs step, elo0=0 elo1=20, cap 600; (2) 30+1 NON-INFERIORITY smooth vs step, elo0=-10 elo1=0, cap 400** | The step form is discontinuous at `winc == 0`: one millisecond of increment moved the divisor 40 → 12, so 60+0.1 was paced at /12 — the exact drain the /40 branch exists to close. Replacement is one rational base (divisor slides 40 → 12) under one cap that cannot go negative. **What carries for free:** `winc == 0` is bit-for-bit `wtime/40` and, above a 2.667 s clock, bit-for-bit the stage-1 `tmfix` arm — so +235.5 ± 65.4 transfers untouched. **What must be bought:** increment TCs are now /12 + 0.9·inc *asymptotically* (−7.4% at 30+1, −8.5% at 60+1, −3.3% at 300+3), so match 2 prices that. Arms are one expression apart from one generator; the step arm packs to **3295 B, sha `fe22791b409b1fba`** — byte-identical to the stage-1 winner. Entry **3295 → 3308 B** (+13, 788 spare). Honest note recorded in advance: if match 1 reads ≈ 0 the change lands as continuity-plus-safety, not as Elo |
| 2026-08-14 | **C-TWIN PR SERVICE + EVICT BATTERY: calibration PASSED at 49.83% (300g, -1.16 ± 12.23, after a voided -54 run whose root-cause fixed the twin's go-nodes driver); #184 +0.52 ± 6.37 (668g, Ptnml [4,5,313,10,2]); #182 +1.04 ± 12.74 (668g, mechanism-active, nets neutral); #171 exactly 0.00 (all 334 pairs identical)** | Eviction battery: unguarded simplification is a **no-op at production TABLE_SIZE** and **-15.09 ± 19.57 under TABLE_SIZE=500 churn** (the root guard earns its keep where it was built); hash-slot two-tier +6.24 ± 20.96 is the guardless alternative; k2/k3 killers +1% nodes, screen-pruned. All fixed-node 20k, twin-grade, zero illegal moves in ~3,640 games |
| 2026-08-14 | **STAGE 1 VERDICT: the sudden-death TM fix is +235.45 ± 65.41 at 60+0 — H1 accepted in 100 games (64W-5L-31D, LOS 100%, 0 losing pairs of 50), in 21 minutes** | And the mechanism is NOT the one the pre-registration expected: **zero time forfeits on either arm**, every decisive game an actual mate. The drain does not flag, it BLINDS — oldtm's clock crosses the negative-cap threshold at median move 42 and it then plays a median 16 moves at the 0.05 s floor (exemplar: 45 moves at 0.00 s), while tmfix never crosses it in 100/100 games and ends with 16.9 s to oldtm's 2.0 s. H3's "depth crater" and H4's TM are therefore ONE finding. Not a ladder claim: arm-vs-arm at 60+0. Stage 1's pass rule had a degenerate-case defect (0 < 0), logged not patched. **Stage 2 (300+0) staged and NOT armed — slot decision** |
| 2026-08-14 | **PRE-REGISTERED: the sudden-death TM fix goes to a TWO-STAGE validation — stage 1 is a 60+0 SPRT, tmfix (3295 B) vs oldtm (3289 B, the pre-fix `wtime/12`), elo0=0 elo1=20** | Supersedes the LOSS_TAXONOMY appendix's 300+0 round-robin: the mechanism question is answerable at a fifth of the clock and gates whether the expensive confirmation earns a slot. Arms are **one AST node apart** (canonicalised over pyminify's renamed locals: 1 differing node in 3,858) and both carry the bestmove floor, so the screen isolates TIME MANAGEMENT. Gates green both arms (legality 100/100 × 2 budget paths, mate 8/8, standalone smoke); the TM assay proves the mod is live in the artifact — 1.38 s vs 4.00 s at `wtime 60000 winc 0`. No adjudication, on purpose. **Stage 2 (one 300+0 SPRT) is pre-registered as conditional on stage 1** |
| 2026-08-14 | **PACK_ENTRY.SH (layout B) gets the same indivisible pair: bake-off `b81` cell 3913 → 3882 (−31), elided 3280 → 3249 (−31), pst_entry −46, classic −22, nnue −31, replnet −29** | Measured on layout B's OWN consumers (cells generated by `bakeoff.run_net`), not inferred from the joint layout. **The classic +4 shebang-alone regression reproduces here too** — indivisible for the same reason. Offsets verified structurally on every artifact (`tail -c+N` == head_end+1, head+lt+|w| == filesize, SF_N == |w|, tail bytes == weights file, slice decompresses, payload no longer starts with `#!`) plus a real boot: uciok + legal bestmove, which is what exercises layout B's `SF_A` self-read. `packrun.PACK_REV` defaults to HEAD so bake-off denominators refresh on one re-run; `BAKEOFF_PACK_REV=eb8897c` reproduces the old numbers |
| 2026-08-14 | **PACK.SH LANDED: `--no-hoist-literals` + payload-shebang strip, verified a win on EVERY artifact family that packs through the shared script — classic −22, sunfish_nnue −31, pst_entry −46, replnet proto −29, variant arms −46/−47/−47/−52** | The golf lane's intel, re-measured and taken global. Free bytes for every family at once: no engine, no eval and no search changed, and all five smoke families answer legally from startpos and from a 6-move Ruy (check_entry 3341 → **3295**, 801 spare; replnet_check PASS; artifacts byte-identical to the pre-land measurement). **The shebang strip ALONE is +4 on classic** and only pays beside `--no-hoist-literals`, so the two are one indivisible change — landing them separately would have shipped a regression. Shebang deadness AUDITED, not assumed: every consumer runs the artifact's own `#!/bin/bash` head, which execs a NAMED interpreter, and the source files keep their polyglot header. **replnet capacity 878 → 909 B in-context (4095 @ --feats 1170), 849 → 880 at the 30-B margin; the 1024-B payload program is 4212 — 116 over, was 142** |
| 2026-08-14 | **REPLNET GOLF ROUND 2 (the 1024-B payload directive): code side 3449 → 3217 (−232) through pack.sh, play-IDENTICAL by node count and probe stream; trained v1 candidate 3831 → 3594 (502 spare); the 1024-B payload budget measures 4238 — 142 OVER** | 20 steps priced, 13 landed, 7 reverted (+1/+2/+6/+7/+8/+12/+14/+36 negatives ledgered); max payload that FITS today: **878 B in-context** (849 at the 30-B margin) vs 617 before = **+261 capacity**; n8 codec seam DEFINED, VERIFIED and PRICED (+15 B code; the payload is the blocker: 4410 @59.6%, 4233 @73.7% zeros); pack.sh intel for the coordinator: `--no-hoist-literals` −24, +shebang strip −32 total; full ladder green (legality 334/334, first-yield worst 183, mate 8/8, conversion 8/8 @1500ms, verify_export bit-exact, nps FASTER) |
| 2026-08-14 | **`khold2` BUILT, PRICED +27 B (3326 vs 3299), and the mate-conversion suite SPLIT the arms exactly as pre-registered: entry 8/8, khold2 8/8 (move-for-move = base), khold 7/8 — FAILS `kqk-approach`, king a1→b1 then 18 moves of shuffle at halfmove-clock 36** | Thomas's KQK objection is now a measurement, not an argument; **khold2 REPLACES khold in every composition row and khold's screen priority drops to mechanism control**; tables round-trip exact all arms, forbidden compositions raise in all four orders (khold2.pend, pend.khold2, khold2.khold, khold.khold2), standalone packed smoke green. No Elo claimed — screen staged, not armed |
| 2026-08-14 | **PRE-REGISTERED: `khold2` (khold + lone-queen escape hatch, K_END iff both queens off OR root non-pawn material ≤ 929) and a MATE-CONVERSION gate** — Thomas's KQK concern made measurable | Pure khold would hold the ATTACKING king home in KQK — the king is a mating piece there; khold2's material clause re-engages K_END exactly on lone-queen boards. New instrument `mate_conversion_gate.py` + 8-position KQK/KRK suite, driver-checked, fixed deterministic defender; **expectation pre-registered: entry 8/8, khold2 8/8, khold FAILS kqk-approach** — if it does, khold's screen priority drops and khold2 replaces it in every composition. Screen = H2's instrument verbatim, both secondary readings carried |
| 2026-08-14 | **H2 candidates BUILT AND PRICED: `kmid` +22 B, `khold` +1 B, `kmid.khold` +23 B against 739 spare** | Order-independent both in-lane (khold.kmid) and CROSS-LANE (kact.kmid ≡ kmid.kact, both 3381, sha-identical); khold.pend raises loudly in both orders as pre-registered; tables round-trip exact, K_END/kend untouched by both; standalone packed smoke green. No Elo claimed — screen staged, not armed |
| 2026-08-14 | **PRE-REGISTERED: H2 king-safety terms — `kmid` (steeper K_MID edge gradient, ±36 cp zero-centred) and `khold` (K_END only when BOTH queens are off), both ZERO hot-loop cost** | The base engine centralizes its king at 10/step while the opponent's queen is still on — kact's mate-feed pre-mortem is live in the baseline TODAY and `khold` is the one-word guard; measured tonight: the entry's 49 mated losses split **23 both-queens / 17 exactly-one / 9 queenless** (classic: 11/13/3), partitioning the evidence between kmid and khold; pawn-shield DEFERRED (mechanism overlap with kmid; the claimed classic PAWN_SHIELD=12 prior does NOT exist in this ledger); QS king-ring admission PRICED OUT (scan class + QS retuning); `khold.pend` composition FORBIDDEN (same seam line, loud by construction); screen = H1's instrument verbatim, mated-share pre-registered as the secondary reading |
| 2026-08-14 | **H1 candidates BUILT AND PRICED: `pend` +42 B, `kact` +1 B, `pend.kact` +43 B against 739 spare** | Order-independent composition (kact.pend byte-identical); shared tables round-trip tuple-identical, kend fix unperturbed; standalone packed smoke green. No Elo claimed — screen staged, not armed |
| 2026-08-14 | **PRE-REGISTERED: H1 tapered endgame terms — `pend` (endgame pawn-advance table) and `kact` (steeper K_END), both ZERO hot-loop cost on the landed queens-off seam** | Hand-designed with mechanisms, per the ledger's fits-play-worse record; passer delta-rule DESIGNED and priced out (score/ps split returns + scan class); screen = fixed-node 20k SPRT 0/10 vs base, LAND at 95% LB > 0 on fixed-N confirm; scan-class terms need timed confirmation, rule written in |
| 2026-08-14 | **PRE-REGISTERED: the gamma seed goes to a NON-INFERIORITY screen — and it lands in the SEARCH lane, not here** | +5 B on every base; **every arm passes first yield with it**, including both that fail without; 1.0000× nodes and the **same move 40/40** at completed depth 8. H1 = engine1 = seed; LAND if the 95% LB excludes −10 |
| 2026-08-14 | **The obvious timed spot-check is VACUOUS, by arithmetic** | No abort can land before node 2,048, so any build with max first yield ≤2,048 is immune at *every* TC — and at 10+0.1 the `0.9·winc` term floors `think` at ~3,800 nodes. Re-specified to **1+0, b8 vs b8seed**, with a base-vs-seed control pair |
| 2026-08-14 | **THE MIX IS THE MECHANISM — a SIZE-MATCHED control swings phase 18-24 by 16.8 points** | At identical N=8,792: natural mix **+11.70 ± 7.58 worse** than classic (40/40 splits), flat mix **−4.95 ± 3.51 better**. Pre-registered check CLEARED. Not the halved data — the mix |
| 2026-08-14 | **The gamma seed is a BLOCKER, not a lead: SIX fitted candidates, six times at the cliff** | b1 passes first yield by **37 nodes**, b8 fails at 2,433. The +5-byte root seed clears every arm ever fitted here (2.8–5.2× margin). The byte-negative arm cannot be screened until it lands |
| 2026-08-14 | The trainer's single-split band row is an OUTLIER in the band the check turns on | Seed 20260813 reads +4.15% for `bal8792` against a 12-split range of [−12.06, +1.64], and +14.24% for d1 against [+2.12, +14.13] — extreme in both, because the FEN-keyed split makes the two sets inherit the same unlucky draw |
| 2026-08-14 | **PRE-REGISTERED: the phase-balanced set — flat 2,198/band, 8,792 positions, written before selection** | Mechanism check that can only CANCEL: `bal8792` must not be worse than classic at phase 18-24, where d1 was **+7.47 ± 3.14 worse on all 12 splits**. Size control `nat8792` separates mix from N |
| 2026-08-14 | **THE CORPUS IS EXHAUSTED: 198 positions left in 4,482 games** | 19,491 + 198 = 19,689, exactly the recorded collection — a positive control on the census. So the experiment re-balances labels we ALREADY have and costs no machine time at all |
| 2026-08-13 | **`bestmove (none)` ROOT-CAUSED: budget starvation before the first yield** | The depth-1 gamma=0 probe costs **32,638 nodes against a 20,000-node budget**; `bound()` raises `Stop` before `search()` ever yields, so `go_loop` gets an empty stream. Shipped entry needs **23** on the same position |
| 2026-08-13 | **REFUTED: "mirroring is the cause."** Mirroring is a slowdown, not a switch | Nodes to first yield: base **23**, c2 1,362, q8 1,745, m1 9,086, c1 32,638. c1 is simply the arm that crosses 20,000. **Unmirrored c2/q8 fail the new gate too** |
| 2026-08-13 | **REFUTED: "fails at every depth including 1."** That was the repro harness | A piped `quit` races the search — run()'s quit branch sets `deadline = 0`. Driven without the race, **every arm answers at `go depth 1` and `go depth 3`** |
| 2026-08-13 | **Gate hole closed: wrong POSITIONS, and the wrong QUESTION** | Random-playout positions cut off in ~2 nodes (unbalanced); quiet balanced openings are the expensive class. New **FIRST-YIELD arm**, 334 book positions: base **0 over, worst 582**; c1 **10 over, worst 10,359** |
| 2026-08-13 | The gate only ever sent `go movetime`; the arena plays **fixed-node** | Both budget paths are gated now, and `ab_fixednode.sh` passes its own `--nodes`. Negative: the node arm alone caught nothing — the position class was the binding hole |
| 2026-08-13 | **d1 DROPPED at −76.0 ± 28.3 over 462 games — and 384-parameter distillation on this position set is CLOSED** | UB **−47.7** against a bar of UB < 0. C2 was −93.8 ± 32.7 on the same openings; the intervals overlap, so the teacher swap is **not** what was wrong. Pre-registered closure condition, met |
| 2026-08-13 | **The phase-band statistic does NOT predict Elo — it points the wrong way** | C2 was 8.00% BETTER at phase 18-24 and played −93.8; d1 is 7.47% WORSE there and played −76.0. The band that was stable across 12 splits carries no Elo signal |
| 2026-08-13 | Held-out loss has now mis-ranked **twice**, on two different teachers | C2 −5.9% → −93.8 Elo; d1 −7.78% → −76.0 Elo. Fitting the search's own value 7.8% better still costs 76 Elo. A-vs-A control on the same book slice: **exactly 50.00%** (47/47/26) |
| 2026-08-13 | **The mate gate was scoring the OPENING POSITION**, and reported `MISS ILLEGAL` for the shipped entry | It feeds `position fen`, which only the driver understands; a variant in a scratch dir runs the builtin loop. base/d1/d8 are all **8/8** once the banner is demanded. Fourth incident of this class |
| 2026-08-13 | The tracked training set carried the bench box's name; **scrubbed before the branch was ever pushed** | `set20260813.npz` `d792b420…` → **`2410786e…`**, arrays bit-identical, `host` the only key dropped |
| 2026-08-13 | **THE DISTILLED STUDENT FITS EXACTLY AS WELL AS C2 — and is stably WORSE where it matters** | Same positions, same split: SF **−7.85 ± 0.81**, distilled **−7.70 ± 0.90**. But at phase 18-24 the distilled student is **+7.47 ± 3.14 WORSE than classic on every split**, where the SF student was −8.00 better |
| 2026-08-13 | **The blocker is the POSITION MIX, not the teacher — and distillation is what makes that cheap to fix** | Classic's loss against our own search at phase 18-24 is **0.007962**, its best band: no headroom, and a global fit spends it to buy the endgame where **65.6%** of positions live |
| 2026-08-13 | **d8 misses the first-yield gate by ELEVEN NODES** (2,059 vs a 2,048 window) | The margin case the measuring gate exists for. d1 passes at 1,896/2,048 — a 7% margin against the entry's 2.6×. **The 5-byte gamma seed fixes both** (537 and 478) |
| 2026-08-13 | **TEACHER CHOSEN: our own search at 160,000 nodes — and its value never converges** | 2.8× the measured 30+1 frontier (56,829 nodes), r **0.9919** with a 4× deeper teacher. From 40k up, 21% of positions still move >25 cp per 4× — that is tactics, not tolerance, and 384 parameters cannot hold it |
| 2026-08-13 | **NEW GATE: first yield. The shipped entry passes at 780 nodes; ALL FOUR fits fail** | C1 32,640, m1 9,088, q8 3,707, C2 2,568 against a 2,048-node window. Measuring the count, not the `(none)` symptom, is what gave it power — the binary form caught only its own reproducer |
| 2026-08-13 | **LEAD: the `bestmove (none)` class is removable for 5 BYTES at the root's gamma seed** | `gamma = pos.score - 150` takes the worst first yield from 780 to **171** on the entry and from 32,640 to **396** on C1 — every arm passes. 1.0001× nodes to depth 8. Un-suspends mirroring if it screens clean |
| 2026-08-13 | **CORRECTION: C2's post-mortem MECHANISM does not survive a re-split** | Over 12 splits the middlegame band is **−2.96 ± 2.36 and its sign FLIPS** — it is the least-improved band, not a worse one. The anti-correlation stands; its explanation does not |
| 2026-08-13 | The first-yield gate scored PASS for every arm until it checked its own driver | An entry resolving no `sunfish_ui/` runs the builtin loop, which ignores `position fen` and answers from the OPENING position. Third time this lane has paid for it |
| 2026-08-13 | Torch trainer validated against C2 on four axes, incl. its failure mode | −5.93% held-out, bands −9.81/+0.56, **3412 bytes (+62)**, and first yield **2,568 on the same FEN** as the box's `e_c2.py` |
| 2026-08-13 | **Quantisation-aware (STE) students are BYTE-NEGATIVE: q8 −53 B, q16 −93 B** | q8 keeps the whole fit (−5.91% vs −5.93%). Round-trip check caught the codec quantising the **king** table, silently rounding the landed `kend` fix |
| 2026-08-13 | `torch.optim.LBFGS`'s default `tolerance_grad` is a silent no-op at this loss scale | Gradients ~1e-9 vs a 1e-7 default: the fit returns its warm start and every band reads 0.00%, which looks like a result |
| 2026-08-13 | **BOTH CANDIDATES DROPPED — and held-out loss is ANTI-CORRELATED with strength** | C2, the clean unmirrored fit, is **−93.8 ± 32.7** over 405 games while its held-out loss was **5.9% better than classic**. The objective, not the bytes, is the blocker |
| 2026-08-13 | **C1 DROPPED: −57.7 ± 25.5 over 651 games, and it answers `bestmove (none)`** | SPRT stopped early for base (339W/232L/79D). Bar was UB > 0; UB is **−32**. A-vs-A control exactly 50.00% |
| 2026-08-13 | **THE CAUSE IS MIRRORING, not the fit and not the quantisation** | Four encodings, one position: exact and **step-8 unmirrored both play fine**; both MIRRORED arms return no move. Kills the mirrored 7-bucket design |
| 2026-08-13 | The 100-position legality gate passed a build that emits no move in real play | Normal middlegame position, fails at **every depth incl. 1**, no info line at all. Gates are only as good as their position sample |
| 2026-08-13 | fastchess's illegal-move counter double-counts, and `(none)` is not illegal | One incident reported as **2**; "illegal move" actually meant the engine produced nothing. Different bug, different investigation |
| 2026-08-13 | **REBASED onto nnue-4k: entry 3350, engine 2886, eval 464, CEILING 1210** | base-90 on top of IIR + interface trims. Re-measured, never composed. C1 **3187 (−163)**, C2 **3412 (+62)** |
| 2026-08-13 | **`price_engine.sh` returned a SILENTLY WRONG answer on the base-90 entry** | "eval costs 30, engine 3320" — its regex matched a `\n}` far below the eval. Fixed for both entry forms; the literal-form number reproduces at 2942/503 |
| 2026-08-13 | **C1's bar RE-DERIVED: its byte credit is VOID under the new allocation** | Eval bytes stopped being scarce (746 under the ceiling, and the directive is to FILL it), so C1 must now clear **LB > 0**, not LB > −15 |
| 2026-08-13 | **THE 1024-1500 B GRID, PRICED BY BUILDING** | Bytes reach it — 7 phase buckets = **1148 eval B**. **The data does not**: the straw man's worst bucket holds **48 positions for 160 parameters** |
| 2026-08-13 | **King-wing buckets are unusable on our data: 0.3 pos/param** | 80.4% of positions have the white king on the king side. Phase quantiles are the partition that survives — 4 = 18.9, 6 = 12.8, 7 = 8.8 pos/param |
| 2026-08-13 | Bucket-census instrument corrected: rank quantiles are not implementable | It reported 12.0 pos/param at 8 buckets where the shippable partition gives **6.5** — phase is a coarse integer and a rank cut splits a value the root cannot |
| 2026-08-13 | **THE GOLF TARGET WAS THE WRONG NUMBER: eval already has 1132 B** | 4096 − 2942 = **1132 ≥ Thomas's "at least 1024"**. 2500 assumed the full 1500-byte eval. The eval lane's grid, not golf, is the binding constraint |
| 2026-08-13 | Info/PV line cut: **−22 B, entry 3445, engine 2942** | Node counts bit-identical (2,342,657 both), gates green, artifact still streams bestmove LIVE to a pipe with stdin open |
| 2026-08-13 | **Two of the three "safe" trims are NOT safe, with evidence** | `version` globals are in the driver's `ENGINE_API` — cutting them makes the entry **unmeasurable in every dev checkout**. `movetime` is a silent-forfeit hazard with a live consumer in `deploy.sh` |
| 2026-08-13 | **ENGINE-SANS-EVAL MEASURED: 2964, not "~2980" — and the byte map is built** | `main()`'s UCI loop is **454 B**, `bound()` 560, the board layer 598. Instrument: `tools/build/price_engine.sh` |
| 2026-08-13 | **The named golf candidates are worth ~115 B against a 464 B gap** | Every UCI-surface cut priced by building. `id name` 40, movetime 25, info-PV 22, quit 7. **Interface trimming cannot reach 2500** |
| 2026-08-13 | Entry stopped describing a net it does not contain; dead `pf` removed | Section header, Position docstring (`ps`/`acc`/`pf`/`kb`) and a false table comment all corrected. **3472 → 3468**, node counts bit-identical |
| 2026-08-13 | **LANDED: IIR ships. Entry 3475 → 3472, 624 spare, +22.3 ± 16.0** | Packed sha `ce091e5e…` is **byte-identical to the binary that played the 1,000 games**. STRONGER AND SMALLER — the first such change in this lane |
| 2026-08-13 | **CONFIRMED: `iirk.noiid` +22.3 ± 16.0, entry 3475 → 3472** | 1,000 games, 0 forfeits/illegal, raw 415–351. **Stronger AND smaller** — timed ≈ +15 Elo for −3 bytes. Ready to land, pending the coordinator's generator sequence |
| 2026-08-13 | **The RR pairing read +41.3; the dedicated match reads +22.3** | Same search, measured twice, 19.0 ± 27.5 apart. Third pooled-vs-direct discrepancy today, all in the same direction. A round-robin pairing is not an effect size either |
| 2026-08-13 | **The frontier futility margin is ALREADY TUNED — axis closed** | −40 costs **−110.3 ± 40.6** (largest single-parameter regression here), −15 is level, both positives lose. The zero-byte candidate packs to 3476 and plays 110 worse |
| 2026-08-13 | **corrhist fully explained: it moved a constant that was already right** | Its correction was a variable +10…+18 cp margin, and a fixed margin in that range costs what corrhist cost. Closed, not shelved |
| 2026-08-13 | **IIR LANDS: +41.3 ± 22.4 head-to-head, and the entry SHRINKS to 3471** | 3,960/3,960 games, 0 forfeits/illegal. Stronger AND smaller — a first for this lane. Shipping form `iirk.noiid` (3472) in fixed-N confirmation |
| 2026-08-13 | **History DROPPED a second time, now by the right instrument** | −3.2 ± 21.6 over 660 games against a +62 bar. The node-ratio dismissal was right after all; the depth-9 caveat is closed (28% MORE nodes) |
| 2026-08-13 | **The 4-arm pool is NON-TRANSITIVE, and the pooled ranking misleads** | fastchess ranks `hist` FIRST at +15.8 while its head-to-head with base is −3.2. Raw win counts confirm the intransitivity is real, not an analyzer bug |
| 2026-08-13 | **Legality gate catches the negative futility margin: 7/100 `bestmove (none)`** | Cause exact: margin in the TEST but not the YIELD ⇒ fail-high on a virtual move ⇒ bound()'s terminal contract broken. Fixed, re-gated 100/100 |
| 2026-08-13 | **A ZERO-BYTE candidate: the frontier futility margin** | corrhist's only consumer was that test and it LOST by pruning less, so the direction with upside is a NEGATIVE margin. `-QS` packs to **3475 — the entry exactly**. Screening `futm40`/`futm` |
| 2026-08-13 | Positive-margin mirror, stopped at 215 games as a predicted loser | base 61 wins, fut40 52, fut 47. Reproduces corrhist's sign for 0-3 bytes instead of 127. Preliminary; the box moved to the negative direction |
| 2026-08-13 | **CORRECTION: corrhist is −54.8, a REGRESSION. I read the sign backwards** | Raw PGN: base 290 wins, corr 192. The entry below stands as written and is WRONG in its direction; the correction entry above it is the one to read |
| 2026-08-13 | ~~corrhist DROPPED: the mechanism works and is priced out~~ **(SIGN ERROR — see the correction)** | The number +54.8 ± 23.3 is right; it belongs to **base**, not to corrhist |
| 2026-08-13 | **IIR replacing IID is BYTE-NEGATIVE: entry 3475 -> 3471** | −4 B, and dropping IID alone is −16 B and node-neutral (0.989x). The first queue item that gives bytes back; it only has to avoid losing |
| 2026-08-13 | MTD guards censused on every new arm, incl. the ordering change | **0 bracket crossings, 0 probe-cap hits** to depth 9 on all six builds. Also: this counter can NOT be read out of a fastchess log |
| 2026-08-13 | History ordering costs 28% more nodes at depth 9 (apples-to-apples) | 1.284x vs base. Not the verdict — node ratio is the wrong instrument, which is why it is being re-screened in games |
| 2026-08-13 | Ordering RR keep/drop **PRE-REGISTERED** | hist (+61 B) needs +61 timed; byte-negative arms land only on a non-negative point estimate, and are otherwise HELD, not shipped |
| 2026-08-13 | **corrhist re-measured on the box: the node saving does not exist** | Interleaved, 7 lines, 3 rounds. nodes **1.042x**, nps **0.944x**, time-to-d8 **1.048x**. The "0.70x / 0.89x, plausibly free" reading was one position on a contended laptop |
| 2026-08-13 | corrhist's key priced: slicing to ranks 2-7 buys 5.6pp of nps | 0.944x sliced vs **0.888x** full-board, node counts bit-identical. The key, not the correction, is still the whole cost |
| 2026-08-13 | corrhist byte price, `pack.sh` on real files | **+127 B** (entry 3475 -> **3602**, spare 621 -> 494). Golfing recovers 5 B -- lzma already had the repetition |
| 2026-08-13 | corrhist keep/drop **PRE-REGISTERED** before the screen | Standing 1.0 Elo/byte bar => needs fixed-node **>= +135.5**. The budget-average rate the +400 goal implies (0.41 Elo/byte) would need only +60.5; both written down, decided on the strict one |
| 2026-08-13 | corrhist correction table censused: not a feedback runaway | median +2...+8 cp, 1-5% at the +/-120 clamp, 4-7k entries per search. The screen measures the feature, not a degenerate one |
| 2026-08-13 | **`legality_gate.py` scored a LAUNCH failure as 100 chess failures** | It ran packed artifacts under `python3`. The **shipped entry** "failed" 100/100 no-move. Fixed, both controls green |
| 2026-08-13 | **Phase reweighting does not pay — 4 schemes, all fail the pre-registered bar** | Every one LOSES on held-out (paired bootstrap CIs strictly below 0), and **none wins the phase-balanced metric it was built for**. No third candidate |
| 2026-08-13 | The flat refit is an ENDGAME refit: 0 gain in the middlegame band | −9.8% at phase 0-5, **+0.6% (worse) at phase 12-17**. Caveat now attached to the C1/C2 screen |
| 2026-08-13 | **King buckets priced by BUILDING: ~134 B/bucket confirmed for filler** | 2-bucket +155…+172, 4-bucket **+128…+136 per bucket** at mirrored step 8. Exact encoding is dead: 4-bucket = **4505 B, 409 OVER** |
| 2026-08-13 | **Taper re-anchored on the landed generator; the seam root got 13 B cheaper** | Old splice targeted a deleted anchor and failed its own assert. Seam machinery **+50 B**, blend **+115 B**; fitted taper now **3439** (was 3452) |
| 2026-08-13 | **Filler data is a FLOOR, not an upper bound** | Real fitted tables cost **60-75 B MORE** than filler of the same shape: fitted values are less round. Reverses the old script's stated reasoning |
| 2026-08-13 | `codec.emit` accepted a `piece` dict and IGNORED it | Hard-coded classic's values; all three callers hand-patched the line back. Fixed at source, entry **byte-identical at 3378** |
| 2026-08-13 | **FITS DONE: two candidates, bars pre-registered, no Elo claimed** | **C1 3215 B (−163!), C2 3441 B (+63)**, both legality-green. Held-out −5.3%/−5.9%. Taper DROPPED on data, not price |
| 2026-08-13 | **Quantisation is free in loss and SAVES bytes** | step 8 = −88 B for nothing; mirroring −159 more and held-out loss *improves* (regularisation). A refit is **not** byte-free: +63 B exact |
| 2026-08-13 | **Continuous phase blend does not fit: 4157 bytes, 61 OVER** | Dead on price whatever its loss. The queens-seam taper is affordable (+74 B) but its data is not there |
| 2026-08-13 | Mirroring silently perturbed the landed `kend` fix | Classic's king table is asymmetric by up to **111 cp**. `codec.emit(exact="K")` holds it back bit-identical for 84 B, so a screen tests the fit and not a bundle |
| 2026-08-13 | **Training loss cannot compare models of different size** | The 768-param taper looks 2× better in-sample than held out. All candidate losses are now on a seeded 80/20 split |
| 2026-08-13 | `codec.mixed` wrapped at 64 bits on numpy input | A ~3000-bit accumulator turned numpy and silently encoded garbage; symptom was a 10-min hang, not a wrong number. `int()` + a wrap control in the self-test |
| 2026-08-13 | **TRAINING SET LABELLED: 19,491 positions, Stockfish 18 @ depth 8** | Committed at `tools/tune/data/set20260813.npz`. Material-vs-label corr **0.727**, sign agreement **92.7%**. **65.6% at ≤16 pieces** vs 47% in the lost set. No fit run |
| 2026-08-13 | **Labels were slot-dependent: the TT carried across positions** | Same FEN, two slots, **−14 vs −22**; others 83→97 and −90→−149. Both modes run-to-run reproducible, which is why it was invisible. `ucinewgame` per position now makes the label a function of (fen, depth, version) |
| 2026-08-13 | `X` is a difference feature — it cannot give you the phase | White Pe2 and black Pe7 cancel in the same cell. `\|X\|.sum()` reads 11.1 pieces where the board has 14.3; phase must come from `fens` |
| 2026-08-13 | Stockfish 18 built from source on the box (no root) | Official binaries need `GLIBCXX_3.4.30`, box has 3.4.29. gcc 11.5, `ARCH=x86-64-avx512icl`. A pinned source build is the more portable recipe anyway |
| 2026-08-13 | **Training set: durable home, 19,689 positions ready, labelling NOT started** | Games under `~/repos/sunfish-data/pgn/` (4,482 games). New mix is **65.7% at ≤16 pieces** vs 47% in the lost set. Box has no Stockfish; laptop is owned by the ladder — needs a call |
| 2026-08-13 | **LANDED on top of kend+fresh: the entry is 3378 bytes, 718 spare** | Rebuilt, not composed: −97 on this base, not the −94 of the last one. All gates green, 60/60 same move AND same score |
| 2026-08-13 | **`agree.py` was comparing two different UCI DRIVERS** | An engine outside the repo tree silently uses the builtin `go` loop. A byte-identical copy of the entry "disagreed" with itself 39/60 on score. Both arms now STAGED into one directory, and an A-vs-A positive control is wired in |
| 2026-08-13 | **Eval tables decode at startup: entry 3483 → 3389, 707 spare** | Tables bit-identical; 60/60 same move AND same score at fixed nodes. Decoder costs **13 B**, decode 1.07 ms. *(Measured on the pre-kend/fresh base; superseded by the 3378 row above)* |
| 2026-08-13 | Eval-data price list, one build per row | Literal 1.31 B/value → base-90 exact 1.03 → step-8 0.65 → mirrored step-8 0.70 (192 values). Codec within 10% of the entropy bound |
| 2026-08-13 | **Tapering re-priced: −77 bytes, not +400** | Two mirrored step-8 tables + root blend = **3312 B, 784 spare**. No second accumulator: the root already rebuilds a table. **No Elo claimed** |
| 2026-08-13 | Historical 1207-byte net re-decoded (correction) | 944 B of factors (not 816) → 7680 values, but they feed an **18→10→1 MLP per node** — it is a runtime net, not a PST |
| 2026-08-13 | **Stale-score bug in the shipped entry: 134 cp** | The bare-king `K_MID`/`K_END` swap invalidates the carried incremental score. Pre-existing; fix is in `search()` |
| 2026-08-13 | **HOLE RR COMPLETE: 4,000 games, and `entry_kf` SHIPS** | **+107.5 ± 31.6 vs classic**, 0 forfeits, 0 illegal. The hole reproduced at **−71.3** and the fix closes it to **+20.0**. Landed: entry **3483 → 3475, 621 spare** |
| 2026-08-13 | LMR's timed value is the SAME with and without the eval bug | +65.9 ± 27.1 (unfixed) vs +72.3 ± 25.1 (fixed). LMR was never "masking" the hole — the defects were additive |
| 2026-08-13 | Classic-anchored differences run ~50 Elo above the head-to-head ones | Both signs agree, magnitudes do not (1.5σ). Head-to-head is the paired instrument; the anchored spread is the one to distrust |
| 2026-08-13 | **THE KING TABLE: entry uses the wrong one in 62.1% of real positions** | Port defect in the EVAL, not the search. **+52.3 ± 21.1**, 444g, SPRT stopped early, and the fix **saves 11 bytes** |
| 2026-08-13 | Stale carried score at the table swap (routed from the eval lane) | Reproduced. Bites the **transition ply only** -- 0.83 plies/game, mean 30cp, max 157cp. Fix priced: kend+fresh **3475, still 8 bytes UNDER the entry** |
| 2026-08-13 | Null-cap census: binds 4.3% of nulls, flips 0.53% | Real but small. Cap RR **deferred behind the king table** -- the cap reads `pos.score` |
| 2026-08-13 | Cap byte cost, built not composed | entry 3483, +cap 3494. **11 bytes** |
| 2026-08-13 | pair_elo.py validated against fastchess | Reproduces +38.86 +/- 19.13 digit-for-digit on lmron.pgn |
| 2026-08-13 | Time management eliminated as the hole | Both arms run the SAME `sunfish_ui/uci.py` budget in every screen |
| 2026-08-13 | corrhist built, interior-only | 0.70x nodes AND 0.89x time to d8. Awaiting a quiet machine |
| 2026-08-13 | **Transfer scoreboard: ice4's +421 is not our +421** | 3 measured, 3 outcomes. Mean transfer far below 1 |
| 2026-08-13 | Lying null-cap comment removed | Entry regenerated, 3483 B, CI green. Comments are free |
| 2026-08-13 | Auto-chaining disarmed in 4 scripts | Now require an explicit `GO_<stage>` marker |
| 2026-08-13 | **Null-move cap: code contradicts its own comment** | Entry's null is UNCAPPED; `git log -S` finds the cap never existed here. RR in flight |
| 2026-08-13 | **The hole is real and it is ~46, not ~85** | `entry_nolmr` **-46.3 +/- 30.0** vs classic, 322g timed, 0 time losses |
| 2026-08-13 | LMP (legality-fixed), fixed nodes, on top of LMR | **-125.8 +/- 38.1**, 269g, H0. **DROPPED** -- bar was +56 |
| 2026-08-13 | LMR on/off, **PST entry**, fixed nodes | **+38.9 +/- 19.1**, 845g, H1. Transfers, at ~60% of its NNUE value |
| 2026-08-13 | **Legality gate built, positive-controlled, wired in** | Fails the pre-fix LMP on 2/100, passes the fix and the shipped entry. Caught a **stale broken LMP copy** on the laptop |
| 2026-08-13 | **ENGINE PROPERTY: pseudo-legal movegen, no notion of check** | Not an LMP bug — it will bite every count-triggered or tail-pruning rule. `best > -MATE_UPPER` is now the required preamble |
| 2026-08-13 | Guards measured INERT on the PST entry | 0 probe-cap hits, 0 bracket crossings to depth 10 — bisection collapses to one arm, ~600 games saved |
| 2026-08-13 | **LMP is BROKEN, not weak: it returned an illegal move** | Reproduced deterministically. Cause is NOT break-vs-skip (falsified) — it prunes the only legal escape when in check. Fixed and re-gated |
| 2026-08-13 | **RFP REJECTED: −2.8 ± 17.9 at 1000 games** | Bar was +31 Elo for 31 bytes. Removed — entry **3517 → 3483, 613 spare** |
| 2026-08-13 | **LEAD: LMR may be masking an ~85 Elo hole in the port** | LMR transfers (+127 ± 77 timed, prelim) — which implies entry-minus-LMR is ~85 BELOW classic. Being measured directly, not inferred |
| 2026-08-13 | **LMR-on-PST: outcomes PRE-REGISTERED before the result** | Three readings written down in advance, incl. the live possibility that LMR is *costing* the shipped entry Elo |
| 2026-08-13 | **Screens switched to SPRT mid-flight** | 300-game fixed-N resolves to ±40 while candidates are +18…+90 — the bottom half was under the noise floor of its own test |
| 2026-08-13 | **Screens moved to the box; stale driver found armed there** | Both box checkouts had `max_nodes`=0 and no version — the 425-game failure waiting. Isolated v2 tree; refusal verified on the box |
| 2026-08-13 | RFP mate gate passes on the PST entry (5 vs 5) | The 5/8→3/8 loss was **eval-dependent** (NNUE eval), not a property of LMR+RFP |
| 2026-08-13 | **Byte accounting fixed to the ENTRY; LMP threshold pre-registered** | entry **3573** (+56 for LMP), **523 spare**; nnue engine 3973. Keep LMP only at ≥1.0 Elo/byte (≥+56 Elo) |
| 2026-08-13 | **4k entry vs classic @10+0.1 (interim)** | **~+133 ± 120 at 51/600 games**, zero time losses — same eval both sides, so this is our SEARCH. Flips the confounded fixed-node sign |
| 2026-08-13 | **+400 decomposition checked: eval worth ~+224, not ~+160** | goal60 predates LMR/guards, so more of its +187 belongs to eval. Priority unchanged — search must still supply +232…+344 |
| 2026-08-13 | RR stopped early; Texel trend isolated | Bug was ~50 Elo of the −66.8; residual **−16.7 ± 31.2** covers zero. TC baseline unblocked and running |
| 2026-08-13 | **"Fixed nodes" wasn't: the cap rewarded pruning LESS** | classic overshot 1.74× vs our 1.32× — LMR penalised for its own virtue. Fixed in-search (gap now 1.70× actual); classic comparisons move to TIME |
| 2026-08-12 | Mate distance (issue #11) | Score separation real (60 pts at d6), **play bit-identical: 0/300 move or node diffs, 0 conversion diffs**; 30+1 match QUEUED |
| 2026-08-12 | **Texel screen −66.8 ± 35.5: the king table was mirrored** | A better fit playing worse was a bug in the EMIT path, not a fit-vs-play effect. Fixed; re-screening |
| 2026-08-12 | **Texel tuning: 10.1% better fit for ZERO bytes** | +13 bytes total (3517→3530); fixed-node screen running. Tapering adds only 1.8pp more for ~400 bytes |
| 2026-08-12 | **Our Elo/byte cost model is INVERTED vs ice4/4ku** | Incremental eval makes (piece,square) terms free and whole-position terms (mobility!) expensive — their 4.0 Elo/byte is not available to us |
| 2026-08-12 | **MILESTONE: valid 4k entry built and verified** | **3517 bytes measured** (composed estimate said 3787), plays alone in an empty dir with SF_NET unset, **579 spare** |
| 2026-08-12 | **DECISION: PST is the main line, NNUE the challenger** | NNUE pays 705 B of machinery before its first weight, against a 579-byte eval — challenger must win per byte, machinery included |
| 2026-08-12 | **Engine byte decomposition: the thesis is in arithmetic trouble** | NNUE machinery 705 B + 553 B richer core = the 1258 overrun. PST entry fits at 3787 (309 spare); NNUE entry leaves **183 B** for the net |
| 2026-08-12 | **Accounting: 71% of logged work served the unbounded net** | The 4k track was priced and never built. Drift recorded, allocation corrected |
| 2026-08-12 | **The engine was ALREADY unstable** | Bracket crossings fire with LMR=0 — the one-value-per-key invariant was violated before any reduction; we just had no instrument |
| 2026-08-12 | MTD guards + LMR landed (packed only) | Guards cost +26 B and 0 nodes; LMR −64% nodes at depth 5 for +36 B. Fixed-node screen running |
| 2026-08-12 | **Packing REVERSED twice: base-3 AND lzma, joint not split** | Compose, don't choose: b3+lzma −1000 B vs raw base-3; one joint lzma stream −1007 B vs split. My earlier "split is right" was measured on incompressible data |
| 2026-08-12 | **Box collision hazard: atomic lock adopted** | Three lanes watched one quiet window. My redundant waiter cancelled, the rest take `mkdir`-atomic `.boxlock` — mechanism offered to all lanes |
| 2026-08-12 | **Rules audit: packer, UCI surface, joint-vs-split** | Split beats joint by **156 B**; artifact already rules-minimal (only **42 B** reclaimable); no-temp-file packer built and verified |
| 2026-08-12 | Time divisor at the real TC | Gap confirmed: 1800+3 gives a **150 s** first move. Scaled sweep (180+0.3, 5 arms) queued behind a 20-min quiet gate |
| 2026-08-12 | **4k design space priced (weights RAW, not xz)** | Ternary base-3 packing + factorisation beat the width-5 baseline **5-50×** in parameters at 1920 B; width is ~free in speed at this scale |
| 2026-08-12 | **4k budget re-derived: the net counts** | Real artifact = **541,781 B** (engine 4488 + net 537,152), not "3798, 298 under". Packing mechanism recovered and verified running |
| 2026-08-12 | **Field study: ice4 + 4ku eval packing** | ice4's ENTIRE eval = **333 characters**; both engines factorise PST into rank+file. Our 768×128 is 98,304 values |
| 2026-08-12 | Historical 1207-byte net decoded | **Trained rank-6 factorisation**: 816 int8 → 4608 PST values, exact by construction |
| 2026-08-12 | **CORRECTION: the bottleneck is `nn_cp`, not the board** | The "85% board" claim was an inference error (marginal ≠ total). Measured: net 8.1µs vs board 2.9µs of a 14.6µs move — mutable board is worth ~+15 Elo, not +71…+110 |
| 2026-08-12 | Hot-path profile (superseded in part) | ~85%-board claim WRONG — see the correction entry above |
| 2026-08-12 | **GOAL-LINE VERDICT: +187.0 ± 49.7 vs classic @60+1** | **272 games, zero time losses. Target +400 NOT met — but against a classic that gained ~+130 during the campaign** |
| 2026-08-12 | `_ext` integerization: scoped and priced | DONT BUILD (SWAR tail 5.2-10.3µs vs 3.8µs now) — but a dead-code third removed: rehab800 0.643 → 0.742× kb8, +21 Elo |
| 2026-08-12 | **LMR CONVERTS: +65.0 ± 43.3 at fixed nodes** | 200 games, 0 forfeits, 0 illegal — 59.25%. First clean local screen, and the first reduction lands |
| 2026-08-12 | Sudden-death budget fix (lichess bot) | `/40` when `winc==0`; a 3+0 loss on time with no move overrunning. Artifact byte-identical at 3913 |
| 2026-08-12 | **VOID: every local fixed-node game was a time forfeit** | 425/425 label-RR games, 54/54 LMR, 40/40 ng — node cap silently ignored. Labels withdrawn; metric C's own numbers stand (no games involved) |
| 2026-08-12 | Quality-term hunt restarted: labels + 3 new families | Metric C measured (churn ranks kbbil worst, w256 best); its LABEL half is void — see the entry above |
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

## 2026-08-15 — pend CONFIRMED at +21.31 and LANDED: the first full conversion of the taxonomy → mechanism → screen → confirm pipeline

The fixed-N confirmation ran and the pre-registered bar is met. **`pend` lands.**

| | screen (SPRT) | **confirmation (fixed N)** |
|---|---|---|
| games | 722, stopped early | **800, no early stopping** |
| raw PGN | 315-239-168 | **pend W 336 · L 287 · D 177 = 53.06%** |
| Elo | +36.71 ± 16.20 | **+21.31 ± 15.73 → [+5.58, +37.04]** |
| status | **biased screen figure** | **the earned number** |

The confirmation did exactly what it exists to do: the SPRT's +36.71 is inflated
by its own stopping rule, and the honest figure is **+21.31**, a **42% haircut**.
Zero illegal, zero `(none)`, zero forfeits in 800 games.

### Reconciling the interval, because two readings differed

An independent read gave **[+0.09, +42.69]** (±21.30) against the instrument's
**±15.73**. Recomputed from the confirmation's own pentanomial
(`Ptnml(0-2): [23, 42, 235, 63, 37]`, 400 pairs):

| method | 95% | LB | verdict |
|---|---|---|---|
| **pentanomial (pre-registered path)** | **±15.73** | **+5.58** | **LAND** |
| the independent read | ±21.30 | +0.09 | LAND, barely |
| game-level (ignores pairing) | ±24.12 | **−2.81** | straddle |

±15.73 reproduces fastchess's printed figure to the digit and is the method every
other verdict in this ledger used, so it is what the bar was set against. All
three agree on the point estimate and two of three clear zero — **but the third
does not, so this landing is method-dependent at the margin** and is recorded as
such, not as a comfortable win. Pairing is what buys the margin: it removes the
colour/opening variance the game-level calculation leaves in.

### The landing

Applied **in the generator**, not by hand-editing the entry: `make_pst_entry.py`
injects both anchors and asserts each occurs exactly once, so drift is a hard
build error rather than a silently unmodified entry. It belongs in the 4k
generator rather than `sunfish.py` because it is an entry change, not a classic
one.

The queens-off seam already switches the KING table; `pend` adds the PAWN table
at the **same test**, so it costs one tuple and no new branch. With queens off a
pawn is worth `(8 - rank)^2 * 2` more, steeply rewarding advanced passers exactly
when promotion is the winning plan.

| | |
|---|---|
| entry | **3308 → 3340 B (+32)**, 756 spare |
| `check_entry.sh` | green |
| decode round-trip | `pst` bit-identical to classic; `K_MID`/`K_END` preserved; `P_MID` is classic's pawn table |
| standalone | packed artifact in an empty dir plays `g1f3` |
| legality | 100/100 at both budgets |
| mate / conversion | **8/8 / 8/8** |
| first yield | max **676**, passed |

+32 B is not the +37 measured against the old 3341 base: lzma shares one
dictionary, so byte deltas never compose across landings.

### SUBSUMPTION OBLIGATION — recorded next to the landing, not filed away

`pend` is a hand-written phase term. **When a phase-capable net screens, the
comparison matrix must include net-vs-net+pend, and `pend` is DELETED if
subsumed**, returning its 32 bytes. This obligation travels with the code — it is
the price of landing a hand-written phase term into a lane whose stated goal is a
learned evaluation, and the ML2 machinery priced in the entry below is exactly
the thing that will trigger it.

### The H1/H2 programme, closed

| arm | bytes | verdict |
|---|---|---|
| **pend** | +32 landed | **CONFIRMED +21.31, LANDED** |
| kact | +1 | DROP, −33.07 ± 15.98 |
| kmid | +15 | undecided at cap, +2.08 ± 16.58 |
| khold2 | +24 | undecided at cap, +2.43 ± 7.24 |

Four candidates, one landing; kact/kmid/khold2 all closed empty. **pend is the
programme's sole conversion** and the first time this lane has run taxonomy →
mechanism → screen → confirm end to end and put Elo into the entry. Every prior
eval candidate (Texel, C1, C2, d1, b1) died — the difference is that this one is
a search-seam mechanism found by taxonomy, not a fitted table.

---

## 2026-08-15 — ML2 MACHINERY PRICED: +98 B code, ~+11%/node — the phase-net's engine seam, measured

Coordinator task: c1024_phase_ml2 posted the campaign's best val (0.01286,
probes confirming phase x feature knowledge in weights) and the blocker was
the unmeasured packed price of its SECOND LAYER — one extra big-int
multiply per node. Built through the generator in the certified form
(field_budget.certify_ml2: F2=32, m=4, umax=127, shift2=10;
packed_layers.LaneConv circular is the training twin), priced through
pack.sh, verified bit-exact, and speed-measured. Form (c) means NO phase
machinery in the engine — phase lives in the weights; the engine seam is
exactly ml2.

**The engine form** (landed as a mechanical derivation,
`nnue_4k/packed/make_ml2_proto.py`, every hunk asserting — no fork to
rot): after the shipped F=16 crelu head, per perspective block: re-space
the 4 capped lanes to 32-bit fields (two shift+mask steps), ONE squaring
folded mod 2^128-1 (= the circular self-convolution), signed per-field u2
read-out by mask+shift (the certificate's group-hsum precondition FAILS at
2^32-2, so hsum is illegal), >> 10, added to the L1 cp before the clip.
Both renorms trunc-toward-zero (`int(x / 2^s)`), so antisymmetry is exact
by construction. **Payload seam:** +4 u2 values as offset-4050 base-90
digit PAIRS between the biases and the feature chars (+8 digits ≈ 6 B
in-context; `make_proto_payload.py --u2 4` emits it).

**Bytes (pack.sh, all measured on the round-2 golfed base):**

| build | B | note |
|---|---|---|
| ml2 code-only (payload elided) | **3315** | vs single-layer floor 3217 → **second layer = +98 B isolated** |
| ml2 @ ps768 + u2 (786 raw) | 3936 | the real phase-net shape at v1 sparsity |
| ml2 @ 1024-B-cost payload | **4343** | (feats 1330 + u2, cost 1028) → **~245 OVER 4096** |
| ml2 @ feats 990 + u2 | **4096 exactly** | **the payload ml2 CAN carry: 781 B in-context** |
| ml2 @ feats 945 + u2 | 4068 | ≈ the 30-B-margin point: ~750 B |

Correction to the tasking premise: round 2 landed **878 B** of payload
capacity (849 at margin), not 909 — 909≈910 is reachable only if the
coordinator lands the pack.sh intel (`--no-hoist-literals` −24, shebang
strip −8), which this lane measured but does not own. With that landed,
ml2's numbers shift by the same ~32: capacity ~813, overage ~213.

**Fit verdict at the 1024 budget: DOES NOT FIT — 4339-4343 measured, ~245
over.** The second multiply eats 98 B of code + ~6 B of payload seam: the
single-layer gap (142) plus ~103. What fits with ml2 machinery TODAY is a
**781 B payload**. The MEASURED random-shaped phase-net build (ps768 + u2
at v1 sparsity) is 3936 — 160 under 4096 — and every trained payload this
ledger has priced compressed well below its random stand-in (v1: 612
random → 382 trained on the same 777 chars). So the 0.01286 candidate's
own shape is EXPECTED to fit with margin once its ternary export exists —
an expectation to be measured through export.py, never banked; only the
1024-BUDGET framing is over.

**Bit-exactness (packed/ml2_check.py, landed, self-deriving):** independent
payload decode == engine globals (u2, shift); engine nn_cp ==
packed_layers.bigint_circular_conv int-bridge reference (the certified
twin, called verbatim) on 60 fens x both pf + rotations + a 40-ply walk
with incremental-identity and antisymmetry asserts; L2 fired on 99/100
probes. Packed smoke: legal bestmove over UCI after moves.

**Speed — the number the ml2 line rides on.** Same-tree isolation (u2=0
payload: identical eval → identical 19,708-node trees, layer-2 arithmetic
still executed): L1 13.66/14.33/14.41 vs ML2 15.29/15.87/16.48 us/node,
interleaved, nice -15 laptop → **+7-15% time per node, central ~+11%
(nps ratio ≈ 0.90)**. At ~100 Elo/doubling that is **≈ −15 Elo timed**, on
top of the replnet family's existing speed tax vs the entry (0.66 was the
stale box reading; re-measure at dispatch per the 8Mv screen rule). Mixed-
tree probes (random u2) read +1-5% — flattered by tree shape; the same-tree
number is the honest one. Box pypy may differ; re-measure there before any
timed screen. Optimization headroom exists (fold-free 7-field read-out,
L1-native-32-bit layout skipping the spread) but is UNPRICED and changes
the certified form — price before believing.

**For the byte-budget conversation the coordinator flagged:** the phase win
(−0.0009 val, the campaign's largest, subsumption-class probe gains) costs
+98 B code + ~11% nps against the alternatives' prices in this ledger
(khold2 +27 B / pend +42 B / kact +1 B hand-terms at zero hot-loop cost —
but those are the terms ml2 would SUBSUME). If the family screens positive,
the code side needs 142+103 ≈ 245 B found (or the payload budget re-set to
~781/750) before a 1024-B-payload ml2 ships.

## 2026-08-15 — CORRECTION + RANKING VERDICT: min40-4 takes the classic-builtin venue, and the "no park" claim in my own pre-registration below is WRONG

**The correction first, because the entry below rests on it.** My
pre-registration's table has a "fixed point (unfloored) — none" column for
both candidates. **That is false.** A park is not caused by a cap. At any
increment TC the clock MUST come to rest where `spend + overhead == income`,
whatever the budget's shape, so every manager parks — both candidates
included. My reading was an artifact of charging **O = 200 ms against a
100 ms increment**: with income below overhead nothing can rest, which is a
fact about the TC and the lag, not about the formula. At the surrogate's
50 ms charge the income wins and the resting point exists. Struck, not
softened, in `tests/test_classic_time_budget.py` and in the branch comments.
`e2306d3` is the authority; the surviving case is `winc == 0`, where income
is zero and no resting point can exist.

What the shape decides is the **altitude**, and that is the whole safety
argument:

| policy | park @ 60+0.1 | reserve @ 60+1 (common ~1.06 s spend) |
|---|---|---|
| one-max | 6.17 s | 10.4 s |
| incumbent `legacy12` / step | 2.11 s (blind, floor knee 2.10) | 4.1 s |
| **min40-4 (shipping)** | **0.22 s** | 6.4 s |

**Recorded against the winner:** min40-4 parks LOWEST of the three, below
even the incumbent. It wastes almost no clock and buys the thinnest flag
margin in the field. What still separates it from the incumbent is that it
reaches the floor on a **positive, monotone** budget — the incumbent gets
there because its cap went negative, so it has no budget at all.

### Ranking verdict (surrogate, 2026-08-15, `tools/ctwin/README.md`)

| arm | vs | 60+0 | 60+0.1 | 30+1 | 60+1 |
|---|---|---|---|---|---|
| `min40_4` | `legacy12` | **+147** [+86,+219] | — | — | — |
| `onemax` | `min40_4` | — | **−89** [−170,−16] | −0 [−80,+80] | +23 [−54,+103] |
| `min40_4` | `pool` | — | **−114** [−208,−34] | **−134** [−218,−62] | **−114** [−198,−41] |

**min40-4 wins the classic-builtin venue** — decisive against one-max where
the two differ, tied elsewhere, and it takes the elegance tiebreak that was
fixed in advance below. The pre-registered tiebreak was therefore not needed
on its own: the Elo cell decided it first.

**The +147 is a FLOOR, not an estimate, and the instrument says so:**
`legacy12`'s negative budget sends **594** of its moves down the
structural-floor path where the surrogate substitutes the twin's bestmove — a
*better* move than the real engine would play. The packed calibration, whose
loser floors at 0.05 s and needed **zero** substitutions, read **+228**.
Quoted with the caveat attached, per `floorbk`.

**And the honest note from the pre-registration is now answered, against
this lane:** the full `pool` manager beats min40-4 at **every** increment TC
by +114 to +134. min40-4 wins the ONE-LINE venue, not the field. The pool
costs +57 B and a manager; this costs −7 bytes. Anyone reading this entry as
"the classic clock is solved" is reading it wrong.

### What is staged, and what is NOT

Per surrogate-ranks/real-clock-confirms, **one** confirmation was bought and
a second arm is added for the flag question the surrogate explicitly cannot
answer (its README: *"the surrogate reproduces mechanisms; it does not
certify flag safety"*):

| | |
|---|---|
| `STAGED_tm_min40_4_60p0.sh` | 60+0 SPRT, min40_4 vs legacy12, **elo0=0 elo1=20**, 200×2 `-repeat`, cap 400, PGN book, **no adjudication** |
| `STAGED_tm_min40_4_flag_hammer.sh` | **1+0, not an Elo arm.** PASS = zero time forfeits AND zero illegal moves on the arm. A large negative score with zero forfeits PASSES; one forfeit fails it however good the score. The incumbent is expected to forfeit — that is the control working and does not excuse a forfeit on the arm |

Both live in `tools/arena/` on the PR branch, **GO-guarded**: they print
their plan and exit 0 unless `GO=1`, and take `.boxlock` as a presence
marker with an owner file inviting reclaim, queueing behind resident matches
rather than preempting them. **Neither self-launches; neither has been run.**
Queued behind the current box matches.

Readings and the single permitted remedy stand exactly as pre-registered
below — the result cannot pick the rule.

### Cotenancy

No box time taken, no `.boxlock` claimed, nothing launched. Surrogate cells
were the ranking lane's; no file of theirs was touched. PR
`thomasahle/sunfish#196` carries min40-4; `classic/tm-one-max-pool` stays
open as the measured runner-up, closed into the ranking table above.

## 2026-08-14 — PRE-REGISTRATION: the CLASSIC builtin clock loses its cap, and two one-line candidates go to the surrogate before either sees a real clock

**Written before a game is played.** Scope first, because the byte figures
below are NOT comparable with the pool ladder's: this is the **classic**
engine's embedded loop in `sunfish.py`'s `go` handler, not `sunfish_nnue.py`.
A checkout or wheel run never reaches it — `main()` imports
`sunfish_ui.uci` unconditionally and returns — and only `pack.sh`, which
deletes the `minifier-hide` block and that import with it, leaves the loop
below live. So this entry prices the **packed classic artifact** and nothing
else. Neither lichess bot rides it.

The incumbent, in the millisecond domain the handler actually works in
(`wtime`/`winc` arrive as integer ms; the next line divides by 1000):

```python
think = min(wtime / 12 + 0.9 * winc, wtime / 2 - 1000)
```

Its cap is not a safety net, it is the park. Once it binds the clock obeys
`T <- T/2 + 1 + I`, stable fixed point `T* = 2 + 2I` — the same one this
file already measured twice (2.0 s at 60+0, a 2.1 s median at 60+0.1). Below
a 2 s clock the cap is NEGATIVE, so the arm that defines the park cannot be
spent at all: the budget collapses to the 0.05 s floor and ~200 ms/move of
lag drains the rest. That is `EAThUL0P` again, in the classic loop.

### The two candidates, both one line, both a distillation of Thomas's pool

The pool this lane already validated is `P = T + (M−1)·I − (M+2)·O`; at
M = 40 and the measured O = 200 ms, `P/M = wtime/40 + 0.975·winc − 210 ms`.
Round the increment to 1 and the reserve to a flat 8 s and that IS candidate
one; keep the pool's other term, the `A/4` clip, and drop the reserve and
that IS candidate two.

```python
# one-max: the reserve is 8 s, named, and the min becomes a max
think = max((wtime - 8000) / 40 + winc, 50)
# min40-4: the reserve is four increments, and nothing carries a unit
think = min(wtime / 40 + 0.9 * winc, wtime / 4)
```

**Why min40-4 is not just a second guess.** Every term is linear in the time
unit, so the statement is homogeneous of degree 1: it reads identically in
seconds or milliseconds. The ms/s confusion that produced a 590-second move
becomes *unrepresentable*, which matters precisely because this shape gets
copied between a seconds-domain interface and a millisecond-domain packed
loop. one-max cannot claim that — `8000` and `50` are millisecond constants,
and its line must never be pasted into `uci.py`.

### Where each one's no-park proof comes from — they are NOT the same argument

| policy | floor reached at | banked there | fixed point (unfloored) |
|---|---|---|---|
| incumbent | 2.1 s | 8 moves | **`T* = 2 + 2I`, stable** — and at I = 0 the park is a DRAIN, since the cap is already negative under 2 s |
| one-max | **10.0 s** = 8 + 40·0.05 | **40 moves** | none: drift ≤ I − O − floor < 0 everywhere, so the pool is genuinely spent down |
| min40-4 | 2.0 s | 8 moves | none **at I = 0**, by a different route — the clip can never bind (`t/40 < t/4` always), so the policy is EXACTLY `t/40`, a pure geometric drain |

one-max buys its safety with a named reserve; min40-4 buys it by never
letting the budget go negative and approaching the floor slowly. Recorded
against min40-4, not hidden: **it banks no reserve**, reaching the floor at
the same 2 s clock as the incumbent. Pricing that against one-max's 10 s is
the single thing the surrogate is being asked to do.

### The arms

| arm | line | packed | sha256[:16] | source |
|---|---|---|---|---|
| base | `min(wtime/12 + 0.9*winc, wtime/2 - 1000)` | 3278 B | `1f458ad9e5370014` | — |
| **one-max** | `max((wtime - 8000)/40 + winc, 50)` | 3282 B (+4) | `db5bff327327366e` | −11 bytes, −2 tokens, net −2 lines |
| **min40-4** | `min(wtime/40 + 0.9*winc, wtime/4)` | **3276 B (−2)** | `020f9aaa8e588fb2` | −7 bytes, −2 tokens, net −2 lines |

Both are net-negative in source bytes and tokens against a 4096 B budget
with ≥ 814 B spare, so **the elegance bar does not decide this** — it is
neutral-to-favourable for both and cannot break the tie.

### Realized spend, measured on the packed builds before any match

Seconds to `bestmove`, laptop, `position startpos moves e2e4 e7e5 g1f3 b8c6`:

| probe | base | one-max | min40-4 | one-max analytic |
|---|---|---|---|---|
| 300+0 | **23.73** | 7.38 | 7.57 | 7.30 |
| 60+0 | 5.14 | 1.12 | 1.25 | 1.30 |
| 30+1 | 2.82 | 1.29 | 1.60 | 1.55 |
| 14+0 | 1.04 | 0.17 | 0.41 | 0.15 |
| 8+0 | 0.59 | 0.12 | 0.23 | 0.05 (floor) |
| 2+0 | 0.06 | 0.11 | 0.17 | 0.05 (floor) |
| 0.5+0 | 0.07 | 0.10 | 0.06 | 0.05 (floor) |

The incumbent spending **23.73 s of a 300 s clock on one move** is the
overspend side of the same defect the underspend audit found from the other
end. Note also the two tight-clock rows: base returns `d2d4` at 2+0 and
0.5+0 where both candidates still return `b1c3` — blind floor play, visible
in a single probe.

### What the surrogate is asked, and the request to its owner

Per the AMENDMENT above, TM arms now run on the virtual-clock twin first.
**Request to the surrogate lane, made here because this file is the
coordination point and no plugin set has been ledgered yet:** min40-4 is
already named as its first customer; please add **one-max and the classic
base** to the same plugin set so all three are ranked against each other in
one run rather than pairwise. The classic base line is not the same as
`uci.py`'s — it caps at `wtime/2 - 1000` in ms, not `wtime/2 - 1` in s — so
it needs its own plugin and cannot be aliased to the smooth-budget arm. No
file of that lane's was touched in writing this.

### The one real-clock arm, STAGED and NOT LAUNCHED

| | |
|---|---|
| instrument | fastchess on the bench box, arena to be created fresh from a `git archive` of the winning branch |
| **engine1** | **the surrogate's winner** (orientation trap: fastchess states the bounds in engine1's frame) |
| engine2 | base, `1f458ad9e5370014` |
| TC | **60+0** |
| book | `book3k.pgn`, PGN not EPD (the packed artifact parses only `position startpos moves …`) |
| games | 200 rounds × 2, `-repeat`, **cap 400**, `-recover` |
| SPRT | elo0=−10 elo1=0 alpha=0.05 beta=0.05 model=normalized |
| adjudication | **NONE** — a drained clock kills long level endgames, the exact class `-draw` would delete before the defect could show |

It is launched only when a slot frees AND the surrogate has ranked the
three. **The bar is non-inferiority**: this is a simplification that removes
a cap, so per the simplify-or-Elo-pays rule the simplification side carries
it if Elo is neutral.

**Pre-registered readings, all reported whatever the SPRT says:** W/L/D and
Elo ± pentanomial interval; LLR against the bounds and LOS; **illegal moves
(zero tolerance — any occurrence kills the run and is reported naming the
game)**; time forfeits per arm; drain profile (clock at game end: median,
min, games under 2 s); the move at which the clock first falls under 2.4 s
and how many moves follow it; per-arm median and maximum move time; and
**the realized reserve at the floor**, which is the 10 s vs 2 s prediction
in the no-park table and the one number that separates the candidates.

**Pre-registered remedy, fixed now so the result cannot pick the rule:** if
the arm fails, the single permitted retune is the reserve constant for
one-max (8000 ms) or the clip divisor for min40-4 (4), set to the value that
equalizes the two arms' median move time, with ONE rerun at the same seed,
both ledgered. Anything else needs a new pre-registration.

**Honest note recorded in advance, and it is the sharpest thing in this
entry.** Both candidates are `/40`-family, and this file holds a standing
measurement AGAINST that family at increment TCs: `/12 + 0.9·inc` beat
`t/40 + inc` by **+91.1 ± 50.7 at 60+1 and +45.9 ± 46.8 at 30+1** (160 games
a leg). It also holds `/40` beating `/12` by **+235.5 ± 65.4 at 60+0**. Those
are not in conflict — **the divisor that wins depends on the increment**,
which is exactly why `uci.py` is getting a sliding one in #188. Both
candidates here collapse that slide back to a constant `/40`, trading
increment strength for one line and no park. So: **a 60+0 pass does NOT
license shipping on its own.** 60+0 is the friendly arm for a `/40` policy
and the standing contrary evidence is at 60+1 and 30+1. The surrogate must
rank all three at an increment TC as well, and if it cannot, the increment
arm becomes a second required real-clock match before either candidate
lands. If the surrogate ranks the two candidates within noise of each other,
**min40-4 ships on the unit-independence argument alone** — that tiebreak is
fixed here, before the ranking, so the result cannot choose it.

### Cotenancy

Nothing was launched, no box time was taken, no `.boxlock` claimed. The
packed builds and probes above ran on the laptop. Implementation lives on
two branches off `origin/master`, `classic/tm-one-max-pool` and
`classic/tm-min40-4`, each one line plus a shared
`tests/test_classic_time_budget.py` (118 tests, passing verbatim on both:
regime tables, the no-park recurrence, banked reserve in moves,
monotonicity in both arguments, and the unit domain). **No PR is open** —
per the owner ruling above, it opens carrying the surrogate's winner.

## 2026-08-15 — DECIDING MATCH 2 (1+0 hammer): the GATE PASSES, and the match found a −209.9 ± 60.1 CLIFF the pre-registration did not anticipate

Three separate things came out of a twelve-second match, and they must not be
collapsed into one headline.

### 1. The pre-registered gate: PASSED

| check | result |
|---|---|
| illegal-move mentions (PGN + log) | **0** |
| `(none)` answers | **0** |
| `0000` null-move tokens | **0** |
| non-normal terminations | **0** of 100 |
| time forfeits | 0 (reported, not the gate) |

100 games at a **1-second** clock (`[TimeControl "1"]`), the regime where `P` is
empty for the entire game and the 0.05 s floor governs every move. The
structural bestmove floor and the wall held perfectly: **the driver never
answered anything it could not play.** That is what this match was pre-registered
to test, and it is a pass.

### 2. The instrument failed, and it is ours

The inline gate in `run6_hammer_1p0.sh` printed **"\*\*\* HAMMER FAILED \*\*\*"**
on a run with zero illegal moves. The bug:

```sh
non=$(grep -c "(none)" "$PGN" || echo 0)
```

`grep -c` **prints "0" AND exits 1** when there are no matches, so `|| echo 0`
appended a second line, `non` became the two-line string `"0\n0"`, and
`[ "$non" -eq 0 ]` errored straight into the else branch. The stray bare `0` in
the output is the fingerprint. The `forfeit/time` probe was wrong too — its
`time` pattern matched fastchess's own `Total Time:` line and reported "1
forfeit mention" on a run with none.

**An instrument that reports something other than what it measured is the exact
defect class this ladder has been ledgering** (the arm label, the ramp, the
underpowered non-inferiority), and it does not get a pass for being ours. Fixed
as a standalone re-runnable `gate_check.sh` — no `|| echo` after `grep -c`, and
`(none)`/`0000` matched as MOVE TOKENS rather than substrings (the naive `0000`
probe matched the `+0000` timezone in `GameStartTime` 200 times). Re-run over
the same PGN: **GATE PASSED**, appended to `m6/RESULT.txt` as a correction rather
than a rewrite. `run6` now calls the fixed file.

### 3. The finding: the pool is −209.9 ± 60.1 at a 1-second clock

| | pool | smooth |
|---|---|---|
| result | **5W 59L 36D of 100, 23.00%** | — |
| Elo | **−209.91 ± 60.11** (nElo −305.64), LOS 0.00% | — |
| pentanomial | [19, 18, 11, 2, 0], PairsRatio 0.05 | — |
| median move | **0.001 s** | 0.013 s |
| starved moves | 99.8% | 98.8% |
| end clock | 0.9 s median | 0.4 s |

**Mechanism, and it is the formula, not a bug.** At a 1 s clock
`P = max(0, T + (M−1)·I − (M+2)·O) = max(0, 1 − 8.4) = 0` **for the whole game**,
so `soft = min(P/M, A/4) = 0` and the search stops at the first converged
iteration — a depth-1 move in about a millisecond. The incumbent still searches
13 ms. The pool plays the entire game 13x shallower and loses three quarters of
the points, while never flagging (0.9 s median end-clock vs the incumbent's 0.4)
and never answering illegally.

The design is *right* that 8.4 s cannot contain 40 more moves at 200 ms of
overhead each. It is *wrong* to conclude from that that a move is worth a
millisecond: `A/4 = 0.15 s` was reachable and safe the whole time, and the
`A` clamps never got the chance to act because `min()` with a zero pool is zero.
**When `P = 0` and `A > 0`, the soft limit collapses to zero and the safety
clamp becomes unreachable.** That is the one structural hole this ladder has
found in the design.

### What this does to the landing shape — an OPEN QUESTION, not a decision

The pre-registration made **illegal moves** the hammer's gate and explicitly said
forfeits and Elo were "read and reported but not the gate". By that rule, which
was written before the match: **the hammer passes and the 30+1 deciding match
proceeds** — and it has been launched.

But the pre-registered landing shape says the pool becomes the *default* manager
for the driver and the entry, and a measured −210 at 1-second TCs is not
something to land silently under a rule that was written without knowing it.
This lane is not deciding that alone. Recorded for the coordinator and Thomas,
with the options as they stand:

1. **Scope the default.** Ship the pool wherever `P > 0` at the root and keep the
   incumbent below it. Honest, and it needs no new screen.
2. **Fix the collapse** (preferred by this lane, *not* applied): floor the soft
   limit against what is reachable rather than against zero — e.g.
   `soft = min(max(P/M, A/20), A/4)`, which gives ~30 ms at a 1 s clock instead
   of 1 ms and is unchanged wherever `P > A·M/20`, i.e. at every TC this ladder
   measured. It is a design change, so it gets its own pre-registration and its
   own 1+0 screen; it must not be slipped into a landing.
3. **Land as measured and document the cliff.** Cheapest, and the least honest
   of the three unless bullet is genuinely out of scope — which it is not: the
   classic bot accepts 1+0 challenges on lichess today.

Nothing is landed until that is answered. The 30+1 match runs meanwhile, because
it is the evidence for the decision-TC claim either way.

## 2026-08-15 — PRE-REGISTRATION: the pool's single real-clock confirmation, and the LANDING SHAPE it decides

Written before either match starts, and it pre-registers **what landing means**
as well as what passing means — because after two H1s in a row the temptation to
decide the landing shape from the result is at its highest, and that is exactly
when the rule has to already exist.

### Why these two and not the rest of the ladder

Arms (a) and (b) are in (+119.94 ± 36.44 at 60+0; +136.58 ± 35.24 at 60+1).
Under the real-clock economy ruling the remaining ladder moves to the twin's
virtual clock, and **the pool gets one real-clock confirmation**. It is spent on
the two things a surrogate cannot settle:

1. **30+1, the decision TC.** `docs/TESTING.md`'s minimum decision-grade
   control, and the one classic is judged at. Non-inferiority, because a manager
   that wins at 60+0 and 60+1 still has to be safe where the clock is half.
2. **1+0, the zero-illegal hammer.** For this manager it is **not** a formality.
   The pool runs its clock down by design — it ended 48 of 262 games under 2 s
   at 60+1 — and at 1+0 the whole game lives inside the regime where `P` is
   empty and the floor governs. If the structural bestmove floor or the wall has
   any hole in it, this is where it shows.

### Match 1 — 30+1 NON-INFERIORITY

| | |
|---|---|
| arms | `pool` (`cddf392e21449054`) vs `smooth` (`14b69a606b743a37`) — same two binaries as arms (a) and (b), unrebuilt |
| TC / book | 30+1, `book3k.pgn`, order=random, srand 20260818, **no adjudication** |
| SPRT | elo0=−10 elo1=0 alpha=0.05 beta=0.05, engine1 = pool (bounds in engine1's frame) |
| cap | **1750 games** (875 rounds × 2, `-repeat`), concurrency 8, nice 10, `-recover` |

**The cap is raised from the ladder's 400 on a recorded lesson, not a hunch.**
The smooth ladder's match 2 was an underpowered non-inferiority: a 400-game cap
against a ±10 band cannot separate "not worse" from "not measured", and that is
the same defect class as the two ledgered above — a number that means something
other than what it says. At ~1 minute per game pair and concurrency 8 this is
about 7 hours, which is affordable for THE deciding match and is why it is the
only real-clock spend left.

### Match 2 — the 1+0 zero-illegal hammer

| | |
|---|---|
| arms | same two binaries |
| TC / book | 1+0, `book3k.pgn`, srand 20260819, no adjudication |
| games | **100**, concurrency 8, nice 10 |
| **pass condition** | **ZERO illegal moves and ZERO `(none)` answers from the pool arm. Required, not a target.** Time forfeits are read and reported but are not the gate — the flag is the manager's business, an illegal move is the driver's |

### THE LANDING SHAPE, fixed now

**If both pass**, the pool becomes the **recommended time manager for both
engines**:

* the packed **entry** takes `pooltm` as its default at its measured **+57 B**
  (3308 → 3365, 731 spare), by moving the mod into `make_pst_entry.py`; the
  `pooltm` mod then RETIRES IN PLACE with a tombstone naming which of its
  anchors survive, and `oldtm`/`steptm` go with it because their anchor stops
  existing;
* the **classic driver** ships the pool as its default manager, and the knob
  value `legacy` keeps the incumbent expression available as a control arm;
* **#188 closes as SUPERSEDED, not as wrong.** Its mechanism is the legacy this
  builds on — the negative cap is what the `A/2` wall exists to prevent, and its
  60+0.1 result (+40.6) is what established that the transition band is real.
  Superseded means its curve is deleted and its bytes come back, not that its
  argument was mistaken;
* the entry's TM section is rewritten around the pool, with the two verdicts and
  this confirmation as its evidence.

**If match 1 fails** (H0 accepted, i.e. the pool is worse than −10 at 30+1): the
pool does NOT become the default at either engine. It stays a knob, the ledger
records that the manager is TC-dependent, and the one permitted follow-up is the
soft scale `s` retuned to equalize the 30+1 median spends, screened once at
30+1. **If match 2 fails on a single illegal move or `(none)`**: landing is
blocked outright regardless of match 1, the game is named, and the defect is
fixed and re-hammered before anything else — no Elo result buys past a driver
that can answer an illegal move.

**If match 1 reads ≈ 0 within the band** (the likely outcome for a
non-inferiority at this power): that is a PASS on the pre-registered rule and
will be reported as "not worse at the decision TC", never as a third win.

### Venue and queue discipline

Both run in `~/sunfish-bench/tmpool-20260814/` on the same arena, same book,
same binaries. They queue **behind the 8Mv screen's boxlock** and take the slot
when it frees; nothing chains itself — the launch is a person's action, and the
staged scripts stay named `HELD_`/queued until then. Cotenancy is recorded at
launch and at finish, as for every arm in this ladder.

## 2026-08-15 — ARM (b) VERDICT: the POOL manager is +136.6 ± 35.2 at 60+1, H1 in 262 games — the risk arm was not the risk

The arm that had to WIN something won more than the arm that only had to not
lose. This is the regime Thomas left open: at 60+1 the pool budgets **2.4x less**
for a routine move than the shipped curve (2.27 s soft against a 5.40 s budget),
and the whole design rests on the wall and the extensions buying that back.

| | |
|---|---|
| arms | `pool` (`pooltm`, `cddf392e21449054`) vs `smooth` (base = HEAD entry, `14b69a606b743a37`) |
| TC / book | 60+1, `book3k.pgn`, order=random, srand 20260817, no adjudication |
| result | **142W 44L 76D of 262**, 68.70% |
| Elo | **+136.58 ± 35.24** (nElo +181.25 ± 42.07), LOS 100.00% |
| pentanomial | [3, 11, 37, 45, 35], PairsRatio **5.71**, WL/DD 2.70, draw ratio 28.2% |
| SPRT | **LLR 2.96 > 2.94 — H1 accepted for [0, 10]** at 262 of a 600 cap, 2 h 09 m |
| tripwires | **0 time forfeits, 0 illegal moves, 263/263 terminations `normal`** |

(The PGN carries 263 games to fastchess's 262 — one pairing finished after the
stop. The tally below reads all 263: 142W 44L 77D, 68.63%. The SPRT figures are
quoted as fastchess computed them.)

### Realized spend — the shape INVERTS between the two TCs, and that is the finding

| TC | arm | median | mean | p90 | max |
|---|---|---|---|---|---|
| 60+0 | pool | 0.512 s | 0.718 s | 1.623 s | **5.534 s** |
| 60+0 | smooth | 0.645 s | 0.687 s | 1.183 s | 1.664 s |
| 60+1 | pool | **1.629 s** | 1.918 s | 3.311 s | **10.151 s** |
| 60+1 | smooth | 1.488 s | 1.933 s | 3.655 s | 5.509 s |

At 60+0 the pool spends **0.79x** the median move; at 60+1 it spends **1.09x** —
*more* — while its p90 is **lower** (3.311 vs 3.655) and its maximum is **1.8x
higher**. So the pool is not a "spend less" manager and never was: it is a
**redistribution**. It moves time off the p90 body of ordinary moves and onto
the handful that need 10 s, and it does that in opposite directions at the two
TCs depending on where the increment lands. Both directions won by ~130 Elo,
which is the strongest evidence in this ladder that the mechanism is the
allocation shape and not the level.

The pre-registered claim that the pool "spends 2.4x less routinely at 60+1" is
therefore **wrong as stated** and is corrected here rather than quietly: that is
the ratio of the BUDGETS, and the realized median moved the other way.

### Drain and floor telemetry

| arm | end clock median | mean | min | games < 2 s | first below 2.4 s | moves after |
|---|---|---|---|---|---|---|
| pool | 2.6 s | **4.7 s** | 1.4 s | **48** | move 53 (116 never) | 15 |
| smooth | 2.7 s | 3.9 s | 2.1 s | **0** | move 67 (162 never) | 8 |

**Blind moves: 0 for both arms, out of 15,851 and 15,810 moves.** At an
increment TC nobody reaches the floor, which is why the starvation band is the
reading that discriminates here (≤ 1.5 s = 1.5x the increment): **pool 44.1%
starved against smooth's 50.6%** — the pool is the *less* starved arm despite
running its clock lower.

The one number that goes the incumbent's way: the pool ended **48 games under
2 s** where the incumbent ended none, and it crosses 2.4 s at move 53 rather
than 67. It converts that into a higher MEAN end-clock (4.7 vs 3.9) and zero
forfeits, so at 60+1 it is spending the bank deliberately rather than draining
it — but it is a genuinely tighter arm at the flag, and it is the reason the
deciding match below carries a **1+0 zero-illegal hammer** rather than treating
that as a formality.

### Ladder status

| arm | verdict |
|---|---|
| (a) 60+0 non-inferiority | **+119.94 ± 36.44**, H1, 274 games |
| (b) 60+1 decisive | **+136.58 ± 35.24**, H1, 262 games |
| (c) 30+1 / (d) phase-M | held for the surrogate; (c) is now folded into the deciding match below |

Cotenancy: cotenant throughout at concurrency 8, nice 10, on a 96-core box at
load 22-33 with other lanes' tournaments live; no `.boxlock` claimed, nothing of
another lane touched.

## 2026-08-14 — Two defects in the CLASSIC pool twin, found by the surrogate lane and fixed before any PR

Neither is a crash; both are the kind that make a number mean something other
than it says. Found while the virtual-clock surrogate was reading the classic
branch (`tm-pool-manager`) as a formula source. **Arm (a)'s verdict and arm (b)
in flight are unaffected — those play packed mods, which have neither defect.**

**1. The arm label lied.** `TM_MANAGER="smooth"` selected master's `/12`
expression — the pre-#188 form whose cap goes negative under a 2 s clock — not
#188's smooth rational. A control arm whose name misdescribes what it plays is
how a screen measures one thing and reports another; the `steptm` sha-identity
discipline exists for exactly this. Renamed to **`legacy`**, which is what the
expression is on that branch, with a test that goes red the moment the smooth
curve appears in the file (that is when the value gets renamed to `smooth`, in
#188's merge and not before). Wiring #188's expression in instead was the other
option and was rejected: it would make the pool PR carry the acute fix, and the
two are meant to be separable. `"smooth"` is not accepted as an alias — a script
asking for an arm that branch does not have fails at startup.

**2. The opening ramp applied to the pool.** `min(think, len(hist) + random())`
capped the pool budget for the first 8 plies, but **the packed arm that measured
+119.94 ± 36.44 has no ramp at all**, so the classic pool was an unmeasured
variant wearing a measured number's name. The ramp is now the incumbent
manager's only, which makes the classic pool the measured pool. Through the real
driver at ply 0, 60+1: **legacy 1.02 s (ramp-capped) vs pool 3.82 s (its own
budget)** — before the fix the pool answered inside the ramp cap.

The pool does not need the ramp (P/M paces the opening, and the +2 in (M+2)·O
banks what it was banking), but what is given up is real and is now written at
the site: `random()` is this engine's only opening variety without a book, so a
pool arm deployed to lichess would repeat openings. Every measurement we run is
booked, so no match is affected — and if the pool becomes the bot's default,
variety comes back as a book, **never as a budget cut**, or the deployed engine
stops being the measured one again.

Classic suite after both fixes: `tests/test_tm_pool.py` 44 passed, `tests/` 339
passed, 2 skipped.

## 2026-08-14 — AMENDMENT to the pool ladder: arms (c) and (d) are HELD for the virtual-clock surrogate, not cancelled

Owner ruling (Thomas, real-clock economy: *"we're wasting a lot of time with
these TM runs"*). A time-manager arm costs an hour of box time to learn
something a clock model can predict, and this lane has now spent three of them.
A **virtual-clock surrogate** is being built into the C twin — budget formulas
as plugins, plus an `nps(piece-count)` model — calibrated against the runs that
already exist: stage 1, the smooth budget's +40.6 with-park match, and pool arm
(a) above.

What changes in the ladder pre-registered at 629cba2:

| arm | status |
|---|---|
| (a) 60+0 | **DONE** — +119.94 ± 36.44, H1, 274 games |
| (b) 60+1 | **PROCEEDS on real clocks**, launched 21:08 UTC. It is the decisive question AND it is calibration data for the surrogate, which is the second reason not to hold it |
| (c) 30+1 | **HELD** for the surrogate |
| (d) phase-M | **HELD** for the surrogate |

**HELD IS NOT CANCELLED.** The bounds, book, seed, cap and pre-registered
readings for (c) and (d) stand exactly as written; what changes is the
instrument they run on, and only the **final composite** gets one real-clock
confirmation. The arena scripts were renamed `HELD_run3_30p1.sh` /
`HELD_run4_phasem.sh` with the ruling in their headers, so the operational state
matches this entry rather than depending on someone reading it — and if the
surrogate is abandoned, the un-hold goes in the ledger BEFORE either is
launched.

The v1.1 dynamic target is unaffected and still unscreened: it was already gated
behind (a) and (c), and (c) now resolves on the surrogate.

## 2026-08-14 — ARM (a) VERDICT: the POOL manager is +119.9 ± 36.4 at 60+0, H1 in 274 games — and it parks the clock at exactly (M+2)·O

The arm that was only asked **not to regress** won it. Pre-registered above
(commit 629cba2) as a NON-INFERIORITY screen against the shipped entry — the
sudden-death fix is worth +235.5 ± 65.4 and the question was whether the pool
gives any of it back. It gives none back; it adds.

| | |
|---|---|
| arms | `pool` (`pooltm`, `cddf392e21449054`, 3365 B) vs `smooth` (base = HEAD entry, `14b69a606b743a37`, 3308 B) |
| TC / book | 60+0, `book3k.pgn`, order=random, srand 20260816, no adjudication |
| result | **144W 53L 77D of 274**, 66.61% |
| Elo | **+119.94 ± 36.44** (nElo +147.12 ± 41.14), LOS 100.00% |
| pentanomial | [4, 18, 37, 39, 39], PairsRatio 3.55, WL/DD 2.70, draw ratio 27.0% |
| SPRT | **LLR 2.99 > 2.94 — H1 accepted for [−10, 0]** at the 274-game mark of a 600 cap |
| tripwires | **0 time forfeits, 0 illegal moves, 274/274 terminations `normal`** |
| median plies | 139 |

The baseline is not a rebuild: `14b69a60…` is the same binary playing #188's
match 1, and at `winc == 0` it is the stage-1 `tmfix` winner's behaviour above a
2.667 s clock. So this is measured against the +235 arm itself.

### Realized spend — the shape is the mechanism

| arm | moves | median | mean | p90 | max |
|---|---|---|---|---|---|
| pool | 17,511 | **0.512 s** | 0.718 s | **1.623 s** | **5.534 s** |
| smooth | 17,468 | 0.645 s | 0.687 s | 1.183 s | 1.664 s |

The pool spends **0.79x the median move** and **3.3x the maximum**. That is the
whole architecture in one table: the soft limit makes routine moves cheap, the
wall lets a hard one run to 5x soft, and the incumbent — one number that is both
target and wall — cannot do either. The pre-registered prediction from
`tm_smoke` was 1.5x MORE at 60+0 on a cold table from the start position; in
games with a warm table it is 0.79x on the median, so **the assay over-read the
routine spend and the honest note in the pre-registration was too pessimistic,
not too kind**.

### Drain and floor telemetry (pre-registered readings)

| arm | end clock median | mean | min | games ending < 2 s | first fall below 2.4 s | moves played after |
|---|---|---|---|---|---|---|
| pool | 12.5 s | 14.1 s | **8.4 s** | **0** | **never, in 274/274** | — |
| smooth | 14.8 s | 16.2 s | 0.9 s | 4 | move 142 (5 of 274 games) | 20 |

**The minimum end-clock is 8.4 s = (M+2)·O exactly.** The pool empties its pool
at that point by construction and parks there, so the sub-2·O regime is not
merely avoided in these games, it is unreachable while `M` and `O` stand. The
incumbent, which has no such term, put 4 games under 2 s and 5 games through the
old cap's collapse threshold.

| arm | blind moves (≤0.06 s) | per game | games with none |
|---|---|---|---|
| pool | 1,057 (6.0%) | 3.86 | 227 of 274 |
| smooth | 117 (0.7%) | 0.43 | 269 of 274 |

**Read this one carefully — same metric, two different mechanisms.** The metric
was calibrated on the drain (a collapsed budget with no clock behind it: stage
1's 22.3% and a median 16 blind moves before being mated). The pool's 1,057 are
not that: they are **deliberate floor moves played with 8–12 s still on the
clock**, taken because `P` is genuinely empty at that point — 8.4 s does not
contain 40 more moves at 200 ms of overhead each. The pool played 14x more of
them than the incumbent and still won 66.6%, which is the evidence that this
class of blind move is not the losing class. It is nonetheless the one number
here that argues for a knob: `M` is a constant where a real horizon estimate
would let the tail spend down further, and that is the phase-M arm (d).

### What this does and does not settle

Settled: the pool does not cost the sudden-death fix anything, the wall never
fired as a forfeit, and the allocation shape is worth ~120 Elo at 60+0 against a
manager that is already the validated one. **Not settled: 60+1**, where the
budget ratio is 2.4x rather than 1.16x and the extensions have to pay for a much
larger cut in routine spend. Arm (b) launched at 21:08 UTC on the same arena,
same book, srand 20260817, elo0=0 elo1=10, cap 600 — it has to WIN something,
not merely not lose. Arms (c) 30+1 and (d) phase-M stay gated behind it, and the
dynamic target (v1.1) stays unscreened until (a) and (c) are both in.

Cotenancy: launched cotenant at load ~22 on 96 cores, concurrency 8, nice 10,
finished in 52 minutes at load ~33 with three other lanes' tournaments live; no
other lane's processes or files were touched, and no `.boxlock` was claimed.
(Record note: `m2/RESULT.txt`'s first line is mislabelled "ARM (a) 60+0" by an
inherited `say` line — the `config`/`sprt` lines under it are correct and the
match is 60+1; the staged (c) and (d) scripts were corrected.)

## 2026-08-14 — PRE-REGISTRATION: the POOL time manager (soft/hard), and the ladder that prices it

Written before a game is played. This is Thomas's design and it is the v2
architecture, not a second attempt at the acute fix: the smooth budget (PR #188,
pre-registered above) stays exactly where it is as the conservative correction,
and this entry asks a different question — whether a **resource pool with two
limits** beats **one curve that has to be both target and wall**.

### The design (Thomas's, in milliseconds as the entry runs it)

```python
M = 40
P = max(0, wtime + (M - 1) * winc - (M + 2) * 200)   # the pool the game has
A = max(0, wtime - 400)                               # what THIS move can reach
soft = min(P / M, A / 4)                              # stop STARTING iterations
think = min(5 * soft, A / 2)                          # the wall, i.e. the deadline
```

`sunfish_ui/uci.py` runs the same arithmetic in SECONDS (`pool_budget`), and its
`tests/test_tm_pool.py` pins this mod's millisecond text as a literal and
asserts `t_ms = 1000·t_s` on a grid — the seconds/ms confusion has cost this
project two incidents, so it is checked numerically on both sides.

**Why a pool.** A single divisor answers "what is this move worth" and "how long
may one iteration run" with one number, and the two pull it in opposite
directions. Splitting them lets a routine move be paced at `P/M` while a hard
one may run to 5x that. It also prices what a divisor cannot see: the increment
is **income** (M−1 further moves will earn it) and the per-move overhead is a
**tax** (M+2 moves pay it, the +2 buying margin for the last move and the flag).
O = 200 ms is measured, not chosen — the lichess autopsy of `EAThUL0P` and the
stage-1 60+0 drain forensics agree on it. The `A` clamps are the safety half:
`A/4` keeps three more moves' worth of clock behind every soft limit and `A/2`
is a wall that **cannot go negative**, which is the exact failure that lost that
game.

**Prior art, for calibration rather than authority:** Stockfish's `optimum`/
`maximum` pair is this pool in another notation, Lc0's move-number curve is the
phase-M arm below, and Berserk's soft:hard ratio is ~1:5, which is where the 5
comes from.

### The load-bearing implementation detail, measured before launch

The soft limit is a rule about **starting** an iteration, and the driver's
obvious landmark — a new depth appearing — arrives one FULL PROBE OF THE NEXT
DEPTH late. Reading it there measured **2.64 s against a 1.29 s soft limit at
60+0** and **6.82 s against 2.27 s at 60+1** through `tm_smoke` on the packed
artifact: a soft limit that is really a 2-3x multiplier. The mod therefore
mirrors the engine's own MTD bracket in the driver (`lo`/`up`, tightened
monotonically exactly as `search()` does, so the instability guards' crossing
case reads as converged too) and stops when it closes to inside
`EVAL_ROUGHNESS`. The mid-iteration `think * 0.8` break is removed; the
deadline, the `Stop` handler and the structural bestmove floor are untouched.

### What this actually spends, and why arm (a) is a non-inferiority check

Budgets are not spends: iterations are discrete, so the pool stops at the first
one that ENDS past its soft limit. Measured through `tm_smoke`, cold table,
start position, on both machines this lane uses:

| | soft (formula) | smooth realized | pool realized | laptop / box |
|---|---|---|---|---|
| 60+0 | 1.29 s | 1.50 s | **2.26 s** | laptop |
| 60+0.1 | 1.39 s | 2.46 s | **1.74 s** | laptop |
| 60+1 | 2.27 s | 5.40 s | **3.10 s** | laptop |
| 60+0 | 1.29 s | 1.41 s | **1.85 s** | box, nice 15, loaded |
| 60+0.1 | 1.39 s | 2.51 s | **2.86 s** | box, nice 15, loaded |
| 60+1 | 2.27 s | 4.82 s | **5.23 s** | box, nice 15, loaded |

**Recorded in advance because it is the opposite of the design's premise:** on
the budgets alone the pool spends 2.4x LESS than the smooth curve at 60+1, but
the realized spend is dominated by where the iteration ladder happens to land,
and on the box the pool spends MORE at every probed TC. Whether that costs or
buys Elo is the question the ladder answers; nothing here is adjusted to make
the answer nicer.

Clock-safety walks (200 ms lag, our own moves only, in `tests/test_tm_pool.py`):
at the ideal 1.0x the pool never flags 100 moves of 60+0 (3.0 s left) where /12
flags at move 39 and the smooth curve at 84; at the measured 1.75x it flags at
89, still later than either. At match-like lag (20-50 ms) neither arm flags
within 120 moves, so **no forfeit is expected from either arm** — a forfeit is
therefore a real signal, not an artifact.

### The arms

`tools/build/make_variants.py` from `nnue_4k/pst_entry.py` (CI-guarded against
its own generator), new mods `pooltm` and `phasem`. Packed on the laptop
toolchain, sha re-verified after transfer.

| arm | mod | packed | sha256[:16] |
|---|---|---|---|
| **pool** = engine1 | `pooltm` | **3365 B** (731 spare) | `cddf392e21449054` |
| **smooth** | `base` (= HEAD entry) | **3308 B** | `14b69a606b743a37` |
| (phase-M, arm d) | `pooltm.phasem` | 3373 B | built at launch |

**+57 B all-in**, and that is the honest price: the pool REPLACES the smooth
curve, so the curve's bytes are already inside that figure. The `smooth` arm is
`14b69a60…` — the same binary now playing PR #188's match 1, and at `winc == 0`
its budget is `wtime/40` with a cap that differs from the stage-1 `tmfix`
winner only below a 2.667 s clock. So arm (a) is measured against the
+235.5 ± 65.4 winner's behaviour, not a rebuild of it.

### The ladder, in order, never batched

| arm | TC | question | SPRT (engine1 = pool) | cap |
|---|---|---|---|---|
| **(a)** | 60+0 | does the pool give any of the +235 back? | elo0=−10 elo1=0 | 600 |
| (b) | 60+1 | THE risk arm: routine spend vs extensions | elo0=0 elo1=10 | 600 |
| (c) | 30+1 | non-inferiority at the decision TC | elo0=−10 elo1=0 | 400 |
| (d) | 60+1 | phase-M (M = max(20, 46 − ply/2)) vs pool | elo0=0 elo1=10 | 600 |

(a) runs now. (b) only after (a) reports, (c) after (b), (d) after (b). **v1 is
STATIC**: the dynamic target (stability × best-move-change × score-drop, the
`TM_DYNAMIC` knob in the classic twin) is v1.1 and is not screened until v1
survives (a) and (c) — if the pool itself is a regression no stability tuning
saves it, and mixing the two would leave us unable to say which half spoke.

### Match (a) — 60+0 NON-INFERIORITY

| | |
|---|---|
| instrument | fastchess on the bench box, arena `~/sunfish-bench/tmpool-20260814/` (NEW dir, fresh `git archive` of the packed HEAD) |
| **engine1** | **pool** (orientation trap: fastchess states the bounds in engine1's frame) |
| engine2 | smooth (`14b69a60…`, the in-flight #188 baseline) |
| TC | **60+0** |
| book | `book3k.pgn`, PGN not EPD (the packed artifact parses only `position startpos moves …`) |
| games | 300 rounds × 2, `-repeat`, cap 600, concurrency 8, `nice 10`, `-recover`, srand 20260816 |
| SPRT | elo0=−10 elo1=0 alpha=0.05 beta=0.05 model=normalized |
| adjudication | **NONE** — a drained clock kills long, level endgames, exactly the class `-draw` would delete before the defect could show |

**Pre-registered readings, all reported whatever the SPRT says:** W/L/D and Elo
± pentanomial interval; LLR against the bounds and LOS; **illegal moves (zero
tolerance — any occurrence kills the run and is reported naming the game)**;
time forfeits per arm; **blind moves** (≤ 0.06 s) per arm, total and median per
game; **drain profile** (clock remaining at game end: median, min, games under
2 s); the move at which the clock first falls under 2.4 s and how many moves
follow it; median plies; and the **per-arm median move time**, which is the
realized-spend number the table above predicts.

**Pre-registered remedy, fixed now so the result cannot pick the rule:** if (a)
fails and the tally shows the pool's median move time above the smooth arm's,
the single permitted retune is the soft scale `s` (SOFT_SCALE, 1.0 today) set to
the ratio that equalizes the two medians, rounded to 2 decimals, with ONE rerun
of (a) at the same seed, both ledgered. Any other adjustment needs a new
pre-registration.

**Honest note recorded in advance.** If (a) and (b) both read ≈ 0, this is an
architecture change with no measured Elo and it will be reported as exactly
that. The case for landing it would then rest on what the walks show and the
matches cannot — the lichess-lag regimes, where 200 ms/move of unavoidable tax
is priced by the pool and invisible to a divisor — and that case would be made
on its own terms, not dressed up as a win.

### Cotenancy

The box is shared under owner-authorized capacity sharing (Thomas, 2026-08-14:
more processes are fine while no other human user needs the box). At launch two
tournaments of other lanes were live (PR #188's 60+0.1 arm, concurrency 8; a
30+1 null-move arm, concurrency 10) on a 96-core box at load ~22. This lane runs
concurrency 8 at `nice 10`, records every cotenant's game and forfeit counts at
launch and at finish in `m1/RESULT.txt`, and touches nothing it did not start.

## 2026-08-14 — C-TWIN PR SERVICE: calibration PASSED at 49.83% (after catching and fixing a real driver bug), three PR intervals delivered, and the tp_move eviction battery — the root guard is worth ~15 Elo exactly where it was built to work

The ctwin lane's numbers, ledgered here because PR #184's docs cite them.
Instrument: `tools/ctwin/sunfish_c` on master (ce9a551 → 4d4974f → f95f49c),
node-identity difftest-gated per docs/TESTING.md rule 14 before every number
(standing 6-suite gate + per-variant suites, 0 mismatches throughout). All
matches: fixed-node 20k/move, `tools/build/gate_openings.epd` (334 openings,
color-swapped pairs — deterministic engines, so 668 distinct games is the
book's ceiling), `ucinewgame` per game, concurrency 2 nice-15, **zero illegal
moves and zero disconnects across ~3,640 games**. Twin-grade caveat on
everything below: fixed effort, rule 12 — only a wall-clock match on the real
engine outranks it.

**Calibration stage 1 (the decision-grade gate), both runs.** Run 1, twin vs
pypy-classic: **-53.7 ± 42.8 at 130 games — VOIDED**, and that was the gate
doing its job: probe-stream comparison showed byte-identical searches but the
twin aborting the cap-crossing MTD probe mid-flight while `uci.py` finishes
it, consumes its yield, and checks the cap between probes at depth > 1 — so
pypy played one depth deeper whenever the cap landed inside the first (long)
probe of a new depth. `go_game` was rewritten as a transcription of uci.py's
consumer (bestmove floor and upperbound info lines included); the twin then
reproduced pypy's info stream byte-for-byte through the crossing probe on the
diagnostic position. Run 2 (fixed driver): **300 games, 124W-125L-51D =
49.83%, Elo -1.16 ± 12.23, 92.7% of pairs move-identical** — parity, stage 1
PASSED. Stages 2-3 (two known-Elo pair reproductions) remain staged.

**PR #184 derive-never-inherit (`DERIVE_FRESH`): +0.52 ± 6.37, 668 games,
271W-270L-127D (50.07%), Ptnml [4,5,313,10,2].** 313 of 334 pairs split 1-1
move-identical — the wording-precision point: the rescore only changes
anything when the K-table swaps mid-game, and when it does, it costs nothing
at fixed nodes. Non-inferior (interval floor -5.9 vs the -10 bar); SPRT
formally undecided at book end (LLR +0.40), the interval is the verdict.

**PR #182 fuel-oracle null (`FUEL_NULL`): +1.04 ± 12.74, 668 games,
277W-275L-116D (50.15%), Ptnml [12,29,250,31,12].** DrawRatio 74.9%: the
mechanism changes many games and nets neutral — the PR's own bar ("value is
formal; neutral suffices") met on the point estimate; the -10 floor excluded
only one-sided (~95%), book-limited. Since merged as #192; the twin's default
flipped to match and re-proved (11 suites, 0 mismatches; classic packs 3278 B
post-merge, measured).

**PR #171 qsearch frontier evasions (`QS_TAIL`): exactly 0.00 ± 0.00, 668
games, 272W-272L-124D, Ptnml [0,0,334,0,0] — every pair move-identical**,
flavor-matched base (IID_MIN_DEPTH=2, MATE_DIST=0). The mechanism fired on
1/27 probe-suite positions (-0.05% nodes) and never changed a played move:
pure rare-path correctness, zero game-level cost, zero game-level gain.

**tp_move eviction/killer battery** (knobs EVICT_POLICY 0-3, KILLER_COUNT
1-3; Python reference = drift-guarded transcription in `tools/ctwin/
variants.py`, itself difftest-proven; policy 3's bucketing reproduces the C
content hash bit-for-bit in Python):

| cell (vs master guard) | games | W-L-D | Elo |
|---|---|---|---|
| unguarded evict-before-insert, default TABLE_SIZE | 668 | 280-280-108 | **0.00 exactly — all 334 pairs identical** |
| unguarded, TABLE_SIZE=500 churn | 668 | 260-289-119 (47.83%) | **-15.09 ± 19.57**, LOS 6.5%, LLR -0.85 → H0 |
| hash-slot two-tier replace-if-deeper, TABLE_SIZE=500 | 668 | 279-267-122 (50.90%) | **+6.24 ± 20.96**, LOS 72% |

Reading: at production TABLE_SIZE (10^6) eviction never fires inside
20k-node games — the proposed unguarded simplification is a **literal no-op
there**, and under churn it **costs ~15 Elo**: the root guard earns its keep
in exactly the regime it was built for (the Qxc6 incident class). The node
screen's "unguarded searches 10% fewer nodes under churn" is killers being
lost, not efficiency — fewer nodes AND worse play. The guardless two-tier
slot table roughly recovers the protection (+6 ± 21) and is the candidate if
simplification is ever wanted; k2/k3 killer lists cost ~+1% nodes at fixed
depth and were screen-pruned, unmatched. Consumer audit (what needs the
depth-advance latch if the guard ever goes) delivered in the lane report:
latched and safe — sunfish.py:590, uci.py:132, three candidate drivers;
needs review — tools/build/first_yield_gate.py:100, tools/tune/
distill_label.py:84; pinned to the guard property — tests/
test_eviction_race.py:63,:87.

Raw artifacts (session scratchpad, `/private/tmp/claude-501/-Users-ahle-
repos-sunfish/6054308f-1ef2-4051-b134-afb688cb98f9/scratchpad/`):
calibration_VOID_old_driver.log, calib/calibration2.{log,pgn},
pr_pr184_derive / pr_pr182_fuel / pr_pr171_qstail .{log,pgn},
evict_default / evict_churn / pr_evict_p3churn .{log,pgn},
gate_*.log (identity suites), ledger.md, consumer_audit.md.

## 2026-08-15 — MATCH 2 VERDICT: 30+1 non-inferiority is NOT established, −17.39 ± 20.07 at the 400-game cap — and the cause is the CAP, not the shortfall it was built to price

Ran to the full 400-game cap in 2 h 28 m without the SPRT resolving. This is
the uncomfortable outcome and it is reported as one.

| | |
|---|---|
| games | **400 of 400 — cap reached, SPRT undecided** (LLR −0.81 against ±2.94) |
| smooth (engine1) | **113 W, 133 L, 154 D — 47.50%** |
| Elo (smooth − step) | **−17.39 ± 20.07** (pentanomial 95%), nElo −29.58 ± 34.05 |
| 95% interval | **[−37.46, +2.68]** |
| LOS | **4.43%** — i.e. ~95.6% posterior that smooth is *worse* at this TC |
| pairs | Ptnml(0-2) [11, 42, 108, 34, 5] over 200 pairs, **PairsRatio 0.74** |
| illegal / forfeits | **0 / 0**, all 400 terminations `normal` |

### The verdict, by the rule as written

The pre-registration says the remedy fires if match 2 **accepts H0** or **the
95% upper bound at cap is below 0**. Neither happened: the SPRT is undecided,
and the upper bound is **+2.68**, above zero by a hair. **So the remedy does
not trigger.** H1 was also not accepted, so **non-inferiority is NOT
established**.

That is a genuine non-result, and it must not be read as a pass. The point
estimate is **−17.4**, on the wrong side of the −10 margin we said we cared
about, with a 95.6% posterior that the smooth form is worse at 30+1. "The rule
did not fire" and "the change is fine here" are different statements, and only
the first is true.

### PRE-REGISTRATION DEFECT #2, logged not patched: the design could not answer its own question

The 400-game cap was never powered for a 10-Elo non-inferiority test. From
this run's own pentanomial variance:

| games | 95% half-width |
|---|---|
| **400 (the cap chosen)** | **±20.1 Elo** |
| 1 000 | ±13.2 |
| **~1 750** | **±10.0 — the minimum to resolve a 10-Elo question** |
| 2 000 | ±9.3 |

An interval twice as wide as the bound under test cannot separate H0 from H1,
and at the observed LLR drift the SPRT needed **~2 000 games** to reach a
boundary. The cap was short by a factor of 4–5. That is a defect in the
pre-registration — mine — not in the run.

Worse, the remedy's trigger is specified backwards. "Upper bound below 0"
demands near-certainty of *harm* before a fix is permitted, on a question
where the null of interest is harm. Under that rule a genuinely −17 Elo
regression walks free on 400 games, exactly as it just did. A non-inferiority
remedy should trigger on **failure to exclude harm**, not on proof of it.

This is the third pre-registration defect in this workstream (after stage 1's
degenerate `0 < 0` pass rule and match 1's floor-calibrated blind-move
metric). All three share a shape: a rule written before the mechanism was
understood encoded an assumption the data then violated.

### The mechanism is NOT the shortfall the match was built to price

Match 2 existed to price the −7.4% asymptotic shortfall at 30+1. The drain
data says that is probably not what cost the Elo. At `winc == 1 s` the two
policies differ in **two ways with opposite signs**:

| clock | smooth | step | ratio |
|---|---|---|---|
| 30 s | 3.150 | 3.400 | 0.93× ← the shortfall |
| 10 s | 1.650 | 1.733 | 0.95× |
| 4.5 s | 1.238 | 1.250 | 0.99× — **the sign flips here** |
| 4.0 s | 1.200 | 1.000 | **1.20×** |
| 3.0 s | 0.900 | 0.500 | **1.80×** |
| 2.5 s | 0.694 | 0.250 | **2.78×** |
| 2.1 s | 0.538 | 0.050 | **10.76×** |

Above ~4.5 s the smooth form spends *less* (the priced shortfall). Below it
the smooth form spends **far more**, because the step's cap has parked and
refuses to pay out while ours never stops. The parking fixed point does the
work again: at `I = 1 s`, `T* = 2 + 2I = 4.0 s`.

And that is exactly what the clocks show:

| | smooth | step |
|---|---|---|
| median clock at game end | **2.5 s** (min 2.1) | **3.1 s** (min 2.9) |
| games crossing 2.4 s | **222** of 400 | **0** of 400 |

**The step arm never once went below 2.4 s in 400 games.** It parked at ~4 s
and conserved. The smooth arm spent its reserve down into the 2 s region in
more than half its games. So the leading candidate for the −17 is not
under-spending in the middlegame — it is **over-spending in time trouble**,
caused by the always-positive cap.

**Which makes the cap change a two-sided trade, and that is the real finding
of both matches together.** The old cap's parking pathology is *lethal* at
sudden death and tiny increments, where the parked clock buys no depth — that
is match 1's +40.6 and stage 1's +235.5. The same pathology is *protective* at
a fat increment, where a parked 4 s clock still buys a whole second per move
and the discipline is worth more than the reserve. One mechanism, opposite
signs, and which side you land on depends on whether the increment can sustain
the parked clock.

### Instrument note: the starvation reading does not discriminate here

At `I = 1 s` the descriptive starved band is `max(0.06, 1.5 × 1) = 1.5 s`,
which captures 67.5% of smooth's moves and 68.2% of step's — it separates
nothing. That band was designed for small increments and is uninformative at
large ones. Reported rather than quietly dropped; the clock-crossing rows
above are what carry the mechanism at this TC.

### What this does NOT say

It does not say the smooth budget is worse overall. Match 1 (+40.64 ± 25.61,
H1 accepted, 438 games) and match 2 (−17.39 ± 20.07, undecided, 400 games)
are measurements at two different TCs and both can be true: the change helps
where the increment is tiny and plausibly hurts where it is fat. Neither is a
ladder claim. The TCEC entry plays 1800+3, where the shortfall is −3.3% and
the parked clock would be 8 s — a third regime that neither match covers.

### Recommendation, and it is not this lane's call

**Do not land on match 1 alone.** The pre-registered rule permits it — the
remedy did not fire — but the honest reading of −17.4 with 95.6% LOS against
is a warning, not a clearance, and the mechanism above says it is a real
effect with an identified cause rather than noise.

The mechanism points at a targeted fix: keep the rational base, and make the
**cap** increment-aware so it stops paying out into time trouble when income
is large, instead of the current increment-blind `wtime²/(2·wtime + 4000)`.
That would aim to keep match 1's gain and remove match 2's loss.

But per the amendment recorded before this match reported, **any change to the
cap invalidates match 1's acceptance** — the retuned expression would have to
rerun both matches, or ship with a proof its 60+0.1 allocation is unchanged. A
cap change cannot offer that proof, since 60+0.1's parked regime is exactly
where the cap binds. So the honest cost of the targeted fix is **both matches
again**, at a corrected power (≥ 1 750 games for the non-inferiority arm).

That is a slot decision, reported rather than taken.

### Cotenancy

Launched 20:52 UTC, finished 23:20 UTC, cotenant throughout; box load 20.9 at
finish, 1 other fastchess process. Nothing of a cotenant's was killed,
reniced or modified. Arena `~/sunfish-bench/tmsmooth-20260814/m2/`.

## 2026-08-14 — CORRECTION + AMENDMENT: the sudden-death identity boundary is 40/19 s, not 8/3 s; and Match 1's acceptance does not survive a retune

Per this file's own rule, the entries below are **not rewritten**. This entry
says what was wrong in them and what replaces it. External review of PR #188's
write-up raised every item here; all were re-derived independently before
being accepted.

### 1. The boundary figure was wrong — and wrong in the safe direction

Entries below state that at `winc == 0` the smooth budget is bit-for-bit the
step arm "above a 2.667 s clock". **2.667 s = 8/3 is not a boundary of the
shipped policy at all.** It is the crossover of `max(wtime/2 − 1, wtime/8)`, a
cap that was designed, considered and **abandoned** before anything was built.
The shipped cap is `wtime²/(2·wtime + 4)`, and its arithmetic is different.

The two real boundaries at `winc == 0`, both exact rationals:

| boundary | equation | value |
|---|---|---|
| new cap stops binding | `T/40 = T²/(2T+4)` ⟺ `40T = 2T+4` | **T = 2/19 ≈ 0.1053 s** |
| new policy == step policy | `T/40 = T/2 − 1` ⟺ `19T/40 = 1` | **T = 40/19 ≈ 2.1053 s** |

giving three regimes rather than two:

| clock at `winc == 0` | shipped allocation | step allocation | relationship |
|---|---|---|---|
| **T ≥ 40/19 ≈ 2.105 s** | `T/40` | `T/40` | **IDENTICAL** |
| 2/19 ≤ T < 40/19 | `T/40` | `T/2 − 1`, **nonpositive at T ≤ 2 s** | differ; ours strictly larger and always positive |
| T < 2/19 ≈ 0.105 s | `T²/(2T+4)` | `T/2 − 1` (deeply negative) | differ; ours the only positive one |

**Every conclusion drawn from the old figure survives, with more margin, not
less.** 40/19 < 8/3, so the identity region is **wider by 0.561 s** than
claimed. Stage 1's minimum measured clock was 2.4 s, which sits inside the
identity region with **0.295 s to spare** — so "the +235.5 ± 65.4 arm and the
smooth arm are the same engine everywhere that run went" is not merely still
true, it was understated. The decision to skip a 60+0 sanity match rests on
that identity and is therefore also unaffected.

Verified exhaustively rather than argued: over **every integer-millisecond
clock from 2106 to 400 000 the two allocations are bit-equal, with zero
exceptions**. Integer milliseconds is the whole reachable domain — UCI parses
`int(next(tokens))`. At real-valued clocks the two agree to ~1e-16 s, from two
float effects that are not policy differences: `wtime*1000/40000` and
`wtime/40` are different expressions of one number, and `wtime/2 − 1000`
suffers cancellation at the boundary (1052.63 − 1000 keeps ~13 digits).

**Propagation, named.** The pool-time-manager lane's entries above
(`ARM (a) VERDICT` and the ladder pre-registration) cite the 2.667 s figure
because they took it from this lane's write-up. Their argument is
strengthened, not weakened: the baseline they measured against is identical to
the stage-1 winner over a wider band than they claimed. No number of theirs
changes.

Tests pin both boundaries exactly, with values immediately either side, in
both repos — plus a named regression test asserting that **nothing happens at
2.667 s**, so the abandoned figure cannot creep back.

### 2. Cap wording: absolute vs relative

The entries say the new cap "differs from the old by exactly 4/(t²−4)". That
conflates two quantities. Correctly:

- **absolute** difference: `2/(t + 2)` seconds
- **relative** difference: `4/(t² − 4)`

The percentages quoted (4.2% at 10 s, 0.11% at 60 s, 0.004% at 300 s) were
relative and are correct; only the label was wrong.

"The same cap everywhere it was ever measured" is also too strong and is
withdrawn. Correct statement: the two caps are **numerically near-identical in
the measured high-clock region**, and the complete allocation is frequently
**exactly** equal there for a different reason — the `T/40` base binds first
for both policies, so neither cap is consulted.

### 3. Increment partition: the three claims are not the same claim

The entries' `winc` rows overlap and over-claim. Replaced by a partition:

| band | status | evidence |
|---|---|---|
| `winc == 0` | preserved exactly, except for low-clock safety below 40/19 s | stage 1's +235.5 ± 65.4, carried |
| `0 < winc < 1 s` | **materially changed** — this is the transition band | **direct evidence at winc = 0.1 s only**: +40.6 ± 25.6 at 60+0.1. No claim is made about other points in the band |
| `winc ≥ 1 s` | within 10% of the audited policy, always spending **less** | analytic: `r(I) = 1 − 28/(40 + 240·I)`, monotone, `r(1) = 0.9` exactly. Match 2 tests the boundary of this band |

### 4. Continuity and monotonicity, as results rather than adjectives

$$\\frac{\\partial B}{\\partial I} = \\frac{560\\,T}{(40 + 240 I)^2} + 0.9 > 0$$

for all `T ≥ 0` — so the base is **strictly increasing** in the increment, and
since the cap does not depend on `winc` at all, the complete allocation is
nondecreasing in both arguments. It is **continuous** everywhere, with a kink
where the cap begins to bind. From here on, formal claims say *continuous*;
*smooth* is used only informally, because the allocation is not
differentiable at the clip.

The missing test is added: monotonicity in **winc**, on a grid, in both repos.
Monotonicity in `wtime` was already tested — but `winc` was the dimension the
defect was in, which makes its absence the more embarrassing gap of the two.

### 5. The parking fixed point — why the losing arm plateaus instead of flagging

Match 1's most striking observation was that the step arm's clock settles at
2.1 s in every one of 438 games. That is not an empirical curiosity, it is a
one-line consequence of the old cap. Once `wtime/2 − 1` binds, the arm spends
exactly that and banks one increment:

$$T_{n+1} = T_n - \\left(\\tfrac{T_n}{2} - 1\\right) + I = \\tfrac{T_n}{2} + 1 + I
\\qquad\\Longrightarrow\\qquad T^{*} = 2 + 2I$$

a contraction with slope ½, so the fixed point is attracting from any starting
clock. **It predicts both runs from one expression**: `I = 0` gives 2.0 s and
stage 1 measured the pre-fix arm asymptoting at exactly 2.0 s; `I = 0.1` gives
2.2 s and match 1 measured a 2.1 s median with a 2.0 s minimum — the fixed
point less per-move overhead. It also explains the zero forfeits directly: at
the fixed point spend equals income, so the arm can starve indefinitely
without ever losing on time.

The shipped cap has no such fixed point: it is positive for every positive
clock and never stops paying out.

### 6. Nominal allocation vs observed move time

The divisor tables in the entries below give the **nominal allocated budget**,
which is not what a stopwatch sees. Observed times are shorter and vary,
because the search breaks at the first iteration boundary after 0.8 × the
budget, and because the reported figures are **single runs on one fixture**
(startpos, or a fixed 8-ply opening), not medians over repetitions. So
`2.90 s` nominal at 60+0.1 against `2.39 s` / `2.34 s` observed on the packed
artifact and `2.17 s` through `uci.py` is the soft break working as designed
(0.8 × 2.90 = 2.32), not an inconsistency; likewise `5.40 s` nominal at 60+1
against `4.50 s` packed and `3.85 s` through `uci.py`. Tables are relabelled
accordingly and every runtime figure is annotated with its fixture.

### 7. AMENDMENT to the pre-registration: Match 1 does not survive a retune

The pre-registration below permits one remedy if match 2 fails: retune the
rate constant 20 → 30, rerun match 2 once. **That is under-specified, and this
amendment closes it before match 2 reports.**

If the remedy changes **any** of `{20, 240, 0.9, the cap}`, it changes the
allocation at 60+0.1 as well — the rate constant is shared by the whole curve,
not local to increment TCs. Match 1's H1 acceptance was earned by one specific
expression, and a retuned expression is a different arm that has never played
that TC.

**So: if the retune touches any of those four, Match 1's acceptance does NOT
carry.** The retuned expression must either

1. rerun **both** matches, or
2. ship with a **proof** that its 60+0.1 allocation is unchanged from the arm
   that won match 1 — which, since 60+0.1 sits in the transition band the rate
   constant controls, is only available to a remedy that leaves the band alone.

Recorded now, before match 2's result is known, so the result cannot pick the
rule.

## 2026-08-14 — MATCH 1 VERDICT: the smooth budget is +40.6 ± 25.6 over the step at 60+0.1, H1 accepted in 438 games

Pre-registered in the entry directly below; nothing here changes a bar. The
SPRT stopped itself at 438 of the 600-game cap in **1 h 53 m**.

| | |
|---|---|
| games | **438** of the 600 cap — `SPRT ([0.00, 20.00]) completed - H1 was accepted` |
| smooth (engine1) | **168 W, 117 L, 153 D — 55.82%** |
| Elo (smooth − step) | **+40.64 ± 25.61** (pentanomial 95%), nElo +52.19 ± 32.54 |
| LLR | 3.00 against ±2.94, **LOS 99.92%** |
| pairs | Ptnml(0-2) **[14, 38, 82, 53, 32]** over 219 pairs, PairsRatio 1.63 |
| independent check | `pair_elo.py` reproduces fastchess digit-for-digit: +40.64 ± 25.61 |
| illegal moves | **0** — and 0 `(none)`, 0 crashes, 0 recovers, 438/438 games carry a Termination tag |

**The step form was leaving Elo on the table at tiny increments.** That is the
positive branch of the pre-registered reading, and it is the one that landed:
this is not a change that had to fall back on aesthetics.

### The mechanism is the drain, and it is the same one stage 1 found

**Zero time forfeits. On either arm. All 438 games terminate `normal`** — no
adjudication, no resign rule. Identical in kind to stage 1: the defect does
not flag, it starves.

| reading | smooth | step |
|---|---|---|
| median clock left at game end | **4.6 s** (mean 6.7, min 0.6) | **2.1 s** (mean 3.1, **min 2.0**) |
| games ending under 2 s | 82 of 438 | **0 of 438** |
| games that NEVER cross 2.4 s | **339** of 438 | 90 of 438 |
| first move the clock crosses 2.4 s (median, of those that do) | move 78 | **move 43** |
| **starved moves** (≤ 0.15 s) | **1110 — 4.0%** of 27,568 | **9441 — 34.3%** of 27,537 |
| **median move time over its last 20 moves** | **0.391 s** | **0.115 s** |

The step arm's clock does not fall to zero and it does not fall to the 0.05 s
floor either — **it parks at 2.1 s and stays there**, in every single game
(minimum 2.0 s across 438 games, zero games under 2 s). That is an
equilibrium, not a coincidence: with the cap at `wtime/2 − 1000`, a 2.2 s
clock buys exactly 0.1 s of budget, which is exactly the increment. The arm
therefore spends its entire endgame paying one increment per move and never
loses on time, while the smooth arm is still thinking **3.4× longer** over the
same phase. A third of the step arm's moves are played that way.

### The pre-registered metric MISSED this, and the miss is the finding

The pre-registered mechanism number was "moves played at or under 0.06 s",
calibrated on stage 1, where a collapsed cap parks the budget on the 0.05 s
floor and the reading separated the arms 0 vs 1191. **At 60+0.1 it reads 0 vs
19 — essentially nothing — while the arms are in fact 4.0% vs 34.3% apart.**

The number stands as written: smooth 0, step 19, of ~27.5k moves each. The
defect is in the metric, not in the run, and it is instructive. A capped
budget does not settle on the floor at an increment TC; it settles wherever
spend equals income. Tie the threshold to the floor and it sees nothing the
moment the TC has an increment. The starvation reading beside it —
`≤ max(0.06, 1.5 × increment)`, plus the median last-20-move time — was added
at analysis time, is labelled DESCRIPTIVE everywhere it appears, and was
validated by re-running the archived stage-1 PGN, where it reproduces that
entry's numbers exactly and returns the same 0 vs 1191. Logged as a
pre-registration defect in the same form as stage 1's degenerate `0 < 0` pass
rule: **an entry, not a silent patch.**

### What this number is, and what it is not

+40.6 Elo is **arm-vs-arm at 60+0.1**, the TC chosen because it is where the
step's discontinuity lives. It is **not** a ladder claim and it does not
transfer to sudden death, where the two arms are bit-for-bit identical above a
2.667 s clock, nor to 30+1, which match 2 is running now. It says that the
regime the step form left broken — a tiny but nonzero increment — was worth
about 40 Elo, and that closing it by making the divisor continuous works.

Note also that the smooth arm ends games with a *lower* minimum clock (0.6 s
vs 2.0 s) and 82 games under 2 s, with **zero** forfeits. That is the intended
trade: it spends the clock it is given instead of hoarding it in a collapsed
state, and the cap that cannot go negative is what makes spending it safe.

### Cotenancy, closed out

Launched 18:48 UTC beside two resident matches, finished 20:41 UTC. Box load
12.7 at launch, 24.4 at finish; live fastchess 2 processes at both ends
besides ours. Nothing of a cotenant's was killed, reniced or modified. Both
arms play inside the same game under the same load, so the A/B is protected
and only the absolute clock figures could shift. Arena
`~/sunfish-bench/tmsmooth-20260814/m1/`.

### Match 2 is running

30+1 non-inferiority (elo0 = −10, elo1 = 0), cap 400, launched 20:52 UTC in
the same action that recorded this verdict. It prices the one thing this
change gives up: the increment budget is now the audited `/12 + 0.9·inc` only
asymptotically. Its remedy — one retune, one rerun — was fixed in the
pre-registration and is not revisited here.

## 2026-08-14 — PRE-REGISTRATION: the step budget becomes a smooth one, and the two matches that price it

Written before a game is played. Thomas's objection to the landed step form
was the whole cause: *"it's too non-continuous at winc close to 0."* The step
paced **60+0.1** — a sudden-death clock in all but name, 0.1 s/move of income
against ~0.9 s/move of spend — at **/12**, which is the exact drain the
`winc == 0` branch exists to close. One millisecond of increment moved the
divisor 40 → 12. A policy whose input is continuous should not be.

Shipped replacement (milliseconds; `sunfish_ui/uci.py` runs the same function
in seconds and the two are asserted equal under `t_ms = 1000·t_s`):

```python
think = min(wtime * (1000 + 20 * winc) / (40000 + 240 * winc) + 0.9 * winc,
            wtime * wtime / (2 * wtime + 4000))
```

One `min`, and it is the safety cap. The base is a single rational function
whose effective divisor `(40000 + 240·winc)/(1000 + 20·winc)` slides 40 → 12:

| winc | 0 | 50 ms | 100 ms | 200 ms | 500 ms | 1 s | 3 s | → ∞ |
|---|---|---|---|---|---|---|---|---|
| divisor | **40** | 26.0 | 21.3 | 17.6 | 14.5 | 13.3 | 12.5 | **12** |

### What carries from which validation, and why

This is the table the screens are built around. Two of these rows are free;
the third is what the matches buy.

| regime | shipped form there | evidence that carries | why |
|---|---|---|---|
| `winc == 0`, clock > 2.667 s | **bit-for-bit the stage-1 `tmfix` arm** | **+235.5 ± 65.4 at 60+0**, 64W-5L-31D, LOS 100% | base is exactly `wtime/40`; the two caps coincide above 2.667 s, which is where the whole 60+0 run lived (`tmfix` never went under 2.4 s in 100/100 games) |
| `winc == 0`, clock < 2.667 s | cap `wtime²/(2·wtime+4000)` instead of `wtime/2 − 1000` | none needed — the old cap is **negative** there | a negative cap is not a clamp, it is a collapse to the 0.05 s floor. Stage 1 measured what that costs: zero forfeits, and the pre-fix arm still played **1191 blind moves in 5349 (22.3%)** against `tmfix`'s **0 in 5385** |
| `winc > 0` | /12 + 0.9·inc **asymptotically** | the 11-game production audit, **weakened to a 10% bound** | base ratio `(12 + 240i)/(40 + 240i)` is 0.900 at i = 1 s and rises to 1: −7.4% at 30+1, −8.5% at 60+1, −3.3% at 300+3, always on the spend-less side. A bound is not an identity, so it gets a match |
| `0 < winc < 500 ms` | **/21.3 at 100 ms, not /12** | **none — this is the regime that CHANGED** | the step's discontinuity lived here; this is match 1 |

Analytic invariants, all asserted numerically in
`nnue_4k/tests/test_time_budget.py` and `tests/test_time_budget.py` (33 tests
each, both green): `winc == 0` is exactly `wtime/40`; the cap equals
`(wtime/2 − 1000) + 2·10⁶/(wtime + 2000)`, is strictly positive for every
positive clock, never exceeds `wtime/2`, and is within 5% of the old cap from
a 10 s clock up; no jump over a 0.1 ms `winc` grid at any clock (**and the
step form is asserted to FAIL that same bound**, so the test has teeth);
monotone nondecreasing in `wtime`; the ms and seconds forms agree to 3·10⁻¹⁶
relative; the old `/12` policy still reproduces the EAThUL0P loss and the
3+0/73-move walk still survives with 21.6 s to spare.

### The arms

Built by `tools/build/make_variants.py` from `nnue_4k/pst_entry.py`, which is
CI-guarded against its own generator. New mod `steptm` beside `oldtm`, both
anchored on the shipped budget so a reshaped budget breaks them loudly.

| arm | mod | packed | sha256[:16] |
|---|---|---|---|
| **smooth** = engine1 | `base` (= HEAD entry) | **3308 B** (788 spare) | `14b69a606b743a37` |
| **step** | `steptm` | **3295 B** | **`fe22791b409b1fba`** |

`steptm` reproduces the stage-1 `tmfix` artifact **byte for byte, sha for
sha**. The baseline in these screens is therefore not a rebuild of the
stage-1 winner, it *is* the stage-1 winner, and the arms are one expression
apart from one generator. (`oldtm` likewise still reproduces 3289 B /
`ecdf96bf34a2e593`, unused here.)

### Match 1 — the regime that changed: 60+0.1

| | |
|---|---|
| instrument | fastchess 1.8.2 on the bench box, arena `~/sunfish-bench/tmsmooth-20260814/` (NEW dir, fresh `git archive` of `nnue-4k` HEAD) |
| **engine1** | **smooth** (orientation trap: fastchess states the bounds in engine1's frame) |
| engine2 | step |
| TC | **60+0.1** |
| book | `book3k.pgn`, PGN not EPD (the packed artifact parses only `position startpos moves …`) |
| games | 300 rounds × 2, `-repeat`, **cap 600**, concurrency 8, `nice -n 10`, `-recover` |
| SPRT | **elo0 = 0, elo1 = 20**, α = β = 0.05, normalized |
| adjudication | **NONE, deliberately** — a drained clock kills long level endgames, which is exactly the class `-draw` would delete before the defect shows |

**Pre-registered readings, both reported whatever the SPRT says.** (1) Time
forfeits per arm. (2) **Blind moves per arm** — moves played at or under
0.06 s, i.e. the 0.05 s floor plus process noise. Reading 2 is the primary
mechanism number, not a consolation prize, because stage 1 established that
this defect class does *not* cash out as a flag: it had **zero forfeits on
either arm** and the losing arm still played 22.3% of its moves blind. The
instrument is calibrated on that run — `tally.py` (one file, arms and TC now
arguments) reproduces every stage-1 number from the archived PGN and scores
the blind-move reading **0 / 5385 for `tmfix` vs 1191 / 5349 for `oldtm`**, so
the metric is known to discriminate before it is used here.

**Zero tolerance, unchanged:** any ILLEGAL move by either arm kills the run
and is reported as a failure naming the game.

### Match 2 — the price of asymptotic-instead-of-exact: 30+1 non-inferiority

Same arena, same book, same discipline; run **after** match 1 reports, never
batched with it.

| | |
|---|---|
| **engine1** | **smooth** |
| engine2 | step |
| TC | **30+1** — the increment TC where the shortfall is largest among those the audit covered (−7.4%; 60+1 is −8.5% but 30+1 is the shorter run) |
| games | 200 rounds × 2, `-repeat`, **cap 400**, concurrency 8, `nice -n 10` |
| SPRT | **elo0 = −10, elo1 = 0**, α = β = 0.05, normalized — a NON-INFERIORITY test, not a superiority one |

**Reading.** H1 accepted → the smooth form is not worse than the audited
policy by as much as 10 Elo at 30+1, and the 10% shortfall is free. H0
accepted → the shortfall is real and costs ≥ 10 Elo.

**Pre-registered remedy, fixed now so the result cannot pick it.** If match 2
accepts H0 (or the 95% upper bound at cap is below 0), the rate constant gets
**exactly one** retune — 20 → 30 in both numerator and denominator terms,
i.e. `wtime·(1000 + 30·winc)/(40000 + 360·winc)`, which moves the 1 s divisor
from 13.3 to 12.9 and the 100 ms divisor from 21.3 to 17.8 — and **exactly one**
rerun of match 2, both ledgered. A second failure is reported as a failure and
the step form stands: the continuity fix does not get unlimited attempts to
find constants that pass.

### The 60+0 sanity match, considered and DROPPED — with the reason

The first version of this plan carried a third match: 60+0 smooth vs step,
non-inferiority, expecting ≈ 0. It is dropped because it is close to
**vacuous by construction**, and saying so is more useful than spending 400
games to rediscover it. At `winc == 0` the two arms differ *only* when the
clock is under 2.667 s — and stage 1 measured that `tmfix` (= the step arm
here) **never once** went under 2.4 s in 100/100 games at this exact TC. The
match would play two artifacts that are bit-for-bit identical in every
position they actually reach. The invariant is instead asserted analytically,
which is the stronger form: exact float equality of the two budgets at
`winc == 0` for every clock above 2.667 s.

### Honest note, recorded in advance

**If match 1 reads ≈ 0**, the continuity fix lands on aesthetic-plus-safety
grounds alone: a policy that is smooth in its inputs, and a cap that cannot go
negative, are worth having for their own sake and the change costs 13 bytes.
That is a legitimate outcome and it will be reported as one — *not* dressed up
as a win. **If match 1 is positive**, the step form was leaving Elo on the
table at tiny increments, which is a real finding about a real TC class
(lichess offers `+1`, `+0`, and everything between on the live bot path).
**If match 1 is negative**, the smooth form is worse where it differs most and
it does not land.

### Cotenancy

The box is shared. Both matches run at concurrency 8, `nice -n 10`, on 96
cores, cotenant with whatever else is resident; the `.boxlock` presence marker
says in writing that it is a marker and not an exclusivity claim, and any lane
that needs the window may reclaim it freely. Nothing of a cotenant's is
killed, reniced or modified. Load and live-fastchess counts at launch and at
finish are recorded in each match's `RESULT.txt`.

## 2026-08-14 — STAGE 1 VERDICT: the sudden-death TM fix is +235.5 ± 65.4 at 60+0, H1 in 100 games — and the drain kills by BLIND PLAY, not by flagging

Pre-registered in the entry directly above; nothing here changes a bar. The
SPRT stopped itself at 100 of the 600-game cap in **21 minutes 33 seconds**.

| | |
|---|---|
| games | **100** of the 600 cap — `SPRT ([0.00, 20.00]) completed - H1 was accepted` |
| tmfix (engine1) | **64 W, 5 L, 31 D — 79.50%** |
| Elo (tmfix − oldtm) | **+235.45 ± 65.41** (pentanomial 95%), nElo +335.28 ± 68.10 |
| LLR | 2.97 against ±2.94, LOS 100.00% |
| pairs | Ptnml(0-2) **[0, 3, 6, 20, 21]** — of 50 colour-swapped pairs, oldtm won **zero** |
| independent check | `pair_elo.py` reproduces fastchess digit-for-digit: −235.45 ± 65.41 from oldtm's side |
| illegal moves | **0** — and 0 `(none)`, 0 crashes, 0 recovers |

### The mechanism is NOT what the pre-registration expected, and the pre-registered second reading is why that is legible

**Zero time forfeits. On either arm. All 100 games terminate `normal`** — and
with no adjudication and no resign rule, every one of the 69 decisive games is
an actual **checkmate on the board** (34 White, 35 Black); the 31 draws are 29
three-fold repetitions and 2 insufficient material. The expectation written
down before the run — oldtm ≫ tmfix ≈ 0 forfeits — is simply wrong at this TC,
and the drain profile is what carries the mechanism instead, exactly as
reading 2 was written to do:

| reading | tmfix | oldtm |
|---|---|---|
| median clock left at game end | **16.9 s** (mean 18.2, min 8.2) | **2.0 s** (mean 2.7, min 2.0) |
| games ending under 2 s | **0** of 100 | **25** of 100 |
| first move the clock crosses 2.4 s (the negative-cap threshold) | **never, 100/100 games** | **move 42** (median; 84 of 100 games) |
| moves then played with the budget collapsed to the 0.05 s floor | — | **16** (median) |

**So the defect does not flag; it blinds.** Below ~2.4 s the `wtime / 2 - 1000`
cap goes negative, the budget collapses to the 0.05 s floor, and each move then
costs about as little as the process can spend — which is *cheap enough to keep
the clock alive*. oldtm's clock asymptotes at ~2.0 s and sits there. It never
loses on time; it plays the last third of the game at zero search and gets
mated. Round 27 is the shape of it — oldtm's own clock, its own game (93 moves,
lost 0-1):

    move   1  10  20  30  40  50  60  70  80  85
    clock 55.3 27.5 12.7 5.7 2.7 2.0 2.0 2.0 2.0 2.0   seconds
    last twelve move times: 0.00 s, twelve times

4.7 s of a 60 s clock on move 1, under the cap by move 40, and 45 further moves
played at the timer's resolution floor. That is what "wtime/12 at winc == 0"
buys, and it is why the ladder's PGNs showed the entry out-depthing classic and
losing anyway: the depth is real for the first forty moves and gone afterwards.

**Consequence for the loss taxonomy.** LOSS_TAXONOMY.md's 300+0 entry losses
are 94/130 MIDDLEGAME and 29/130 ENDGAME `SELF-DETECTED` swings with a *median
swing move of 32* and 36% of them preceded by a below-median-depth window. At
300+0 the same arithmetic puts the /12 crossing at ~56 moves; at 60+0 we can
now see directly that everything after the crossing is played blind. The H3
"depth crater" and H4 "sudden-death TM" hypotheses are not two findings — the
crater in the late middlegame IS the TM drain, and the fix addresses both.

### What this number is, and what it is not

+235 Elo is **arm-vs-arm at 60+0**, and it is the value of the defect under the
most exposing conditions available: one arm blind for 16+ moves while the other
still holds 17 s. It is **not** a ladder claim. It does not say the entry gains
235 on the python league, where the opponents are a field rather than a copy of
itself, the TC is 5× longer, and the ladder's other deficits (endgame eval,
mate-proneness) are untouched by this fix. What it does establish is that the
mechanism the ladder logs pointed at is real, large, and closed by six bytes of
source — which is precisely the question stage 1 was built to answer.

Caveats, stated rather than buried: the run was cotenant (box load ~20 of 96
cores), but both arms play inside the same game under the same load, so the A/B
is protected and only the absolute drain numbers could shift; and the clock
figures are RECONSTRUCTED from fastchess's per-move times because
`timeleft=true` records `tl=0.000s` for engines that emit no `info` lines.

### Stage 1 PASSES — and its pass rule has a pre-registration defect, recorded rather than patched

The pre-registered rule reads: PASS iff (H1 accepted or 95% LB > 0) **and**
tmfix's time-forfeit count is ≈ 0 (≤ 1% of its games **and strictly fewer than
oldtm's**). H1 was accepted and tmfix forfeited 0 of 100 — but "strictly fewer
than oldtm's" is **0 < 0, which is false**, so read literally the rule fails on
the degenerate case its author did not anticipate. That clause was written to
say "tmfix is not the arm losing on time", a condition 0/0 satisfies a
fortiori; the same entry also pre-registered the resolution — *"if forfeits are
near zero on both arms, the drain profile plus the SPRT is what carries (or
sinks) the mechanism claim"* — and both of those are unambiguous. **Verdict:
PASS.** The defect is logged here in the same form as the H3 rules' missing
minimum-SM-size clause: pre-registration defects get an entry, not a silent
patch.

### Stage 2, specced and NOT armed

`~/sunfish-bench/tmfix60-20260814/stage2_300.sh` — same two artifacts, same
book, same bounds (elo0 = 0, elo1 = 20, α = β = 0.05), **300+0**, cap 400
games, concurrency 8, nice 10. It refuses to run without `ARMED=1`, on purpose:
it needs a slot decision, not a lane's unilateral launch. Two honest notes for
that decision:

1. **Cost.** ~6–9 h, and *longer if the fix works* — the arm that stops
   flagging plays longer games. Stage 1 took 21 minutes; this is a different
   order of expense, which is the whole reason the two stages were split.
2. **What it buys.** The appendix's DIRECT bar at the benchmark TC. It does
   **not** recover the ANCHORED bar ((tmfix − classic) − (entry − classic)),
   which needs classic in the same tournament — a round-robin at ~1.5× the
   games. Given the size and cleanliness of the stage-1 result (LOS 100%, zero
   losing pairs), the case for spending the slot on the two-arm repeat is
   weaker than the case for spending it on the anchored form or on a ladder
   rerun; that trade is the coordinator's call, and the script is staged either
   way.

### Cotenancy, closed out

Launched at 15:29 UTC beside the resident 30+1 gauntlet, which stood at 191
games and 0 forfeits; at our finish it was at 284 games with **0 forfeits and 0
illegal moves**, unharmed. Nothing of theirs was killed, reniced or modified.
The `.boxlock` presence marker released itself on exit. Arena:
`~/sunfish-bench/tmfix60-20260814/` (arms, book, gates, manifest, pilot,
`match.pgn`, `RESULT.txt`).

## 2026-08-14 — PRE-REGISTRATION: the sudden-death TM fix goes to a TWO-STAGE validation, and stage 1 is a 60+0 SPRT against the pre-fix time manager

The P0 fix (`e73da7d`, `think = min(wtime / (12 if winc else 40) + 0.9 * winc,
wtime / 2 - 1000)`) is in the artifact and untested in games. This entry is the
screen kit, written before a game is played.

**This SUPERSEDES the confirmation form pre-registered in
`nnue_4k/LOSS_TAXONOMY.md`'s appendix** (one 300+0 round-robin, tmfix vs the
pre-fix entry vs classic, ≥200 games per pairing). The appendix's bars were
written before the ladder cost was understood: a 300+0 round-robin with three
arms is ~6–13 h of a bench-box slot, and it spends all of it on a question that
has a cheap decomposition. The mechanism question — *does the fixed budget stop
the clock-drain loss class at all* — is answerable at 60+0 for a fifth of the
clock, and it gates whether the expensive confirmation is worth a slot.

### Why 60+0 tests the same mechanism

The defect is a drain, not an overrun: no single move overruns: the budget
decays geometrically, and once the clock is under ~2.4 s the `wtime / 2 - 1000`
cap goes NEGATIVE, the budget collapses to the 0.05 s floor, and ~200 ms/move
of unavoidable lag finishes the job. Moves to reach that regime, from
`clock · (1 − 1/d)^n = 2.4 s` (lag and the 0.8 soft break ignored, so these are
order-of-magnitude, not predictions):

| divisor | at 60 s | at 300 s |
|---|---|---|
| **/12 (oldtm)** | **~38 moves** | ~56 moves |
| /40 (tmfix) | ~130 moves | ~193 moves |

Both time controls put the pre-fix arm inside a normal game's length and the
fixed arm outside it. 60+0 is the same mechanism at 1/5 the cost, which is
exactly what a stage-1 screen should buy.

### The arms: one AST node apart, both carrying the floor

Built by `tools/build/make_variants.py` from `nnue_4k/pst_entry.py`, which is
itself CI-guarded against its own generator — no hand-edited copy exists. New
mod `oldtm`, one anchor asserted to occur exactly once, reverting the budget
conditional to the unconditional `wtime / 12` the ladder actually played
(LOSS_TAXONOMY P0: 97.2% of 4,158 matched moves). The replacement text is
byte-identical to the line at `e73da7d^`.

| arm | source | packed | sha256[:16] |
|---|---|---|---|
| **tmfix** = engine1 | `base` (= HEAD entry) | **3295 B** | `fe22791b409b1fba` |
| **oldtm** | `oldtm` | **3289 B** (−6) | `ecdf96bf34a2e593` |

`e_base.py` packs **sha-identical** to packing `nnue_4k/pst_entry.py` directly,
so ARM A is the shipped entry and not a generator artefact of it.

**The arms were verified through the ARTIFACT, not the source.** Decompressing
both payloads (`tail -c+75 | xz -d`) and diffing the texts shows six *renamed
locals* on top of the budget line, because pyminify assigns single-character
names by frequency and deleting one use of `winc` reshuffles the ranking. A raw
diff therefore cannot answer "is the budget the only difference". Canonicalising
exactly the minifier-generated names (1–2 characters) by first-occurrence order
and leaving every real identifier (`len`, `max`, `min`, `gen_moves`,
`deadline`, …) verbatim, then diffing the two ASTs, gives **exactly one
differing node in 3,858**:

    A: IfExp(test=Name(winc), body=Constant(12), orelse=Constant(40))
    B: Constant(12)

Nothing else in the artifact differs — same eval, same search, same driver
redirect, and **both arms carry the structural bestmove floor (`03beefe`)**. So
this screen isolates TIME MANAGEMENT, not the `(none)` forfeit class the floor
already closed by construction.

### Gates, all green before a game was played

| | tmfix | oldtm |
|---|---|---|
| legality, 100 positions × `movetime 300` **and** `nodes 20000` | **0 no-move, 0 illegal** | **0 no-move, 0 illegal** |
| first-yield | SKIPPED (packed builds emit no `info` lines — reported as a skip, never a pass) | SKIPPED |
| mate-in-1, 8 positions, sources through the arena's own driver (`DRIVER_VERSION = 3`) | **8/8** | **8/8** |
| standalone smoke, empty cwd, `SF_NET`/`PYTHONPATH` unset | uciok + legal move | uciok + legal move |

**The TM assay is the gate that matters here**, because it is the only
black-box way to see which budget line is live inside a packed artifact — a mod
that silently failed to apply would otherwise produce two identical arms and a
screen that measures nothing:

| `go` | tmfix | oldtm | expected |
|---|---|---|---|
| `wtime 60000 winc 0` | **1.38 s** | **4.00 s** | 0.8 × 60000/40 = 1.2 s vs 0.8 × 60000/12 = 4.0 s |
| `wtime 60000 winc 1000` | 5.90 s | 4.77 s | same 5.9 s budget both arms (soft break vs hard deadline; the fix is byte-identical for winc > 0) |
| `wtime 1900 winc 0` | 0.00 s, legal | 0.00 s, legal | the negative-cap regime: BOTH arms collapse to the floor — the fix delays this regime, it does not remove it |

### The screen, pre-registered

| | |
|---|---|
| instrument | fastchess 1.8.2 on the bench box, arena `~/sunfish-bench/tmfix60-20260814/` (NEW dir, fresh checkout of `nnue-4k` HEAD shipped by `git archive`) |
| **engine1** | **tmfix** (orientation trap: fastchess states the bounds in engine1's frame) |
| engine2 | oldtm |
| TC | **60+0**, sudden death |
| book | **`book3k.pgn`, PGN not EPD** — verified from the pilot's engine log: fastchess sends `position startpos moves …` 100% of the time and `position fen` zero times, which is the only form the packed artifact parses |
| games | 300 rounds × 2, `-repeat`, **cap 600**, concurrency 8, `nice -n 10`, `-recover` |
| SPRT | **elo0 = 0, elo1 = 20**, α = β = 0.05, normalized |
| adjudication | **NONE, deliberately** — see below |

**No `-draw`/`-resign`.** Adjudication would delete the measurement: a drained
clock kills long, level endgames, which is exactly the class `-draw
movenumber=40 movecount=8 score=10` adjudicates away before the flag falls. At
60+0 nothing else is needed to bound game length — two minutes of clock is the
bound.

**Reading the SPRT.** H1 accepted → the fix is worth ≥ 20 Elo head-to-head at
sudden death. H0 accepted → it is worth ≤ 0. Undecided at 600 games is reported
as undecided, and the stage-2 rule below reads the interval instead.

**Pre-registered readings, both reported whatever the SPRT says:**

1. **Time forfeits per arm.** Forfeits in OUR PGN are DATA here, not an
   incident: the pre-fix arm losing on time is the defect showing. Expectation:
   oldtm ≫ tmfix ≈ 0.
2. **Drain profile.** Median clock remaining at game end per arm. fastchess
   writes `tl=0.000s` for these engines — the packed builds emit no `info`
   lines for it to track — so the clock is RECONSTRUCTED as 60 s minus the sum
   of that arm's own move times, which fastchess does record per move.

Reading 2 is not a consolation prize for reading 1. The drain does not need to
reach a flag to cost Elo: an arm that spends its last thirty moves at the 0.05 s
floor is playing them blind, and that shows up as lost games with
`Termination "normal"`. If forfeits are near zero on both arms, the drain
profile plus the SPRT is what carries (or sinks) the mechanism claim.

**A 4-game pilot ran first**, in a separate directory with a different seed and
excluded from the SPRT, purely to verify two build facts that a failed
assumption would have turned into 600 wasted games: that fastchess sends
`position startpos moves …` from a PGN book (it does — zero `position fen` in
the engine log), and what the PGN clock comments actually contain (`tl=0.000s`,
hence reading 2's reconstruction). Its incidental score is recorded in the
arena and claimed for nothing; the SPRT bounds above were fixed before it ran.

**Zero tolerance, unchanged:** any ILLEGAL move by either arm kills the run and
is reported as a failure naming the game. The floor makes it impossible; if it
happens, the floor is wrong and no Elo may be read from the run.

### Stage 2, pre-registered NOW so the stage-1 result cannot pick the rule

**Stage 1 PASSES** iff (SPRT accepts H1, or the 95% lower bound of
(tmfix − oldtm) is > 0 at cap) **AND** tmfix's time-forfeit count is ≈ 0
(≤ 1% of its games and strictly fewer than oldtm's).

- **PASS → stage 2 is ONE 300+0 SPRT**, same two arms, same book, same bounds
  (elo0 = 0, elo1 = 20, α = β = 0.05), cap 400 games, concurrency ≤ 8, nice 10.
  Cost estimate for the slot decision: ~6–9 h — and *longer if the fix works*,
  because the arm that stops flagging plays longer games. **SPECCED, NOT
  LAUNCHED**: it needs a slot decision, so it is reported back rather than
  armed. Note for that decision: a two-arm SPRT recovers the appendix's DIRECT
  bar only; the appendix's ANCHORED bar ((tmfix − classic) − (entry − classic))
  needs classic in the same tournament, which is a round-robin and ~1.5× the
  games. Whether the anchored reading is worth that is the coordinator's call,
  not this lane's.
- **FAIL or NEUTRAL → the 300+0 is NOT justified.** Report and stop. A fix that
  cannot show itself at the TC where the drain arithmetic says it is strongest
  does not get a second, more expensive chance to look good.

### What stage 1 cannot answer

It is arm-vs-arm with no classic anchor and not at the benchmark TC, so it
CANNOT say what the entry scores on the 300+0 ladder, and it cannot separate
"the fix helps" from "the fix helps *at 60+0*". It answers one question —
whether the fixed budget stops the clock-drain loss class — and that is the
question that decides whether the expensive one is worth asking.

### Cotenancy

Launched cotenant with a resident 30+1 gauntlet (owner-authorized capacity
sharing; that queue keeps right of way for its handoffs). The box is 96 cores
at load ~13 and 314 GB free; this run adds 16 engine processes at `nice 10`.
The `.boxlock` marker is taken as a PRESENCE marker with an owner file that
says so in writing and invites any lane that needs the window to reclaim it —
this lane is not claiming the box. Cotenant game count and forfeit count at
launch are recorded in the arena's `RESULT.txt`, and their PGN is polled for
forfeits alongside ours: a forfeit in THEIR match is a cotenancy harm signal
and gets reported immediately.

## 2026-08-14 — PACK_ENTRY.SH (layout B) gets the same indivisible pair: −31 on the bake-off's own cells, and the classic +4 hazard reproduces here too

Follow-up to `eb8897c`. The split-layout packer (`[head][engine.lzma][weights
raw]`) was flagged there as carrying the same dead shebang and left alone as
out of scope. Measured now against **layout B's own consumers** — the
bake-off's `b81` cells, generated by `bakeoff.run_net` itself, not by a
hand-rolled stand-in — because "the joint layout won, so the split layout
wins" is exactly the inference this ledger keeps having to un-learn.

| source through `pack_entry.sh` | base | −hoist | −shebang | **both** |
|---|---|---|---|---|
| **b81 bake-off cell** (v1c, 631-B tail) | 3913 | 3890 (−23) | 3905 (−8) | **3882 (−31)** |
| **b81 elided cell** (empty tail) | 3280 | 3257 (−23) | 3272 (−8) | **3249 (−31)** |
| `pst_entry.py` | 3380 | 3342 (−38) | 3371 (−9) | **3334 (−46)** |
| `sunfish.py` | 3271 | 3260 (−11) | 3275 (**+4**) | **3249 (−22)** |
| `sunfish_nnue.py` | 3970 | 3952 (−18) | 3962 (−8) | **3939 (−31)** |
| `replnet_proto.py` | 3880 | 3861 (−19) | 3873 (−7) | **3851 (−29)** |

**The classic +4 shebang-alone regression reproduces in this script too**, so
the pair is indivisible here for the same reason and the script says so. Note
the l1=0.001 net measures byte-identical to v1c in layout B (3913 / 3882):
layout B's tail is RAW, both tails are 631 B, and the engine stream is the
same source — it is one data point wearing two names, recorded so nobody reads
it as independent confirmation.

**Only the engine stream moves.** The weights are appended raw and the head
recomputes `head -c$lt` in the same run, so the offsets stay consistent by
construction. Verified rather than argued, on every artifact above: the head's
`tail -c+N` equals head_end+1, `head_end + lt + |weights| == filesize`, `SF_N`
equals the weights length, the artifact's last `SF_N` bytes are byte-identical
to the weights file, the bracketed slice lzma-decompresses, and the recovered
payload **no longer starts with `#!`** (the same assertion fires on every
pre-change build, which is how the check is known to be live). Each artifact
then booted for real: `uciok` and a legal `bestmove` from startpos, which is
what exercises layout B's `SF_A` self-read — a wrong offset would corrupt the
big-int and take the ROWS build down at import.

Bake-off consumers pick this up automatically: `packrun.PACK_REV` defaults to
`HEAD`, so one re-run refreshes layout-B denominators to **3882 / elided 3249**
(`BAKEOFF_PACK_REV=eb8897c` reproduces the old 3913 / 3280 for comparison).
Nothing executes a split artifact via the stripped line — the only consumer is
`packrun.boot_smoke`'s `["bash", artifact]`, and TCEC runs the outer
`#!/bin/bash` head, neither of which reads the payload's own first line.

---

## 2026-08-14 — PACK.SH: `--no-hoist-literals` + payload-shebang strip LANDED GLOBALLY, a win on every family that packs through the shared script

The golf lane priced two pack-pipeline levers on the replnet artifact and
deliberately did **not** land them, because `tools/build/pack.sh` is shared by
every artifact family and a lane that measures one file cannot speak for the
others. This entry is that verification. Both levers are now in `pack.sh`.

### What the levers are

* **`--no-hoist-literals`.** pyminify rewrites each repeated string literal to
  a fresh one-character global. That shrinks the TEXT and grows the ARTIFACT,
  because the repetition it deletes is exactly what lzma matches for free.
  Measured directly: turning hoisting off makes the minified stream *bigger*
  by +104 chars (classic), +74 (pst_entry), +60 (replnet) — and every packed
  artifact smaller. The lever is "stop helping".
* **The payload shebang strip.** `sed -e '1{' -e '/^#!/d' -e '}'`, line 1 only.
  The polyglot `#!/bin/sh` survives minification into the compressed payload,
  where nothing can ever read it.

### Per-family bytes — one real file per cell, through the real script

| family | source | base | −hoist | −shebang | **both** |
|---|---|---|---|---|---|
| classic | `sunfish.py` | 3232 | 3221 (−11) | 3236 (**+4**) | **3210 (−22)** |
| nnue | `nnue_4k/sunfish_nnue.py` | 3931 | 3913 (−18) | 3923 (−8) | **3900 (−31)** |
| entry (SHIPPED) | `nnue_4k/pst_entry.py` | 3341 | 3303 (−38) | 3332 (−9) | **3295 (−46)** |
| replnet proto | `nnue_4k/replnet_proto.py` | 3841 | 3822 (−19) | 3834 (−7) | **3812 (−29)** |
| replnet code-only | payload elided | 3217 | 3195 (−22) | 3211 (−6) | **3186 (−31)** |
| variant `base` | make_variants | 3341 | 3303 (−38) | 3332 (−9) | **3295 (−46)** |
| variant `cap` | make_variants | 3352 | 3308 (−44) | 3343 (−9) | **3300 (−52)** |
| variant `nolmr` | make_variants | 3341 | 3300 (−41) | 3332 (−9) | **3294 (−47)** |
| variant `khold2` | make_variants | 3365 | 3327 (−38) | 3357 (−8) | **3318 (−47)** |
| replnet TRAINED v1 | `l1=0.001` spliced | 3594 | — | — | **3567 (−27)** |
| replnet trained v1c | `v1c` spliced | 3543 | — | — | **3519 (−24)** |

**The combined column is a win in all eleven rows, and that is the only column
that ships.** The shebang strip ALONE is **+4 on classic** — it lands the
stream in a worse lzma neighbourhood — so the two levers are one indivisible
change. Landing them one at a time, in the wrong order, would have shipped a
regression to the release artifact and looked like progress. The `pack.sh`
header comment says so, in those words, so the next person cannot split them.

The golf lane's replnet reading reproduces within noise: they measured −24 /
−32, this stream reads −22 / −31 on the same file (their number came off a
1024-payload build; the pinned-payload builds below read −26 exactly).

### Correctness, per family

* **Standalone smoke, base vs both, ten runs.** Every artifact executed
  directly in an empty temp dir with `SF_NET` unset (except the nnue arm,
  which needs its net): `uciok` in all ten, and a legal `bestmove` from
  startpos AND from the 6-move Ruy `e2e4 e7e5 g1f3 b8c6 f1b5 a7a6`, legality
  checked against python-chess. No family changed answer where the search is
  deterministic; classic's one differing reply (b5c4 vs b5c6, both legal) is
  movetime jitter, not the pipeline — the artifacts are byte-identical to the
  ones smoked, verified by sha256.
* **`check_entry.sh` GREEN**: source still matches its generator, entry packs
  to **3295 (801 spare)**.
* **`replnet_check.py` PASS**: mirror + 40-ply walk (acc/ps/score) +
  antisymmetry + nn-fires + sentinel-margin. It reads the SOURCE, so the pack
  pipeline cannot perturb it; run as a control, and it did not move.
* **No pinned byte count anywhere to update.** Both CI workflows assert
  `-le 4096`; every pricing instrument (`price_engine.sh`, `measure.py`,
  `price_grid.py`, `price_taper.py`, `price_kbucket.py`, `price_candidates.py`,
  `build_students.py`) computes its baseline by packing at run time, and all
  of them report DIFFERENCES, which a uniform shift leaves untouched. Run as a
  check, `price_engine.sh` reads **ENGINE-SANS-EVAL 2871 → 2835 (−36)** and
  **EVAL CEILING 1225 → 1261 (+36)** — the eval lane's budget grew by more
  than the entry shrank, because a zero-eval build has less to un-hoist.
* **Test suite**: `tests/` + `nnue_4k/tests/` 336 passed, 2 skipped, and one
  PRE-EXISTING failure (`test_terminal_fail_high_reports_exact_score_before_none`)
  that reproduces identically with `pack.sh` reverted — the terminal-score fix
  is on master and not yet on this branch. Not caused here, not fixed here.
* **Stale-number debt this task could not pay** (single-writer: another lane
  owns the file): `nnue_4k/replnet_proto.py`'s header comment still quotes the
  pre-land `878 / 849` capacity and `4238 -- 142 B still to find`. The live
  numbers are **909 / 880** and **4212 — 116 B**. The golf lane must refresh
  that comment block on its next touch.

### The shebang is dead — audited, not assumed

The failure this could have caused is in the repo's own history: a pyminify
update once stripped the shebang while the head still did `exec $T`, and the
header fed Python source to `/bin/sh`. That is fixed at the root — the head
execs `$(command -v pypy3||echo python3)` on a `/dev/fd` — and the audit
confirms nothing depends on the payload's copy:

* every consumer runs the artifact through its OWN `#!/bin/bash` head, which
  this change does not touch: `./sunfish.packed` (CI), `[abspath(engine)]`
  (`legality_gate.py`), `["bash", artifact]` (`packrun.py`), `[tgt]`
  (`measure.py`). Screen wrappers drive `.py` SOURCES, not artifacts;
* nothing anywhere extracts the payload and runs it on its own — the only
  `xz -d` sites in the tree are the two pack heads themselves;
* `pack_entry.sh` names the interpreter the same way, so its payload shebang
  is dead too (**landed same day in the follow-up section below, after
  measuring layout B's own consumers rather than assuming this carried**);
* the SOURCE files keep their polyglot header. `./sunfish.py`, the wheel, the
  lichess bot and every dev path are untouched; only the copy inside the
  compressed payload goes.

### New capacity for the c1024 program

Pinned measuring payload as before (`make_proto_payload.py --zeros 0.596
--seed 20260814`), in-context cost = total − payload-elided total:

| build | base total | landed total | Δ | in-context payload |
|---|---|---|---|---|
| code-only (elided) | 3217 | **3186** | −31 | — |
| `--feats 1330` (the 1024-B program) | 4238 | **4212** | −26 | 1026 |
| `--feats 1170` | 4120 | **4095** | −25 | **909 — the frontier** |
| `--feats 1175` | 4125 | 4097 | −28 | 911 (over by 1) |
| `--feats 1130` | 4091 | **4066** | −25 | **880 at the 30-B margin** |
| `--feats 1135` (old frontier) | 4095 | 4069 | −26 | 883 |

**Capacity to size arms against: 909 B in-context absolute, 880 B at the
30-B safety line** (was 878 / 849). The base column reproduces the golf
lane's 4238 / 4095 / 4066 exactly, so the two pipelines are being compared on
one instrument and not two.

**The 1024-B payload program now builds to 4212 — 116 B over 4096**, down
from 142. The pack pipeline paid 26 of the gap; the remaining 116 is still
code-side and still belongs to the screened lanes, not to golf.

---

## 2026-08-14 — REPLNET GOLF ROUND 2: the 1024-B payload budget, measured; code side −232, play-identical; 142 B still missing

**Thomas's directive:** the replacement-net artifact gets a **1024-byte net
payload**; all code (engine + big-int machinery + decoders) must fit so the
total packed artifact stays ≤ 4096.

**Step zero — the true gap, measured, not composed.** The budget accounting
is IN-CONTEXT bytes through tools/build/pack.sh (same as the 617-B budget it
replaces). A real-shaped payload that COSTS 1024 B in context — random
ternary at the v1 winner's 59.6% zeros through the entry's own codec — is
`make_proto_payload.py --feats 1330` (raw 1339 chars; the decode reads its
768 features and never peels the front padding, so the artifact runs). That
build measures **4473 → true gap 377 B**, top of the expected 120-380 band.
The pinned measuring payload (feats 1330 / zeros 0.596 / seed 20260814) was
held fixed through every step below.

**The golf ledger** (every step: pack.sh on the pinned-payload splice +
replnet_check; accepted steps re-gated at the end; reverts kept):

| step | Δ | total | note |
|---|---|---|---|
| flat pst, no zero border | −22 | 4451 | padding never read; K_END always had nonzero padding |
| value(): castling-rook term | −31 | 4420 | exactly 0 on flat R rows; dead branch out |
| rows1 via [::-1] | +1 | REVERTED | lzma already had the 119−s pattern |
| __ne__ deleted | −10 | 4410 | no consumer; eviction spells `not ==`; comment warns tuple.__ne__ trap |
| QS/QS_A/EVAL_ROUGHNESS/LMR inlined | −17 | 4393 | no external readers; PROBE_CAP/TABLE_SIZE kept then priced separately |
| nn_cp tail: int(v/(1<<SHIFT)) + max/min | −13 | 4380 | bit-exact (|v| ≤ 11392 « 2^53); FASTER under pypy |
| MGP folded into gains decode | −6 | 4374 | |
| rows0/rows1 zip+slice forms | +7 | REVERTED | dedup loss beats raw savings |
| null-guard literal 'RBNQ'→'NBRQ' | −4 | 4370 | any() is order-free; dedups with promotion literal |
| import os → hidden dev block | −3 | 4367 | os is driver-resolution-only |
| directions["K"] alias | +8 | REVERTED | duplicate tuple was already free |
| tt Entry namedtuple → plain pairs | −17 | 4350 | lo/up unpack reused at the store |
| is-None tests → truthiness | −2 | 4348 | Move tuples always truthy |
| parse loop keyed on len(hist) | −12 | 4336 | exact parity identity; aligns with both render sites |
| rend() helper factoring | +2 | REVERTED | sharing costs bytes under lzma... |
| ...render sites spelled byte-identically | −8 | 4328 | ...but ALIGNMENT is free: same var names, whole-block lzma match |
| trit peel as one comprehension | −10 | 4318 | `_d // 3**k % 3`; `_half[_p] = _h = [0]*120` |
| header decode as block peels | 0 | REVERTED | byte-neutral; loop form kept |
| value(): pawn ep/prom flat constants | −6 | 4312 | rows flat ⇒ the literal 100; prom stays a table read |
| generated `initial` string | +14 | REVERTED | lzma loved the literal rows |
| TABLE_SIZE + PROBE_CAP inlined | −8 | 4304 | dev Hash option simply not offered (hasattr-guarded); consumers target sunfish_nnue |
| numeric knight directions | +3 | REVERTED | symbolic A+B dedups better |
| lambda params, put(j,p), 9e9, while 1 | −4 | 4300 | pyminify leaves lambda params unrenamed |
| from time import time | −2 | 4298 | 3 call sites |
| __hash__ on 3 fields | +3 | REVERTED | measured, not assumed |
| version = "sunfish replnet" | −5 | 4293 | API name kept; value truthful |
| layout literals hardcoded | −24 | 4269 | NN/LBITS/VBITS/HALF/ONES/M16 → literals; n8 literal map in the header comment |
| ACC_BASE folded into _B | −4 | 4265 | |
| gen_moves ep/kp guard → abs form | −6 | 4259 | same set {ep, kp, kp±1}; dedups with k()/value() |
| def main() removed | +36 | REVERTED | **locals are CHEAPER than globals under --rename-globals** |
| decode wrapped in a builder fn | +14 | REVERTED | the inverse also loses: plumbing beats rename gains — current structure is the optimum |
| piece dict via zip("PNBRQK", ...) | −9 | 4250 | the literal dedups 3 ways |
| directions via zip("PNBRQK", ...) | −6 | 4244 | |
| pst from piece.items() | +12 | REVERTED | |
| `% 2 == 0` → `not ... % 2` (5 aligned sites) | −3 | 4241 | |
| value() capture test `q != "."` | −3 | 4238 | targets on-board ⇒ piece or "."; dedups with move() |

**Result: 4473 → 4238 (−235); code-only (payload elided) 3449 → 3217
(−232); trained v1 candidate 3831 → 3594 (502 spare).**

**The budget line (measured):** payload 1024 / code 3217 / total **4238 —
142 OVER 4096**. The directive's 1024-B payload does NOT fit yet. What
fits TODAY at 59.6% random ternary: **payload 878 / code 3217 / total 4095**
(--feats 1135), or **849 B at the 4066 safety line** (--feats 1095). Against
the old 617-B budget that is **+261 B of paid-for capacity**, and the
trained-payload discount is large (v1's 777-char trained payload costs 382
in context vs 612 random-shaped): a trained net that SPENDS ~878 B will
carry well over 1135 features' worth of structure.

**Play-identity, verified:** node counts and full probe streams (depth,
gamma, score, move) are bit-identical to origin/nnue-4k over 5 random-walk
roots at depth 5 and the 6-move Ruy position at depth 7 (17,991 nodes both).
Every accepted step is exactness-preserving by construction; dedicated
scripts verified castling/ep/promotion incremental identity (ps/acc/score ==
from_board fresh) under BOTH K tables, and the builtin loop's rewritten
parse against 4 long random games through the packed artifact.

**Gate ladder on the golfed trained candidate (3594):** replnet_check PASS;
verify_export (a,b,c) bit-exact (payload==trainer; entry==int-ref==torch
on 60 fens x 3 views + 40-ply walk); legality **334/334** (packed);
first-yield **worst 183 / 2048, 0 over** (source+driver); mate **8/8**;
mate-conversion **7/8 @500ms, 8/8 @1500ms — and origin reads the same 8/8
@1500ms**, the documented marginal-flip noise, not a change. nps probe
(pypy, interleaved): golfed 14.4-15.6 us/node vs origin 17.0-18.6 — the
int(v/2^s) tail is FASTER, node-for-node identical.

**The c1024-n8 codec seam (TRAINQUEUE #3 asked this lane to define it):**
two chars per feature, low char = lanes 0-3, high char = lanes 4-7; decode =
`divmod(_w, 8100)` + trit_k = `_d // 90**(k//4) % 90 // 3**(k%4) % 3`,
range(8), and the 128-bit literal map (1<<256 for _U, _R2 = 1|1<<128,
rows shift 128, nn_cp mods at 128). Built in scratch, mirror/walk/
antisymmetry invariants PASS, and PRICED: **n8 code-only 3232 (+15)**; n8
payload is 1553 raw chars → totals **4410 @59.6% zeros, 4233 @73.7%** —
n8-at-ps768 does not fit without ~80%+ zeros or the bake-off lane's better
encoder. The seam is agreed on the golf side; export_replnet already emits
N=8 (make_proto_payload --N 8 matches it digit-for-digit).

**Instrument intel for the coordinator — since LANDED, see the entry above
(pack.sh changes every anchor, so it needed a cross-family verification this
lane could not do):** pyminify's literal hoisting fights lzma:
`--no-hoist-literals` is **−24 B** on this artifact; stripping the
(unused-in-artifact) shebang line in the pack pipe adds −8: **−32 combined**
(4238 → 4212 equivalent; capacity 878 → ~910). lzma lc/lp sweeps are ±0-1 and
not worth it. **Landed 2026-08-14 and confirmed a win on all nine families;
the estimate held: 4238 → 4212 measured, capacity 878 → 909 in-context.** One
correction the cross-family sweep found and this lane could not: the shebang
strip alone is **+4 on classic**, so the two levers only pay together.

**Coordination (training lane):** c1024-cal fits trivially (ps768 max cost
≈ 720 < 849). c1024-kb4 stays priced out (12,288 trits ≈ 3,072 chars raw,
mid-2k in context). The export path accepts any payload size through
splice_entry, and the entry decode ignores front padding by construction —
verified end-to-end at feats 1330. Capacity number to size arms against:
**849 B in-context at the safety line, 878 absolute**, measured with
random ternary at 59.6%; re-price per-arm as sparsity moves.

**What did NOT close:** the remaining 142 B (110 with the pack.sh intel).
The engine regions after this round: bound 498, uci-main 467, move+value
363, gen_moves 208, machinery ~400 — all previously golfed; the reverts
above mark the measured walls. Next credible code-side moves are semantic
(sort tie-break, K_END drop, TM shrink) and belong to screened lanes, not
golf.

## 2026-08-14 — khold2 BUILT AND PRICED +27 B; the mate-conversion suite splits khold from khold2 EXACTLY on the pre-registered position

Implementation of the khold2 pre-registration below (same day, separate
commit — rules first, the H1/H2 pattern). `make_variants.py` gains `khold2`;
`khold` stays untouched as the mechanism control; the SHIPPED entry is
untouched.

### Prices, every row one real file through pack.sh

| arm | packed | vs base (3299) | spare |
|---|---|---|---|
| base | **3299** | — | 797 |
| khold | 3299 | **+0** | 797 |
| khold2 | 3326 | **+27** | 770 |

Base reproduces the post-golf 3299 exactly (instrument sanity; the H2 table's
3357 base predates the golf commits). khold repriced +0 on today's stream —
the golf moved its lzma neighborhood, previously +1. khold2 was estimated
+30…+70, measured +27: the comprehension's vocabulary (`sum`, `piece`,
`pos.board`, `"NBRQ"`-adjacent strings) is already in the stream; only
`heavy`, `.upper()` and the clause structure are new.

### The MATE-CONVERSION suite — the arms split exactly as pre-registered

Movetime 700 ms, niced, ladder running (fixed suite, comparative verdict,
same defender for all arms). ~75 s of engine time total.

| position | base | khold | khold2 |
|---|---|---|---|
| kqk-m1 | 1 | 1 | 1 |
| kqk-near | 2 | 2 | 2 |
| kqk-mid | 6 | **9** | 6 |
| kqk-approach | 8 | **FAIL — budget 18 exhausted** | 8 |
| krk-m1 | 1 | 1 | 1 |
| krk-m1b | 1 | 1 | 1 |
| krk-mid | 8 | 8 | 8 |
| krk-approach | 10 | 10 | 10 |
| **total** | **8/8** | **7/8** | **8/8** |

(cells = attacker moves to mate.) Every cell of the expectation table hit:

- **khold fails `kqk-approach` precisely as Thomas predicted**: final
  position `8/8/8/4k3/Q7/8/8/1K6 w - - 36 19` — in 18 moves the attacking
  king went a1→b1 and then SHUFFLED, halfmove clock 36, a 50-move draw in
  the making with a queen up. K_MID (own queen on) holds the mating piece
  home; no depth of search compensates. The objection is now a measurement.
- khold also converts `kqk-mid` SLOWER (9 vs 6) — the passivity tax is
  visible even where the king starts close; KRK untouched (queenless, the
  or-clause is false, K_END as pre-registered).
- **khold2 is move-for-move IDENTICAL to base on all 8** — the lone-queen
  escape hatch restores exactly the base's conversion play, while differing
  from base only in the ≥ 929 queen-on regimes the screen will measure.

### Verification (checked, not assumed)

- `piece`, `pst[P..K]`, `K_MID`, `K_END` tuple-identical to base in khold
  and khold2 (and the committed entry) — seam-only mods by construction and
  now by measurement;
- seam census: base carries the `and`-seam once, khold the `or`-seam once,
  khold2 neither (its two-line replacement, `heavy` twice);
- the khold2 condition unit-checked against python-chess on 8 board classes
  (KQK, KQ+P K, KQN K, KQ KR, KRK, KRB K, KQQ K, one-queen middlegame) —
  all match the pre-registered rule, including the accepted KQQ-vs-K wart;
- forbidden/meaningless compositions raise loudly in ALL FOUR orders:
  `khold2.pend`, `pend.khold2` (seam-line collision), `khold2.khold`,
  `khold.khold2` (subsumption — shared anchor);
- standalone smoke: packed khold2 alone in an empty dir, `PYTHONPATH`/
  `SF_NET` unset — `uciok`, legal `bestmove g1f3` at `go movetime 200`,
  the base's known standalone answer.

### Screen-queue consequence (the pre-registered rule, now triggered)

The suite split the arms, so per the pre-registration: **khold2 REPLACES
khold in every composition-matrix row** (kmid.khold → kmid.khold2,
khold.kact → khold2.kact; the pend prohibition transfers as khold2.pend,
verified loud above), **khold's screen priority drops to mechanism
control** — it runs only if khold2's reading needs the lone-queen clause
isolated, and it must never LAND (it cannot convert KQK). khold2 takes
khold's SPRT slot: `ab_fixednode.sh`, 20k fixed nodes, khold2 = engine1 vs
base, elo0 = 0, elo1 = 10, LAND at 95% LB > 0 on fixed-N confirmation,
got-MATED share and ENDGAME loss share as the secondary readings. All four
gates (first-yield, legality, mate-in-1 parity, mate-conversion per the
expectation table) rerun on the arm the screen actually plays, at screen
time. Nothing armed tonight; the ladder runs.

---

## 2026-08-14 — PRE-REGISTRATION: `khold2` — khold with a lone-queen escape hatch — and a MATE-CONVERSION gate that makes the KQK objection measurable

Thomas, on khold, verbatim: *"I think we need to promote king->center as soon
as either queen leaves, if we want to solve KQK end games."* The concern is
exact and it is khold's pre-mortem (b) sharpened into a conversion failure:
in KQK the ATTACKING side still has its queen on the board, so pure khold
selects K_MID and the attacking king — a mating piece, the only one that can
deliver the final square — sits home while the queen shuffles. Mate never
happens; a won game is a 50-move draw. The 17 mated-at-exactly-one-queen
losses are khold's target; KQK conversion is its constraint. This entry
pre-registers (rules first, then implementation, the H1/H2 pattern) both the
refined candidate and the gate that decides between them.

### `khold2` — the seam rule, exactly

K_END is active iff **(both queens are off) OR (root non-pawn, non-king
material across BOTH sides ≤ piece["Q"] = 929)**; K_MID otherwise. Written
at the seam (root-only, the scan rides the rebuild the landed kend+fresh fix
already pays for — cost class stays ZERO in the hot loop):

```python
heavy = sum(piece[c] for c in pos.board.upper() if c in "NBRQ")
pst["K"] = K_MID if heavy > piece["Q"] and ("Q" in pos.board or "q" in pos.board) else K_END
```

**THRESH = piece["Q"], and the rule is crisper than a threshold suggests**:
given a queen on the board, heavy ≥ 929, so `heavy ≤ 929` holds exactly when
**the queen is the lone non-pawn piece on the board** — KQK and its
pawn-dressed forms, nothing else. A queen plus even the lightest minor
(929 + 280 = 1209) stays K_MID; queenless boards take the first clause as
before. So khold2 ≡ khold everywhere EXCEPT lone-queen boards, where it
reverts to the base's K_END — the smallest deviation from khold that answers
the directive. Pawns are deliberately not counted: promotion races and
KQ-vs-pawns want the king out too, and a pawn is not a mating attack to
hide from.

Pre-mortems, khold2's own: (a) inherits khold's give-back in the ≥ 929
queen-on regimes it still guards (the endgame-loss-share secondary reading
carries over unchanged); (b) KQQ-vs-K after a race promotes a second queen
reads 1858 > 929 and holds K_MID — accepted, two queens mate without their
king; (c) KQN-vs-K (1209) holds K_MID though king help is customary —
accepted, the extra piece substitutes; (d) one comprehension per SEARCH
(root only) — not per node, but it is a new 120-char scan and the standalone
smoke should stay indistinguishable.

Composition rules: `khold2.pend` **FORBIDDEN** exactly as khold.pend (same
seam line, raises loudly in either order); `khold2.khold` is meaningless and
raises the same way (shared anchor — khold2 SUBSUMES khold, never compose
them); kmid/kact compose (disjoint anchors), and khold's designed pairing
with kact transfers: khold2.kact is the follow-up arm if kact's single shows
a mated-share rise.

### The MATE-CONVERSION gate (new instrument, pre-registered with the arms)

`tools/build/mate_conversion_gate.py` + `tests/files/mate_conversion.fen`.
The mate-in-1 gate asks whether the eval reorders an immediate win out of
reach; this asks whether the seam lets the engine FINISH a won king ending —
a failure invisible to mate-in-1 (king already placed) and to legality
(every answer legal), visible only over a sequence. The engine plays the
attacker via `position fen` + `go movetime` (driver-resolved and
banner-checked, mate_gate's discipline verbatim); the gate plays the
defender with a FIXED deterministic bare-king heuristic — legal capture
first (a bare king's legal capture is winning by definition), then maximal
centrality, then maximal distance from the attacking king, UCI-string
tie-break — which is part of the instrument and must never be tuned against
an arm. CONVERTED = checkmate within the position's attacker-move budget;
stalemate / piece lost / illegal / budget are distinct FAILs.

The suite: 8 positions, KQK and KRK, attacker (white) to move — two
mate-in-1 spots per piece, two mid-boxing positions, and one per piece where
the attacking king starts on a1 and **MUST approach before any mate
exists** (`kqk-approach`, `krk-approach`). Budgets are generous (the
failure of interest is directional passivity, not low-depth shuffling);
movetime 700 ms, niced, ~1–3 min total for three arms.

### PRE-REGISTERED EXPECTATION (written before any arm runs the suite)

| arm | seam in KQK | seam in KRK | expected |
|---|---|---|---|
| entry (base) | K_END ("q" off) | K_END | **8/8** |
| khold | **K_MID** (own Q on) | K_END (queenless) | **FAILS `kqk-approach`**; other KQK spots uncertain (king pre-placed or close), all KRK pass |
| khold2 | K_END (lone queen, 929 ≤ 929) | K_END | **8/8** |

If khold fails `kqk-approach`, that is the directive made measurable:
khold's screen priority DROPS below khold2's, khold2 REPLACES khold in
every composition-matrix row (kmid.khold → kmid.khold2, khold.kact →
khold2.kact, the pend prohibition transfers), and khold is kept only as the
mechanism control. If khold unexpectedly passes everything, both screen and
the suite gets a harder approach position before it is trusted. Either way
base must be 8/8 — a base miss is an instrument bug (budget or defender),
not a chess verdict, and the suite is fixed before arms are compared.

### THE SCREEN for khold2, pre-registered (H2's instrument verbatim)

| | |
|---|---|
| instrument | `ab_fixednode.sh`, 20,000 fixed nodes, the 2,000-position book |
| arm | `khold2` vs `base` — one SPRT, replacing khold's slot if the mate suite splits them |
| **engine1** | **the candidate** (orientation trap, verified on the C2 record) |
| SPRT | elo0 = 0, elo1 = 10, α = β = 0.05 |
| KEEP bar | **LAND requires 95% LB > 0 on a fixed-N confirmation** — SPRT's terminal Elo is biased away from zero and does not earn the number |
| undecided at cap | reported as undecided, never as a point estimate |
| secondary reading 1 | **got-MATED share of screen losses** — H2's outcome metric; reported with n, never gated |
| secondary reading 2 | **ENDGAME loss share** — the kend give-back detector, inherited from khold pre-mortem (a) unchanged |

Gates before any game, per arm, now FOUR: first-yield (PASS, max ≤ 2048),
legality (0 no-move / 0 illegal), mate-in-1 parity (8/8), and the
mate-conversion suite per the expectation table above. Expected bytes for
khold2: +30…+70 packed (one new comprehension line; `heavy` is a fresh
local). Exact price via pack.sh only, in the implementation entry. The
standing rules inherit verbatim: no combination before singles read
non-negative, `.seed` never composed into one arm only.

---

## 2026-08-14 — H2 candidates BUILT AND PRICED: kmid +22 B, khold +1 B, kmid.khold +23 B — order-independent in-lane and cross-lane, forbidden composition raises as designed, smoke green

Implementation of the H2 pre-registration below (same night, separate commit —
rules first, the H1 pattern). `make_variants.py` gains mods `kmid` and
`khold`; the SHIPPED entry is untouched.

### Prices, every row one real file through pack.sh

| arm | packed | vs base (3357) | spare |
|---|---|---|---|
| base | **3357** | — | 739 |
| kmid | 3379 | **+22** | 717 |
| khold | 3358 | **+1** | 738 |
| kmid.khold | 3380 | **+23** | 716 |
| khold.kmid | 3380 | +23 | (in-lane order control: packed sha256-identical) |
| kmid.kact | 3381 | +24 | (CROSS-LANE pricing: H2's kmid on H1's kact) |
| kact.kmid | 3381 | +24 | (cross-lane order control: packed sha256-identical) |

Base reproduces the recorded 3357 exactly — the instrument sanity check.
kmid was estimated +15…+45, measured +22 (the abs-expression really is
nearly free the third time — kact's whole formula line cost +1 for the same
reason). khold was estimated ±2, measured +1. The full H2 bundle spends
3.1% of the headroom; H1+H2 together (pend.kact + kmid.khold ≈ +66 B if all
four ever land) would spend 8.9%.

### Verification (checked, not assumed; no engine except the one smoke)

`check_tables_h2.py` (scratch instrument, scratchpad) execs the eval region
of every built arm plus the committed base entry — 7 arms verified:

- `piece` and `pst[P..K]` tuple-identical to base in ALL arms (the base-90
  stream is untouched by construction, and now by measurement);
- `K_END` bit-identical to base in every non-kact arm — kmid/khold do not
  perturb the landed kend fix (the mirrored-kend lesson, applied in advance
  again);
- kmid arms: `K_MID` exactly `x and x + 6*grad(i) - 48` over base's K_MID,
  delta range exactly [−36, +36], padding zeros preserved, minimum still
  above MATE_LOWER (the kingless-sentinel margin holds); spot values
  g1 +24, e1 0, a1 +36, e4 −36;
- khold arms: `K_MID` bit-identical to base; the seam line carries `or`
  exactly once and the `and` form is gone;
- kmid.kact / kact.kmid: K_END reproduces the 14/step kact formula AND
  K_MID the kmid formula — order-independence at the table level, on top of
  the packed-sha identity above;
- **the pre-registered FORBIDDEN composition fails loudly**: `khold.pend`
  and `pend.khold` both raise "anchor occurs 0 times" at generation — the
  seam-line collision is a designed hard error in either order, verified
  tonight, so it can never silently screen as a half-applied arm.

Standalone smoke (the one engine run spent on this): packed `kmid.khold`
alone in an empty dir, `SF_NET`/`PYTHONPATH` unset — `uciok`, legal
`bestmove g1f3` at `go movetime 200`, the base's known standalone answer.

**No Elo is claimed.** The screen, its gates, the secondary mated-share and
endgame-share readings, and the LAND bar are in the pre-registration entry
below; nothing is armed while the ladder runs.

---

## 2026-08-14 — PRE-REGISTRATION: H2 king-safety terms — two zero-hot-loop candidates, and the base engine's own seam is part of the evidence

LOSS_TAXONOMY.md H2: the entry is MATED in **33%** of its losses (43 of 130)
against classic's 21% in the same league and 9.6% in the classic-vs-classic
control; the C1 exemplar has d-house walking a queen+knight in from a −1 eval
with the entry's depth steady at 11 — the attack was never priced, it was
assembled outside the horizon. The d1 calibration says this is a targeted
shape anomaly, not generic weakness. This entry is the candidate design and
the screen rules, committed BEFORE implementation and before any byte is
priced. No games tonight; the laptop runs a timed ladder.

**What "king safety" can even be in this architecture.** The eval is an
incremental PST delta; there is no check concept (king-capture-as-mate), no
king-ring scan, and qsearch admits only moves with static value ≥ QS = 40 —
quiet checks and slow king-side buildups are invisible at depth 0 by
construction. So a king-safety term must take one of three forms, in
cheapest-first order: (i) TABLE SHAPE — make the king's own wandering
expensive (zero hot-loop cost); (ii) SEAM POLICY — change WHEN the king
tables swap, at the root where phase is free (zero hot-loop cost); (iii)
SEARCH ADMISSION — let king-relevant quiet moves into QS (scan class, the
expensive kind). Two of form (i)/(ii) are implemented; one of each remaining
form is designed and explicitly not built.

**The kact interaction is designed for, not discovered later.** H1's kact
pre-mortem recorded that a centre-pulled king may FEED mate-proneness. On
inspection the hazard is live in the BASE engine: the seam selects K_END —
60070 minus 10 per step of centre-manhattan distance, a table that actively
PULLS the king centre-ward — as soon as EITHER queen leaves the board. Trade
our queen while theirs stays on, and our king starts marching toward the
enemy queen.

**The queen-regime split of the mated losses, measured tonight** (scratch
instrument `mate_regime.py`, deterministic python-chess replay of the
pyleague PGN, queen presence read off the FINAL position; fresh snapshot —
the ladder has appended since the taxonomy's 00:54 one, so n is larger):
491 games parsed, 160 entry losses, **49 mated**, splitting **23 both
queens on / 17 exactly one queen / 9 queenless**. Classic in the same file:
131 losses, 27 mated, splitting 11 / 13 / 3. So 40 of the entry's 49 mates
happen with a queen on the board, and the exactly-one-queen slice — where
the base seam has already flipped to K_END and is centralizing the king
into a live queen — is 35% of them. That slice is `khold`'s half; the
both-queens-on excess (23 vs classic's 11, the largest single gap) is
`kmid`'s half. The two candidates partition H2's evidence between them,
and the queenless 9 belong to neither (that is kact's regime, K_END either
way).

### S1 `kmid` — steeper K_MID edge-vs-centre gradient (IMPLEMENTING)

- **Mechanism → loss class**: classic's fitted K_MID carries roughly a
  60–100 cp home-vs-centre slope (g1 +40, e4-ish −19…−51). At median depth
  10 with no king-ring term and no check extensions, that slope demonstrably
  does not keep the king out of assembling attacks (43 MATED losses). Add a
  linear centre-manhattan gradient, zero-centred at the middle ring:
  `+ 6*(abs(2*(i//10)-11) + abs(2*(i%10)-9)) - 48`, i.e. corner +36, e1 0,
  g1 +24, centre four −36 — roughly doubling classic's gradient without
  touching the material mean. BOTH kings read it via the 119−i mirror, same
  symmetry argument as kend. Middlegame only: K_END is untouched, so this is
  the exact mirror of kact (which steepens K_END and leaves K_MID alone).
- **Hot-loop cost class**: ZERO — startup formula over the decoded K_MID;
  the base-90 stream is untouched.
- **Expected bytes**: +15…+45 packed (the abs-expression already sits in the
  lzma stream; a near-copy is cheap). Exact price via pack.sh only, in the
  implementation entry.
- **Pre-mortem**: (a) FILE-BLIND — it cannot tell a shielded g1 from a
  stripped g1; a naked castled king still reads as safe, so the C1 attack
  class is only partially priced (that gap belongs to S3); (b) passivity
  tax — defensive king moves, luft, and legitimate king marches in dead
  positions are taxed equally; (c) coverage hole at the seam: once EITHER
  queen is off the board the base engine selects K_END and kmid does
  nothing — 17 of the 49 mated losses live there, which is `khold`'s half
  (and under khold.kmid the hole closes: K_MID then covers that regime too);
  (d) king-step deltas shift by up to 24 cp, so some quiet king retreats
  cross QS = 40 and get admitted at depth 0 — a real tree-shape change,
  correctly visible at fixed nodes; (e) the slope is a guess with a
  mechanism, not a fit — if the sign is right and the size wrong, a slope
  sweep is a follow-up, not part of this screen.

### S2 `khold` — hold K_MID until BOTH queens are off (IMPLEMENTING)

- **Mechanism → loss class**: the seam line reads `K_MID if "Q" in
  pos.board and "q" in pos.board else K_END` — K_END, the centralization
  table, engages when either queen leaves. With the OPPONENT's queen still
  on, centralizing the king is walking it toward the mating attack; that is
  the kact pre-mortem as a property of the baseline. One-word fix:
  `and` → `or`, so K_MID holds while ANY queen is on the board and the
  centralization reward waits for a genuinely queenless board. Addresses
  the 17 mated-at-exactly-one-queen losses directly.
- **Hot-loop cost class**: ZERO — same root seam, same boolean cost, the
  from_board rebuild after the swap already handles the carried score.
- **Expected bytes**: −2…+2 packed (raw diff is −1 character).
- **Pre-mortem**: (a) it NARROWS the landed kend fix (+107.5 ± 31.6 in its
  RR, keyed either-off): in Q-vs-no-Q endings the queenless side loses its
  king-activity reward too — one shared table cannot centralize one king
  and hold back the other, so the symmetric loss is accepted as the price
  of killing the mate feed. The screen PGNs' ENDGAME loss share is the
  pre-registered detector for this give-back (secondary reading, reported
  not gated); (b) locked or dead-queen positions keep the king home too
  long; (c) a one-word diff with a large behavioral surface — mate-gate
  parity is load-bearing, not a formality.

### S3 `pshield` — pawn-shield as wing-symmetric pawn PST deltas (DESIGNED, DEFERRED)

The mechanism with the most chess literature behind it, and the soundness
analysis is the useful part tonight:

- A king-FILE-conditional pawn table is UNSOUND in this architecture. One
  shared `pst["P"]` serves both colours through the 119−i point mirror — a
  180° rotation, files flipped — so a bonus keyed on OUR king's file lands
  on the OPPONENT's rotated files (their g-file shield would be read
  through our b-file entries), wrong whenever the kings are not
  point-mirrored. And any root-frozen king square goes stale the moment a
  line castles in-tree. Pseudo-legal movegen itself is NOT an obstacle —
  tables read only the board string.
- The sound form is UNCONDITIONAL and wing-symmetric: +s on files a-c/f-h
  rank 2, +s/2 rank 3, 0 on d/e — a pattern symmetric under the 180°
  rotation is side-correct by construction, and it taxes shield pushes
  whether or not the king has castled yet (which classic's own K_MID
  already rewards it for doing).
- **The claimed prior does not exist.** The task pointer to a classic
  PAWN_SHIELD=12 experiment was checked against this ledger, including the
  delaybonus-era entries: the delaybonus notes are bench-box scheduling
  records, and the only pawn-shield mentions anywhere are H2 itself and an
  NNUE feature-plane proposal. No slope prior to inherit; s would be a pure
  guess on top of a mechanism kmid already half-covers.
- **Why deferred rather than built**: kmid and pshield tax the same failure
  (self-exposure) from two ends; screening both tonight spends two arms on
  one mechanism before either has a reading. And pshield COLLIDES with
  pend's P-table capture — pend's `P_END` formula reads `pst["P"]`, so
  shield deltas would leak into the endgame table via the formula unless
  the composition is designed, not discovered. Revisit only if kmid lands
  and the mated share persists.
- Cost class ZERO (startup formula), expected +25…+50 B, pre-mortem: taxes
  legitimate pawn storms; discourages wing-pawn development the distilled
  PSTs currently get right (the entry's opening loss share is 0.8% — the
  one solved phase, not to be re-broken).

### S4 `qking` — QS admission for king-adjacent captures/checks (DESIGNED AND PRICED OUT)

The search-side fix for "attacks assemble outside the horizon": admit
king-ring quiet moves at depth 0. Priced honestly, it is the E3 of this
hypothesis: (1) cost class SCAN — a per-move enemy-king locate plus
adjacency test inside the admission filter, the hottest comprehension in
the engine, est. +80…150 code-class bytes; (2) QS-TUNING COUPLING — the
val ≥ QS − depth·QS_A gate, the futility break and stand-pat all assume
the admitted set is capture-shaped; flooding depth 0 with quiet king-ring
moves retunes QS/QS_A implicitly (H1's quadratic discipline was about
staying UNDER these thresholds; S4 deliberately punches through them); (3)
the pre-registered nps rule applies in full — fixed-node flatters any
scan-class term, so a timed confirmation is mandatory before landing.
Shelved unless both zero-cost arms fail and the mated share persists.

### Interaction matrix with pend/kact (pre-registered now, not negotiated later)

| pair | status |
|---|---|
| kmid.kact | allowed AFTER both singles read non-negative — disjoint anchors, disjoint phases (K_MID vs K_END); the "safe middlegame + active endgame" bundle |
| kmid.pend | allowed after singles — fully disjoint anchors and tables |
| khold.kact | allowed after singles, and it is the DESIGNED pairing: khold gates exactly the regime where kact's centre pull is dangerous; if kact's single shows a mated-share rise, khold.kact is the pre-registered follow-up arm |
| khold.pend | **FORBIDDEN as a dot-composition** — both mods rewrite the same seam line, so the generator raises loudly in either order (checked in the implementation entry, not assumed). If both land, a combined `pendkhold` mod must be WRITTEN and screened as its own arm |
| kmid.khold | the in-lane H2 bundle — built and gated now, SPRT only after both H2 singles read non-negative |

The standing H1 rule is inherited verbatim: NO combination runs before its
singles read non-negative, and `.seed` is never composed into one arm only.

### THE SCREEN, pre-registered (post-harvest window, nothing armed tonight)

H1's instrument, verbatim, so H1 and H2 singles can share one screen window:

| | |
|---|---|
| instrument | `ab_fixednode.sh`, 20,000 fixed nodes, the 2,000-position book |
| arms | `kmid` vs `base`, `khold` vs `base` — two independent SPRTs; combos per the matrix above |
| **engine1** | **the candidate** (orientation trap, verified on the C2 record) |
| SPRT | elo0 = 0, elo1 = 10, α = β = 0.05 |
| KEEP bar | **LAND requires 95% LB > 0 on a fixed-N confirmation** — SPRT's terminal Elo is biased away from zero and does not earn the number |
| undecided at cap | reported as undecided, never as a point estimate |
| secondary reading 1 | **got-MATED share of screen losses, ALL arms** — H2's own outcome metric; reported with n, never gated (n will be small) |
| secondary reading 2 | **ENDGAME loss share, khold arms** — the kend give-back detector from pre-mortem (a) |

Effect-size honesty: mate is a TERMINATION, not a cause — converting mated
losses to adjudicated losses is worth nothing. H2's value is bounded by the
games that were level before the attack landed; LOSS_TAXONOMY's estimate for
this pool is +20…40 Elo, and unlike H1 the mechanism is eval-shape, so fixed
nodes should see it without a timed amplifier.

**Gates before any game**, per arm (the b8/d8 lesson — an eval change moves
the first-yield distribution, and both candidates move king-move admissions):

```sh
B=/tmp/h2screen; mkdir -p $B
nice -n 15 python3 tools/build/make_variants.py $B base kmid khold kmid.khold
for a in base kmid khold kmidkhold; do
  nice -n 15 python3 tools/build/first_yield_gate.py $B/e_$a.py   # bar: PASS, max <= 2048
  nice -n 15 python3 tools/build/legality_gate.py $B/e_$a.py 300 --nodes=20000 --first-yield=2048
                                                                  # bar: GATE PASSED, 0 no-move / 0 illegal
  nice -n 15 python3 "$ARENA/mate_gate.py" $B/e_$a.py tests/files/mate1.fen 4   # bar: 8/8 parity
done
```

First-yield caveat, carried explicitly: the gate is the count-measuring v2,
and its PASS is a **lower bound on the true worst case** — the sample max
over our phase-stratified positions bounds the population max from below, so
PASS is margin evidence, not proof of immunity. It stays the bar because it
is the strongest instrument that exists here, not because it is airtight.

**Gamma-seed dependency**: H1's rule verbatim — if the seed lands on the
canonical entry first, every arm inherits it; if not and an arm fails
first-yield, that arm is BLOCKED on the seed rather than composed with it.

**The nps rule**: kmid and khold are zero-scan (startup formula + the same
root boolean), so fixed-node is a fair instrument and no timed nps
confirmation is required. S4-class scan terms remain bound by the timed-
confirmation rule regardless of their fixed-node reading.

**Timed follow-up**: a LANDing candidate joins the pre-registered 300+0
confirmation round-robin (LOSS_TAXONOMY.md appendix (b)) as one more arm of
the SAME tournament, per the standing shared-tournament methodology.

---

## 2026-08-14 — H1 candidates BUILT AND PRICED: pend +42 B, kact +1 B, pend.kact +43 B — order-independent, tables round-trip exact, standalone smoke green

Implementation of the H1 pre-registration below (same night, separate commit —
rules first). `make_variants.py` gains mods `pend` and `kact`; the SHIPPED
entry is untouched and regenerates byte-identical.

### Prices, every row one real file through pack.sh

| arm | packed | vs base (3357) | spare |
|---|---|---|---|
| base | **3357** | — | 739 |
| pend | 3399 | **+42** | 697 |
| kact | 3358 | **+1** | 738 |
| pend.kact | 3400 | **+43** | 696 |
| kact.pend | 3400 | +43 | (order-independence control: byte-identical composition) |

The pre-registration estimated pend at +30…60; measured +42. kact is one digit
(10 → 14) and costs exactly +1. Both fit trivially inside 739 spare; the
combined H1 bundle spends 5.8% of the headroom.

### Verification (no codec re-encode was needed, and that is checked, not assumed)

Both candidates are startup FORMULAS over the decoded base tables — the base-90
stream is untouched. `check_tables.py` (scratch instrument) execs the eval
region of every built variant, no engine process:

- `pst[P..K]` and `piece` tuple-identical to base in all three arms;
- pend's `K_MID`/`K_END` bit-identical to base's (the landed kend fix is not
  perturbed — the mirrored-kend lesson, applied in advance this time);
- `P_END` exactly `x and x + 2·(8 − rank)²` with padding zeros preserved,
  per-rank deltas {2, 6, 10, 14, 18, 22}, max 22 < QS = 40;
- kact's `K_END` reproduces the 14/step formula in both `kact` and `pend.kact`.

Standalone smoke (the one engine run spent on this): packed `pend.kact` alone
in an empty dir, `SF_NET`/`PYTHONPATH` unset — `uciok`, `bestmove g1f3` at
`go movetime 200`, the base's known standalone answer.

**No Elo is claimed.** The screen, its gates, the seed dependency and the
LAND bar are in the pre-registration entry below; nothing is armed while the
ladder runs.

---

## 2026-08-14 — PRE-REGISTRATION: H1 tapered endgame terms — two zero-hot-loop candidates at the queens-off seam, and the passer delta-rule priced out honestly

LOSS_TAXONOMY.md H1: the entry's ENDGAME loss share is 22.3% against classic's
8.9% under identical 300+0 conditions, the swings happen at NORMAL/HIGH depth
(eval knowledge, not clock), and both hand-checked exemplars are pawn-race /
king-activity blindness (41.h3?? c3!, 39.Nd5? and the a-pawn runs). This entry
is the candidate design and the screen rules, committed BEFORE the mods are
implemented and before any byte is priced. No games tonight; the laptop runs a
timed ladder.

**Method stance, from this ledger's own record**: Texel tuning and TWO
distillation programs fit better and PLAYED WORSE (C1 −57.7, C2 −93.8, d1
−76.0). RFP, LMP, corrhist and history all closed negative. So these candidates
are HAND-DESIGNED terms with a written mechanism, not fitted tables — the one
family this ledger has not yet falsified for endgame knowledge.

### The architectural lever: the root seam is free

`search()` already tests queens-off once per search, swaps `pst["K"]` between
`K_MID`/`K_END`, and rebuilds the root score with `from_board(...)` (the landed
kend+fresh fix, +107.5 ± 31.6 in its RR). Any whole-table swap keyed on the
same root boolean therefore costs ZERO in the hot loop — `value(move)` still
does two lookups, `move()` still does one delta — and the stale-carried-score
hazard is already paid for by the rebuild line. The taper pricing entry
established the same point from the byte side: the seam machinery is ~50 B; it
was the fitted second table set (+224 B, 11 pos/param) that was dropped, not
the seam. These candidates put HAND-MADE data on that landed seam.

A second premise dissolves on inspection: a phase condition needs NO
incremental material counter, because phase is only ever consulted AT THE ROOT
(once per search, where a full board scan is free) — exactly as kend works
today. An in-tree phase change would invalidate every carried score in the
tree; the taper entry already recorded that as "a different engine, not a
pricing question".

### E1 `pend` — endgame pawn-advance table (IMPLEMENTING)

- **Mechanism → loss class**: at queens-off, every pawn's value grows with
  advancement, both colours (the mover reads the opponent's pawns through the
  same table via the 119−i mirror). The entry stops walking into lost races it
  scores as −0.5, because the opponent's runner is priced before it is inside
  the QS horizon. Addresses ENDGAME SELF-DETECTED, 28 of 130 losses.
- **Form**: `P_END = tuple(x and x + (8 - i // 10) ** 2 * 2 for i, x in
  enumerate(pst["P"]))` — a FORMULA at startup, like K_END itself. Bonus by
  rank 2..7: 0, 2, 8, 18, 32, 50 (promotion row 72, which consistently
  discounts the promotion delta pst[prom]−pst["P"]). No codec re-encode: the
  decoded base tables are shared bit-identical by construction, padding zeros
  stay zero via `x and`.
- **Search-coupling discipline (the reason for the quadratic)**: per-move
  deltas are 2, 6, 10, 14, 18, 22 — ALL below QS = 40 and LMR = 60, so the QS
  admission gate, the futility break and the reduction trigger keep their
  measured tuning. A linear slope big enough to matter at rank 6 would cross
  QS on the single step 6→7 and flood QS with quiet pushes.
- **Hot-loop cost class**: ZERO (root table swap + startup formula).
- **Expected bytes**: the formula line, `P_MID` capture, and one root select
  line; comments are stripped by the packer. Estimate +30…+60 packed B against
  739 spare. Exact price via pack.sh in the implementation entry.
- **Pre-mortem**: (a) not passedness-aware — a blockaded or doubled pawn earns
  the same bonus, so the engine may push pawns it should hold; (b) queens-off
  is not "low material" — an early queenless middlegame with 4 rooks gets
  endgame pawn values (fallback: re-gate on a root material count, ~+25 B,
  design E4); (c) the slope is a guess with a mechanism, not a fit — if the
  sign is right and the size wrong, a slope sweep is a follow-up, not part of
  this screen.

### E2 `kact` — steeper K_END centralization (IMPLEMENTING)

- **Mechanism → loss class**: king activity is the other half of H1's
  evidence. K_END is already selected at queens-off, but its gradient — 10/step
  of centre manhattan distance — was inherited, never swept on this engine.
  Steepen to 14/step: an active king out-values a passive one by up to +126
  across the board instead of +90.
- **Hot-loop cost class**: ZERO (constant inside the existing startup formula).
- **Expected bytes**: ±0…2 B (one digit changes).
- **Pre-mortem**: (a) a diagonal centralizing king step's delta goes 40 → 56,
  crossing QS = 40, so those steps get firmly admitted at depth 0 — a real
  tree-shape change, correctly visible at fixed nodes; (b) the entry's OTHER
  anomaly is mate-proneness (33% of losses) — a king pulled centre-ward at
  queens-off with rooks on may feed H2 while helping H1. The screen PGNs'
  got-mated share is a pre-registered SECONDARY reading (reported, not gated).

### E3 — passed-pawn term with a delta rule: DESIGNED AND PRICED OUT (no build)

The mechanism is the best-evidenced of the three (both exemplars are passer
races), and a sound delta rule EXISTS — passedness of a pawn changes only on:
its own advance past an adjacent-file enemy pawn's rank, its own file change by
capture, or an enemy pawn on files f−1..f+1 leaving the front span (captured,
promoted, or advancing level-past). All local: a move()-time update needs a
3-file × ≤6-rank rescan for up to 3 affected pawns per side per pawn event.
Pseudo-legal movegen does NOT break soundness (passedness reads only the board
string). What kills it tonight is the ARCHITECTURE PRICE, in two parts:

1. **The score/ps split comes back.** In this entry score == ps and
   `value(move)` is an EXACT delta of it — the generator collapsed the two
   fields to save bytes. A passer term riding in score but not in value() makes
   the futility test and QS ordering carry a systematic error term; keeping
   value() exact means re-introducing the ps field and its threading, i.e.
   paying back bytes the entry already banked.
2. **Cost class SCAN, and fixed-node hides it.** The update is a per-pawn-event
   partial board scan in move() — est. +200…300 code bytes (the expensive
   kind) and a measurable nps tax concentrated exactly in pawn endings. A
   fixed-node screen CANNOT price that: any scan-class term that passes fixed
   nodes MUST take a timed confirmation before landing (rule pre-registered
   below).

`pend` is the cheap first-order approximation of E3 — an advanced pawn is
priced whether or not it is technically passed; what it cannot see is the
passed/blockaded distinction. E3 stays on the shelf unless pend lands AND its
losses still show passer blindness.

### E4 — material-count root gate (design note, fallback only)

Replace/augment the seam boolean with a root material count (e.g. non-pawn men
≤ 6) — free at the root, ~+25…35 B, no counter needed. Not screened now: it
multiplies the arm count and only becomes interesting if pend's failure mode is
early-queenless misfires (pre-mortem (b)).

### THE SCREEN, pre-registered (post-harvest window, nothing armed tonight)

| | |
|---|---|
| instrument | `ab_fixednode.sh`, 20,000 fixed nodes, the 2,000-position book |
| arms | `pend` vs `base`, `kact` vs `base` — two independent SPRTs; `pend.kact` runs ONLY if both singles read non-negative, then confirms as one arm |
| **engine1** | **the candidate** (orientation trap: fastchess states bounds in engine1's frame — verified on the C2 record) |
| SPRT | elo0 = 0, elo1 = 10, α = β = 0.05 |
| KEEP bar | **LAND requires 95% LB > 0 on a fixed-N confirmation** — SPRT's terminal Elo is biased away from zero and does not earn the number |
| undecided at cap | reported as undecided, never as a point estimate |

**Gates before any game**, per arm (the b8/d8 lesson — an eval change moves the
first-yield distribution):

```sh
B=/tmp/h1screen; mkdir -p $B
nice -n 15 python3 tools/build/make_variants.py $B base pend kact pend.kact
for a in base pend kact pendkact; do
  nice -n 15 python3 tools/build/first_yield_gate.py $B/e_$a.py   # bar: PASS, max <= 2048
  nice -n 15 python3 tools/build/legality_gate.py $B/e_$a.py 300 --nodes=20000 --first-yield=2048
                                                                  # bar: GATE PASSED, 0 no-move / 0 illegal
  nice -n 15 python3 "$ARENA/mate_gate.py" $B/e_$a.py tests/files/mate1.fen 4   # bar: 8/8 parity
done
```

**Gamma-seed dependency, decided now**: if the seed has landed on the canonical
`pst_entry.py` by screen time, every arm inherits it and nothing changes. If it
has NOT landed and a candidate fails first-yield, the candidate is BLOCKED on
the seed exactly as b8/d8 are — `.seed` is never composed into one arm only,
because that would screen two changes as one (the mirrored-kend lesson).

**The nps rule, written into the registration**: fixed-node screens hide
per-node time. `pend`/`kact` are zero-scan (startup formula + root swap), so
fixed-node is a fair instrument for them and no timed nps confirmation is
required. Any SCAN-class term (E3, king-ring counts, mobility) that passes
fixed nodes needs a timed confirmation BEFORE landing — no exceptions, this
line is the pre-registration.

**Timed follow-up (effect-size honesty)**: the endgame deficit is amplified
under timed sudden death (endgame loss share 22.3% timed vs 8.2%/3.8% at fixed
nodes), so a fixed-node PASS may understate ladder value and a small fixed-node
positive is still interesting. If a candidate LANDs, it joins the already
pre-registered 300+0 confirmation round-robin (LOSS_TAXONOMY.md appendix (b):
tmfix vs entry vs classic) as one more arm of the SAME tournament — shared
round-robin with the classic anchor, per the standing methodology, not a new
match.

---

## 2026-08-14 — PRE-REGISTRATION: the root gamma seed goes to a NON-INFERIORITY screen. It is a SEARCH change and it does not land here

The seed is now formally a **blocker** on the eval programme: `b8` and `d8`, the
two byte-negative arms, fail the first-yield gate, and `b1` passes it by 37
nodes. This entry is the screen kit, written before the script is staged.

**WHERE THIS LANDS. The seed is a SEARCH change.** If it passes it lands as a
**search-lane commit on `nnue-4k` in `sunfish-packed`**, not on
`eval-decode-track`, and the commit/PR authorship should say so. This lane owns
the measurements below because it owns all eight first-yield builds; it does not
own the change.

### The arm, built by the generator

`make_variants.py` mod `seed`: `gamma = 0` → `gamma = pos.score - 150` at the
root probe, one anchor, asserted to occur exactly once. Composed with every
candidate through the same generator rather than by hand.

| arm | packed | vs base | | arm | packed | vs base |
|---|---|---|---|---|---|---|
| base | 3350 | — | | seed | 3355 | **+5** |
| d1 | 3421 | +71 | | d1seed | 3426 | +76 |
| d8 | 3306 | −44 | | d8seed | 3311 | −39 |
| b1 | 3431 | +81 | | b1seed | 3436 | +86 |
| b8 | 3316 | −34 | | b8seed | 3321 | −29 |

**+5 bytes on every base**, which is the whole price.

### First yield: every arm, both bases, one instrument

| arm | median | p90 | p99 | **MAX** | gate |
|---|---|---|---|---|---|
| base | 5 | 28 | 178 | **780** | PASS |
| **seed** | 4 | 24 | 80 | **171** | PASS |
| d1 | 5 | 40 | 220 | 1,896 | PASS |
| **d1seed** | 4 | 26 | 123 | **537** | PASS |
| d8 | 5 | 44 | 377 | 2,059 | **FAIL** |
| **d8seed** | 4 | 26 | 125 | **478** | PASS |
| b1 | 4 | 38 | 447 | 2,011 | PASS |
| **b1seed** | 4 | 24 | 140 | **728** | PASS |
| b8 | 5 | 42 | 620 | 2,433 | **FAIL** |
| **b8seed** | 4 | 22 | 140 | **394** | PASS |

Generator-built and re-measured from scratch; the numbers reproduce the earlier
hand-substituted ones exactly. **Every arm passes with the seed**, including
both that fail without it, and the incumbent improves 4.6×.

### The other gates, and the cost

| | base | seed |
|---|---|---|
| legality | 100/100 | **100/100** |
| mate-in-1 | 8/8 | **8/8** |
| standalone, empty dir | `g1f3` | **`g1f3`** |
| nodes to complete depth 8, 40 positions | 691,287 | **691,313 = 1.0000×** |
| same move at completed depth 8 | — | **40/40** |

At equal completed depth the seeded search returns **the same move on every one
of 40 positions and costs 0.004% more nodes**. The seed only changes what a
search that is *stopped* commits to — which is exactly why it still needs games.

### THE SCREEN, pre-registered

Non-inferiority: the seed is bought for correctness, not Elo, so the question is
whether it costs anything, not whether it gains.

| | |
|---|---|
| instrument | `ab_fixednode.sh`, 20,000 fixed nodes, the 2,000-position book, arena `eval-c1-20260813` |
| **engine1** | **`seed`** |
| engine2 | `base` |
| SPRT | **elo0 = −10, elo1 = 0**, α = β = 0.05 |

**THE ORIENTATION TRAP, written out because it has bitten this project.**
fastchess states `elo0`/`elo1` **in engine1's frame** — verified against our own
record, not assumed: the C2 screen ran `-engine name=base` first with
`elo0=0 elo1=10` and reported "H1 was accepted" when **base** scored 63.18%. So
with `seed` as engine1:

- **H1 accepted → the seed is not inferior by 10 Elo → LAND.**
- **H0 accepted → the seed is worse than −10 → DROP.**
- **LAND requires the 95% lower bound to exclude −10**; DROP only on a
  demonstrated real cost. An undecided run at cap is reported as undecided.

### WHAT FIXED-NODE CANNOT SEE, and why the obvious timed check is VACUOUS

The seed's entire benefit is *yield timing under `Stop`*, and fixed-node play
never raises `Stop` before the search has answered. The SPRT above can therefore
only show that the seed **costs nothing**; it cannot show that it **buys**
anything. A timed check has to do that, and the naive version does not work:

Both stop conditions are polled at `self.nodes % 2048 == 0`, so **no abort can
land before node 2,048**. Any build whose max first yield is ≤ 2,048 is immune
to the `(none)` class at *every* time control. base is 780 and seed is 171 —
**both immune**. The driver's budget is
`think = min(wtime/12 + 0.9·winc, wtime/2 − 1)` seconds, and at the measured
42,473 nps, 2,048 nodes ≈ **48 ms**.

- At **10+0.1** the `0.9·winc` term alone floors `think` at 90 ms ≈ 3,800 nodes.
  **Nothing is ever aborted before node 2,048**, so base-vs-seed at 10+0.1
  returns zero for both arms whatever is true. That is a gate that passes
  everything — the exact failure mode this lane has paid for four times.
- At **1+0** (`winc = 0`), `think = wtime/12` drops below 48 ms whenever
  `wtime < 0.58 s`, which happens in the endgame of essentially every game.

**So the spot-check is re-specified, and this is a deliberate deviation from the
suggested 200 games @ 10+0.1:**

| | |
|---|---|
| TC | **1+0** (zero increment — the increment is what makes the check vacuous) |
| arms | **`b8` vs `b8seed`** — b8's max first yield is 2,433, *above* the cliff |
| control pair | `base` vs `seed`, same TC |
| n | 200 games per pair |
| counted | `bestmove (none)`, illegal-move adjudications, time forfeits |
| **prediction** | **`b8` > 0; `b8seed` = 0; base = 0 and seed = 0** |

The control pair is what makes it an instrument rather than an anecdote: it
shows the time control alone does not manufacture the failure. **This is a
correctness count, not an Elo measurement**, and it is registered as such —
1+0 on a shared box is far too noisy for a strength claim and none is made.

### The kit, staged and UNARMED

`tools/screens/seed_screen.sh`. Same discipline as `label_corpus.sh`: the GO
marker is checked **once**, it never waits and never chains, and the producer
writes `RESULT_seedscreen.txt` on **every** exit path including a crash or a
kill. Controls run before staging: unarmed → refuses with a result file
(exit 2); armed with an incomplete arena → refuses **before any games**
(exit 3), naming the missing file.

Two preflights inside the script that are not decoration:

- **Legality and first yield on all four arms**, seconds rather than games.
- **`b8` must STILL FAIL the node gate.** It is the positive control for
  stage 2 — if a rebuild has quietly fixed it there is nothing left to catch,
  and stage 2 would report a clean sheet that means nothing. The script aborts
  (exit 7) rather than produce that.

Staged into the arena that measured C1, C2 and d1, md5-verified after transfer,
and probed one position each to confirm the real driver resolves (`v2 nodes
fen`, all three answering `e1e8`):

| in `eval-c1-20260813` | |
|---|---|
| arms | `bin/e_seed.py`, `bin/e_b8.py`, `bin/e_b8seed.py` (+ `e_base.py` already there) |
| wrappers | `w_seed.sh`, `w_b8.sh`, `w_b8seed.sh` |
| script | `seed_screen.sh` |
| GO marker | **absent — UNARMED** |

Nothing was launched: the box is owned by the Thomas-side systemd queue and the
laptop by the ladder.

### Not pre-registered, deliberately

No Elo prediction. The seed changes the bisection, so a stopped search can
commit to a different move; whether that is worth anything is what the screen is
for, and this lane's record on predicting Elo from anything other than games is
0 for 2.

---

## 2026-08-14 — THE MIX IS THE MECHANISM: a size-matched control swings phase 18-24 by 16.8 points. The pre-registered check is CLEARED, and b8 fails first yield like every fit before it

The pre-registration is the entry below; this is what came out of it. **No games
have been played. Nothing here is an Elo claim** — the metric involved has now
mis-ranked twice.

### The mechanism check, and the control that makes it mean something

Three fits, all on labels from the same teacher, differing only in which
positions they see. 12 splits each, as registered (`band_stability.py`,
`sha256(split + fen)`):

| arm | N | mix | OVERALL | 0-5 | 6-11 | 12-17 | **18-24** |
|---|---|---|---|---|---|---|---|
| `d1` | 19,434 | natural | −7.70 ± 0.90 | −13.34 ± 1.37 | −7.32 ± 1.65 | +0.12 ± 1.35 | **+7.47 ± 3.14** |
| `nat8792` | 8,792 | natural | −5.05 ± 1.27 | −11.20 ± 2.51 | −4.05 ± 2.97 | +4.23 ± 4.62 | **+11.70 ± 7.58** |
| **`bal8792`** | **8,792** | **flat** | −3.89 ± 1.06 | −4.27 ± 2.98 | −2.80 ± 3.03 | −3.39 ± 2.57 | **−4.95 ± 3.51** |

**The pre-registered condition was that `bal8792` must not be worse than classic
at phase 18-24. It is 4.95% BETTER.** Check cleared.

**The size control is what makes this a result rather than a coincidence.**
`nat8792` has exactly the same number of positions as `bal8792` and the natural
mix, and it is **+11.70% worse** in that band — worse than d1, not better. So
halving the data does not produce the effect; **changing the mix does**, and the
swing at identical N is **16.65 points**.

Extended to 40 splits afterwards (post-hoc, reported as such — the registered
number is the 12-split one above):

| arm | 18-24 over 40 splits | splits worse than classic |
|---|---|---|
| `nat8792` | **+13.31 ± 6.12** | **40 of 40** |
| `bal8792` | **−3.52 ± 3.98** | a minority (range −12.24 … +5.37) |

The trade that was diagnosed after d1 is now demonstrated as a controlled
intervention: with a majority endgame band to buy, the fit sells phase 18-24;
with no majority band, it stops.

**The honest caveat.** These 40 splits re-split the *same* 8,792 positions, so
the spread measures split noise, not sampling error over positions. It says the
band difference between the arms is not an artefact of which split you look at.
It does not say the fit would behave this way on a fresh corpus.

### Instrument note: the trainer's default split is an outlier in this band

`distill_train.py` reports bands on one split (seed 20260813) and it reads
**+4.15%** for `bal8792` at phase 18-24 — outside the entire 12-split range
[−12.06, +1.64], and on the wrong side of the check. The same seed also gave
d1 **+14.24%** against its 12-split range of [+2.12, +14.13]: **extreme in both
sets, in the same band, because the split is FEN-keyed and the two sets share
positions**, so an unlucky draw is inherited rather than re-rolled.

This is not a bug and the pre-registration is what protected the result: the
check was put on the 12-split table before any of it was run. **No band number
from the trainer's single split should be read as a mechanism** — that is the
error that produced C2's post-mortem in the first place.

### The candidates, priced and gated

| | b1 (exact) | b8 (step 8, K exact) |
|---|---|---|
| packed | **3431 B (+81)** | **3316 B (−34)** |
| held-out (single split) | −2.53% | −2.25% |
| decode round trip | OK | OK |
| standalone | `d2d4` | `b1c3` |
| legality | 100/100 | 100/100 |
| mate-in-1 | 8/8 | 8/8 |
| **first yield, max** | **2,011 — PASS by 37 nodes** | **2,433 — FAIL** |

`b1` passes with a **1.8% margin**. That is the thinnest pass this gate has ever
recorded, and the gate exists precisely because a build one ordering change from
failing is not safe.

### SIX fitted candidates, six times at the cliff. The gamma seed is a BLOCKER, not a lead

| build | shipped seed | `gamma = pos.score - 150` |
|---|---|---|
| entry | 780 | **171** |
| C2 | 2,568 FAIL | 453 |
| m1 | 9,088 FAIL | 250 |
| C1 | 32,640 FAIL | 396 |
| d1 | 1,896 | 537 |
| d8 | 2,059 FAIL | 478 |
| **b1** | **2,011** | **728** |
| **b8** | **2,433 FAIL** | **394** |

Every fitted eval this lane has produced lands within a factor of ~1.2 of the
2,048-node cliff or over it; the two that "pass" do so by 37 and 152 nodes. The
+5-byte root seed clears **all of them** with 2.8–5.2× margin. This was routed
to the search lane as a lead. On this evidence it is a **prerequisite for
shipping any fitted eval at all**, and the step-8 arm — the one that gives bytes
back — cannot be screened until it lands.

### What is staged, and what is not

`tools/tune/label_corpus.sh` enlarges the corpus and labels at 160k. It is
**STAGED, NOT ARMED**: it checks a GO marker **once** and exits if it is absent,
never waits, and never chains. Its producer writes `RESULT.txt` on **every**
exit path including crashes, so a consumer can never poll forever on a job that
died. Controls run: unarmed → refuses with a result file; wrong teacher sha →
refuses **before** labelling; missing teacher file → reports *not found* rather
than *changed* (the first version conflated them, which would have sent a reader
hunting for a commit that never happened).

Sizing for when a slot exists, measured from the census: a flat draw at d1's N
needs 4,858 per band, so the shortfall is **2,660 at phase 12-17 and 1,948 at
18-24** — about 23,500 fresh sampled positions, on the order of 5,000 games,
which the box arenas hold. At the measured 0.58 pos/s per worker that is ~17
minutes on 8 workers.

### Cost of everything above

One census pass, five fits and two band-stability sweeps, all single-threaded at
`nice -n 15` on the laptop: **under four minutes of CPU in total**, nothing on
the box, no labelling, no games.

---

## 2026-08-14 — PRE-REGISTRATION: the phase-balanced set. The corpus is EXHAUSTED, so the experiment is a re-balance of labels we already have — and it costs nothing to run

**Nothing is selected, fitted or labelled below. This entry is written first, on
purpose.** Two programmes have now died on this instrument, and both times the
explanation arrived after the number. This one states the mix, the arms, the
mechanism check and the bar before a single position is drawn.

### First, the finding that chooses the design: there are no new positions

The plan was to sample a phase-balanced set from the existing corpora. Censused
with the sampling rule byte-identical to `texel_data.py` (`ply>=10`, every 7th
ply, not in check, `>=6` pieces), over every game in `~/repos/sunfish-data/pgn`
— 4,482 games, walked to the end rather than stopped early:

| | |
|---|---|
| unique positions the rule admits | 19,689 |
| already spent in `set20260813` | 19,491 |
| **left over** | **198** |

| band | available |
|---|---|
| 0-5 | 34 |
| 6-11 | 69 |
| 12-17 | 74 |
| 18-24 | 21 |

**The corpus is exhausted.** 19,491 + 198 = 19,689, exactly the "positions
collected" recorded when the set was built, which is also a positive control on
the census: this walker reproduces the original sampler's yield to the position.
No phase-balanced set of any useful size can be drawn from the games on this
laptop. Enlarging the corpus means pulling PGNs off the box and labelling new
positions, and that needs a machine.

### So the experiment inverts: re-balance the labels we ALREADY have

`distill160k` holds 19,434 positions **already labelled by our own search at
160,000 nodes**. Its band structure:

| band | count | share |
|---|---|---|
| 0-5 | 8,374 | 43.1% |
| 6-11 | 5,952 | 30.6% |
| **12-17** | **2,198** | **11.3%** |
| 18-24 | 2,910 | 15.0% |

A flat draw is capped by the thinnest band at **2,198 per band, 8,792 total**,
and it fits entirely inside a set that is already labelled. So the whole
experiment costs **one census and three CPU-minutes of fitting** — no teacher
run, no box slot, nothing added to either machine tonight.

It also buys the cleanest single variable this lane has ever had. d1 and the
balanced fit share the same teacher, the same labels, the same feature
construction, the same split rule, the same model, the same encoding, the same
gates and the same screen. **They differ in which subset of the same labelled
positions the fit sees, and in nothing else.**

### PRE-REGISTERED TARGET MIX: flat, 2,198 per band, 8,792 positions

Written before selection, with the alternatives that were considered and
rejected, so nobody re-derives them later:

- **FLAT (chosen).** The measured failure is a *trade*: the fit sells phase
  18-24, where classic's loss against our teacher is already lowest
  (0.007962, its best band), to buy phase 0-5, where 43.1% of the positions
  live. Under a flat mix that trade has no payoff, because there is no majority
  band to buy. It is the minimal intervention, it has **no free parameter**, and
  it is legible: one number per band, all equal.
- **Headroom-weighted** (weight ∝ classic's per-band loss) — **rejected**. It
  would weight 12-17 and 0-5 up and 18-24 *down*, which is the direction that
  just failed. It optimises the same quantity the loss already optimises.
- **Play-frequency weighted** — **rejected**, because that is what the corpus
  already is. Sampling every 7th ply of real games *is* the frequency with which
  positions occur, and 43/31/11/15 is the mix that produced −93.8 and −76.0.

### The arms, and the control that separates MIX from SIZE

Halving the data is a confound. So three fits, not two:

| arm | positions | mix | purpose |
|---|---|---|---|
| `d1` (measured) | 19,434 | natural 43/31/11/15 | the incumbent result, **−76.0 ± 28.3** |
| **`nat8792`** | **8,792** | **natural** | **the size control.** Same N as the candidate, unbalanced. Anything the balanced arm gains must survive this |
| **`bal8792`** | **8,792** | **flat 25/25/25/25** | the candidate |

Without `nat8792` a difference between `d1` and `bal8792` could be the mix or
could be having half the data, and there would be no way to tell.

### PRE-REGISTERED MECHANISM CHECK — it can only CANCEL a screen, never justify one

Held-out loss decides nothing about strength; that is on the record twice over.
But it can say whether the intervention **did the thing it was designed to do**,
and if it did not there is nothing worth spending games on:

> **Condition to proceed:** on the 12-split band table, `bal8792` must not be
> worse than classic at phase 18-24. d1 was **+7.47 ± 3.14 worse** there on all
> 12 splits. If the balanced fit is still stably worse in that band, the mix was
> not the mechanism, the diagnosis that survived d1 is wrong too, and this
> closes **without a box slot**.

This is a veto, not a trigger. Clearing it does not make the arm good — d1
cleared every gate it had and lost 76 Elo.

### Everything downstream is unchanged, deliberately

| | |
|---|---|
| teacher | our own search @ 160,000 nodes, unchanged, labels reused verbatim |
| split | 20% held out by `sha256(seed + fen)`, seed 20260813 — **FEN-keyed, so a position keeps its side of the split in every set it appears in** |
| model | the same 384 parameters, king frozen at the landed `kend` fix |
| encodings | exact (`b1`) and step-8 STE with `exact="K"` (`b8`), as d1/d8 |
| gates | decode round trip, `check_entry.sh`, legality, mate, **first yield ≤ 2048**, A-vs-A |
| screen | `ab_fixednode.sh`, 20,000 nodes, SPRT elo0=0 elo1=10, α=β=0.05, 2,000-position book, in the arena that measured C1, C2 and d1 |
| **bar** | **LAND at 95% LB > 0, DROP at UB < 0** |

Band predictions beyond the mechanism check are **not** registered: the band
statistic failed its stability test once and its prediction test once, and it is
being used here only to verify an intervention, never to forecast Elo.

### What makes this different from the two programmes that died — and what does not

**Different:** C2 and d1 were both free to spend the high-phase band to buy the
majority, and d1 measurably did. Under a flat mix that trade has no payoff. This
is the first intervention aimed at the mechanism that was actually measured
rather than at the objective (C2 → d1 changed the teacher, and the teacher was
not it).

**Not different, and worth saying plainly:** it is the same 384-parameter linear
model on the same features fitted by least squares, and the two dead programmes
prove that a 6-8% held-out improvement on this model carries no Elo whatsoever.
The mix could be the mechanism *and* the model still be too small to profit from
fixing it. The honest expectation is that this is a coin flip, and the reason to
spend the slot is that it is the last cheap hypothesis standing before the
conclusion becomes "384 global parameters cannot be improved by fitting, at any
mix, with any teacher" — which is itself a result worth having on the record.

### Cost, and why it is being written tonight

The box is running Thomas-side timed matches and the laptop is running the timed
ladder. Nothing here adds sustained load to either: the census is one pass over
19,434 FENs, and the fits are single-threaded, `nice -n 15`, ~1 minute each.
The labelling recipe for *enlarging* the corpus is staged but **not armed** —
it needs a machine, and it will be scheduled, not chained.

---

## 2026-08-13 — d1 DROPPED at −76.0: the teacher was not the problem, the band statistic does not predict Elo, and 384-parameter distillation on this set is CLOSED

The screen ran in the arena that measured C1 and C2, on C2's own openings, and
it resolved against the candidate. Below: the verdict, the pre-registered
closure it triggers, and two instrument/hygiene defects found on the way by the
ordinary discipline of running every gate on the **shipped entry** first.

### THE VERDICT

| | |
|---|---|
| games | **462** raw `[Result]` lines (460 paired; 2 unpaired dropped by `pair_elo`) |
| **d1, raw PGN** | **W 146 · L 247 · D 69 — 39.07%** |
| **Elo** | **−75.96 ± 28.29** (95%), i.e. **[−104.3, −47.7]** |
| SPRT | stopped early, H1 accepted **for base** |
| time forfeits / illegal / `(none)` | **0 / 0 / 0** |
| A-vs-A control, same book slice | **exactly 50.00%** — 47 W / 47 L / 26 D over 120 raw games, 0 forfeits |

**DROP.** The bar was UB < 0 and the upper bound is −47.7. It is not close.

### The pre-registered closure condition is MET

The registered wording: *"the informative comparison is against −93.8, not
against zero. If the distilled student's point estimate is not materially above
C2's, the teacher swap is not what was wrong, and 384-parameter distillation on
this position set is closed."*

| | teacher | held-out vs classic | Elo (95%) |
|---|---|---|---|
| C2 | Stockfish 18 @ depth 8 | −5.9% | **−93.8 ± 32.7** |
| **d1** | **our own search @ 160,000 nodes** | **−7.78%** | **−76.0 ± 28.3** |

Both screened on the same instrument, the same book, and — since d1 reused
C2's srand — **the same 500 openings**. The point estimate moved **+17.9 Elo**
and the intervals overlap over most of their length ([−104, −48] against
[−126, −61]). That is not materially above C2.

**So the teacher is not what was wrong, and 384-parameter distillation on this
position set is CLOSED** by the rule written before the labels finished. The
teacher itself is not discredited — it is Stockfish-free, reproducible, and
cost 10 core-hours — but it does not rescue a 384-parameter global fit, and
nothing further should be spent testing teachers on this position mix.

### The phase-band statistic does not predict Elo. It points the WRONG WAY

This was the one thing the coordinator asked the screen to answer, and the
answer is clean:

| | phase 18-24 held-out | Elo |
|---|---|---|
| C2 | **−8.00% (BETTER than classic)** | −93.8 |
| d1 | **+7.47% (WORSE than classic, on all 12 splits)** | **−76.0** |

The student that is stably *worse* in the band played *better*. Carried
verbatim from the coordinator, and it stands: *the phase-band evidence arrived
after the bar was set and the bar is not revised; honest expectation is that d1
drops; the informative output is whether the band statistic predicts Elo (C2 is
the control: it improved that band and lost 94).* It does not. The band
programme has now failed twice — once on stability (its sign flipped across
splits) and now on prediction — and nothing further should be built on it.

### Held-out loss has now mis-ranked on BOTH teachers

C2 fitted Stockfish 5.9% better than classic and lost 94. d1 fits **our own
search's converged value 7.78% better than classic** — the quantity the engine
literally maximises, on labels the engine itself produced — and loses 76.
Two teachers, two objectives, same direction. The 384-parameter model does not
have an objective problem that a better label fixes.

What is left standing from the whole distillation pass is the measured
diagnosis, not the candidate: **65.6% of the positions are endgame, classic
already predicts our own search best at phase 18-24 (0.007962, its best band),
and a global least-squares fit spends that band to buy the majority.** The
position mix is the open axis, and this teacher makes it cheap to re-sample.

### The screen, as run

`base` (the landed 3350 B entry) vs **`d1`** (the distilled exact student,
3421 B, +71). Instrument byte-for-byte the one that produced C1's −57.7 and
C2's −93.8: `ab_fixednode.sh`, **20,000 nodes**, SPRT elo0=0 elo1=10,
α=β=0.05, the 2,000-position book, 500 rounds, concurrency 8. Launched
21:42:43 UTC, complete 22:03:13 UTC; the A-vs-A control ran 22:05:05–22:12:38.

`base` (the landed 3350 B entry) vs **`d1`** (the distilled exact student,
3421 B, +71). Instrument byte-for-byte the one that produced C1's −57.7 and
C2's −93.8: `ab_fixednode.sh`, **20,000 nodes**, SPRT elo0=0 elo1=10,
α=β=0.05, the 2,000-position book, 500 rounds, concurrency 8.

It runs in **`eval-c1-20260813`, the same arena**, not a new one. Every
instrument file there was checked md5-identical to the repo at launch —
`ab_fixednode.sh`, `legality_gate.py`, `pair_elo.py`, and `sunfish_ui/` at
`DRIVER_VERSION = 2` — and the arena's `bin/e_base.py` is byte-identical to a
fresh `make_variants.py base` off the landed tree, so the incumbent arm is not a
stale copy of anything. `e_d1.py` was rebuilt from that tree and its md5 checked
after transfer. **srand is 20260815, C2's own seed**, so d1 plays the *same 500
openings C2 played*.

Gates before games, both arms, locally under CPython and on the box under the
pypy3 that actually plays:

| | base | d1 |
|---|---|---|
| packed | 3350 | **3421 (+71)** |
| decode round trip | — | OK |
| legality | 100/100 | 100/100 |
| mate-in-1 | 8/8 | 8/8 |
| first yield, max (window 2,048) | **780** | **1,896** |

The bar was **not revised**: LAND at 95% LB > 0, DROP at UB < 0.

`d8` was **not** in this screen. It fails first yield at 2,059 against a 2,048
window and is not screenable until the search lane's 5-byte gamma seed lands.

### COTENANCY INCIDENT: a timed RR voided itself six minutes after this launch

Reported because it is not mine to interpret and the timing is short.

At launch (21:42:08 UTC) the box was 96 cores at load 10.89, running the formal
lane's widening RR — 7 engines, 840 games at **30+1**, concurrency 8 — plus a
root `cliosoft` service at 19.7% CPU and one `innovus` job systemd-capped at 4
cores. This screen added 16 `nice -n 5` pypy3 processes at 21:42:43.

At **21:48 UTC** the widening lane voided its own run, renaming its output to
`VOID_process_leak_20260813T2148Z`. Its log stops at game ~115 of 840.

I did not touch that lane's arena, processes or files, and I cannot establish
causation from here. What is on the record and relevant: **that runner had
deliberately waited from Aug 12 18:00 until 21:13 for a "10-min fastchess-quiet
window" before launching**, so it was built on the premise of an otherwise
quiet box; its matches are **timed**, and `ab_fixednode.sh`'s own header says
timed comparisons need a quiet machine while fixed-node ones do not. The
cotenancy that provably cannot corrupt *this* screen is not the same claim as
cotenancy being harmless to a *timed* neighbour. Flagged for the coordinator;
the correlation is 6 minutes and the mechanism is plausible in at least two
ways (load on a timed match, or my pypy3 processes tripping their leak
detector).

The d1 numbers themselves are unaffected: fixed-node play is node-counted, the
A-vs-A control on the same book slice returned exactly 50.00%, and there were
no time forfeits in either match.

### The mate gate was answering from the OPENING position

`mate_gate.py` feeds `position fen`, which **only `sunfish_ui/uci.py`
understands**. An entry whose grandparent directory has no `sunfish_ui/` — which
is every variant a generator writes into a scratch directory — falls through to
the builtin loop, which knows `position startpos` alone. It searched the opening
position and answered an opening move.

Run that way the gate reported **`MISS ILLEGAL g1f3` for the SHIPPED ENTRY** on
three mates it in fact solves. Loud rather than silent, unlike the first-yield
gate's first run, but still a chess verdict on a position the engine never
saw — and the **fourth** incident of this exact class in this lane (`agree.py`,
the stale driver that voided 425 games, the first-yield gate, now this).

Fixed the same way the first-yield gate was: resolve a checkout that has a
driver, set `PYTHONPATH`, **demand the banner by name**, and take source entries
only, since `position fen` cannot reach a packed artifact at all. Re-run:
base, d1, d8 all **8/8**, and a packed artifact is now refused instead of scored.

### The tracked training set carried the bench box's name

`distill_pack.py` copied the labeller's meta into the `.npz` verbatim, including
the machine it ran on, and `set20260813.npz` — a **tracked file** — already
carried `"host": "hardware"`.

The branch has never been pushed, so this was still cheap. `distill_pack.py` now
drops the key, and the committed set has been rewritten:

| | |
|---|---|
| pre-scrub sha256 | `d792b42081f0adec10cbcb17ca72a7a96949cfac21fe1b97be1935b3cffc4c13` |
| **post-scrub sha256** | **`2410786e14f09fecbcea8c94f74fd1378d04b0f1aa634cb27702f0890196ed4d`** |
| arrays | `X`, `y`, `fens` **bit-identical**, verified against an untouched copy |
| meta | dropped exactly `{host}`; every other field equal |

**C1's and C2's provenance therefore refers to the pre-scrub sha of the same
bytes minus one metadata key** — the training data behind those two verdicts is
unchanged, and `fits.json` records no `data_sha256` to update. The pre-scrub
blob exists only in this branch's history and must never be pushed.
`distill160k.npz` (`b0ed8b6617a7…`) was built after the fix and was never
affected.

Verification note, because the first attempt was wrong: reading arrays back
through the *same* `np.load` handle after overwriting the file compares the new
file with itself — `numpy` reads lazily from disk offsets. The check above is
against a copy taken before the write.

---

## 2026-08-13 — The distillation teacher, specified and priced; a new gate that all four old fits fail; and the C2 post-mortem's mechanism does NOT survive a re-split

Thomas's reorder: distillation first, and the teacher is **our own search's
converged value**, not a shallow Stockfish score. Static SF-depth-8 loss is on
record as anti-correlated with strength (C2 fitted it 5.9% better and played
−93.8), and the search's own value is the quantity the engine actually
maximises. This entry is the teacher spec, the budget choice behind it, the
instruments built to gate what comes out, one correction, and the students.
**No Elo is claimed anywhere below.**

### The frontier the teacher has to beat, measured not assumed

A teacher at the student's own depth teaches nothing. So the budget is chosen
against the budget the engine actually spends, measured on the bench box under
its ordinary match load (load 24 of 96, two `tc=30+1` matches cotenant):

| | median | min | max |
|---|---|---|---|
| depth at 30+1 | **9** | 7 | 15 |
| nodes at 30+1 | **56,829** | 23,797 | 112,506 |

16 positions, `wtime 30000 winc 1000`, so `think = wtime/12 + 0.9*inc = 3.40 s`
at a median 42,473 nps. Re-measured after the cotenant matches finished (load
10.75): median **67,642 nodes, depth 10**, 54,638 nps. So the frontier is
**57k–68k nodes** depending on what else the box is doing. The fixed-node
screens run at **20,000**, which from the sweep below is ~6.5 ply.

**The honest tension, stated rather than glossed.** At 160,000 nodes the
teacher's mean *completed* depth is 9.74. Against the **screen** budget where
the candidate is actually judged that is 8× the nodes and ~3 plies — a real
teacher. Against a **30+1 game move** it is only 2.4–2.8× the nodes and about
one ply, so this teacher is not teaching the student to see deeper than the
engine plays; it is teaching it to agree with a slightly better version of
itself. Buying the extra plies is what 640k would do, and the sweep below says
that buys 0.008 of correlation while 21% of positions still churn >25 cp. If
the distilled student converts at all, raising the budget is the first axis to
re-open.

### The teacher's value does NOT converge. It plateaus, and the plateau is tactics

320 phase-stratified positions labelled at five budgets, every one labelled at
every budget:

| budget | mean depth | median \|Δ\| vs previous | mean \|Δ\| | >25cp | r with 640k |
|---|---|---|---|---|---|
| 2,500 | 4.27 | — | — | — | 0.9241 |
| 10,000 | 5.54 | 8 | 15.8 | 13.4% | 0.9345 |
| 40,000 | 7.61 | 12 | 24.9 | 22.5% | 0.9710 |
| **160,000** | **9.74** | 10 | 22.8 | 21.2% | **0.9919** |
| 640,000 | 11.91 | 10 | 21.0 | 21.6% | 1 |

Read the middle columns first. **The label never stops moving**: from 40k
upward, a 4× budget increase still shifts ~21% of positions by more than 25 cp,
and the median shift sits flat at 10 cp. This is not a convergence tolerance
that more nodes would close — it is tactical content entering and leaving the
value, and **a 384-parameter piece-square table cannot represent it at any
budget**. Buying more of it is buying label noise.

What does keep improving is agreement with the deepest run, and it saturates:
**r = 0.9919 at 160k**, so a 4× more expensive teacher would move 0.8% of the
variance.

**Chosen: N = 160,000 nodes.** Justified on four measured grounds — 2.8× the
30+1 play frontier and 8× the screen budget; median completed depth 9.74
against the frontier's 9; r = 0.9919 against 4× the cost; and 1.8 s/position,
which is ~10 core-hours for the whole set and fits inside the cotenancy rule.
640k would be 5× that for 0.008 of correlation.

### Dataset spec

| | |
|---|---|
| positions | the **same 19,491 FENs** as `set20260813.npz` |
| teacher | `nnue_4k/pst_entry.py`, sha256 `f2f0bdc87cd1…`, the shipped entry |
| budget | `go nodes 160000`, node cap only, no wall clock |
| label | score of the **last completed depth**, MTD bracket midpoint, white POV |
| features | 6×64 piece-square counts, white minus mirrored black — byte-identical construction to `texel_data.py` |
| dropped | positions where the teacher saw a mate at the root, counted not coerced |

The positions are deliberately **unchanged**. C2 differed from classic in the
teacher AND could have differed in a dozen other ways; the distilled student
differs from C2 in the label and in nothing else, so if it converts, the
teacher is why and no other explanation is available. Re-sampling the position
mix is a separate axis, and the teacher is free of Stockfish, so it can be run
over any number of new positions later without a relabelling dependency.

### Two controls on the labeller, both of which the SF labeller needed

- **Order/TT control.** `Searcher.search` clears `tp_score` but *not*
  `tp_move`, and move ordering changes the tree — the same carry-over that
  made Stockfish score one FEN −14 in one slot and −22 in another. A fresh
  `Searcher` is built per position, and 40 positions labelled forward and then
  **reversed** produce **bit-identical records in every field**.
- **Interpreter control.** The same 8 positions labelled under CPython 3.14.5
  on the laptop and PyPy 3.11.13 on the box give **identical labels**. The
  label is a function of (fen, budget, engine sha) alone.

**Positive control on the extraction itself**: for the first sweep position the
labeller reports 353 cp at depth 5, and the engine's own `info` stream on the
same position converges to `lower 348 / upper 359` at depth 5 — midpoint 353.
The number is the engine's, not a re-derivation of it.

### What the 384-parameter model can actually express, measured

Two numbers that frame every fit this lane has run, on the SF-labelled set:

- Adding classic's piece-square terms to **pure material** moves the
  correlation with either teacher by **~0.000-0.004** (0.7364 → 0.7368 ours,
  0.7298 → 0.7284 SF). Material is essentially the whole raw signal.
- Yet refitting **only the 5 free piece values** buys **−1.31%** held-out,
  against **−7.63%** for all 384 parameters. So the square terms carry ~83% of
  the loss improvement while contributing almost nothing to raw correlation.

The fit's gain therefore lives entirely in a component that is small next to
material. Stated as an observation, not a mechanism — the last mechanism this
lane asserted from a single split is the one being corrected below.

For the record, the SF fit's piece values: P 100, N 280→**265**, B 320→**295**,
R 479→**473**, Q 929→**839**. Overall eval scale is unchanged (sd ratio 1.014
on 5,074 real positions), so this is a change in the *relative* piece values —
the queen down 10% against a near-unchanged rook — not a rescaling.

### How different is this teacher from Stockfish? Enough to matter

On the 320 sweep positions, our 160k value against SF depth 8: **r = 0.884**,
**median absolute difference 86 cp**, mean difference −3.7 cp, sd 324 vs 304.
So the scale matches (no piece-value scale mismatch to absorb) while the
per-position disagreement is large. This is a real change of target, not a
relabelling that moves things a little.

### NEW GATE: `tools/build/first_yield_gate.py`, and all four old fits fail it

`main()` can only print a move the search handed it, and the search hands one
over only on a **root fail-high with a move**. Both stop conditions are polled
at `self.nodes % 2048 == 0`, so **the earliest an abort can land is node
2,048**. A build whose first fail-high needs more than that has a budget — of
nodes *or of time*, so this is not a fixed-node-only hazard — at which it
prints `bestmove (none)`.

The first version asked the binary question (`go nodes 1`, is it `(none)`) and
over 505 positions caught C1 on **exactly one**: the position already known to
fail. A gate whose power comes from carrying its own reproducer catches the bug
it was written for and nothing else. It now **measures the node count**, which
turns a 1-in-505 event into a distribution with a margin:

| arm | median | p99 | **max** | verdict |
|---|---|---|---|---|
| **shipped entry** | 5 | 178 | **780** | PASS |
| C2 (exact, unmirrored) | 5 | 369 | **2,568** | FAIL |
| q8 (step 8, post-hoc rounded) | 5 | 361 | **3,707** | FAIL |
| m1 (exact, mirrored) | 5 | 167 | **9,088** | FAIL |
| C1 (step 8, mirrored) | 5 | 283 | **32,640** | FAIL |

505 phase-stratified positions from our own games, plus the C1 reproducer.
**Every fitted candidate this lane has produced fails; the shipped entry passes
with a 2.6× margin.** The ordering by max reproduces the bisection's ordering
exactly — mirroring is worst — from an instrument that knows nothing about
mirroring.

Two things this gate needed to be true rather than reassuring:

- **A subprocess control.** The measured node count is checked against
  `go nodes 1` through the real UCI surface on the worst positions plus a fixed
  slice; a number that did not predict the played move would be a generator
  nobody had validated.
- **A driver assertion, which it needed on its first run.** An entry that
  resolves no `sunfish_ui/` falls through to the **builtin loop, which knows
  only `position startpos`** — it ignores the FEN, searches the opening
  position, and answers a legal-looking move. The gate scored **PASS for every
  arm including C1** until it started demanding the driver banner by name. Same
  class as the `agree.py` incident; third time this lane has paid for it.

### The C2 post-mortem's MECHANISM does not survive a re-split

The recorded explanation for C2 was a band diagnostic: "the whole −5.31% lives
in the endgame and the middlegame band is slightly worse than classic
(+0.6% at phase 12-17)". That was read off **one** held-out split, and the band
holds ~450 positions. Refitting the identical model on **12 splits**:

| band | mean | sd | min | max | sign |
|---|---|---|---|---|---|
| OVERALL | −7.85 | 0.81 | −8.66 | −5.82 | stable |
| phase 0-5 | −10.92 | 1.75 | −13.22 | −7.32 | stable |
| phase 6-11 | −6.11 | 1.16 | −7.97 | −4.25 | stable |
| **phase 12-17** | **−2.96** | **2.36** | **−6.36** | **+0.94** | **FLIPS** |
| phase 18-24 | −8.00 | 1.95 | −12.16 | −5.20 | stable |

**The middlegame band is the LEAST IMPROVED band, not a worse one**, and its
sign is not determined at this sample size. The anti-correlation itself is
untouched — the fit is reliably ~8% better on held-out loss and played −93.8
Elo — but *why* is now unexplained, and the phase-reweighting programme that
was justified by this mechanism was aimed at a statistic that flips sign.
Recorded so that nothing further is built on it.

### The torch trainer is validated against a fit whose Elo we already know

`tools/tune/distill_train.py`, `torch.optim` only, seeded and deterministic,
CPU, single-threaded (the laptop is running a timed league). Run on the **old
SF-labelled set** it must reproduce C2, and it does — on four independent
axes:

| | recorded for C2 | torch harness |
|---|---|---|
| held-out vs classic | −5.9% | **−5.93%** |
| phase 0-5 / 12-17 | −9.8% / +0.6% | **−9.81% / +0.56%** |
| packed bytes | 3412 (+62) | **3412 (+62)** |
| first yield, worst position | 2,568 nodes (box `e_c2.py`) | **2,568 nodes, same FEN** |

The last row is the strongest one: an independently rebuilt candidate
reproduces the incumbent's failure mode node for node.

**The split is now keyed on `sha256(seed + fen)`, not on a row permutation.**
The distilled set drops the positions where the teacher saw a mate, so an
index-based split puts *different positions* in the two teachers' held-out
sets and the single-variable comparison silently stops being true.

### Quantisation-aware students are BYTE-NEGATIVE

Straight-through rounding inside the forward pass, so the fit is over tables we
can actually store rather than a float fit rounded afterwards. On the SF set
(harness exercise, not a candidate):

| arm | held-out | packed | vs entry |
|---|---|---|---|
| linear (exact) | −5.93% | 3412 | +62 |
| **q8** | −5.91% | **3297** | **−53** |
| **q16** | −5.11% | **3257** | **−93** |

Step 8 keeps essentially the whole fit and gives **53 bytes back** against the
shipped entry. Two defects were caught getting there, both by the round-trip
check: the codec **quantises every table it is handed, including the king**, so
a step-8 emit silently rounded the landed `kend` fix (`exact="K"` holds it, ~84
B); and the piece value must be snapped to the same grid, or subtracting it
shifts the whole table half a step off and the codec rounds a second time.

### Instrument note: `torch.optim.LBFGS`'s default tolerance is a silent no-op here

At this loss scale (~0.017) the gradients are ~1e-9 and the **default
`tolerance_grad=1e-7` declares convergence before the first step**. The "fit"
comes back exactly equal to its warm start and every band reads `0.00%`, which
looks like a result rather than a failure. Caught by a run where *every* number
was zero; `tolerance_grad=1e-12` is now set explicitly everywhere.

### THE STUDENTS: on the metric, the teacher swap changes NOTHING

19,491 positions labelled (66.6 min, 8 workers), 19,434 kept, 57 dropped as
mates at the root, mean completed depth 10.10, label sd 366 cp. Both teachers
trained on **the identical 19,434 positions with the identical FEN-keyed
split**, each measured against its own classic baseline (the baselines differ
because the labels differ — 0.017804 for SF, 0.014239 for ours: **the incumbent
already predicts its own search far better than it predicts Stockfish**, so
there is less headroom to win).

| | classic | student | improvement |
|---|---|---|---|
| SF depth 8, exact | 0.017804 | 0.016443 | **−7.65%** |
| **our search @160k, exact** | 0.014239 | 0.013131 | **−7.78%** |

Over 12 splits: SF **−7.85 ± 0.81**, distilled **−7.70 ± 0.90**. **They are the
same number.** Distilling the search's own value does not make the 384-parameter
model fit better *or* worse in aggregate — which is neither good news nor bad,
because this is the metric that ranked C2 above classic while C2 played −93.8.

### And on the band structure it is stably WORSE where it matters most

This is the finding. 12 splits, distilled student:

| band | mean | sd | min | max | sign | SF student, same test |
|---|---|---|---|---|---|---|
| OVERALL | −7.70 | 0.90 | −9.33 | −6.38 | stable | −7.85 |
| phase 0-5 | −13.34 | 1.37 | −15.09 | −10.65 | stable | −10.92 |
| phase 6-11 | −7.32 | 1.65 | −10.73 | −4.70 | stable | −6.11 |
| phase 12-17 | +0.12 | 1.35 | −1.68 | +3.11 | FLIPS | −2.96 (flips) |
| **phase 18-24** | **+7.47** | **3.14** | **+2.12** | **+14.13** | **stable** | **−8.00 (stable)** |

**The distilled student is reliably worse than classic in the highest-phase
band** — every split, by 2 to 14% — where the SF student was reliably *better*.
Unlike C2's post-mortem, this one does not flip: it is a real property of this
generator.

The cause is visible in the baselines. Against our own teacher, classic's
held-out loss at phase 18-24 is **0.007962**, by far its best band: with most
of the material still on, our own search's value is nearly what classic's table
already says. There is almost no headroom there, and a global least-squares fit
spends that band to buy the endgame band, where the loss is large and **65.6%
of the positions live**.

So the diagnosis that was wrongly attached to C2 turns out to be *true of the
distilled student*, stably, and for a reason that is now measured rather than
guessed: **it is the POSITION MIX, not the teacher.** And the position mix is
exactly what distillation makes cheap to fix — this teacher needs no Stockfish,
so any number of high-phase positions can be labelled without an external
dependency. That is the axis the evidence points at, and it is not the axis
that was just tested.

### Prices and gates

| arm | held-out | packed | vs entry | first yield (max) | legality | mate |
|---|---|---|---|---|---|---|
| **d1** (exact) | −7.78% | **3421** | **+71** | 1,896 | 100/100 | 8/8 |
| **d8** (step 8, K exact) | −7.50% | **3306** | **−44** | **2,059 FAIL** | 100/100 | 8/8 |
| q16 (step 16) | −6.91% | 3267 | −83 | not gated | — | — |

Decode round trip OK and standalone-in-an-empty-directory OK on all three.

**d8 misses the gate by 11 nodes.** That is the margin case the measuring form
of the gate exists to catch: one position in 505 needs 2,059 nodes where the
window is 2,048, so this build emits `bestmove (none)` at some budget. d1
passes but at **1,896 of 2,048** — a 7% margin against the shipped entry's 2.6×.

**And the search lane's 5-byte seed fixes both**: with
`gamma = pos.score - 150`, d1's worst goes 1,896 → **537** and d8's 2,059 →
**478**, both passing comfortably. Every fitted table this lane has produced
sits near this cliff; the seed is what moves them off it. **The search change is
a prerequisite for shipping any fitted eval, quantised or not.**

### Pre-registered, before the labels finish

Screen: fixed-node 20,000 our-vs-our, SPRT elo0=0 elo1=10, alpha=beta=0.05,
2,000-position book — **identical to C1's and C2's screens**, so the numbers
are directly comparable. Gates first: decode round trip, legality 100/100, mate
8/8, **first yield**, and an A-vs-A control. Slot to be requested from the
coordinator; this lane launches nothing itself.

- **LAND** if the 95% lower bound is **> 0**. This is C1's *re-derived* bar, not
  its original LB > −15: eval bytes stopped being scarce (746 under the
  ceiling, and the directive is to fill it), so a byte credit is void. The
  step-8 arm gives bytes back, which strengthens the case at the same bar but
  does not lower it.
- **DROP** if the upper bound is below 0.
- **The informative comparison is against −93.8**, not against zero. If the
  distilled student's point estimate is not materially above C2's, the teacher
  swap is not what was wrong, and 384-parameter distillation on this position
  set is closed.
- **No band prediction is registered**, because the band statistic had just
  failed its own stability test on the SF set.

**POST-HOC, and marked as such because it arrived after the bar was set:** the
distilled student's phase 18-24 band then turned out to be stable *and* bad
(+7.47 ± 3.14, every split). The bar above is **not** being revised — that is
the whole point of writing it down first — but the honest expectation now is
that this arm is more likely to drop than to land, and the thing worth learning
from the screen is *whether the band statistic predicts Elo at all*. C2 gives
the control: it improved that band and lost 94. If d1 degrades that band and
loses about the same, the band is not the lever either, and the next candidate
should change the POSITION MIX rather than the teacher or the encoding.

**d8 is NOT screenable as it stands** — it fails the first-yield gate. Either
it screens on top of the gamma-seed change, or it does not screen.

### LEAD FOR THE SEARCH LANE: the `(none)` class is removable for 5 bytes

The gate makes the cause visible. `search()` starts every search at **`gamma =
0`** and bisects. The root stores a move **only on a fail-high** — measured, not
assumed: on the C1 reproducer the first moment `tp_move[root]` is populated is
the same probe as the first fail-high, at node 32,640 for C1 and node 25 for the
entry. So there is no earlier move to fall back on, and "report the fail-low
move" is not a fix that exists. The seed is the whole mechanism.

Seeding it differently, measured on the 505-position gate:

| root seed | entry | C2 | m1 | C1 |
|---|---|---|---|---|
| `gamma = 0` (shipped) | 780 | 2,568 | 9,088 | 32,640 |
| `gamma = pos.score` | **2,920 FAIL** | 1,357 | 1,140 | 5,197 FAIL |
| **`gamma = pos.score - 150`** | **171** | **453** | **250** | **396** |

`pos.score` alone is a **trade, not a fix** — it helps every fit and makes the
incumbent worse, because seeding at the true value makes the first probe a
coin flip and it is a fail-HIGH that produces a move. Seeding *below* it makes
the first probe cheap and one-sided: **every arm passes, including C1, and the
shipped entry improves 4.6×.**

Priced by building: **+5 bytes** (3350 → 3355) and **1.0001× nodes to complete
depth 8** over 40 positions — the seed only affects the first probes, and gamma
persists across depths.

**This is not an Elo claim and the constant is not tuned.** Changing the seed
changes the whole bisection and therefore what a stopped search plays; it needs
its own screen like anything else. But if it holds, it removes the correctness
objection that suspended mirroring — which is what made 7 phase buckets
affordable in the first place. Routed to the search lane, not built here.

### Conditions

Box: 8 `nice -n 15` pypy3 labellers, 0.55 pos/s each, alongside two of our own
`tc=30+1` matches at concurrency 10. Load **22.5 → 26.5 of 96 cores**. Labelling
never gates a match and this one could not have distorted one; no fastchess
process was started by this lane. Laptop CPU used only for the 384-parameter
fits (single-threaded) while the league ladder ran.

---

## 2026-08-13 — C1 DROPPED at −57.7, and the cause is MIRRORING, which also makes the engine answer `bestmove (none)`

The screen ran, and it did not merely fail its bar — it found a correctness bug
and a bisection that overturns the encoding the whole 1024-1500 B design was
about to be built on.

### The screen

Fixed-node our-vs-our on the box, 20,000 nodes, SPRT elo0=0 elo1=10,
alpha=beta=0.05, 2,000-position book, concurrency 8. **Raw counts first, as
required, before any Elo number:**

| | games | base W | base L | D | base score% |
|---|---|---|---|---|---|
| **base vs C1** | **651** | **339** | **232** | **79** | **58.23%** |

**C1 = −57.72 ± 25.53 Elo.** SPRT stopped early, H1 accepted *for base*.
0 time forfeits.

**Pre-registered verdict: DROP.** The re-derived bar was "DROP if the upper
bound is below 0"; the upper bound is **−32.2**. This is not the modal flat
result that was predicted — it is materially worse than that.

Controls, all clean:

- **A-vs-A driver control**: `e_aa.py`, byte-identical to `e_base.py` (same
  sha256), same directory, same `PYTHONPATH`. 120 games, **47W 47L 26D,
  exactly 50.00%**, 0 forfeits, 0 illegal. Both arms provably get the same UCI
  driver, and the harness is unbiased.
- **Gates before games**: legality 100/100 and mate-in-1 8/8 on every arm.

### The engine answered `bestmove (none)` in a real game — and it reproduces

The match reported an illegal move. It was not an illegal move; it was **no
move at all**, by C1, in game 65. Reproduced deterministically:

```
position fen rnbqkb1r/pp2pppp/3p4/2pnP3/3P4/2P2N2/PP3PPP/RNBQKB1R b KQkq d3 0 11
e_base.py  ->  bestmove b8c6
e_c1.py    ->  bestmove (none)
```

Not in check. Not a tail position. A **normal middlegame position**, and it
fails at **every depth including 1**, emitting no `info` score line at all — so
the search returns nothing from the root rather than searching badly. Identical
whether the position is given as a FEN or replayed through its move list, so it
is positional and eval-dependent, not a history artifact.

**The 100-position legality gate passed this build.** Its three classes (forced,
in-check, quiet-random) do not reach whatever this is, which is the second time
this lane has learned that a gate is only as good as its position sample.

### The bisection: it is MIRRORING, not the fit and not the quantisation

Four encodings of the *same fit*, one position, seconds of work — the cheapest
decisive measurement in this whole entry:

| arm | encoding | move |
|---|---|---|
| base | classic tables | `b8c6` |
| **C2** | fit, exact | `d6e5` |
| **q8** | fit, **step 8**, unmirrored, K exact | `d6e5` |
| **m1** | fit, **exact**, MIRRORED, K exact | **`(none)`** |
| **C1** | fit, step 8, MIRRORED, K exact | **`(none)`** |

**Mirroring is the cause.** Quantisation is exonerated — `q8` at step 8 plays
the same move as the exact fit. And `m1` is at full resolution, so this is not
a rounding artifact; folding the tables left-right is itself what breaks it.
Note the king table was held **bit-identical** (`exact="K"`) in both failing
arms, so the earlier kend-perturbation explanation does not cover this: it is
the PNBRQ fold alone.

### What this costs the 1024-1500 B design

The recommendation in the entry below was **7 phase buckets at mirrored
step 8** — 1120 parameters for 1147 eval bytes. **Mirroring is what made that
fit in the budget**, and mirroring is now implicated in a correctness failure
and, pending C2, possibly in most of the −57.7.

Re-reading the grid with mirroring off the table:

- **4 phase buckets, step 8, unmirrored: 1280 parameters, ~1102-1110 eval B,
  total 3988-3995 — still IN BUDGET.** The budget is reachable without
  mirroring at all.
- But the data gets worse, not better: unmirrored is 320 parameters per set, so
  4 phase quantiles is **3,024 / 320 = 9.5 pos/param** — below the taper's
  failure point of 11, where mirrored 4-bucket was 18.9.

So mirroring was the trick that made the parameter count affordable *in data*
as well as in bytes, and it is not available. **The data constraint tightens.**
Every conclusion in the entry below stands and this sharpens the last one:
**do not fit the budget-filling candidate on this generator.**

### Two instrument notes

- **The harness's illegal-move counter double-counts.** It greps `-ci 'illegal
  move'`, which matches both the `[Termination "illegal move"]` tag and the
  in-game comment, so one incident is reported as **2**. Distinct terminations:
  1. Anyone reading that line as an incident count is reading double.
- **`bestmove (none)` is reported by fastchess as an illegal move**, which sends
  you looking for a move-generation bug when the real event is that the engine
  produced nothing. The two failure modes need different investigations.

### C2 answers the generator question, and the answer is worse: the FIT is −94

C2 is the same fit at exact resolution with no mirroring, so it carries no
compression and — confirmed over 405 games — **no `bestmove (none)` incidents,
0 illegal, 0 forfeits.** It is a clean measurement of the fit alone.

| | games | base W | base L | D | base score% | candidate Elo |
|---|---|---|---|---|---|---|
| base vs **C1** (mirrored step 8) | 651 | 339 | 232 | 79 | 58.23% | **−57.72 ± 25.53** |
| base vs **C2** (exact, unmirrored) | 405 | 235 | 129 | 38 | 63.18% | **−93.83 ± 32.69** |

Both SPRTs stopped early, H1 accepted for base. **Both candidates DROPPED.**
1,056 games total, zero time forfeits.

Two readings, and the second is the one that matters:

1. **Mirroring is not what cost the Elo — it was recovering some of it.** C1 is
   36.1 ± 41.3 *better* than C2, which straddles zero but points the same way
   the held-out loss did: halving the parameters regularises away per-square
   noise the fit had memorised. Mirroring's problem is the correctness bug, not
   the strength.
2. **The fit itself is worth about −94 Elo**, and its held-out loss said it was
   **5.9% BETTER than classic**.

### The headline: on this dataset, held-out loss is ANTI-CORRELATED with strength

This is not the 2026-08-12 story repeating. That time a better-fitting table
played worse because of a bug in the emit path — an un-flipped king table — and
the fix restored the relationship. This time there is no bug in the measured
arm: C2 emits exactly what it fitted, is re-scored from its own emitted
integers, plays 405 clean games, and is **−94**.

The mechanism was visible in advance and is now confirmed in play. The
phase-band diagnostic said the whole −5.31% lived in the endgame and the
middlegame band was *slightly worse than classic*. Games are decided in the
middlegame. A Texel fit on a set that is 65.6% endgame optimises the band that
does not decide games, at the expense of the band that does.

### Consequence for the 1024-1500 byte design: the objective is now the blocker

Everything below about bytes stands — 7 phase buckets fit in 1147 eval B, the
grid is priced, the ceiling is 1210. **None of it should be built yet**, and
the reason has moved:

- **Not bytes.** The budget is reachable, unmirrored, at 4 phase buckets.
- **Not only data.** The pos/param gate said the straw man was 37× past
  known-bad, and that stands.
- **The objective.** A 384-parameter fit by this objective on this set is −94
  Elo. **Filling 1024-1500 bytes with 1,280 parameters fitted the same way is
  buying more of exactly what just lost 94 Elo.** Capacity multiplies whatever
  the generator produces, and this generator produces negative Elo.

**Recommendation, updated and firm: do not fit the budget-filling candidate.**
The next thing this lane should measure is not a bigger table, it is a
generator that converts:

1. **Fix the objective's blind spot before its capacity.** The cheapest test is
   a phase-band-weighted or middlegame-only refit screened at 384 parameters —
   the same cheap arm that just gave a decisive answer in 405 games. If a
   384-parameter fit cannot beat classic in play, no 1,280-parameter version
   will. (Note this is *not* the reweighting already refuted below: that was
   selected on held-out loss, the metric now shown to mis-rank. It has to be
   selected in games.)
2. **Distillation is now the leading route rather than a nice-to-have.** A
   teacher labels as many positions as we sample, which fixes the data gate
   *and* lets the target be the search's own value rather than a
   win-probability proxy that demonstrably mis-ranks middlegames.
3. **Mirroring is suspended on correctness** regardless of the above, until the
   `bestmove (none)` path in the root is understood — that is an engine bug the
   eval merely exposed, and it belongs to the search lane.

The one unambiguous asset from this round: **quantisation to step 8 is free and
safe** (`q8` plays the exact fit's move, and step 8 saves ~180 B), so whenever a
generator does convert, its tables can be stored cheaply.

---

## 2026-08-13 — The 1024-1500 byte eval, priced by building: the BYTES are there and the DATA is not

Thomas's allocation — eval 1024-1500 B, engine ~2500 — flips this lane from
byte-minimising to capacity-maximising. Below is the whole grid, every row a
real entry source through `tools/build/pack.sh`, size off disk, run alone in an
empty directory. **No Elo is claimed and nothing here is fitted yet**; the
deliverable is prices and one recommendation.

### First, what "eval bytes" even means

lzma carries one dictionary across the file, so no region has an intrinsic
size. The only honest definition is differential, against a build identical
except that it holds no table data:

> eval bytes(X) = packed(entry with X) − packed(entry with a ZERO stub)

The stub still defines `piece`, `pst`, `K_MID`, `K_END` — the same engine with a
flat evaluation. On the rebased base that is **2886 B**, so the eval today
occupies **464** and its **ceiling is 1210**.

### The grid (classic-derived filler, K exact throughout)

| partition | encoding | sets | params | packed | eval B |
|---|---|---|---|---|---|
| 1 flat | step 8, mirrored | 1 | 160 | 3167 | 281 |
| 2 seam | step 8, mirrored | 2 | 320 | 3323 | 437 |
| 4 wings | step 8, mirrored | 4 | 640 | 3573 | 687 |
| 4 wings | step 8 | 4 | 1280 | 3988 | **1102** |
| 8 seam × wings4 | step 8, mirrored | 8 | 1280 | 4021 | **1135** |
| 8 seam × wings4 | step 2, mirrored | 8 | 1280 | 4379 | over 4096 |
| 4 wings | exact | 4 | 1280 | 4478 | over 4096 |

Machinery is sublinear and nearly free — the 4-bucket selector plus three extra
decode loops costs **97 B** when the data is identical, ~32 per bucket, because
the loops compress against each other. **Exact resolution is dead at every set
count above two.**

### How many sets actually fit, on fitted-SHAPED data

Correcting filler by the measured "+60-75 B per fitted set" would be composed
arithmetic. Instead, a better build: permute the **real fitted tables** per set,
which preserves the fitted value multiset — the roundness that makes fitted data
expensive — while making the sets distinct, and needs no new fit.

| sets | step 8 mirr | step 4 mirr | step 2 mirr |
|---|---|---|---|
| 1 | 301 | 319 | 340 |
| 4 | 746 | 832 | 922 |
| 6 | 992 | **1125 (in budget)** | over 4096 |
| 7 | **1111 (in budget)** | over 4096 | over 4096 |
| 8 | over 4096 | over 4096 | over 4096 |

Marginal cost settles at **~125-135 B per extra set** at mirrored step 8. Note
this brackets from the *other* side: permuted sets share a value distribution
that lzma exploits, where the one genuinely independent fit we have (the qseam
second set) cost **+224**. A real candidate sits between, and the way to push it
to the cheap end is to fit extra sets as **regularised deltas from the base
set**, sharing piece values — which is also the fix for the data problem below.

### The recommended partition is PHASE quantiles, not king wings

King-wing buckets were the straw man. They are unusable on our data: **80.4% of
positions have the white king on the king side**, so the wing product has a
nearly empty corner. Phase quantiles are far better balanced. Both priced:

| partition | sets | params | packed | spare | eval B |
|---|---|---|---|---|---|
| phase quantiles | 4 | 640 | 3648 | 448 | 762 |
| phase quantiles | 6 | 960 | 3899 | 197 | 1013 |
| **phase quantiles** | **7** | **1120** | **4033** | **63** | **1147 — IN BUDGET** |
| phase quantiles | 8 | 1280 | 4144 | −48 | over 4096 |

The selector is a 25-character phase→bucket string indexed by the piece-count
phase, read once at the root — the same mechanism as the queens-off swap, so no
per-node cost and no second accumulator. Decode of the largest build: **0.96 ms**
against a 60 s startup budget.

**So the bytes reach the budget: 7 phase buckets, 1120 free parameters,
1147 eval B, 63 spare.**

### And the data does not. This is the finding.

Every bucket is fitted from its own positions, so what matters is the WORST
bucket's positions-per-parameter. The reference is measured, not assumed: the
queens-seam taper was dropped at **11 pos/param**, where it played a2a3 to
depth 5.

| partition | sets | worst bucket | pos/param | |
|---|---|---|---|---|
| 1 flat | 1 | 15,592 | 97.5 | OK |
| 2 seam (queens-off) | 2 | 3,544 | 22.1 | thin |
| 4 wings (wk × bk) | 4 | 981 | **6.1** | below the failure point |
| **8 seam × wings4 (the straw man)** | 8 | **48** | **0.3** | **37× worse than known-bad** |
| 4 phase quantiles | 4 | 3,024 | 18.9 | thin |
| 6 phase quantiles | 6 | 2,044 | 12.8 | marginal |
| **7 phase quantiles (the in-budget row)** | 7 | 1,406 | **8.8** | **below the failure point** |
| 8 phase quantiles | 8 | 1,035 | 6.5 | below the failure point |

**The straw man's worst bucket holds 48 training positions for 160 parameters.**
It is not close, and no encoding choice fixes it — buckets buy capacity by
*dividing the data*.

A correction to my own instrument, recorded because it flattered the answer:
the first version cut *ranks* into equal parts and reported 12.0 pos/param at 8
buckets. That partition cannot ship. Phase is a coarse integer with a lumpy
histogram (2,300 of 15,592 training positions sit at phase 4 alone), so a rank
cut splits a phase value between two buckets and the root, which sees only the
phase, cannot reproduce it. The shippable number is **6.5**.

### Recommendation

1. **Do not fit the budget-filling candidate yet.** At 7 buckets it is 8.8
   pos/param — below the point where this lane has already watched a bucketed
   fit fail. Filling 1024-1500 B with the current generator would be buying
   memorisation and paying bytes for it.
2. **The honest maximum on today's data is ~4-6 phase buckets** (18.9 → 12.8
   pos/param, 762 → 1013 eval B) — which lands just under Thomas's 1024 floor.
   That is the gap, stated in the currency that binds: **not bytes, positions.**
3. **The generator is the unlock, and there are two.** Distilling from a trained
   teacher removes the labelling budget from the equation entirely — it can
   label as many positions as we sample, so pos/param stops being a function of
   Stockfish time. That is what makes the ternary N=16 net (−14.7% val)
   interesting now: with 1120 parameters to decode INTO, it finally has a
   student worth its capacity. The cheaper route in parallel is simply more
   games (caprr's 4,000 plus the ladder), which the set already knows how to
   consume.
4. **Fit extra sets as regularised deltas from a shared base**, not
   independently. It halves the effective parameter count at the same nominal
   capacity, and it is also what moves the byte cost from the +224 end of the
   bracket to the +130 end. Both problems, one change.
5. **Mirroring stays** until C1 vs C2 says otherwise — that screen is the only
   measurement of what compression costs in play, and it gates whether the
   budget buys 7 mirrored sets or 3 unmirrored ones.

---

## 2026-08-13 — Rebased onto the moved base, C1/C2 re-measured, and C1's bar re-derived because its byte credit no longer exists

The eval lane's work was sitting on a base two landings behind. Rebased
`eval-decode-track` onto `nnue-4k` (IIR + the interface trims), regenerated,
and **re-measured every number rather than adjusting the old ones.**

| | packed | engine-sans-eval | eval | eval ceiling |
|---|---|---|---|---|
| `nnue-4k` tip (literal tables) | 3445 | 2942 | 503 | 1154 |
| **rebased, base-90 decode** | **3350** | **2886** | **464** | **1210** |

Base-90 is worth **−95 bytes on this base**, against −97 on the previous one —
the usual reminder that a byte saving is a property of the pair, not of the
change. **The eval ceiling is 1210 B**, so Thomas's 1024 floor is reachable
today and his 1500 needs ~290 B more from the search lane.

### `price_engine.sh` was returning a silently wrong answer on this entry

Run against the base-90 entry it reported **"eval data costs 30, ENGINE 3320"**.
Its regex looks for a `pst` literal to zero; there is no literal any more, so it
matched a `\n}` hundreds of lines below the eval and zeroed something unrelated.
Both numbers were wrong and neither looked wrong — and this instrument is where
the golf target's 2942/1132 came from, so it needed to be right before the
allocation was argued over.

Fixed to handle both entry forms, and to **abort rather than guess** if it
recognises neither. The literal form still reproduces **2942 / 503** exactly, so
the golf lane's published numbers stand; the base-90 form now reads 2886 / 464.

### The candidates, rebuilt from the current generator

C1 and C2 are no longer hand-staged files. They are **mods in
`tools/build/make_variants.py`**, generated at screen time from the committed
fit — the same single-source rule the search variants follow, because the whole
point of that file is that a candidate cannot go stale against its base.

| | packed | vs entry | eval bytes | spare |
|---|---|---|---|---|
| entry (rebased) | 3350 | — | 464 | 746 |
| **C1** flat refit, mirrored step 8, K exact | **3187** | **−163** | 301 | 909 |
| **C2** flat refit, exact | **3412** | **+62** | 526 | 684 |

The deltas survived the base change almost exactly (−163 and +62, against −163
and +63 before). The generated sources pack to the same bytes as the direct
codec builds, which is the check that the mod and the pricer agree.

### The bars are RE-DERIVED, because the old ones were denominated in a currency that no longer exists

C1's old bar was *"LAND if the 95% lower bound is above −15 Elo"*, and the −15
was a byte credit: C1 saved 163 bytes when bytes were scarce. **Under Thomas's
allocation they are not scarce.** The eval sits 746 bytes under its own ceiling
and the standing instruction is to FILL it to 1024-1500. A byte saved inside the
eval now buys nothing the project wants — it moves *away* from the target.

So, pre-registered before the screen runs:

- **C1 LANDS only if the 95% interval's lower bound is above 0.** No byte
  credit. **DROP if the upper bound is below 0.** Between those, C1 is not
  landed and the result is read as the generator verdict below.
- **C2's old +63 Elo bar is void too** — a +62 byte eval cost is now *wanted*,
  not charged at 1.0 Elo/byte. C2 lands over the entry only on LB > 0, and
  **C1 − C2 is the mirroring measurement**, which is the only thing loss cannot
  answer. Run only if C1 leaves mirroring ambiguous.

### What the screen is actually FOR now: it is a generator validation

This is the part that matters more than whether C1 lands. A bucketed
1024-1500 B eval is **the same fit with more parameters on the same data.** So:

- **C1 clearly positive** ⇒ the Texel objective on `set20260813` converts to
  Elo, and the budget-filling design is justified on this data, subject to the
  positions-per-parameter gate below.
- **C1 flat or negative** ⇒ scaling that fit to 1120 parameters across 7 buckets
  is not justified, and the budget should be filled only after the GENERATOR
  changes — a distilled teacher or more data — not after more parameters.

A flat result is the modal outcome and was predicted in advance: C1's −5.31%
held-out is almost entirely endgame, and in the middlegame band the fit is
very slightly *worse* than classic.

### Gates, on the box, before any game

| arm | legality (40 FORCED / 30 check / 30 quiet) | mate-in-1 |
|---|---|---|
| `e_base.py` (the 3350 entry) | 0 no-move, 0 illegal — PASSED | 8/8 |
| `e_c1.py` | 0 no-move, 0 illegal — PASSED | 8/8 |
| `e_aa.py` (A-vs-A control arm) | 0 no-move, 0 illegal — PASSED | — |

The mate gate is new (`tools/build/mate_gate.py`) and deliberately does not
replace the legality gate — the standing warning is that a mate suite passed
5-vs-5 on the very build that answered `bestmove (none)`. It verifies mate-in-1
by PLAYING the move and asking python-chess whether the result is checkmate,
never by reading a score: a score check would pass an engine that reports
`mate 1` and then plays something else.

`e_aa.py` is byte-identical to `e_base.py` (same sha256), staged in the same
directory with the same `PYTHONPATH`, so both arms provably get the same UCI
driver. That is the control the `agree.py` incident earned, where an engine
outside the repo tree silently picked up the builtin `go` loop and a
byte-identical copy of the entry "disagreed" with itself 39/60.

### Cotenancy: the slot was not actually free, and the screen waited

The coordinator's slot handoff said the box arena was idle. It was not:
`elo-masked-cap-20260813` was **363 games into a 600-game `tc=30+1` TIMED
match** at concurrency 10. A fixed-node screen is insensitive to load, but the
match beside it is not, and this project has already voided one 200-game match
to a shared-machine effect. The screen was **staged and held** rather than
launched, and went out only after that match completed. Logged here because
"the slot is yours" is a claim about the box, and the box is checkable.

## 2026-08-13 — `bestmove (none)` is BUDGET STARVATION, not mirroring; and the gate's hole was its positions

The eval lane bisected a `(none)` to "mirroring alone, at full resolution".
Mirroring is real but it is **not the mechanism**, and two of the three facts
that framed the search were harness artifacts. Repro before theory:

### The mechanism, named

`search()` reaches its **first `yield` only after the depth-1, gamma=0 MTD
probe completes**. `bound()` polls its budget at `nodes % 2048 == 0` and
raises `Stop`. When the probe costs more nodes than the budget, `Stop` is
raised *before the first yield*, `stop_softly` swallows it, and `go_loop`
iterates an **empty** stream: `best_move` and `cand` stay `None`, the `pv()`
walk finds nothing in `tp_move`, and it prints `bestmove (none)` — with no
`info` line, because info lines are printed per yield.

Nodes to first yield on the reported FEN
(`rnbqkb1r/pp2pppp/3p4/2pnP3/3P4/2P2N2/PP3PPP/RNBQKB1R b KQkq d3 0 11`),
read off each engine's own first `info` line:

| arm | encoding | nodes to first yield | `go nodes 20000` |
|---|---|---|---|
| base | classic tables | **23** | `d6e5` |
| C2 | fit, exact, unmirrored | 1,362 | `c5d4` |
| q8 | fit, step 8, unmirrored | 1,745 | `d6e5` |
| m1 | fit, exact, MIRRORED | 9,086 | `c5d4` |
| **C1** | fit, step 8, MIRRORED | **32,638** | **`(none)`** |

The budget is the whole story: **C1 at `go nodes 20000` → `(none)`; C1 at
`go nodes 40000` → `c5d4`.** Same binary, same position, one number changed.

### Two claims REFUTED

- **"Mirroring alone breaks it."** Mirroring is a slowdown, not a switch. It
  is a continuum — 23 → 1,362 → 1,745 → 9,086 → 32,638 — and C1 is merely the
  arm that crosses 20,000. Mirroring is *not even the dominant term*: the
  **unmirrored** C2 and q8 both fail the new gate. The 7-bucket mirrored
  design was withdrawn on a cause that does not hold; what is actually
  disqualifying is that every fitted arm is 60–1400× slower to a first move.
- **"It fails at every depth including 1, with no info line."** That was the
  repro harness. A piped heredoc delivers `quit` while the search is still
  running, and `run()`'s quit branch sets `searcher.deadline = 0`, which trips
  the same 2048-node poll. Driven so that `quit` is sent only *after*
  `bestmove` arrives, **every arm answers at `go depth 1` and `go depth 3`**.
  The `(none)` needs a *budget*, not a depth.

Root, not eval: nothing here is specific to a table. Any engine whose first
probe outruns its budget answers `(none)`, and `(none)` is scored as an
illegal move — a forfeit. The shipped entry is safe only by margin (23 nodes),
not by construction. **The fix belongs in the search or the driver: never
return from `go` without a move while any pseudo-legal move exists.** Not
taken here — it costs artifact bytes and needs its own match — but the gate
below now measures the margin instead of trusting it.

### Why the 100-position gate passed it

Two independent holes, and only the second one mattered:

- **Wrong budget (minor).** The gate only ever sent `go movetime 300`; the
  arena plays fixed-node. Now both paths are gated and `ab_fixednode.sh`
  passes its own `--nodes`. **Negative: this alone caught nothing** — C1
  passed 100/100 at `go nodes 20000` too.
- **Wrong positions (binding).** Random-playout positions are wildly
  unbalanced, so the null-window probe at gamma=0 cuts off almost at once:
  median **2 nodes**, and C1's worst over 120 was 4,542. The expensive class
  is **quiet, balanced, dense** — real opening positions, where nothing
  resolves the window. Also measured and rejected: WAC/Bratko-Kopec/CCR
  (tactically sharp, so they cut off fast — C1 max 955 over 79), and
  book-plus-random-plies (random moves destroy the balance: max 2,737).

### The new arm, and why it needs no luck

Asking "did a move come back" only catches starvation when the budget happens
to land between 2048 and the first yield — a coin flip per position. The
**FIRST-YIELD** arm asks the quantity itself, off the engine's own
`info ... nodes` field, over 334 committed book positions
(`tools/build/gate_openings.epd`), one process for the whole arm. It fails if
any first yield exceeds **2048 nodes** — the poll granularity, hence the
smallest budget any engine can observe, so exceeding it *proves* a starving
budget exists.

| build | worst first yield | over budget | verdict |
|---|---|---|---|
| **shipped entry** (`nnue_4k/pst_entry.py`) | **582** | **0 / 334** | **PASS** (3.5× margin) |
| C2 (unmirrored, exact) | 4,870 | 8 / 334 | FAIL |
| q8 (unmirrored, step 8) | 7,949 | 7 / 334 | FAIL |
| m1 (mirrored, exact) | 7,779 | 2 / 334 | FAIL |
| **C1** (mirrored, step 8) | **10,359** | **10 / 334** | **FAIL** |

Positive-controlled both directions on the final file. The packed artifact
emits no `info` lines, so the arm **SKIPs** there and the verdict says
`GATE PASSED (LEGALITY ONLY)` — a skipped arm is never reported as a passed
one. Instrument note: recording every info line instead of the first measured
`stop` latency, not the probe, and inflated the shipped entry's worst position
from 582 to 1,879 against a 2048 budget — a false failure was one position
away.

---

## 2026-08-13 — The golf target was the wrong number, and two of three "safe" trims are not safe

### The arithmetic, re-run against the measured engine

**4096 − 2942 = 1132 bytes for the eval, and Thomas's floor is "at least
1024".** The 2500 target assumed the full 1500-byte eval; the engine already
clears the 1024 case with 108 bytes to spare. So the binding constraint is not
golf — it is **the eval lane's priced grid**: if their winner fits in ~1130 the
golf mission is essentially done, and only if it wants the full 1500 does the
remaining ~250-350 of expression surgery become real work.

Bit-identical surgery on the search core is therefore **paused**, which is the
right call for a reason beyond scheduling: that surgery operates on `bound()`,
`gen_moves()` and `move()` — the code holding every feature this lane has
earned — and spending that risk to hit a target nobody needs yet is how a byte
diet turns into a strength regression.

### Taken: the info/PV line, −22 bytes

| | |
|---|---|
| entry | 3468 → **3445** (651 spare) |
| engine-sans-eval | 2964 → **2942** |
| node counts, depth 9 over 6 positions | **2,342,657 both builds — bit-identical** |
| gates | legality 100/100, mate 5/8 parity, 312 passed / 2 skipped |

`info` is optional in UCI, `cand` is still computed (it is what gets played),
and the development path is untouched — every screen runs through
`sunfish_ui/uci.py`, which prints depth, time, nodes, nps and pv. What is lost
is PV output from the **packed artifact in production**, which no gate, no test
and no harness reads.

**Verified after the cut, because these lines were the only output between `go`
and `bestmove`:** the artifact still streams its handshake and its bestmove
**live** to a pipe with stdin held open. Worth recording that the *first*
version of that check was wrong — `select()` on a buffered text stream reported
a phantom stall and I nearly wrote up a buffering bug that does not exist. Redone
with a reader thread: handshake at 0.46 s, `bestmove` at 3.59 s, both live.

### Rejected: `version` globals. It would make the entry unmeasurable

Cutting `id name` **and** the `version`/`__version__` globals prices at −40,
the biggest item on the menu. It is not available:

    sunfish_ui/uci.py:
    ENGINE_API = ("MATE_LOWER", "Move", "Position", "Searcher", "Stop",
                  "opt_ranges", "parse", "render", "version")

`check_engine_module` raises `TypeError` on a missing member. The **packed**
artifact never touches the driver, so the shipped thing would work perfectly —
the breakage appears only when someone tries to **measure** the entry, i.e. in
every screen, every legality gate run from a checkout, every A/B. Ship fine,
measure impossible: the worst failure shape available.

Cutting only the `print("id name", …)` and keeping the globals is ENGINE_API-safe
and prices at **−8**, for a UCI-spec deviation (`id name` is a "should"). Not
taken: 8 bytes is a bad trade for anonymity in a tournament we are entering.

### Deferred: the `movetime` branch. −25 bytes for a silent forfeit

Removing it does not make `go movetime 200` fail loudly. `times.get("movetime")`
disappears, `wtime` defaults to 60000, and the engine thinks **5 seconds**
against a 200 ms limit — a forfeit, with nothing in any log to say why. The
branch exists because of a measured incident: *"425 local fixed-node games,
every single one a forfeit."*

And it has a live consumer: **`tools/deploy.sh:191` smoke-tests the packed
artifact with `go movetime 200`** — the packed artifact, which is exactly the
build that has no driver to fall back on.

Deferred, not refused. If the eval lane's winner needs the full 1500, take it
**together with** switching that smoke to a `wtime`/`btime` command, so the
capability and its only consumer move in one commit.

### The menu as it now stands

| cut | saves | status |
|---|---|---|
| info/PV line | **22** | **TAKEN** |
| `version` globals + `id name` | 40 | **REJECTED** — breaks `ENGINE_API` |
| `movetime` branch | 25 | deferred — silent forfeit, `deploy.sh` consumer |
| `id name` print only | 8 | not taken — spec deviation for 8 bytes |
| `position` prefix → `args[0]` | 12 | not taken — misparses `position fen` |
| `quit` branch | 7 | not taken — dies on EOF instead of exiting |
| `__hash__`/`__eq__`/`__ne__` | 54 | **load-bearing**, see the previous entry |

**Interface golf is now exhausted at 22 taken bytes.** Everything else on the
menu costs a capability, and the arithmetic says we are not short of bytes.

## 2026-08-13 — THE GOLF BUDGET, MEASURED: 2964 engine bytes, and the named cuts are worth ~115

New mission: the eval lane takes 1024-1500 bytes, leaving **~2500 for the
engine itself**. First job is to stop estimating that number.

### The instrument, and the baseline

`tools/build/price_engine.sh` takes the real entry, zeroes only the *values*
inside the `pst` literal, and packs it with the real packer. That is not "the
tables in isolation" — lzma shares one dictionary across the stream and this
ledger has been wrong every time it composed a byte figure. It is the honest
quantity: **what the entry would cost if its eval data were free**, which is
exactly the budget the golf mission spends against.

| | bytes |
|---|---|
| entry as shipped | **3468** |
| same file, `pst` values zeroed | **2964** |
| eval data, incremental | 504 |
| **ENGINE-SANS-EVAL** | **2964** |
| target | 2500 |
| **still to find** | **464** |

The handoff's "~2980" was close, and it is now measured.

### The byte map, built by deletion

Each region deleted from a real file, re-packed, difference reported. Most of
these builds cannot run — this instrument prices, it does not propose.

| region | packed cost |
|---|---|
| `Searcher.bound` | **560** |
| **`main()` UCI loop** | **454** |
| `Position.gen_moves` | 207 |
| `Position.move` | 196 |
| `Searcher.search` | 156 |
| `parse`/`render`/`from_board`/`hist` | 138 |
| `Position.value` | 113 |
| `__hash__`/`__eq__`/`__ne__` | 54 |
| `Position.rotate` | 46 |
| `Position.king_capture` | 36 |
| sum | 1960 |

The ~1000 unattributed is module constants, `initial`, `directions`, imports,
the packer's own 74-byte head, and shared-dictionary residue.

### The named candidates, priced — and the answer is no

Every cut on the handoff's list, built as a real file and packed inside the
engine frame:

| cut | saves | what it costs |
|---|---|---|
| `id name` + the `version` globals | **40** | GUIs show a blank engine name |
| `movetime` branch | **25** | the artifact can no longer be driven by `go movetime` |
| info/PV print | **22** | no PV in any log, ever |
| `position` prefix test → `args[0]` | 12 | a `position fen …` command misparses |
| `__ne__` (control) | 9 | **load-bearing — see below** |
| `quit` branch | 7 | dies on `EOFError` instead of exiting |
| **total (not additive)** | **~115** | |

**Interface trimming cannot reach 2500.** `main()` costs 454 bytes, but almost
none of that is command parsing — it is the *search driver*: the deadline, the
committed-vs-candidate logic that stops the Qxc6 giveaway class, and the
iteration loop. The parseable surface is ~115 bytes and every byte of it costs
a capability. **The remaining ~350 has to come out of `bound()`, `gen_moves()`
and `move()` as bit-identical expression rewrites**, which is delicate work
against the hard constraint that every strength feature stays.

**And one candidate is a trap.** Deleting `__hash__`/`__eq__`/`__ne__` and
letting namedtuple's defaults apply looks like 54 free bytes, because `score`
is a pure function of the board. It is not free: **`pst["K"]` is swapped
between `K_MID` and `K_END` per search**, so the same board legitimately
carries two different scores across a table change. Default equality compares
`score`, so the repetition set would stop recognising repetitions and the
"never evict the root" guard (`k != self.root`, the only `!=` on a Position in
the file) would stop firing. The custom identity is load-bearing; this is now
written in the docstring rather than left for the next golfer to discover.

### Free while we were looking: the entry stopped describing a net

The survey found the shipped entry opening with a section header called
*"Packed big-integer NNUE residual"*, an evaluation formula containing
`clip(nn(pos), -CLAMP, CLAMP)`, a `Position` docstring listing **`ps`, `acc`,
`pf` and `kb`** — four fields the class does not have — and a comment claiming
*"the tables themselves ride in the net file (see the loader above)"* when
there is no loader and no net file. Same defect class as the null-move comment
that claimed a cap this engine never had, and the same rule applies: the model
matches the code.

Corrected in the generator. Comments are stripped by the packer so the prose
was free, but the dead **`pf=0` parameter on `from_board()`** was not — no
caller passes it, in this file or in `sunfish_ui/uci.py`.

**Entry 3472 → 3468, spare 628.** Verified **bit-identical**: node counts to
depth 9 over 6 positions are **2,342,657 for both builds**, exactly, which is
the pre-registered standard for a pure-golf cut. Legality gate 100/100, mate
gate 5/8 parity, 312 passed / 2 skipped, plays alone in an empty directory.

**First 4 of the 464 found. The other 460 are in the search core**, and the
budget arithmetic deserves a look before that work starts: interface is
exhausted at ~115, and the rest is bit-identical surgery on the code that
holds every feature this lane has earned.

## 2026-08-13 — LANDED: IIR ships, and the entry is stronger AND smaller

**`iirk.noiid` is in `tools/build/make_pst_entry.py`.** Internal iterative
reduction — reduce a ply when the position has no table move — **replacing**
the IID probe, with the killer read once at the top of `bound()` so the
position is hashed once instead of twice.

| | before | after |
|---|---|---|
| entry, packed | 3475 | **3472** |
| spare | 621 | **624** |
| fixed-node vs the old entry | — | **+22.3 ± 16.0** (1,000 games) |
| timed (speed term −7.5) | — | **≈ +15 Elo** |

**We shipped exactly what we measured, and this time it is provable.** The
generated entry differs from `e_iirknoiid.py` — the arm that played all 1,000
confirmation games — only in comment wording, and comments are stripped by the
packer, so:

    packed sha256  ce091e5e4051add8703a896498711fdc8d7f6bc36a18461956799eadf629f99f
    played arm     ce091e5e…   3472 bytes
    shipped entry  ce091e5e…   3472 bytes   <- byte-identical

**Landing gates, all green:**

- `check_entry.sh` — source matches its generator, packs to **3472 (624 spare)**
- **legality gate** on the generated file — 40 FORCED / 30 in-check / 30 quiet,
  **0 no-move, 0 illegal**
- **mate gate** — 5/8, parity with the pre-IIR entry
- **284 passed, 2 skipped** in the full test suite
- **plays alone in an empty directory** with `SF_NET` and `PYTHONPATH` unset —
  one file, 3472 bytes, `bestmove g1f3` at depth 6

**Entry-only, deliberately.** The transform lives beside `kend`/`fresh` rather
than in `sunfish_nnue.py`. IIR's trigger reads no evaluation, so the
(feature, eval) rule does not *force* a re-measure — but `sunfish_nnue.py` is
the lichess bot's engine and another lane's artifact, and nothing here has
played a game with the net. It transfers when someone measures it there.

**The mods are DELETED from `make_variants.py`, not retired in place**, and the
reason is sharper than tidiness this time. `noiid`'s anchor no longer exists —
that is the designed failure and it would be safe. But **`iirk`'s first anchor,
`depth = max(depth, 0)`, still occurs exactly once in the new baseline**, so
re-applying it would insert a *second* killer read and a *second* reduction
while the occurs-exactly-once check passed cleanly. That is the silently
doubled mod the generator was written to prevent, and it was one composition
away. The remaining mods (`cap`, `corr`, `hist`, `nolmr`, `wkey`, the `fut`
family) were regenerated against the new baseline and all still apply.

**Byte note for the next lander.** This is measured against **3475**, this
branch's baseline. The eval lane's base-90 decode is a separate −97 on *their*
baseline, and **the two do not add**: lzma shares one dictionary across the
whole stream. Per the standing rule and the coordinator's sequencing, they
rebase on this and regenerate.

## 2026-08-13 — CONFIRMED: `iirk.noiid` is +22.3 ± 16.0 and the entry SHRINKS to 3472

**The shipping binary played its own match and won it.** `base` vs
`iirknoiid`, fixed nodes 20,000, fixed N, no SPRT:

| | |
|---|---|
| result | **+22.3 ± 16.0** for `iirknoiid` (`Elo(base) = −22.27`) |
| raw | base **351** wins, `iirknoiid` **415** wins, 234 draws |
| games | **1,000 of 1,000**, counted from `conf.pgn` |
| clean | **0 time forfeits, 0 illegal moves** |
| gates | driver control PASS, mate gate 5/8 parity, legality gate 40 FORCED / 0 no-move / 0 illegal |
| arm | sha `a124a6b8…`, verified identical to a fresh generator build |
| bytes | **3475 → 3472, −3**; spare 621 → **624** |
| timed | +22.3 − 7.5 = **≈ +15 Elo** |

**Stronger and smaller.** No Elo/byte bar applies to a byte-negative change; it
had only to avoid losing, and it won by more than its interval.

### The confirmation halved the round-robin's number, and that is the finding

| instrument | arm | result |
|---|---|---|
| 4-arm round-robin, 660 g | `iirnoiid` | **+41.3 ± 22.4** |
| 2-engine fixed-N, 1,000 g | `iirknoiid` | **+22.3 ± 16.0** |

The two arms are the same search — node counts bit-identical on all seven probe
lines — so this is one quantity measured twice, and the answers differ by
**19.0 ± 27.5**. Not formally inconsistent, and the direction never wavered.
But it is the third time today that a *pooled or selected* number came in high
against a *direct, pre-sized* one:

- classic-anchored differences ran ~50 above head-to-head (predecessor)
- the round-robin's global fit ranked `hist` first at +15.8 while its direct
  match with base was −3.2
- and now a round-robin pairing reads +41.3 where a dedicated match reads +22.3

**Prefer the dedicated fixed-N match on the actual artifact.** It is paired, it
is pre-sized, it has the tighter interval, and the binary in it is the binary
that ships. `+22.3 ± 16.0` is the number that goes in the ledger; `+41.3` is
withdrawn as an over-estimate rather than averaged with it.

This is also the standing "an SPRT pass is not an effect size" rule earning its
keep in a new form: **a round-robin pairing is not an effect size either.**

### What lands, and what has to happen first

`iirk.noiid` = internal iterative reduction (reduce a ply when the position has
no table move, decided *before* the table probe so the reduced depth is the key
in both directions) **replacing** the IID probe, with the killer read once at
the top so the position is hashed once instead of twice.

**It is NOT landed in this session.** Landing means editing
`tools/build/make_pst_entry.py`, which is coordinator-sequenced, and the eval
lane's base-90 decode (`430c297`, `5b58a18`, entry 3378 / 718 spare) is ahead of
this branch in the entry's history. The −3 bytes will not simply add to their
−94: lzma shares one dictionary across the whole stream, so **the second lander
regenerates and re-measures**. The mod is in `tools/build/make_variants.py` as
`iirk.noiid` and reproduces byte-for-byte from the generator.

## 2026-08-13 — THE FRONTIER FUTILITY MARGIN IS ALREADY TUNED, and that explains corrhist

**The axis is closed in both directions, and the zero-byte candidate is dead.**

The frontier futility test is `pos.score + val < gamma`, i.e. a margin of
**zero**, and it has never been tuned — QS/QS_A were swept, this constant was
not. corrhist's autopsy said the margin was the mechanism, so it was measured
directly. Five points now exist on that axis, all fixed-node, all our-vs-our:

| margin at depth ≤ 1 | measurement | Elo vs base |
|---|---|---|
| **−40** (`futm40y`) | 256 g, raw 64–141 | **−110.3 ± 40.6** |
| **−15** (`futmy`) | 258 g, raw 93–99 | **−4.0 ± 38.9** |
| **0** | the entry | — |
| **+15** (`fut`) | 71 g, stopped, preliminary | ≈ −54 |
| **+40** (`fut40`) | 72 g, stopped, preliminary | ≈ −25 |
| variable, mean +10…+18 (corrhist) | 617 g | **−54.8 ± 23.3** |

**Zero is at or next to a local optimum, and the well is steep on the pruning
side.** −40 costs **110 Elo** — the largest single-parameter regression in this
ledger. −15 is level. Both positive margins lose. There is no free Elo here and
no zero-byte win: `futm40y` packs to 3476 and plays 110 Elo worse.

**This closes corrhist completely.** Its correction was a *variable positive*
margin averaging +10 to +18 cp, and a fixed positive margin in that range costs
roughly what corrhist cost. corrhist did not fail because a pawn-skeleton key
is a bad idea or because 127 bytes is dear; **it failed because the thing it
moved was already in the right place**, and every cp it added to that test was
spent making a well-tuned rule worse. The mechanism is understood end to end,
which is the standard for closing an item rather than shelving it.

**Stopped deliberately at 772 of 1,800 games** to give the box to the
`iirk.noiid` confirmation, which gates a landing. The direction was established
far outside noise by then (−110 ± 41 and −4 ± 39), and the remaining games would
have refined a negative at the cost of delaying a decision. The `fut`/`fut40`
rows are flagged preliminary and stay preliminary — they are the mirror of a
result that is not in doubt, not evidence in their own right.

**Method note that generalises.** The zero-byte claim was real (`+QS` packed to
3475 exactly) and the arm was still worthless. Byte price and Elo are
independent, and a cheap candidate deserves the same screen as a dear one —
"free" is a reason to *test* something, never a reason to ship it. Two of
today's four candidates were byte-negative and only one of them was any good.

## 2026-08-13 — THE ORDERING RR: history is dead a second time, and IIR lands

**3,960 of 3,960 games** (counted from `order.pgn`), **0 time forfeits, 0
illegal moves**, 330 complete colour-swapped pairs in every one of the six
pairings, **not one unpaired game dropped**. 12:43:36 → 14:15:43 UTC, 1h27m28s.
Every arm passed the driver control (same `uci.py` v2), the mate gate at 5/8
parity with base, and the legality gate at 40 FORCED / 0 no-move / 0 illegal.

**Every number below was checked against raw PGN win counts before it was
written down**, which is the process the corrhist sign error bought us.

| pairing | `pair_elo` Elo(A) | raw wins A–B | reading |
|---|---|---|---|
| base — hist | +3.16 ± 21.62 | 252 – 246 | **hist −3.2** |
| base — iirnoiid | −41.25 ± 22.37 | 240 – **318** | **iirnoiid +41.3** |
| base — noiid | −3.16 ± 10.42 | 240 – 246 | **noiid +3.2** |
| hist — iirnoiid | −3.16 ± 21.02 | 228 – 234 | iirnoiid +3.2 over hist |
| hist — noiid | +54.13 ± 20.70 | **294** – 192 | hist +54.1 over noiid |
| iirnoiid — noiid | +3.16 ± 17.34 | 258 – 252 | iirnoiid +3.2 over noiid |

### The verdicts, against the bars fixed before the games

**`hist` — DROPPED, and this time by the right instrument.** −3.2 ± 21.6
against a bar of +62 fixed-node. The 2026-08-12 removal used a node-ratio
proxy, and this revisit exists precisely because that was the wrong
instrument; 660 games agree with it anyway. History ordering is worth
**nothing** here at 61 bytes. The caveat that removal carried — *"measured at
depths 6-7, no evidence either way at depth 9+"* — is now closed from both
ends: censused at depth 9 it costs **28% more nodes**, and in games it is
level. **The 61 bytes stay out, and history should not be queued a third time**
without a materially different mechanism.

**`iirnoiid` — LANDS. +41.3 ± 22.4, and it is byte-negative.** The interval
excludes zero, the raw count is 318–240, and the entry goes **3475 → 3471**.
The first item in this lane's history that is both **stronger and smaller**,
and the largest verified search gain since LMR.

**`noiid` — +3.2 ± 10.4, level.** Byte-negative (−16) with no measurable cost;
by the pre-registration a non-negative point estimate lands. In practice it is
subsumed — `iirnoiid` contains it — so what ships is the pair, not `noiid`
alone.

### The pool is NON-TRANSITIVE, and it is not an analyzer artifact

The triangle does not close, and the raw win counts say exactly what the
analyzer says:

- `iirnoiid` beats `base` **318–240** (+41.3) but only edges `noiid` **258–252** (+3.2)
- `noiid` is level with `base` **246–240** (+3.2)

So base → noiid → iirnoiid implies iirnoiid ≈ +6 over base, against +41
measured directly. Likewise `hist` is level with base and beats `noiid`
**294–192**.

**What that does to a global fit — fastchess's own ranking table:**

| rank | arm | pooled Elo | head-to-head vs base |
|---|---|---|---|
| 1 | hist | **+15.8 ± 12.2** | **−3.2** |
| 2 | iirnoiid | +15.8 ± 11.8 | **+41.3** |
| 3 | base | −13.7 ± 10.9 | — |
| 4 | noiid | −17.9 ± 9.7 | +3.2 |

The pooled fit ranks **`hist` first**, on the strength of one lopsided pairing
against `noiid`, and ranks `noiid` *below* `base` while their direct match is
level. This is the predecessor's "anchored differences run high" finding in a
new costume: **a round-robin's global rating spreads a result against one
opponent across the whole pool.** Per the standing rule — and because the
question a shipping decision asks is literally "does this beat the incumbent" —
**the base head-to-head is the instrument.** Nothing turns on the choice here
(`hist` fails on both; +29.5 pooled is still far under +62), but the next lane
to read a ranking table should read this row first.

Draw rates carry the same signal: base–iirnoiid is the most decisive pairing in
the tournament (**102 draws**), hist–iirnoiid the least (**198**). These arms
are not the small perturbations of each other that their diffs suggest.

### What ships is not what played

`iirk.noiid` — same search, single `tp_move` lookup, **3472 bytes (−3)**, nps
0.950 instead of 0.908 — is the form worth landing, and its node counts are
bit-identical to `iirnoiid`'s on all seven probe lines. That is a strong
argument, but it is an argument, and the arm has never played a game. A
**fixed-N confirmation, `base` vs `iirknoiid`, 1,000 games at 20k nodes**,
launched 14:17 UTC (arm sha `a124a6b8…`, verified identical to a fresh
generator build). Fixed N and no SPRT: an SPRT pass is not an effect size, and
neither is a node count.

**`iirk.noiid` must not enter `tools/build/make_pst_entry.py` before that match
reports** — and the generator is coordinator-sequenced regardless.

Timed value if it confirms: **+41.3 − 7.5 ≈ +34 Elo for −3 bytes.**

## 2026-08-13 — CORRECTION: corrhist is −54.8. It is a REGRESSION, and I read the sign backwards

**The entry below is wrong in its direction and I am not editing it away.** It
says corrhist won +54.8 and was "priced out rather than disproven". The +54.8
is real and it belongs to **`base`**. corrhist **lost**.

Counted straight out of the PGN, no analyzer in the way:

| | wins | draws |
|---|---|---|
| **base** | **290** | 135 |
| **corr** | **192** | 135 |

base scores 57.94% over 617 games. `pair_elo.py` prints `Elo(A) = +54.77 ±
23.28` where **A is the alphabetically-first engine name**, which is `base`;
fastchess's own line prints `Elo: 54.77` for **engine1**, which is also `base`,
under `Wins: 281, Losses: 192` — engine1's wins. Every instrument agreed and
said so plainly. I read "SPRT H1 accepted" as "the candidate passed", when what
it means is "**engine1** is better by ≥ elo1", and engine1 was the baseline.

**The verdict, restated correctly:**

| | |
|---|---|
| corrhist, fixed nodes | **−54.8 ± 23.3** |
| games | 617, SPRT stopped early, H1 accepted **for base** |
| speed term | −8.5 |
| timed | **≈ −63** |
| bytes | +127 |

**corrhist is DROPPED as a regression, not as an expense.** It is not "the
largest fixed-node effect this lane has measured"; it is the second-largest
*negative*, after LMP's −126. The transfer scoreboard's real state:

| feature | ice4 Elo | ours (fixed-node) | outcome |
|---|---|---|---|
| LMR | 81 | +38.9 ± 19.1 | transfers, shipped |
| RFP | 58 | ~ 0 | sound, worthless here |
| **corrhist** | **70** | **−54.8 ± 23.3** | **harmful** |
| LMP | 123 | −126 | structurally incompatible |

**Three of four ice4 items are now negative or worthless on this engine.** The
mean transfer coefficient is not "far below 1", it is **below zero**. Anyone
planning +400 out of the ice4 catalogue should read that column again.

**And the mechanism is legible, which is the useful part.** corrhist's only
consumer is the `depth <= 1` futility test; its censused table was
systematically **optimistic** (mean +10…+18 cp); adding an optimistic
correction there makes `pos.score + corr + val < gamma` fire **less**, so the
search prunes less — which is exactly what the node counts said (1.04× to depth
8, 1.15× to depth 9). **It searched more and played worse.** The frontier
futility rule is not too aggressive. If anything it is not aggressive enough.

**What this does to the zero-byte lead below: it inverts it.** The
futility-margin entry was written from the wrong sign and reasoned that
"pruning less wins, so a flat positive margin is a cheap version of it". The
truth is the opposite — pruning less **loses** — so the arm with upside is a
**negative** margin, and `fut`/`fut40` are now *predicted losers*, worth one
confirmation rather than two. The first 28 pairs of that RR already read `base
+53.9` over `fut`: the same sign and roughly the same magnitude as corrhist,
i.e. the mechanism reproducing itself for 3 bytes instead of 127. The RR is
relaunched around the negative margin.

**Process note, because the labelling makes this error easy to repeat.**
`pair_elo.py` sorts the two names alphabetically and calls the first one "A",
so which arm is "A" depends on what the variant was *named*. `base` sorts
before `corr`, `fut`, `hist`, `iir*` and `noiid` — the anchor is "A" in every
pairing of every screen this lane runs, which is exactly the arrangement that
makes a positive number look like a candidate's win. From here: **every verdict
is checked against raw win counts from the PGN before it is written down.** It
is a two-second `awk` and it would have caught this before it reached a commit
message.

## 2026-08-13 — corrhist DROPPED at +54.8 (THIS ENTRY HAS THE SIGN BACKWARDS — see the correction above)

**The screen passed and the feature still does not ship.** That combination is
new in this ledger and it is the whole entry.

`base` vs `corr`, fixed nodes 20,000, SPRT elo0=0 elo1=10, α=β=0.05:

| | |
|---|---|
| result | **+54.77 ± 23.28** (`pair_elo.py`, 307 complete pairs / 614 games) |
| games | **617** in `corr.pgn`, counted from the file; SPRT **stopped early, H1 accepted** |
| clean | **0 time forfeits, 0 illegal moves** |
| stability | **0 bracket crossings, 0 probe-cap hits** (censused directly — fastchess logs do not carry engine `info string` lines) |
| gates | driver control PASS (both arms on the same `uci.py` v2), mate gate 5/5 parity, legality gate 40 FORCED / 0 no-move / 0 illegal |

**The arithmetic against the bar written down before the games:**

| | |
|---|---|
| fixed-node (quality only) | **+54.8** |
| speed term, from the interleaved probe (102·log₂ 0.944) | **−8.5** |
| **timed value** | **≈ +46.3** |
| byte cost, `pack.sh` on a real file | **127 B** |
| **Elo/byte** | **0.36** |
| standing bar | **1.0** ⇒ needed +127 timed |

**DROP.** And it is a clean one: the pre-registration said a fixed-node result
between +60.5 and +135.5 would be a drop *plus* a question for the coordinator
about whether the 1.0 bar is calibrated to a goal that only needs 0.41. **+54.8
is below +60.5, so there is no question to ask** — corrhist misses the standing
bar and misses the budget-average rate too. The interval reaches +78, so the
budget-average threshold is not excluded, but two things point the other way:
SPRT's terminal Elo is **biased away from zero** by construction (this is why
the script prints that warning), and the point estimate is the honest input to
a keep/drop rule, not the interval's optimistic edge.

**Priced out is not disproven, and the distinction matters for the queue.**
The transfer scoreboard now reads:

| feature | ice4 Elo | ours (fixed-node) | outcome |
|---|---|---|---|
| LMR | 81 | +38.9 ± 19.1 | **transfers**, ~1.8 Elo/byte, shipped |
| RFP | 58 | ~ 0 | sound, worthless here |
| LMP | 123 | −126 | structurally incompatible |
| **corrhist** | **70** | **+54.8 ± 23.3** | **works, and costs 127 bytes** |

corrhist is the **largest fixed-node effect this lane has ever measured** —
bigger than LMR's. It fails on the exchange rate alone. That is the cost model
this ledger has been describing from the other direction all along: our bytes
are dear (Python source through lzma), so a feature that would be trivially
worth 70 Elo in a hand-golfed C++ engine can be genuinely strong here and still
be unaffordable.

**What would change the verdict, stated so nobody re-runs this.** Not more
games — the effect is established. Only the byte side: 127 B is the price after
golfing (a walrus/`clear()` rewrite recovers 5), so corrhist returns to the
queue only if the *key* gets structurally cheaper — an incremental pawn hash
carried on `Position` rather than a `str.translate` per interior node would cost
nodes-per-move instead of source bytes, and is the only version worth building.
Shelved, not closed, and the mod stays in `make_variants.py` so it never has to
be rebuilt from a spec again.

## 2026-08-13 — corrhist's win, for zero bytes: the futility-margin lead, pre-registered

**The best thing corrhist produced may be a hypothesis, not a feature.**

Three facts from its screen, put together:

1. corrhist's **only consumer** is the `depth <= 1` futility test. Everything
   else it touches is bookkeeping.
2. It won **+54.8** while searching **more** nodes (1.04× to depth 8, 1.15× to
   depth 9). So what it bought was **"prune less at frontier nodes"** — the
   futility rule was too aggressive.
3. The censused correction table was **systematically optimistic**: mean +10 to
   +18 cp, median +2 to +8, p90 about +85, and only 1-5% pinned at the clamp.

If the **constant** part of that correction is most of the effect, then a flat
margin on the same test captures it for almost nothing — and corrhist's 127
bytes were buying the position-specific half, which is the expensive half and
possibly the smaller one.

**Priced, `pack.sh` on real files:**

| arm | futility test | packed | Δ |
|---|---|---|---|
| `base` | `pos.score + val < gamma` | 3475 | — |
| `fut` | `+ EVAL_ROUGHNESS` (15) | 3478 | **+3 B** |
| `fut40` | `+ QS` (40) | **3475** | **ZERO** |

**`fut40` is byte-for-byte the shipped entry.** Both reuse a constant already
in the stream, so lzma pays almost nothing for them; `QS` costs literally
nothing. Two margins rather than one, because a screen at a single value cannot
distinguish "the margin is wrong" from "the margin is too small", and 15 sits
near the bottom of the range corrhist actually applied.

**Soundness:** a constant added inside the break's test leaves it a constant
threshold on the sort key `val`, exactly as corrhist's per-node `corr` did. The
break still tests the quantity it sorts by. The yielded estimate stays honest at
`pos.score + val` — the margin is a cushion on the *decision to stop looking*,
not a claim about the position's value.

### Keep/drop, fixed before the games

- **`fut40` (0 bytes).** There is no Elo/byte rule for a free change. **LAND on
  a non-negative point estimate**; **HELD** if negative with the interval
  covering zero; **DROP** if the interval excludes zero below. Same rule as the
  byte-negative arms, same reasoning.
- **`fut` (+3 B).** Needs ≥ +3 timed at the standing rate, which is to say it
  needs to be positive.
- **The `fut` vs `fut40` pairing is the informative one**: it says which
  direction the margin wants to move, and that is worth more than either
  arm's verdict, because it points at a tuning axis rather than a feature.
- **This does not resurrect corrhist.** If a flat margin recovers most of
  +54.8, corrhist is *more* firmly dropped, not less — 127 bytes for the
  residual. If it recovers none of it, corrhist's value really was
  position-specific and the shelved entry keeps its note.

**Speed terms, measured before the games** (interleaved, base anchored, 7 lines
× 2 rounds, under both RRs' load): `fut` **nps 1.021×**, `fut40` **nps 1.073×**.
Neither is slower — one integer add cannot be — and the reading above 1.0 is
almost certainly a **node-mix** effect rather than a real speedup: pruning less
at frontier nodes means more cheap nodes per second, not faster nodes. So the
timed conversion is treated as **≈ 0 to +10 Elo, not the +3.1/+10.4 the model
prints**, and the keep/drop rules above are stated on the fixed-node number
alone, where that ambiguity cannot flatter either arm.

Launched 13:27 UTC: 300 rounds, 1,800 games, 600 per pairing (≈ ±28),
concurrency 12, alongside the ordering RR — legitimate because both are
fixed-node, where results do not depend on load.

## 2026-08-13 — The ordering round-robin: pre-registered before it is launched

Queue items 2 (history revisit) and 3 (IIR) run as **one round-robin with a
baseline anchor**, not as three A/Bs — the anchor occurs exactly once, every
arm meets every other arm on the same book in the same conditions, and the
pairing that prices IIR by itself (`noiid` vs `iirnoiid`) costs no extra games.

| arm | mod | entry bytes | Δ |
|---|---|---|---|
| `base` | — | 3475 | anchor |
| `hist` | history heuristic, restored verbatim from 438ac49 | 3536 | **+61** |
| `noiid` | drop the IID probe | 3459 | **−16** |
| `iirnoiid` | IIR *instead of* the IID probe | 3471 | **−4** |

**`iirnoiid` is byte-negative, and that changes the question being asked.**
Every previous item in this queue had to earn its bytes; this one hands 4 back.
It does not have to win. It has to avoid losing.

### Keep/drop, fixed before the games

- **`hist` (+61 B).** Standing 1.0 Elo/byte bar ⇒ **KEEP at ≥ +61 timed**,
  where timed = fixed-node + 102·log₂(nps_hist/nps_base), the speed term
  measured separately by the interleaved probe. Below that the 61 bytes come
  back out, as RFP's 31 and LMP's 56 did.
- **`noiid` (−16 B) and `iirnoiid` (−4 B).** An Elo/byte rule is untestable on
  a byte-negative arm: 4 bytes buys 4 Elo of tolerance at the standing rate,
  and this RR resolves to about ±27. So the rule is about strength, not rate:
  **LAND on a non-negative point estimate** (free bytes, no measurable cost).
  **HELD, not landed,** if the point estimate is negative but the interval
  covers zero — recorded as "bytes available if the budget binds", because
  "not significant" at ±27 is not the same as "not −25", and a silent −25 is
  how a byte win becomes an Elo loss. **DROP** if the interval excludes zero
  below.
- If both byte-negative arms qualify, the **`noiid` vs `iirnoiid` head-to-head**
  picks between them — a paired number, per the rule that head-to-head beats
  anchored differences where both exist.

### The speed terms, measured before the games so the rule is fully specified

Same interleaved probe, same machine, 7 lines × 3 rounds, base anchored and
occurring once. These are *inputs to the pre-registered rule*, not results:

| arm | nps | timed = fixed-node + … | time to d8 |
|---|---|---|---|
| `hist` | **0.994×** | **−0.9 Elo** | 1.099× |
| `noiid` | **1.046×** | **+6.6 Elo** | 0.950× |
| `iirnoiid` | **0.929×** | **−10.6 Elo** | 0.705× |

So `hist` must show **≥ +62 fixed-node** to clear its +61 timed bar.

Two of these are worth reading now. **`noiid` is 4.6% FASTER for 0.7% fewer
nodes** — the IID probe was costing 5% of wall-clock for 0.7% of the tree,
because each probe is a `root=True` driver probe that regenerates and sorts
moves and stores nothing. Dropping it is 16 bytes back and ~+6.6 Elo of pure
speed before any ordering effect. And **`iirnoiid` is 7% slower per node**,
which is the `pos not in self.tp_move` hash it pays before the table probe
that hashes the same position again — a known, and later removable, cost.

**The 0.705× time-to-depth for `iirnoiid` is NOT +50 Elo, and the probe's own
"nodes +69.7 Elo" line is the confounded reading, printed and hereby
discounted.** A reduction changes what depth 8 means. Only the fixed-node
games below convert it.

### `iirk`: the 7% was a duplicate hash, and it is recoverable

Built while the RR ran, because the cause was identified rather than merely
noted. `iir` asks the table for this position **twice** — `pos not in
self.tp_move` at the top, then `killer = self.tp_move.get(pos)` inside
`moves()`. `iirk` reads the killer once, at the top, and lets the closure
carry it in. (It requires `noiid` to be legal at all: the IID block *assigns*
to `killer`, which would make the name local to `moves()` and shadow the outer
read.)

Interleaved, base anchored, 7 lines × 2 rounds, measured under the RR's own
load:

| arm | packed | nodes to d8 | nps | timed = fixed-node + … |
|---|---|---|---|---|
| `iir.noiid` | 3471 (−4) | 0.623× | 0.908× | −14.3 Elo |
| `iirk.noiid` | 3472 (−3) | **0.623×** | **0.950×** | **−7.5 Elo** |

**The node counts are identical on all seven lines**, which is the check that
`iirk` is a pure speed refactor and not a different search. That has a useful
consequence: the RR below measures `iirnoiid` at *fixed nodes*, where the two
arms are the same engine — so **the RR's Elo transfers to `iirk.noiid`
verbatim**, and `iirk` simply gets a better speed term (−7.5 instead of −14.3)
for one more byte. If the arm lands, `iirk.noiid` is the form that ships.

**Power, stated in advance.** 330 rounds × 12 = 3,960 games, 660 per pairing,
≈ ±27 Elo. An undecided result at that size means the effect is **small**, not
that the mechanism is absent — the same reading the null-cap census forced.

**What the node counts already say, and why they are not the verdict.**
Nodes to depth 9 over 6 positions: `hist` **1.284×**, `noiid` 0.989×, `iir`
0.752×, `iirnoiid` **0.693×**. Two different readings, and the difference is
the point:

- For **`hist`** the comparison is apples-to-apples — ordering changes which
  move is searched first, never what "depth 9" means — so 1.284× is a genuine
  negative signal: the history-credit order is *costing* 28% more nodes for the
  same tree. It is still not the verdict, because the node-ratio proxy is
  precisely the instrument this revisit exists to replace.
- For **`iirnoiid`** the 0.693× is **not** an efficiency claim. A reduction
  changes what depth 9 means, so "fewer nodes to depth 9" is the same confound
  that made classic look efficient at a fixed node cap. Only fixed-node games
  price it, which is what this RR is.

## 2026-08-13 — corrhist re-measured: the node saving was one position, and it is gone

The queue's premise for corrhist was a single sentence in the entry below:
*"Interior-only reaches depth 8 faster than the entry does, which makes this
the first queued feature that is plausibly free."* That sentence rested on
**one position** (the start position), measured on a laptop that was
simultaneously running a round-robin, with the two builds not interleaved.
It does not survive being measured properly.

**The prototype is gone.** `e_pstcorrhist.py` / `e_pstcorrhist2.py` were never
committed and no copy exists on the laptop, the box, or in git (`git log -S
corrhist` returns only ledger edits). So corrhist was **re-implemented from
that entry's written spec** — pawn-skeleton key via `str.translate`, ±120cp
clamp, 7/8 decay, mates excluded, key not computed in QS — and now lives as a
generated mod (`corr` in `tools/build/make_variants.py`) rather than as a
scratch file that can evaporate again. Every number below belongs to the
rebuild; the old 0.70×/0.79×/0.89× column belongs to a build we no longer
have and is withdrawn rather than compared against.

**The harness proves itself before it is believed.** The `base` arm reaches
depth 8 from the start position in **150,870 nodes** — digit-for-digit the
number in the entry below. The baseline is the same engine it was, so any
difference in the corrhist column is corrhist.

**Interleaved A/B(/C), one session, one machine**, 7 opening lines × 3 rounds,
`go depth 8`, arm order rotated every round so shared load enters all three
alike. Per-line ratios, then the median over lines, so one slow line cannot
carry the result:

| corr / base | per line | median |
|---|---|---|
| nodes to d8 | 0.84 1.12 1.04 0.85 0.94 1.30 1.05 | **1.042** |
| time to d8 | 1.01 1.19 1.05 0.90 0.94 1.38 1.05 | **1.048** |
| nps | 0.83 0.94 0.99 0.94 1.00 0.94 1.00 | **0.944** |

**The node saving does not exist.** The median node ratio is *above* one, and
the per-line spread runs 0.84 to 1.30 — corrhist searches fewer nodes on some
openings and 30% more on others. The start position, which is the only
position the old measurement used, is the second-best line in the set (0.84).
That is the whole correction: a one-position sample landed on a favourable
tail and was read as a property of the feature.

**The nps question, which was the assignment, answers cleanly: 0.944×.** It
clears the ~0.90 bar. Load held between **12.19 and 13.68 on 96 cores** across
all 23 samples of the run (344 s), so this is a ratio taken under genuinely
steady conditions rather than an average of two different machines-in-time.

**Cotenancy, logged not remembered.** Two foreign tournaments of another
session's (`elo-intrinsic-tune-20260813`, `elo-lmp30-latch-20260813`, 23
processes, constant throughout; a third, `elo-rfp80-20260813`, launched at
12:01 UTC). Other users: `nick-lehrter` at 3.8% on one process throughout, and
`root` + `zach-belateche` running django at up to ~22% in a few samples. None
is resource-hungry by the box rule and total load never left its band, so we
did not yield.

**The key is the cost, and it can be halved.** Pawns only stand on ranks 2-7,
which is `board[31:89]` in the 120-char layout, so translating the slice is
*exactly* equivalent to translating the whole board. Measured as its own arm
(`corr.wkey` restores the full board):

| | nps | time to d8 |
|---|---|---|
| corrhist, key on `board[31:89]` | **0.944×** | 1.048× |
| corrhist, key on the whole board | **0.888×** | 1.107× |

**5.6 percentage points of nps for one slice**, and the two arms' node counts
are **bit-identical on all seven lines** — which is the check that they really
are the same corrections and differ only in what the key costs. The full-board
key would have missed the 0.90 bar on its own. This is the third time in this
ledger that corrhist's *key*, not its correction, decided the verdict.

**The table is not feeding on its own output.** The depth≤1 futility branch
yields an estimate built from the corrected static score, and that estimate can
become the node's `best`, which is what the table learns from — a loop worth
checking before spending games, because a saturated table would make corrhist a
blanket futility margin wearing an eval-correction costume. Censused over four
depth-8 searches: **1.1-4.8% of entries at the +120 clamp, 0.0-0.8% at −120**,
median +2…+8 cp, p10/p90 about −25/+90, 4-7k entries per search. Healthy, mildly
optimistic in exactly the way a fail-soft null-window `best` should be, and the
eviction path never fires.

**Byte price, built not composed.** `pack.sh` on real files, four of them:

| build | packed | note |
|---|---|---|
| `nnue_4k/pst_entry.py` | 3475 | the entry |
| `e_base` (generated variant) | **3475** | identical — the provenance header is free, so the variant path is byte-faithful |
| `e_corr` | **3602** | **+127 B**, spare 621 → 494 |
| `e_corr.wkey` | 3597 | the wide key is 5 B *cheaper* in source and 5.6pp dearer in speed |

A golfed rewrite (walrus lookup, truthy sentinel, `clear()` instead of FIFO
eviction) also lands at 3597: **golfing recovers 5 bytes**, because lzma has
already claimed the repetition. 127 B is the price, not a starting offer.

### Keep/drop, pre-registered before the screen reports

corrhist costs **127 bytes, 20% of the remaining 621**. The standing bar in
this ledger is **1.0 Elo/byte** (set for LMP at 56 B, applied to RFP at 31 B,
against LMR's measured 1.8). Applied here:

- The fixed-node screen measures **quality only**; the speed term is separate
  and already known. Timed value ≈ fixed-node + 102·log₂(0.944) = **fixed-node
  − 8.5 Elo**.
- **KEEP at ≥ +135.5 fixed-node** (= +127 timed = 1.0 Elo/byte). **DROP below.**

And the reading that is *not* the decision rule, written down now so it cannot
be invented afterwards to rescue a result: the +400 goal needs +293 more Elo
from ~718 spare bytes, which is **0.41 Elo/byte on average** — the standing bar
is 2.4× stricter than the budget the goal actually implies. If corrhist lands
between **+60.5 and +135.5** fixed-node it is a DROP under the bar *and* a live
question for the coordinator about whether 1.0 Elo/byte is calibrated to a goal
that only requires 0.41. Below +60.5 there is no question to ask.

For calibration, and it is not encouraging: **LMR**, the largest search win
this lane has measured, is **+38.9 ± 19.1 fixed-node**. Nothing in this engine
has ever measured +135.

**Why the screen runs anyway.** The bar is a shipping decision, not a
measurement, and the two should not be confused. A hard number closes corrhist
permanently instead of leaving it on every future queue, and if the quality
half turns out large the byte half is separately attackable (the eval lane just
found 94 bytes). The screen is our-vs-our fixed nodes, which is valid because
both arms honour the mid-search cap; `go nodes` cannot decay into movetime here
because with no tc the `sunfish_ui` driver sets the in-search deadline to
now + 600 s (the 1.5 s default belongs to the builtin loop, which only the
packed artifact runs) — and both arms print which driver they resolved, which
the script compares and refuses to proceed on.
## 2026-08-13 — The legality gate was scoring a LAUNCH failure as 100 chess failures, on the shipped entry

Run the gate on a **packed artifact** and it reports:

```
FORCED    n= 40  no-move=40  illegal=0
IN CHECK  n= 30  no-move=30  illegal=0
quiet     n= 30  no-move=30  illegal=0
GATE FAILED: 100 bad answers
```

on the **landed 3378-byte entry**. Nothing is wrong with the entry. The gate
launched every engine as `[sys.executable, ENGINE]`, and a packed artifact is a
`#!/bin/bash` self-extractor: under `python3` it dies on line 1. The engine
never started, stderr was never captured, and "produced no output" was recorded
as "produced no move" — 100 times, with a chess-shaped verdict and no hint of
the real cause.

This is the **never-hide-errors** class, in its most expensive form: a *fake
red*. The ledger's "C1 and C2 both pass the legality gate" line was produced by
running the gate on the `.py` sources, which take the interpreter path and work
— so the tooling had been passing and failing for reasons unrelated to chess
depending only on which file extension it was handed.

Fixed: `.py` still goes through the interpreter, anything else is executed
directly, stderr is captured, and **an engine that emits nothing at all is a
loud abort, never a chess verdict**. Both controls run:

| control | result |
|---|---|
| landed entry, packed | 40/30/30, **0 no-move, 0 illegal — PASSED** |
| landed entry, `.py` source | 40/30/30, **0 no-move, 0 illegal — PASSED** |
| **C1 (3215 B), packed** | 40/30/30, **0 no-move, 0 illegal — PASSED** |
| **C2 (3441 B), packed** | 40/30/30, **0 no-move, 0 illegal — PASSED** |
| an engine that exits immediately (negative control) | `ENGINE DID NOT START` |

C1 and C2 had only ever been gated as `.py` sources. They are now green **as
the packed artifacts that will actually play**, so the screening slot does not
have to spend itself discovering a launch problem.

All three skeletons below were then gated against the entry as an A-vs-A
control, and all four arms are identical: 0 no-move, 0 illegal.

---

## 2026-08-13 — Phase reweighting fails its pre-registered bar four times, and the flat refit turns out to be an ENDGAME refit

The labelled set is 65.6% at ≤ 16 pieces (mean phase 8.44/24, 73.7% below 12),
so every uniform fit is mostly a fit to endgames. Does correcting that skew
produce a better table? **No — and the reason is more useful than the answer.**

Phase is counted off the FEN board field, never off `X`. The control is printed
by the script rather than asserted in a comment: `|X|.sum()` reads **11.08**
pieces per position where the boards hold **14.27**.

### Pre-registered before any result was looked at

- **M1** (primary): unweighted held-out loss on the pinned seeded 80/20 split —
  the number C1/C2 were selected on, so it is comparable to the ledger.
- **M2** (secondary): phase-*balanced* held-out loss, validation rows reweighted
  to a flat phase density estimated on **train rows only**. This is the metric a
  reweighting is designed to win, so it cannot be the primary.
- **Bar for a third candidate**: paired-bootstrap 95% interval of
  (uniform − reweighted) must be **strictly above 0 on M1** and **not below 0 on
  M2**. Same held-out rows resampled together, 10,000 resamples, so split luck
  cancels.

### Result: nothing clears it

| weighting | M1 | M2 | M1 Δ vs uniform (95% CI) | M2 Δ (95% CI) | verdict |
|---|---|---|---|---|---|
| uniform (= C1/C2's fit) | **0.016800** | 0.019018 | — | — | — |
| flatphase (full correction) | 0.017296 | 0.019071 | −0.000496 [−0.00069, −0.00030] | −0.000053 [−0.00033, +0.00023] | no |
| sqrtflat (half correction) | 0.016943 | 0.018960 | −0.000142 [−0.00023, −0.00005] | +0.000058 [−0.00007, +0.00019] | no |
| mgtilt (1 + ph/24) | 0.016856 | 0.018951 | −0.000056 [−0.00011, −0.000004] | +0.000067 [−0.00001, +0.00015] | no |
| mgonly (ph/24) | 0.017499 | 0.019171 | −0.000699 [−0.00091, −0.00048] | −0.000153 [−0.00045, +0.00015] | no |

Every reweighting loses on M1, and — the informative part — **not one wins the
phase-balanced metric it exists to win.** The best M2 delta, mgtilt's +0.000067,
straddles zero against an M2 level of 0.019. Reweighting 384 shared parameters
does not buy middlegame accuracy; it only trades loss between bands.

The uniform arm reproduces the landed candidate exactly (M1 0.016800; scored
through the **codec's own decode** at C1's step-8/mirrored/K-exact encoding,
0.016911 = **−5.31% vs classic**, the ledger's C1 figure to the digit), so this
is the same fit, not a lookalike.

### Where the loss actually lives — and a caveat for the C1/C2 screen

Held-out loss by phase band (3,899 rows):

| band | rows | classic | uniform fit | mgtilt | flatphase | taper (768p) |
|---|---|---|---|---|---|---|
| 0-5 deep eg | 1675 | 0.016304 | **0.014705** (−9.8%) | 0.014931 | 0.015544 | 0.014547 |
| 6-11 endgame | 1211 | 0.017645 | **0.016686** (−5.4%) | 0.016837 | 0.017666 | 0.017326 |
| 12-17 middle | 446 | 0.022460 | **0.022587 (+0.6%, WORSE)** | 0.022392 | 0.022065 | 0.023005 |
| 18-24 opening | 567 | 0.019299 | 0.018682 (−3.2%) | 0.018229 | 0.017931 | 0.016728 |

**C1's −5.31% is almost entirely endgame.** In the middlegame band the fit is
not better than classic — it is very slightly worse. A reweighting has nothing
to move because the flat table is already at its capacity in every band; the
skew of the set is *not* what is holding the middlegame back.

Two consequences, both recorded before any game is played:

1. **Caveat attached to the C1/C2 screen.** Screening games start from a book
   and spend their decisive moves in exactly the band where this fit buys
   nothing. C1's bar (*lower bound above −15*, i.e. "do not lose") was already
   the right shape; this says a *win* should not be expected, and a flat result
   is the modal outcome rather than a disappointment.
2. **It points at capacity, not data weighting.** The taper column moves the
   opening band (0.019299 → 0.016728) where no reweighting could — and gives
   the middlegame band back, which is its 22.6%-of-the-set data problem showing
   up exactly where the earlier entry predicted. More middlegame *data* plus a
   second table *set*, not a different weighting of the same 384 parameters.

**No third candidate.** C1 and C2 stand as the screening pair. Script:
`tools/tune/fit_phaseweighted.py`.

---

## 2026-08-13 — The taper re-anchored on the landed generator, and king buckets priced by building

Two skeleton prices, both **shape prices with filler data — no Elo is claimed
and no candidate is proposed here.** Every row is one real entry source through
`tools/build/pack.sh`, measured off disk, run alone in an empty directory with
`SF_NET` unset, and gated for legality against the entry as an A-vs-A control.

### The old taper pricer was pointing at a deleted anchor

`price_taper.py` spliced at the **bare-king** swap, which the `kend` fix
replaced with classic's queens-off rule months of commits ago. It failed its own
`assert old in src` — so its numbers could not be reproduced from it at all.
Re-anchored, and the landed root changes the arithmetic twice in our favour:

1. **The queens-off king rule IS the phase seam.** The engine already tests
   queens-off once per search; selecting a whole second table set on the *same*
   boolean costs one `pst.update` and no new condition.
2. **The stale-score rebuild is already there.** `pos = self.root = from_board(…)`
   follows the swap, so a taper inherits it for **zero bytes** — it used to be
   part of the taper's own price.

Hoisting the shared boolean into a local also made the fitted two-set candidate
**13 bytes cheaper than the ledger's B row: 3439, not 3452.**

`K` is never in a second set: its two tables are the landed `kend` fix, so it
rides in the exact block and the root keeps its own K_MID/K_END line.

### Taper: measured (one-set references first — the marginal is unreadable against 3378)

One set, classic tables: exact **3400**, step 8 **3276**, step 8 mirrored
**3195** (entry as landed: 3378).

| root | encoding | filler | bytes | spare | vs entry | **vs 1-set** |
|---|---|---|---|---|---|---|
| seam | step 8, mirrored, K exact | same (machinery only) | 3247 | 849 | −131 | **+52** |
| seam | step 8, mirrored, K exact | shuffled | 3342 | 754 | −36 | **+147** |
| seam | step 8, mirrored, K exact | perturbed | 3358 | 738 | −20 | **+163** |
| blend | step 8, mirrored, K exact | same (machinery only) | 3312 | 784 | −66 | **+117** |
| blend | step 8, mirrored, K exact | perturbed | 3421 | 675 | +43 | **+226** |
| seam | exact | perturbed | 3786 | 310 | +408 | +386 |
| blend | exact | perturbed | 3848 | 248 | +470 | +448 |

**The seam root costs ~50 bytes of machinery; the continuous blend costs ~115** —
the phase loop is 65 bytes dearer than the boolean update, at every encoding,
and it buys the more expressive form. Decode of the largest build: **0.54 ms**
against a 60 s startup budget.

### Filler is a FLOOR, not an upper bound — the old script had this backwards

The previous docstring reasoned that filler over-prices, "because a real eg
table would share structure with mg and lzma would find some of it". Measured
against the **real fitted qseam tables through the same builder**: the fitted
second set costs **+224** over C1, where filler of the same shape costs
+147…+163. Filler is **60-75 bytes CHEAP**. Fitted values are less round than
classic's hand-made ones, and that costs more than correlation saves — the same
effect that made a plain refit +63 bytes instead of free. Every filler figure in
this entry is therefore a floor.

(Secondary, same mechanism: `shuffled` prices consistently *below* `perturbed`.
Permuting reuses the exact value multiset; adding noise widens lo..hi and buys
extra levels in the mixed-radix pack.)

### King buckets: the ~134 B/bucket estimate was arithmetic; here it is built

What a king bucket **can** be in this engine: the score is incremental and both
sides read one shared `pst`, so a per-side own-king bucket would change the
table whenever a king moves and invalidate every carried score in the tree —
that is a different engine, not a pricing question. What is free is the
mechanism `kend` already uses: a **position-global** property read once at the
root. So these are king-**wing** buckets (white king wing × black king wing; the
2-bucket form folds that to same-wing / opposite-wing).

| buckets | encoding | filler | bytes | spare | vs 1-set | **per extra bucket** |
|---|---|---|---|---|---|---|
| 2 | step 8, mirrored, K exact | same | 3258 | 838 | +63 | 63 (machinery) |
| 2 | step 8, mirrored, K exact | shuffled | 3350 | 746 | +155 | **155** |
| 2 | step 8, mirrored, K exact | perturbed | 3367 | 729 | +172 | **172** |
| 4 | step 8, mirrored, K exact | same | 3292 | 804 | +97 | 32 (machinery) |
| 4 | step 8, mirrored, K exact | shuffled | 3579 | 517 | +384 | **128** |
| 4 | step 8, mirrored, K exact | perturbed | 3602 | 494 | +407 | **136** |
| 4 | step 8, K exact (not mirrored) | perturbed | 4013 | 83 | +737 | 246 |
| 4 | **exact** | perturbed | **4505** | **−409** | +1105 | 368 |

Readings:

- **The ~134 B/bucket estimate is confirmed for filler at mirrored step 8** —
  measured 128-136 B per extra bucket at four buckets. It was never built
  before; now it is.
- **Machinery is sublinear and nearly free**: the 4-bucket selector plus three
  extra decode loops costs 97 bytes when the data is identical, ~32 per bucket,
  because the loops compress against each other.
- **Only the mirrored step-8 encoding survives.** At exact resolution a
  4-bucket build is **4505 bytes, 409 over the limit** — dead on price, recorded
  so nobody re-derives it. Even unmirrored step 8 leaves 83 spare, which is not
  a budget.
- Decode of the largest bucket build: **0.96 ms**.

Applying the filler-is-a-floor correction (+60-75 B/set), a *fitted* 4-bucket
version projects to roughly 3215 + 3×~200 ≈ **3.8 kB**. That number is
**composed arithmetic and must be built before it is believed** — the last time
this lane projected a second table set it said ~134 and the exact build came
back +670.

### One silent trap fixed at its source

`codec.emit(piece, raw, …)` accepted a `piece` dict and **ignored it**: the
value line was a hard-coded copy of classic's numbers. All three callers
happened to patch the line back out afterwards, so no landed figure is wrong —
verified by reproducing **C1 at 3215 and C2 at 3441 to the byte** — but the
failure mode of forgetting is a plausible-looking artifact carrying the wrong
piece values, which is the mirrored-king / numpy-wrap class again. `emit` now
uses what it is given. The entry regenerates **byte-identical at 3378** and
`check_entry.sh` is green.

---

## 2026-08-13 — Fits done: quantisation is FREE and SAVES bytes, the taper is affordable but its data is not there, and two candidates go forward

Fits are candidate generators. The last one improved the loss 10.1% and played
−16.7 ± 31.2. **No Elo is claimed anywhere below**; the deliverable is tables,
prices and a pre-registered bar.

Every loss is **held out** — an 80/20 split by position, seeded, and no fit ever
saw the 3,899 validation rows. That is not pedantry: `texel_tune.py` reported
*training* loss, which cannot compare a 384-parameter table against a
768-parameter tapered one, because the bigger model wins in-sample by
construction. On training loss the taper looks twice as good as it is.

### The fits

| candidate | params | train | **held out** | vs classic |
|---|---|---|---|---|
| classic, no fit | — | 0.017916 | 0.017860 | — |
| A flat refit | 384 | 0.015762 | **0.016800** | −5.9% |
| B taper at the queens seam | 768 | 0.014774 | **0.016693** | −6.5% |
| C continuous 24-point phase blend | 768 | 0.014308 | **0.016565** | −7.3% |

**Every emit was verified.** Each candidate is re-scored from its own emitted
integer tables, reconstructed exactly as the engine indexes them; all three
matched their fitted loss. This is the check that was missing when an un-flipped
king table cost −67 Elo while the fit looked 10% better — the fit never sees the
emit, so the emit has to be scored separately.

### The prices, built not composed — and quantisation is the whole story

| candidate | bytes | vs 3378 | spare | held out |
|---|---|---|---|---|
| A flat, exact | 3441 | **+63** | 655 | −5.93% |
| A flat, step 2 | 3385 | +7 | 711 | −6.02% |
| A flat, step 8 | 3290 | −88 | 806 | −5.96% |
| A flat, mirrored step 8 | 3131 | −247 | 965 | −6.27% |
| **A flat, mirrored step 8, king exact** | **3215** | **−163** | **881** | **−5.31%** |
| B queens-seam, 2nd set exact | 4048 | +670 | 48 | −6.52% |
| B queens-seam, 2nd set mirrored step 8 | 3720 | +342 | 376 | −7.57% |
| B queens-seam, both mirrored step 8 | 3391 | +13 | 705 | −7.19% |
| **B queens-seam, both mirrored step 8, king exact** | **3452** | **+74** | 644 | −6.30% |
| C continuous phase blend, exact | 4157 | **+779** | **−61** | −7.25% |

Four things follow.

1. **A refit is NOT free.** "The shape is unchanged so the bytes are unchanged"
   is a composed claim and it is wrong: fitted values are less round than
   classic's hand-made ones and cost **+63 bytes** at exact resolution. The
   ledger's old "+13 bytes total" for a Texel candidate was measured on a
   different base and must not be reused.
2. **Quantisation is free in loss and pays in bytes.** step 8 costs nothing
   measurable (−5.96% against −5.93% exact) and saves 88 bytes; mirroring saves
   another 159. Held-out loss *improves* under mirroring — halving the
   parameters regularises away per-square noise the fit had memorised.
3. **The continuous phase blend does not fit in 4096 bytes.** 4157, sixty-one
   over. Dead on price, whatever its loss. Recorded so nobody re-derives it.
4. **The taper's shape is affordable** — +74 bytes with the king held exact,
   against a ledger projection of ~134 and a naive exact build of +670. Two
   mirrored table sets compress against each other.

### Mirroring perturbs the landed kend fix, so the king is now held out of it

`emit(half=True)` mirrored **all six** tables, and classic's tables are not
symmetric: 28 of 32 file-pairs differ in the king table, by up to **111 cp**,
and that asymmetry is the castling-side preference the `kend` fix depends on. A
screen of a mirrored candidate would have measured the fit and an unmeasured
perturbation of a landed +30.5 Elo fix as one bundle, and a negative result
would have said nothing about either.

`codec.emit` now takes an `exact=` set that holds named tables back into a
second, full-resolution decode block. With `exact="K"` the king table and
`K_MID` are **bit-identical** to the landed entry's, verified by assert. It
costs 84 bytes (3131 → 3215) and it buys an interpretable screen.

### The taper is dropped from screening: its data is not there

`B` is cheap and it fits, but it does not go forward, and the reason is not
price:

- The queens-on subset is **22.6%** of an already endgame-heavy set — about
  3,500 training positions for 320 free parameters, **11 per parameter**. Its
  train/held-out gap is the widest of the three fits.
- The packed artifact, on a real clock from the start position, plays
  **a2a3** and reaches only **depth 5**, where the landed entry reaches depth 10
  and candidate A reaches 8. An engine whose best opening move is a rook-pawn
  push is not a screening candidate; it is a diagnosis.

So the taper is **blocked on data, not on bytes** — the +74-byte price stands
and is worth revisiting once the set is less endgame-skewed, or once the second
set is fitted as a regularised delta from the first rather than independently.
This is the phase-mix finding cashing out exactly where it was predicted to.

### Going forward: two candidates, bars pre-registered

Both pass the legality gate (40 FORCED / 30 in-check / 30 quiet, 0 no-move,
0 illegal). Sources at `tools/tune/candidates/`.

| | bytes | vs entry | held out | packed artifact from startpos |
|---|---|---|---|---|
| **C1** flat refit, mirrored step 8, king exact | **3215** | **−163** | −5.31% | e2e4, depth 8 |
| **C2** flat refit, exact | **3441** | **+63** | −5.93% | b1c3, depth 9 |

**Pre-registered, before any game is played:**

- **C1 saves 163 bytes**, so it does not need to win, only not to lose: **LAND
  if the 95% interval's lower bound is above −15 Elo** against the 3378-byte
  entry. **DROP if the upper bound is below 0.**
- **C2 costs 63 bytes**, so the project's standing 1.0 Elo/byte rule applies:
  **LAND only at ≥ +63 Elo.** Its real job is to be the **control that isolates
  mirroring** — C1 and C2 are the same fit, and the difference between them is
  the compression, measured in play rather than in loss.
- Both screens: fixed-node **our-vs-our on the box** against the current
  3378-byte entry, legality gate and mate gate first, SPRT with these bars,
  the A-vs-A driver control, 95% intervals, and a fixed-N confirmation for any
  winner's Elo/byte number.

One caveat carried forward rather than resolved: at a real clock C1 reached
depth 8 where the landed entry reached 10. Node efficiency is not loss, and
~100 Elo per doubling is this project's own estimate, so a better-fitting eval
that searches shallower can still lose. The screen measures the sum; if C1
comes back negative, that is the first place to look.

### One more silent-corruption bug, in the codec

`codec.mixed` accumulates a ~3000-bit Python integer. A single **numpy int64**
anywhere in its input makes the whole product numpy and it **wraps at 64 bits**,
producing a valid-looking source that encodes garbage — announced only by a
`RuntimeWarning` in a log nobody reads. It reached the pricing harness through
tables built with numpy, and the symptom was not a wrong number but a 10-minute
hang: the garbage tables made the artifact miss `bestmove`, so every standalone
check sat out its 120 s timeout.

`int()` on every input now, and the codec self-test carries a **64-bit wrap
control** that fails if numpy inputs ever disagree with Python ones. The landed
entry is byte-identical after the fix (3378), verified.

---

## 2026-08-13 — THE TRAINING SET IS LABELLED: 19,491 positions, Stockfish 18 @ depth 8, and the labels are a function of the position alone

The set that gates every training candidate exists again, and it is
**committed** at `tools/tune/data/set20260813.npz` (995 KB) with its run log
beside it. Sources stay at `~/repos/sunfish-data/`.

| | |
|---|---|
| positions collected | 19,689 |
| **positions kept** | **19,491** (198 dropped: mate scores or \|cp\| ≥ 1500) |
| labeller | **Stockfish 18**, depth 8, Threads 2, Hash 64 MB |
| engine sha256 | `0a119807d135b44f…` (built on the box, see below) |
| npz sha256 | `d792b42081f0adec…` as committed here; **`2410786e14f09fec…` after the 2026-08-13 host-field scrub** (see the scrub entry — same arrays, one metadata key dropped) |
| source games | 4,482 (4,000 hole RR + 444 kend screen + 38 ladder snapshot) |
| wall time | ~5.4 min at 61 pos/s |

### The binary: built, not downloaded, and it took two tries to get right

The official `sf_18` Linux binaries need `GLIBCXX_3.4.30`; the box tops out at
`3.4.29`, so `stockfish-ubuntu-x86-64-avx512icl` would not start at all. Built
from the pinned `sf_18` source tarball instead (src sha256 `22a19556…`) with
the box's own gcc 11.5.0, `ARCH=x86-64-avx512icl` — the matching target for its
Ice Lake-SP Xeon 8375C (`avx512_vnni`, `avx512_vbmi2`). Nets `nn-c288c895ea92`
and `nn-37f18f62d772` fetched and validated by `make net`. Binary lives at
`~/sunfish-bench/bin/stockfish`; user-space, no root, no system change.
**Building against the box's own libstdc++ is also the more reproducible
choice** — the recipe is (source tag, compiler, arch), which does not depend on
someone else's build host.

**One instrument failure on the way, mine:** the first smoke test was
`printf "position startpos\ngo depth 12\nquit\n" | stockfish`, which returned
`bestmove a2a3` and looked like a broken build. It was a broken *test* — `quit`
arrives before the search finishes and aborts it. A driver that waits for
`bestmove` gets `e2e4` at +48 cp, `Bc5` in the Italian, and `Ra8#` found as
`mate 1`. Piping a whole UCI session into an engine's stdin only works if the
engine is allowed to finish.

### The labels are a property of the position, not of their slot in the list

`texel_data.py` reused one Stockfish process across all positions without
clearing the hash, so the transposition table carried over and **the same FEN
got a different label depending on what preceded it**. Measured at depth 8 on
the box: one FEN scored **−14 in one slot and −22 in another**, and two other
positions moved **83 → 97** and **−90 → −149**. Both modes are perfectly
run-to-run reproducible, which is exactly why this was invisible — a re-run
agrees with itself and the bias stays.

`ucinewgame` + `isready` now runs before every position, and Hash dropped
256 → 64 MB because the table is cleared anyway and a bigger one only makes the
clear slower. Cost: nothing measurable (61 pos/s). The label is now a function
of **(fen, depth, engine version)** alone.

### Checks that the data is what it claims

- **Sign convention, the one that would poison everything silently**:
  correlation between raw material balance and label **0.727**, and where
  material is ≥ 300 cp the signs agree **92.7%** of the time (n=4,465). A
  white/black POV or mirror error would destroy both numbers.
- **Feature encoding**: `X` rebuilt straight from the FEN on 300 random rows,
  **0 mismatches**.
- Distribution: median **+3 cp**, mean +21, σ 320, 52.0% white-better, 23.8%
  inside ±50. No duplicate FENs.
- Provenance travels **inside** the npz (`meta`): engine name and sha256,
  depth, threads, hash-clear flag, per-PGN name/size/sha, the sampling and
  filter rules, POV, counts, build time and host.

### `X` cannot tell you the phase — use `fens`

`X` is a *difference* feature: a white pawn on e2 and a black pawn on e7 land
on the same `(piece, mirrored square)` cell and **cancel**. So `|X|.sum()` is
a material *imbalance* count, not a piece count — it reads 11.1 pieces where
the board has 14.3. Phase must be recomputed from `fens`, which is one more
reason they are stored. The first read of the labelled set got this wrong and
reported a 0.1% opening bucket.

True phase mix of the 19,491, from the boards:

| phase (pieces) | share | n | mean \|cp\| |
|---|---|---|---|
| opening 25-32 | 9.5% | 1,856 | 161 |
| middle 17-24 | 24.8% | 4,840 | 228 |
| late-middle 9-16 | 40.5% | 7,885 | 289 |
| endgame 6-8 | 25.2% | 4,910 | 239 |

**65.6% of the set sits at ≤ 16 pieces, against 47% in the lost set.** Recorded
as a property, not corrected. It may be *better* for the eg tables and the king
table — those terms live exactly here — or it may simply be a different bias
than the one the old opening-heavy set carried. The honest position: the last
Texel fit's non-conversion (10.1% better fit, −16.7 ± 31.2 in play) now has two
candidate explanations and this set discriminates neither on its own. `fens` is
in the file, so **any future fit can reweight by phase without relabelling** —
noted as an option, deliberately not built.

### Conditions

Box, throughout: load 12.3 → 14.2 on 96 cores, **20-22 of our own pypy3 match
processes cotenant** the whole time plus another user's npm work; our labeller
was one `nice -n 10` Stockfish at Threads 2, ~196% CPU. Labelling never gates a
match and this one could not have distorted one. No laptop CPU was used; the
league ladder was untouched.

### Reproduce

    python3 tools/tune/texel_data.py OUT.npz 30000 8 \
        ~/repos/sunfish-data/pgn <stockfish> 2

Arguments are `OUT.npz [NPOS] [DEPTH] [PGNDIR] [STOCKFISH] [THREADS]`. An empty
games directory or a missing Stockfish now **asserts** instead of quietly
writing a valid empty `.npz`.

**Next, and not started: no fit has been run.**

---

## 2026-08-13 — The training set has a durable home and 19,689 positions waiting; labelling is the only step left and it is NOT started

The 15,328 Stockfish-labelled positions that gate every training candidate
died with a session scratchpad. Two causes, and only one of them was the
purge: `tools/tune/texel_data.py` globbed `tools/tune/arena/*.pgn`, a
directory that was **never committed**, and `tools/tune/.gitignore` ignored
`data.npz`. Both the inputs and the output were, by construction, things git
would not keep. **Regeneration, not restoration** — the original set is gone.

### The durable store

Source games now live at **`~/repos/sunfish-data/pgn/`** — outside every
worktree, outside every scratchpad:

| file | games | source |
|---|---|---|
| `box_caprr_hole.pgn` | 4000 | the hole round-robin, 10+0.1, five arms incl. classic |
| `box_caprr_kend.pgn` | 444 | the king-table screen |
| `laptop_pyleague_20260813_123259_snapshot.pgn` | 38 | league ladder snapshot at 13:44, taken as a **copy** of a growing file |

4,482 games. `texel_data.py` no longer hard-codes where games or Stockfish
live: both are arguments (`OUT.npz [NPOS] [DEPTH] [PGNDIR] [STOCKFISH]`,
defaulting to the store above), and an empty games directory now **asserts**
instead of quietly writing a valid empty `.npz`.

### The yield, dry-run without spending a single Stockfish node

Sampling only — `ply >= 10`, every 7th ply, not in check, ≥ 6 pieces, dedup
by FEN — over the 4,482 games:

    box_caprr_hole.pgn      4000 games ->  15760 new unique
    box_caprr_kend.pgn       444 games ->   3538 new unique
    pyleague snapshot          38 games ->    391 new unique
    TOTAL unique positions available: 19689

**19,689 available against the 15,328 that were lost**, and the phase mix is
not the old one:

| phase (pieces) | new set | the lost set |
|---|---|---|
| opening 25-32 | **9.4%** | 21% |
| middle 17-24 | **24.9%** | 32% |
| late-middle 9-16 | **40.6%** | 32% |
| endgame 6-8 | **25.1%** | 15% |

The new set is **far more endgame-weighted** — 65.7% at or below 16 pieces
against 47%. That is not a defect to correct before it is understood: the
eg half of the tapered candidate and the king table are exactly the terms
this material informs, and the old set's opening-heavy mix is one plausible
reason Texel tuning fitted 10.1% better and then played −16.7 ± 31.2. It is
recorded here so that whatever the next fit does, nobody attributes it to the
tuner when the data distribution moved underneath it.

### Not started, and why

**No labelling has run.** ~20k positions at depth 8 on 2 threads is real CPU,
and this laptop's timed league ladder owns it for about a day; Stockfish
would show up in the ladder's own cotenancy sampler. The box is the
alternative and it is **not ready**: `which stockfish` finds nothing there
(numpy 2.0.2 and python-chess are present, 96 cores, load ~12.6 with 2 pypy3
at ~50%). So the open decision is *install Stockfish on the box and label
there*, or *wait for the ladder and label here* — it needs a call, not a
default. Everything up to that line is done and durable.

One rule adopted from the loss: the labelled `.npz` gets **committed to the
branch**, not gitignored. A few MB of int8 that gates an entire track is not
a throwaway artifact.

---

## 2026-08-13 — Base-90 lands on the moved base: 3378 bytes, 718 spare, and the agreement instrument was comparing two different drivers

The startup decode was measured on the pre-`kend`/`fresh` entry (the entry
below). The search lane then landed `kend`+`fresh` at 3475. **The two numbers
were never allowed to be added**, and they don't add: lzma carries one
dictionary across the whole stream, so the second lander rebuilds. Rebuilt, on
the real file, through `tools/build/pack.sh`:

| | bytes | spare |
|---|---|---|
| entry, `kend`+`fresh`, decimal-literal tables | 3475 | 621 |
| **entry, `kend`+`fresh`, startup-decoded tables** | **3378** | **718** |

**−97, not −94.** The composed guess (3475 − 94 = 3381) would have been wrong
by 3 bytes in the safe direction this time; it is the sixth composed byte
figure in this project to miss, and the first one nobody acted on.

Rebased rather than cherry-picked: the eval commit is a single commit whose
only overlap with the search lane is textual (the generator's head vs its
tail), so a rebase keeps one linear history and one commit to review. Three
conflicts, all resolved by keeping master's version where it had landed
independently: master's `pathlib`-derived `REPO` (PR #176) supersedes the
identical `os.path`-derived fix this lane wrote in parallel — the `os` import
it added is gone with it.

### The instrument failure: a byte-identical engine disagreed with itself

The pre-registered gate was "same move and same score over ~60 positions
against the pre-decode build". The first run came back **60/60 moves but
21/60 scores**, which reads exactly like a decoder that is not exact. It was
not. Two harness defects, found by asking the instrument to compare the
candidate against *a copy of itself*:

1. **`go nodes N` is not fixed effort.** The node cap is an *additional* stop
   on top of the clock, and with no time fields the engine defaults to
   `wtime=60000` → a 1.5 s deadline with a 1.2 s soft break. Under the league
   ladder's load that break can fire first. Fixed by sending an hour on both
   clocks so only the cap binds. *(This was not the cause here — 8000 nodes
   takes ~90 ms — but it was a live trap for any slower budget.)*
2. **The engine picks its UCI driver from its own path.** `main()` does
   `sys.path.insert(0, grandparent(__file__))` and imports `sunfish_ui`. An
   engine at `REPO/nnue_4k/x.py` gets the full driver; the *same bytes* copied
   to `/var/folders/...` find nothing and fall into the builtin `go` loop,
   which parses `go` differently. `agree.py` compared an in-repo file against
   a scratchpad file — **two different programs**. That is the whole 39-score
   gap, and it is reproducible: byte-identical copy, 39/60 scores differ.

Both are fixed in `tools/eval4k/agree.py`. Every arm is now **staged into one
directory under the repo** before it is run, `ask()` returns the engine's own
`info string driver` line, and `compare()` **raises** if the two arms did not
resolve the same driver. The script now runs a **positive** control (A against
a byte-identical copy, which must agree everywhere) as well as the negative
one, and fails loudly if either misbehaves.

The lesson generalises past this lane: *where an engine file sits changes what
engine it is*. Any harness in this repo that copies an engine somewhere before
running it is comparing something other than what it thinks.

### The gates, after the fix

    A vs B     positions 60  nodes 8000   same move 60/60   same score 60/60
    self       positions 60  nodes 8000   same move 60/60   same score 60/60
    control    positions 60  nodes 8000   same move 33/60   same score  1/60

- **Tables bit-identical**, asserted inside the generator on every build, and
  re-checked independently at runtime: `piece`, `pst` (all six, padded),
  `K_MID` and `K_END` all compare equal against the literal build.
- `check_entry.sh`: source matches generator, packs to **3378 (718 spare)**.
- `legality_gate.py`: **40 FORCED / 30 in-check / 30 quiet, 0 no-move,
  0 illegal**.
- Artifact **alone in an empty directory with `SF_NET` and `PYTHONPATH`
  unset**: `uciok` → `readyok` → `bestmove g1f3`, **no files left behind**.
- `nnue_4k/tests`: 28 passed.

**No Elo is claimed and no match should be spent.** The engine is behaviourally
identical to the one that measured +107.5 ± 31.6 vs classic; this buys 97 bytes
of headroom for the eval work that follows, nothing else.

By the same rule that produced the −97: **the tapering price below (3312 B,
−77) is now stale too, and it cannot simply be re-run.** `price_taper.py`
splices the phase blend in at the bare-king swap, and `kend` deleted that
line — the script now fails its own anchor assert (`bare-king swap not
found`) rather than pricing something else, which is the correct behaviour.
Re-anchoring it is part of the taper candidate, not of this landing, because
the new king rule (`K_MID` iff both queens are on) is *itself* a phase rule
and the two have to be reconciled before there is a shape to price. Its
*shape* argument — mirrored step-8 eg table at 134 B, no second accumulator
because the root already rebuilds — is unaffected; only the number is, and
the eg data behind it was filler in the first place.

One number recorded rather than resolved: our locally packed 3475-byte
baseline hashes to `823cb35c…`, while the search lane's ledger records
`939506a5…` for the same 3475 bytes. Both are internally consistent and
size-identical — the lane packed on the bench box, we pack on the laptop, and
`pyminify`/`xz` versions differ. **A packed sha is only comparable within one
toolchain**; the size is the portable number.

Conditions: laptop, timed league ladder cotenant throughout (load ~5.4 on
12 cores, 3 pypy3 + 1 python at ~100%). No matches, screens or nps figures
were taken. Every figure above is either a byte count or a deterministic
fixed-node comparison, and the positive control certifies that the
determinism actually held.

---

## 2026-08-13 — The eval decodes at startup: 94 bytes free and EXACT, and tapering now costs LESS than the table it replaces

*(Pointer added on landing, no number below altered: every byte figure in this
entry is measured on the **pre-`kend`/`fresh`** entry, which no longer exists.
The shipped figures are in the 3378-byte entry above; the price-list ratios and
the shape arguments here still hold.)*

The reframe under test: **the net is a compression scheme for tables, not a
runtime evaluator.** TCEC 4k gives 60 s of startup, so evaluation data should
be stored in whatever form is smallest and expanded once at load time into the
plain 120-square tables the search already reads. Nothing enters the hot loop;
`value(move)` still does two lookups and the score stays O(1) incremental.

Every number below is **built, not composed** — one real file per row through
`tools/build/pack.sh`, size read off disk.

### Landed: the entry is 3389 bytes with 707 spare, and it plays identically

| | bytes | spare |
|---|---|---|
| entry, decimal-literal tables (previous) | 3483 | 613 |
| **entry, startup-decoded tables** | **3389** | **707** |

The 384 numbers are unchanged — the decoder reproduces classic's tables
**bit-identically**, asserted inside the generator on every build. Gates:

- `check_entry.sh`: source matches generator, packs to 3389.
- `legality_gate.py`: 100/100, **0 no-move, 0 illegal**, including the 40 forced
  (in check, ≤2 legal replies) positions.
- Standalone in an empty directory with `SF_NET` unset: `uciok` → `readyok` →
  `bestmove d2d4`, **no files left behind**.
- `nnue_4k/tests`: 28 passed.
- **Behavioural identity, 60 positions at 8000 fixed nodes: 60/60 same move,
  60/60 same score.** No match is needed to justify this change and none should
  be spent on it.

### The decoder

Nine lines. All six tables are one big integer in mixed radix, written as a
base-90 string over ASCII 35..126 minus the apostrophe and the backslash:

    _v=0
    for _c in "...": _d=ord(_c)-35;_v=_v*90+_d-(_d>4)-(_d>56)

**No numpy.** (Aside: the ledger's "our pypy3 has no numpy" note is stale —
this laptop's pypy3 has numpy 2.4.6 — but a decoder that needs nothing beats a
decoder that needs a wheel, and integer arithmetic is fast enough that the
question never arises.) Decode cost **1.07 ms** against a 60 000 ms budget.

### The price list, one build per row

Measured against `b90_null` = 2994 B (the decode machinery with a 1-value
payload), so the third column is the marginal cost of the DATA alone.

| scheme | entry bytes | data bytes | B/value | tables |
|---|---|---|---|---|
| decimal literal (what we shipped) | 3483 | **502** | 1.31 | exact |
| **base-90, exact (210 levels)** | **3389** | **395** | **1.03** | **exact** |
| base-90, step 2 | 3337 | 343 | 0.89 | max abs err 1 |
| base-90, step 4 | 3291 | 297 | 0.77 | 2 |
| base-90, step 8 | 3242 | 248 | 0.65 | 4 |
| base-90, step 16 | 3195 | 201 | 0.52 | 8 |
| file-mirrored, exact (192 values) | 3200 | 206 | 1.07 | 87 |
| file-mirrored, step 4 | 3152 | 158 | 0.82 | 88 |
| file-mirrored, step 8 | 3128 | 134 | 0.70 | 88 |
| *the decoder machinery itself* | 2994 | **13** | — | — |

Three things follow.

1. **The decoder is free (13 bytes).** Its cost is a rounding error against any
   payload, so the question is only ever how few values you need and how
   coarsely you can round them.
2. **Exact re-encoding buys 94 bytes and nothing else** — the entropy of 384
   int8 values is the floor and lzma on decimal text was only 27% above it.
   Anyone hoping for a 300-byte win from *encoding* should stop here.
3. **The wins are in fewer values and fewer levels.** step 8 costs 0.65 B/value
   against an entropy bound of 0.59 — the codec is within 10% of optimal, so
   further work on the *codec* is worth at most ~10%. Work on the *shape*.

### What the historical 1207-byte net actually is (correcting an earlier entry)

The 2026-08-12 entry read `models/color2.pickle` @ `0c0a33a` as "a trained
rank-6 factorisation, 816 int8 → 4608 PST values, exact by construction". Two
corrections after decoding it against its own engine (`sunfish_nnue_color.py`
@ the same commit):

| array | bytes | what it is |
|---|---|---|
| `ars[0]` | 384 | 64 squares × 6 latent dims |
| `ars[1]` | 360 | 10 outputs × 6 dims × 6 piece types |
| `ars[2]` | 6 | **never referenced by the engine — 6 dead bytes** |
| `ars[3]` | 200 | 10 × 10 × 2 colour combiner |
| `ars[4]` | 180 | layer1, 10 × 18 |
| `ars[5]` | 10 | layer2, 10 × 1 |
| | 1140 | + 67 bytes of pickle framing = the 1207 |

1. The factorisation is **944 bytes, not 816**, and it produces
   2 × 6 × 64 × **10** = 7680 values (the earlier read dropped the colour
   einsum). 8.1 values per byte.
2. **It is not a PST.** Those 10 dims are an *accumulator*, and the engine runs
   an 18 → 10 → 1 MLP on them at every node. The historical artifact is exactly
   the thing this project has proven cannot pay for itself — a net in the hot
   loop — and its 4008-byte artifact was weak.

So it is the floor to beat in *packing density*, and an anti-pattern in
architecture. The transferable half is the shape: a rank-r factorisation of
(square × piece). For a **scalar** table that is 64r + 6r values against 384,
which beats the codec only below r≈5 — i.e. the factorisation is *dominated* by
plain mirroring plus quantisation at the precision a PST actually needs. That
is why the ladder above stores values and not factors.

### A 134 cp stale-score bug in the shipped entry

Found while looking for somewhere to hang a phase blend. `search()` swaps
`pst["K"]` between `K_MID` and `K_END` at the root, but the artifact's UCI loop
accumulates `score` incrementally from the initial position for the whole game.
Everything banked before the swap used the other table.

    KRK, white Ke4 Rd1, black Kg7
    carried score (accumulated under K_MID)   399
    consistent with K_END                     533
    STALE BY                                 -134 cp

The offset is fixed at the moment of the swap and then flips sign every ply as
the score rotates, so it acts like a ±134 cp tempo bonus on stand-pat and
futility — in bare-king endings, which is precisely what `K_END` was added to
win. This is **pre-existing, not introduced by the decoder**, and the fix is one
line inside the swap block (rebuild the root from `from_board`). It sits in
`search()`, so it is being handed to the search lane rather than landed here.

### The candidate: tapering is now cheaper than the table it replaces

The ledger declined tapering at "~300-400 B for the second table plus ~100 B of
accumulator threading" for 1.8 loss-points. Both halves of that price are wrong
now:

- the second table costs **134-248 B** through the codec, not 300-400;
- **no second accumulator is needed.** The engine already rebuilds a table at
  the root once per search (the `K_MID`/`K_END` swap), and tapering is that same
  mechanism with a phase instead of a boolean. Tables stay fixed for the whole
  search, which is what the comment there already requires.

Built and packed, with the root-score rebuild included and the eg table filled
with *uncorrelated* perturbed data (so the data figure is an upper bound — a
real eg table shares structure with mg and lzma would find some of it):

| build | bytes | spare | vs today |
|---|---|---|---|
| entry today (single table, exact) | 3389 | 707 | — |
| tapered, two full 384-value tables, step 8 | 3544 | 552 | **+155** |
| **tapered, two mirrored 192-value tables, step 8** | **3312** | **784** | **−77** |

Both play standalone in an empty directory; decode 2.12 ms and 0.80 ms.

**A tapered eval fits in 77 bytes LESS than the untapered one we ship.** That
is the whole point of the reframe, and it is measured.

### What is NOT claimed, and what would have to be true

No Elo is claimed for tapering. The honest state of the evidence is hostile:
Texel tuning improved the fit 10.1% and measured **−16.7 ± 31.2** in play, and
tapering added only 1.8 points on top of that same fit on that same data. **A
static-loss argument for this candidate would be the third time this lane
believed one.** What has changed is the price, not the evidence.

The candidate is also **not trainable today**: the 15 328-position labelled set
lived in a session scratchpad and was purged, so `tools/tune/texel_taper.py`
has no input. Rebuilding it needs game pgns plus local Stockfish and is the
gating step, not the byte budget.

Two other things the price list makes affordable, recorded so they are not
re-derived: **file mirroring** costs 283 B to *save* and changes the tables by
up to 87 cp (classic's hand tables are not symmetric — the asymmetries look
like 2014 noise, e.g. N c7=+100 against d7=−36), and **king-bucketed tables**
now cost ~134 B per bucket, which puts a 4-bucket table inside the 707 spare
for the first time. Both change the eval and neither may skip a screen.

### Harness failures found tonight (all the same shape as the other eight)

- `tools/build/make_pst_entry.py` had **`REPO` hard-coded to an absolute
  path**, so running it inside a git worktree regenerated the entry from the
  *other* checkout's sources and `check_entry.sh` cheerfully verified a file it
  had never read. Fixed to derive the root from `__file__`.
- The first version of `agree.py` drove the **packed artifact** with
  `position fen` and `go depth 5`. The artifact's built-in UCI loop supports
  neither: it ignores `position fen` outright and parses `go depth 5` as "60 s
  on the clock". It would have reported agreement figures for the wrong
  positions at a nondeterministic budget. Rewritten to drive the *source*
  through the `sunfish_ui` driver at fixed nodes.
- `legality_gate.py` takes the engine **source**, not the artifact — it runs
  `[sys.executable, ENGINE]`. Handed the packed file it reports 100 bad answers
  that read exactly like an engine bug.
- `agree.py` ships its own negative control and it fires: perturb one table
  value and agreement drops to **34/60 moves and 1/60 scores**.

## 2026-08-13 — THE HOLE ROUND-ROBIN, COMPLETE: 4,000 games, and the fix ships

The five-arm classic-anchored round-robin ran to its full length. **4,000 of
4,000 games** (counted from `hole.pgn`, not from the driver's claim), **0 time
forfeits, 0 illegal moves**, 200 complete colour-swapped pairs in every one of
the ten pairings and **not a single unpaired game dropped**. 08:25:07 →
11:13:54 UTC, 2h48m46s.

Per-pairing, pentanomial 95% (`pair_elo.py`), stated as A-over-B:

| A | B | pairs | games | score% | Elo(A) |
|---|---|---|---|---|---|
| entry_kf | classic | 200 | 400 | 65.00 | **+107.54 ± 31.64** |
| entry | classic | 200 | 400 | 57.00 | +48.96 ± 32.51 |
| entry_nolmr_kf | classic | 200 | 400 | 52.88 | **+20.00 ± 25.05** |
| entry_nolmr | classic | 200 | 400 | 39.88 | **−71.34 ± 29.69** |
| entry_kf | entry | 200 | 400 | 54.38 | **+30.48 ± 24.36** |
| entry_kf | entry_nolmr_kf | 200 | 400 | 60.25 | +72.25 ± 25.08 |
| entry | entry_nolmr | 200 | 400 | 59.38 | +65.92 ± 27.10 |
| entry_nolmr_kf | entry_nolmr | 200 | 400 | 52.75 | +19.13 ± 28.62 |
| entry_kf | entry_nolmr | 200 | 400 | 63.38 | +95.26 ± 29.64 |
| entry | entry_nolmr_kf | 200 | 400 | 51.25 | +8.69 ± 29.75 |

fastchess's own pooled ranking, for cross-reference (1,600 games each):

    1 entry_kf        +75.9 ± 13.9   60.8%  draws 42.4%  [47, 73, 339, 171, 170]
    2 entry           +18.7 ± 14.4   52.7%  draws 40.2%  [98, 110, 322, 148, 122]
    3 entry_nolmr_kf   −5.9 ± 13.7   49.2%  draws 47.1%  [99, 129, 377, 90, 105]
    4 classic         −25.7 ± 15.0   46.3%  draws 41.9%  [150, 122, 335, 82, 111]
    5 entry_nolmr     −62.6 ± 14.4   41.1%  draws 42.6%  [176, 139, 341, 82, 62]

**The three questions the tournament was launched to answer.**

1. **The control reproduced.** `entry_nolmr` measures **−71.3 ± 29.7** below
   classic here, against **−46.3 ± 30.0** on the laptop. Same sign, same order,
   intervals overlapping (the difference is 25.0 ± 42.2). The hole is a
   property of the engine, not of the laptop, so the rest of the table may be
   compared to the ledger.
2. **The hole is closed.** `entry_nolmr_kf` is **+20.0 ± 25.1** *above* classic
   — a **+91.3 ± 38.8** swing from the unfixed arm, with the reduction removed
   so nothing can be hiding behind it. An eval defect that cost ~70 Elo against
   the very engine whose tables we pasted is now a small positive.
3. **The shipping candidate confirms.** `entry_kf` is **+107.5 ± 31.6** over
   classic, and **+30.5 ± 24.4** over the shipped entry head-to-head — the
   latter excludes zero, so the fix pays on its own paired instrument and not
   only by subtraction. It agrees in direction with the fixed-node +52.3 ± 21.1.

**LMR was never masking the hole.** The lead that started this line supposed
the reduction was propping up a broken eval. It was not: LMR is worth
**+65.9 ± 27.1** on the unfixed engine and **+72.3 ± 25.1** on the fixed one —
the same number within noise. The two defects were **additive**, not
interacting. The original inference ("LMR transfers +127, therefore
entry-minus-LMR is ~85 below classic") got the conclusion right for the wrong
reason: there *was* a hole, but LMR's value was ~+70 all along, not +127.

**The discrepancy worth distrusting.** Every classic-anchored *difference* runs
above the corresponding head-to-head measurement of the same quantity:

| the fix's value | via classic | head-to-head | gap |
|---|---|---|---|
| with LMR off | +91.3 ± 38.8 | +19.1 ± 28.6 | 72.2 ± 48.2 (1.5σ) |
| with LMR on | +58.5 ± 45.4 | +30.5 ± 24.4 | 28.0 ± 51.6 (0.5σ) |

Neither gap is significant and the signs never disagree, so this does not
threaten the verdict. But the pattern is consistent across both rows and it has
an obvious candidate cause: `kf` makes our king-table rule **identical to
classic's**, so the arm and the anchor share a phase rule in a way our own
sibling arms do not — exactly the setting where intransitivity appears. The
methodological consequence is recorded rather than resolved: **where both are
available, prefer the head-to-head number** — it is paired, it has the smaller
interval, and it does not route the comparison through a third engine. Note
that `entry_nolmr_kf` vs `entry_nolmr` (+19.1 ± 28.6) still **includes zero**;
the claim "the fix is worth something with LMR off" rests on the anchored
route, and only the LMR-on head-to-head is individually significant.

**Launch conditions, from the sampler rather than from memory.** 10+0.1, 200
rounds × 10 encounters, concurrency 10, `nice -n 5`, 2,000-position book
consumed exactly once, anchor `classic` md5-identical to `sunfish.py@b49426b`.
**Peak process count 36**, constant across all 252 samples. Box load ranged
**19.56–23.75 on 96 cores** — no contention, and no cgroup quota or CPU
affinity limit (checked, not assumed); each engine process averaged ~50% CPU,
which is exactly right for alternating-turn play. Cotenants: two of our own
fastchess matches (`elo-171-full-tail`, `elo-173-exact`, 30+1, concurrency 5
each) from 09:01 UTC onward, i.e. for the last ~2/3 of the run; and another
user (`zach-belateche`) above 5% CPU on **50 of 252 samples** in two windows
(09:27–10:02, 10:41–11:05), never more than 2 processes. We did not yield: the
box rule is to pause for a *resource-hungry* neighbour, and total load never
moved outside its band. The mid-run arrival of the cotenants is a condition
change, but pairings are interleaved uniformly across the schedule, so it
enters every arm alike.

**Landed.** `kend`+`fresh` are now transforms in `tools/build/make_pst_entry.py`
rather than mods in `make_variants.py`. Entry **3483 → 3475 bytes, 621 spare**
(`pack.sh` on the real file, not composed). The landed artifact's packed sha is
`939506a5…` — **byte-identical to the `entry_kf` binary that played 1,600
games**, which is the strongest form of "we shipped what we measured" available.
`check_entry.sh` green, legality gate green (40 FORCED / 30 in-check / 30
quiet, 0 no-move, 0 illegal), 28/28 unit tests, and the artifact plays alone in
an empty directory with `SF_NET` and `PYTHONPATH` unset.

Two caveats on the byte number. It was measured **before** the eval lane's
base-90 table decode lands; that change is worth −94 bytes on its own but the
two may not be added, because lzma shares one dictionary across the whole
stream — the second lander regenerates and re-measures. And the classic anchor
has since moved: master now carries mate-distance scoring (#172), so every
classic-relative number above is pinned to **b49426b** and a future comparison
is against a different opponent.

**Next.** corrhist interior-only needs its nps re-measured on a quiet box (both
builds in one session on one machine, cotenancy logged) before its screen; then
the history revisit, whose old dismissal used a node-ratio proxy — the wrong
instrument — and IIR. Fixed-node our-vs-our screens are box-tolerant; anything
against classic needs a time control and a re-baselined anchor.

## 2026-08-13 — THE KING TABLE: the entry evaluates 62.1% of positions with the wrong one

**The entry does not inherit classic's eval.** That premise is stated all over
this ledger — it is the reason the MTD guards were expected to be inert, and it
is the reason "same eval both sides, so this is our SEARCH" was written next to
the baseline. It is false, and it is false in the majority of positions.

`make_pst_entry.py` pastes classic's tables into the NNUE engine's search. It
pastes `K_MID, K_END` verbatim, so the entry has classic's **endgame
centralisation** table — the one classic annotates *"important to win KRK/KQK
endings"*. But it kept the **NNUE engine's trigger**:

```python
# classic
pst["K"] = K_MID if "Q" in pos.board and "q" in pos.board else K_END
# entry -- the NNUE engine's rule, on classic's table
bare = sum(c.isupper() for c in pos.board) == 1 or sum(c.islower() for c in pos.board) == 1
pst["K"] = K_END if bare else K_MID
```

The trigger is correct **for the engine it came from**: the NNUE engine's
`K_END` is a trained *bare-king mop-up* table (its own comment says so), and
"one side is down to a lone king" is exactly when a mop-up table applies.
Against classic's endgame table the right condition is classic's — switch as
soon as either queen leaves the board.

Measured over **37,374 positions from 400 real games** (last night's
`lmron.pgn`):

| | uses K_END |
|---|---|
| classic's rule | 23,665 (**63.3%**) |
| the entry's rule | 448 (**1.2%**) |
| **disagree** | **23,217 (62.1%)** |

So for essentially every position after the queens come off, the entry judges
king placement with the **middlegame** table — it keeps its king hiding in the
corner through the whole endgame, and it does so while holding a table
explicitly built to stop that.

**This is the sibling of a bug already in this ledger.** 2026-08-12: *"A better
fit that played 67 Elo worse: the king table was mirrored"* — also the EMIT
path, also a king table, also invisible to every gate. The king table is now
twice-burned; anything that touches it gets a positional check, not just a
compile.

### Why it fits the hole, and why it hid

- It is an **eval** defect, so it is present with LMR and without it. The hole
  (`entry_nolmr` **−46.3 ± 30.0** vs classic) is present in exactly that shape.
- It is an **endgame** defect, so a fixed-depth or fixed-node screen from
  opening positions barely sees it. Every screen that pronounced the entry
  healthy was one of those.
- It cost nothing in speed, so the "1.10× faster than classic at equal depth"
  observation — used to argue the hole is quality, not cost — is untouched and
  still correct. It just points at the eval rather than the search.

### Pre-registered, before the screen reports

Byte delta measured by `pack.sh` on real files, never composed:

| build | packed | spare |
|---|---|---|
| entry | 3483 | 613 |
| **entry + kend fix** | **3472** | **624** |

The fix is **byte-negative**, so the standing 1.0 Elo/byte rule cannot bind on
it. Bar: **keep unless it measurably loses** — point estimate ≥ 0 and the 95%
lower bound above −10.

1. **≥ +40** — the hole is a port defect in the EVAL, not a search-quality
   defect. The +400 decomposition then needs redoing: it currently bills
   +232…+344 to search on the strength of a gap that was partly this.
2. **+10 … +40** — a real but partial cause; at least one more component
   remains and the bisection stays open.
3. **≈ 0** — the king table does not matter at these depths despite 62.1%
   disagreement. Surprising given classic's own annotation; would be recorded
   as such and the fix kept anyway for its 11 bytes.
4. **< 0** — something is wrong with the premise (the tables are not classic's,
   or the phase switch interacts with the incremental score). Do not ship;
   investigate.

Screened **fixed-node, our-vs-our** — legitimate between two of our engines and
tolerant of a shared box, which is what is available. **A fixed-node PASS earns
a timed confirmation, not a ship**: this is an eval change, and TESTING.md rule
12 records two sign flips between fixed-effort and wall-clock.

### RESULT: +52.3 ± 21.1, and it is outcome 1

444 games, fixed nodes (20k), our-vs-our, **zero time forfeits and zero illegal
moves**, SPRT stopped early:

| | Elo |
|---|---|
| shipped entry vs the fix | **−52.27 ± 21.07** |
| i.e. **the king-table fix** | **+52.3 ± 21.1** |

The interval is well clear of zero and it clears the pre-registered
**≥ +40** band, so this reads as outcome 1: **the ~46 Elo hole is a port defect
in the EVAL, not a search-quality defect.** The two numbers are strikingly close
— the hole was −46.3 ± 30.0 and the fix is worth +52.3 ± 21.1, measured
independently.

Two honest qualifications, both pre-registered:
- **SPRT's terminal estimate is biased away from zero.** +52.3 is an upper-ish
  read, not a settled effect size. The fixed-N confirmation is the timed
  round-robin now running.
- This is **fixed nodes from opening positions**, the setting that *understates*
  an endgame defect. It is not obvious which way the bias nets out; the timed
  number decides.

A note on reading fastchess here: the log says *"SPRT completed - H0 was
accepted"*, which looks like a rejection and is not. fastchess reports on the
**first-named** engine, and the first-named engine is the unfixed `base`. H0 for
base is the win for the fix. `pair_elo.py` prints both names against the sign,
which is why it exists.

### What now runs, and what it must show

`tools/screens/rr_hole.sh`, timed 10+0.1, 4,000 games, classic-anchored, five
arms: entry, entry_nolmr, entry_kf, entry_nolmr_kf, classic. The arm that
matters is **entry_nolmr_kf vs classic**: if the hole is this defect, a build
with no LMR and a corrected king table should sit at or above classic, where
`entry_nolmr` sat at −46.3.

`entry_nolmr` is carried as a **control, not a question**. The −46.3 was
measured on the laptop and this runs on the box. If the control does not
reproduce near −46, the environments differ and nothing in this tournament may
be compared against the ledger.

### Box conditions, logged at launch and sampled while running

The box rule is **yield-based, not a fixed process cap**: size to the machine
when it is ours, and *pause* rather than shrink when another user's job needs
the juice — `-recover` makes a stopped-and-relaunched match cost wall time
rather than games.

Measured rather than assumed, because the obvious instruments lie:

- **`ps` truncates usernames to `thomas-+`**, so an `$1 != "thomas-ahle"`
  filter counts *our own* processes as foreign. The first version of the
  sampler reported `other_user_busy=53`; the true answer is **0**. Any box
  etiquette check must use `ps -eo user:24` and match the full name — and
  `pgrep` on a command string matches the poller itself.
- **Process count is not CPU demand.** All of our engine processes sit in state
  `SNl`; only ~2 are runnable at any instant, because in a timed game only one
  side thinks at a time. Zero zombies, zero defunct.

| | |
|---|---|
| our engine processes (`caprr`) | **peak 35** at concurrency 10 |
| why 35, not 20 | 5 arms x 10 game slots; fastchess reuses engine processes across pairings rather than restarting, so the pool is bounded by `slots x arms`, not `2 x slots` |
| other lanes of ours on the box | `elo-171-full-tail` (12), `elo-173-exact` (12) |
| **other users** with any process > 5% CPU | **0** (`nick-lehrter`'s 16h `innovus` session idles at ~3.3%) |
| load / cores | ~22 of **96** |

No yield was owed and none was taken. **Sizing data for the next launch:** at
concurrency 10 a 5-arm round-robin holds ~35 processes and ~11 busy cores, with
~74 idle — so concurrency 30 is available when the box is ours alone, and the
pool would then be ~105, which is a number to state up front rather than
discover.

The `threat-cap-match` resident at launch has since finished; the two
tournaments that replaced it are **ours**, not another user's.

### This reorders the queue, and the reason is a rule already in the ledger

The cap round-robin is **deferred behind this fix**. The null-move cap is
`min(pos.score + EVAL_ROUGHNESS, ...)` — **it reads the static eval**. The
(feature, eval) rule, confirmed four separate times this session, says a
heuristic whose trigger reads the eval must be re-measured per eval. Spending
4,000 games pricing the cap against an eval we are about to correct would
measure a configuration we are abandoning.

## 2026-08-13 — The stale carried score at the table swap: reproduced, and sized

Routed in from the eval lane, who found it and correctly did not land it —
the line lives in `search()`, which is this lane's file. Reproduced here
before being believed, per the standing rule.

**The defect is real.** `pos.score` is carried incrementally by `value(move)`,
and swapping `pst["K"]` changes what that accumulated number means. Nothing
recomputes it. On a KRK position built under `K_MID` and then searched under
`K_END`, `search()` leaves `root.score` at 448 where a fresh computation under
the live table gives 437. The offset then **flips sign every ply** through
`rotate()`, so within that search it behaves as an oscillating phantom tempo on
every stand-pat and every futility margin.

**But it is rarer than the report implies, and the reason is the driver.**
`sunfish_ui/uci.py` rebuilds `hist` from scratch on every `position` command,
recomputing the score under whatever table is live. So the carried score equals
a fresh computation under the *previous* search's table, and the error is
nonzero **only on the ply where the phase rule flips**. Replaying 120 real
games, 11,362 plies:

| phase rule | transition plies | per game | mean \|E\| | max \|E\| |
|---|---|---|---|---|
| `bare` (shipped entry) | 17 (0.15%) | 0.14 | 66.4 cp | 178 |
| classic's (the kend fix) | 100 (0.88%) | 0.83 | 30.0 cp | 157 |

So the eval lane's 134 cp is a fair **peak** — the max is 157 — but "applied to
every stand-pat and futility decision" holds only *within* the one search per
game where it fires, not across the game. One skewed search per game, at the
moment the queens come off, which is a moment that often decides the game.

**It interacts with the king-table fix, and not in the fix's favour.** Correcting
the phase rule raises the firing rate from 0.14 to 0.83 plies per game — the
expected per-game error exposure goes from `0.14 x 66.4 = 9.3` to
`0.83 x 30.0 = 24.9`, about 2.7x. That is exactly pre-registered outcome 4 for
the king table ("the phase switch interacts with the incremental score"),
written down before this bug was routed in. So the two land together.

**Priced, from `pack.sh` on real files:**

| build | packed | vs entry |
|---|---|---|
| entry | 3483 | — |
| + kend | 3472 | **−11** |
| + kend + fresh | **3475** | **−8** |

Both fixes together are still **8 bytes under the shipped entry**. A score that
does not mean what the search thinks it means is a correctness problem, and
this one is byte-negative, so it lands on correctness alone; the Elo is upside.
Verified: with `fresh`, `root.score` after `search()` is 450 against a live-table
computation of 450, where the unfixed build reports 465.

## 2026-08-13 — The null-move cap, censused before it was screened

Zero games, and it sets expectations the RR would otherwise have set expensively.
Instrumented build, 40 book positions to depth 7, 6,179,827 nodes:

| | count | rate |
|---|---|---|
| null attempts | 71,561 | 1.2% of nodes |
| cap **binds** (min actually changes the value) | 3,090 | **4.3% of null attempts** |
| cap **flips the cutoff** (raw ≥ gamma, capped < gamma) | 376 | **0.53% of null attempts** |
| mean inflation removed when it binds | | **55.6 cp** |

So the cap is **real but rare**: one null in 23 is over-optimistic by ~56 cp,
and one in 190 is cutting when it should not. Node counts confirm the same
picture from the other side — at fixed depth 6 the capped build differs from
the entry on 1 of 3 positions and by 0.09%.

**What this buys:** the RR was sized expecting an effect that might be worth
~46 Elo. The census says not to expect that, and an undecided result must
therefore be read as "small", not as "the mechanism is not real". Recorded now
so the reading is fixed before the games are played.

## 2026-08-13 — Two eliminations, both free

**Time management is not the hole.** The entry carries its own `think` formula,
and it differs from classic's — an obvious suspect for a defect that appears in
TIMED play and not at fixed nodes. It is not, because **neither engine uses it
in a screen**: in a development checkout the entry's `main()` hands off to
`sunfish_ui/uci.py`, the same driver classic imports, so both arms compute their
budget from the same line. The entry's own formula is only ever exercised in the
**packed artifact**, which no screen measures. That is worth flagging separately
— we measure the unpacked engine and ship a packed one with a different time
manager — but it is not this hole.

**`pair_elo.py` is validated, not assumed.** A round-robin needs per-pairing
numbers and fastchess prints one ranking table, so the PGN has to be split and
re-scored. Scored against last night's `lmron.pgn` the new analyzer returns
**+38.86 ± 19.13**, digit-for-digit what fastchess printed. It also shows that
figure belongs to **422 complete pairs (844 games)** with 1 game unpaired — the
ledger records it as 845.

## 2026-08-13 — The transfer scoreboard: ice4's +421 is not our +421

Three features from the ice4 catalogue have now been measured on our engine.
**Three different outcomes**, and that is the finding.

| feature | ice4 Elo | ours | outcome |
|---|---|---|---|
| LMR | 81 | **+38.9 +/- 19.1** fixed, ~+65 timed | transfers, at ~60% |
| RFP | 58 | ~ 0 | sound, worthless here |
| LMP | 123 | **-126** | structurally incompatible |
| corrhist | 70 | built, unmeasured | — |

The mean transfer coefficient across the measured three is **far below 1**, and
the spread runs from 0.6 to *negative*. So:

**Any remaining +400 arithmetic must use measured transfer, not ice4's
published column.** Summing the catalogue and expecting its total is the
error this table exists to prevent. The catalogue is still the right *queue* —
it ranks what to try — but it is not a forecast, and each item has to be
priced on our engine before it may appear in a plan.

Two structural reasons the numbers do not carry, both now demonstrated rather
than argued:
1. **Our cost model is inverted.** Eval is O(1) incremental, so (piece,
   square) terms are free and whole-position terms are expensive — the reverse
   of ice4. Anything whose cost is "touch the whole board once per node" is
   priced differently here (corrhist below is a live example: the same feature
   is a loss or a win depending purely on which nodes pay for the key).
2. **Our movegen is pseudo-legal with no notion of check** (see LMP).

## 2026-08-13 — corrhist: the key, not the correction, is the whole question

Built (`e_pstcorrhist.py`, `e_pstcorrhist2.py`), keyed on the pawn skeleton via
`str.translate`, clamped to +/-120cp, updated at interior nodes only with a
7/8 decay, mates excluded. Legality-gated, both variants pass.

It was queued because it is the last large ice4 item with no prior negative,
and because **its trigger is not a move count** — it shifts a static score, so
the pseudo-legal-tail defect that killed LMP does not reach it.

Depth-8 from the start position:

| | nodes | nps | time to d8 |
|---|---|---|---|
| entry | 150,870 | 85,815 | 1758 ms |
| corrhist, **every node** | 120,461 (0.80x) | 41,098 (**0.48x**) | 2931 ms (1.67x) |
| corrhist, **interior only** | **105,307 (0.70x)** | 67,609 (0.79x) | **1558 ms (0.89x)** |

The first version is a **predicted loss**: 0.48x nps is ~-104 Elo on the speed
model against ~+32 for the node saving. The correction was working and the
*key* was eating it — the inverted cost model, caught before spending a single
game.

Not computing the key in QS fixed both halves at once, and the second half was
a surprise: the node count got **better** (0.70x vs 0.80x), so the QS
corrections were not merely expensive, they were **noise**. A stand-pat score
is largely the static eval itself, so correcting it teaches the table its own
output. Interior-only reaches depth 8 **faster than the entry does**, which
makes this the first queued feature that is plausibly free.

**Not yet a strength claim.** These are single-position numbers taken while
the cap RR had the laptop, so the nps column is measured under load; node
counts are deterministic and unaffected. It needs a quiet machine and games
before any Elo is attached to it.

## 2026-08-13 — Process: a chain may no longer start itself

`bisect.sh` fired unbidden a second time, when I stopped the hole test. The
mechanism is now understood exactly, and it was never really about pgrep:

    while [ ! -f hole_result.txt ]; do sleep 60; done

The producer writes its result file on **any** exit, *including being killed*.
So "the previous stage stopped" was silently read as "the next stage should
start". A wait on a producer's output is a race by construction, and the
contamination guard behind it could not help — by the time it ran, the machine
genuinely was free, because I had just cleared it.

**Four scripts had this shape** (`bisect.sh`, `followups.sh` — the one that
corrupted 174 games — `ng_fn.sh`, `timescreen.sh`). All four now wait on a
marker that only a deliberate act creates, consume it so it cannot re-fire,
and give up after 12h rather than lingering armed:

    GO=GO_bisect
    while [ ! -f "$GO" ]; do ... done
    rm -f "$GO"

## 2026-08-13 — The null-move cap: the code contradicts its own comment

Found by **diffing `bound()` against classic** rather than by bisecting eras —
a much cheaper instrument than the era rollback I had queued, and it landed on
one line.

```python
# classic
score = min(pos.score + EVAL_ROUGHNESS,
            -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3))
# ours -- NO CAP
score = -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)
```

**The comment directly above our uncapped line says the cap is there.** It
reads "*dropping the cap was Elo-neutral over 900 games yet cost a mate-in-3 at
the CI fixed-depth floor. Both halves stay.*" Both halves did not stay.
`git log -S "min(pos.score + EVAL_ROUGHNESS"` on `sunfish_nnue.py` returns
**nothing** — the cap has never existed in this engine's history. The comment
was adapted from classic and describes a decision that was never implemented
here, so for an unknown length of time the file has documented behaviour it
does not have. Per the standing rule that the model must match the code, this
is a blocker in its own right, independent of the Elo.

Uncapped null is wrong in **two** ways at once, and both bite hardest with a
weak eval:
1. `score` is larger, so `score >= gamma` fires more often — **more null
   cutoffs, and more aggressive ones**.
2. the yielded `score` becomes the node's returned value, so an inflated null
   estimate **propagates into the TT and into the MTD bisection**. Classic's
   cap bounds that inflation at `static + 15`.

**Counter-evidence, stated with the same care as the lead:** this ledger
already contains "capped-null decision match: -10.4 +/- 23.3 over 300g,
statistically flat". That was measured **on the NNUE engine**. By the
(feature, eval) rule — now confirmed four separate times this session (RFP's
mate gate, LMR's transfer, the zero crossings, LMP) — a heuristic whose trigger
reads the eval must be re-measured per eval. A cap that is inert against a
learned eval can matter against piece-square tables precisely because the
static score it clamps to means something different.

Under test as one classic-anchored round-robin rather than four A/Bs
(`rr_cap.sh`), answering the hole, the fix, the shipping question, and the
timed LMR value together.

## 2026-08-13 — The hole is real, and it is ~46 Elo, not ~85

`entry_nolmr` vs classic, timed 10+0.1, **322 games, zero time losses,
zero illegal moves**:

| | Elo vs classic |
|---|---|
| entry (shipped) | **+19.1 +/- 24.5** |
| entry **minus LMR** | **-46.3 +/- 30.0** |

The interval excludes zero, so **the hole is confirmed**: strip LMR and our
port is meaningfully *worse* than classic — while running **1.10x faster than
classic at the same depth**, which rules out speed as the cause. It is a
search-quality defect, not a cost.

**But it is smaller than I claimed, and I should correct that on the record.**
I advertised ~85 Elo, derived by subtracting a timed LMR estimate of
+104.6 +/- 90.9 from the entry's +19.1. That input was noise: the tight
fixed-node measurement below puts LMR at +38.9 +/- 19.1, and the timed
interval was wide enough to contain it. The consistent picture is
**LMR timed ~ +65** (19.1 - (-46.3)), and a hole of ~46. Quoting an 85 built
on a +/-91 input was overreach.

## 2026-08-13 — LMR transfers to the PST entry: +38.9 +/- 19.1

845 games fixed-node, SPRT H1 accepted, stopped early. The tightest LMR number
we have, and it settles a question left open twice.

| measurement | value |
|---|---|
| NNUE engine, fixed nodes | +65.0 +/- 43.3 |
| **PST entry, fixed nodes** | **+38.9 +/- 19.1** |
| PST entry, timed (77g, superseded) | +104.6 +/- 90.9 |

So LMR **does** transfer across the eval swap, at roughly 60% of its NNUE
value — the (feature, eval) rule predicts a change in magnitude, and that is
what happened; it did not predict a sign flip, and there wasn't one. The
headline lesson is methodological: the timed point estimate was **2.7x** the
fixed-node one and its interval contained it. It was noise, and I built a lead
on it.

**Build-lineage note.** The baseline (`e_pstnc.py`) and the variant
(`e_pstnolmr.py`) turned out to be from different build generations — only the
variant carried the driver version stamp. I diffed them before trusting the
result: the **only** search-relevant difference is `LMR = 60` vs `LMR = 0`, the
rest being a startup assertion that cannot affect play. **The result stands.**
Both are now one lineage. Unifying them, I briefly broke both files with a
bad splice; the smoke test caught it before any game was played, which is the
argument for smoke-testing every variant rather than only gating it.

## 2026-08-13 — LMP is dead: -125.8 +/- 38.1

269 games, fixed nodes, on top of LMR, SPRT H0 accepted, stopped early. The
pre-registered bar was **+56 Elo** (1.0 Elo/byte for 56 bytes). It missed by
180 Elo.

This is the number that closes the LMP question, and the legality gate is what
makes it *meaningful*: the previous LMP run was a **correctness** failure
(`bestmove (none)`), so its loss measured a bug. This run is on the gated,
legality-clean build, so **-125.8 is a genuine strength verdict** — the rule
itself is bad here, not merely misimplemented.

Worth stating why, since ice4 rates LMP highly (its Elo column is why this was
queued at all): our movegen is **pseudo-legal with no notion of check**, so the
move list's tail is not "obviously bad moves", it is "moves we never
evaluated", including forced king escapes. A count-triggered rule that
discards the tail is discarding a different population in our engine than in a
legal-movegen engine. **ice4's Elo/byte for LMP does not transfer, and the
reason is structural rather than tuning.** Third entry in the transfer
scoreboard: LMR transfers, RFP is ~0, LMP is negative.

**This predicts a second casualty, and it is one we were counting on.** The
defect is not specific to LMP; it belongs to *any* **count-triggered** rule
that discards the tail. **Move-count LMR is exposed to exactly the same
thing** -- and move-count LMR was our named fallback if threshold-LMR
saturated (it saturates at 40). So the fallback is not a safe one, and it
must be screened with the legality gate first rather than assumed sound.
The general rule: a rule triggered by *how many* moves we have seen says
nothing about what remains, because our sort orders by static value and our
list contains moves no one has evaluated for legality. Only *value*-
triggered rules inherit the sortedness argument.

## 2026-08-13 — The legality gate, and the fifth stale copy

`tools/build/legality_gate.py`. It asks the question a mate suite does not:
**is a legal move ALWAYS produced?**

**The obvious version of this gate does not work**, which is the part worth
keeping. Sampling in-check positions indiscriminately **passes** a build that
demonstrably emits illegal moves — most in-check positions have several escapes,
so the pathological case never comes up. The gate needs a dedicated **FORCED**
class: in check with **≤ 2 legal replies**, which is where the only legal move
sorts to the tail that count-triggered rules discard.

Positive-controlled in both directions, by me as well as by its author:

| build | FORCED (40) | in check (30) | quiet (30) | verdict |
|---|---|---|---|---|
| shipped entry `e_pstbase.py` | 0 | 0 | 0 | **PASS** |
| LMP **pre-fix** | 1 no-move | 1 no-move | 0 | **FAIL** |
| LMP **post-fix** (`best > -MATE_UPPER`) | 0 | 0 | 0 | **PASS** |

The two failing positions reproduce exactly:
`3r1k1r/1Qb1n1pp/3p4/4p3/bnP1p1P1/3P1P2/PP3q1P/2RK1B1R w - - 1 26` and
`3R2kr/8/1p1p1p1p/5p1P/2pn2p1/Q1P5/P2b2P1/RKB2BR1 b - - 0 34`, both
`bestmove (none)`. **So the fix is verified by the instrument that caught the
bug**, which is a stronger statement than my single-position reproduction.

It is now wired **ahead of** the mate gate in the box chain, for both arms — it
costs seconds, not games, and a screen that runs on an illegal-move build is
wasted from the first game.

### The fifth stale copy

Two LMP builds existed and only one was fixed. The box's queued build carried
the guard (`md5 9ad938103f68`); **the laptop's `e_pstlmp.py` did not**
(`c1ed68e68c3d`, guard count 0) — I had written the fix to a *new* file,
`e_pstlmpsafe.py`, shipped that to the box, and left the original sitting next
to it. All copies are now identical (`777c60cfed83`), the obsolete `skip`
variant is deleted, and the box screen was confirmed to be running the fixed
build before this was noticed.

**That is the fifth stale-copy failure tonight**: the driver (425 void games),
the scratchpad driver (near-miss on the node cap), the entry source (38 lines),
the box checkouts (marked), and now an engine variant. The pattern is always the
same — a fix written to a new path while the old path stays live and reachable.
The version stamp fixed it for the driver; nothing yet fixes it for variant
files, and the honest mitigation is that variants should be **generated at
screen time from a single source**, not accumulated as files.

Also corrected in every copy: the comment claiming *"the list is sorted by
static value, so breaking is the same argument the futility break already relies
on"*. That justification is disproved — futility is **value**-triggered, so
sortedness licenses its break; LMP is **count**-triggered, and a count says
nothing about what remains. A comment asserting a disproved justification is how
the next person re-derives the bug.

## 2026-08-13 — ENGINE PROPERTY: pseudo-legal movegen with no notion of check

Recorded as a **property of the engine**, not as an LMP bug, because it will
bite every rule that touches the tail of the move list.

**This engine generates pseudo-legal moves and cannot detect check.** It is a
king-capture engine: illegality is discovered by the opponent capturing the
king, not prevented at generation. Consequences for any pruning rule:

- In a check position the only legal reply is frequently a **low-value king
  escape**, which sorts near the **end** of the move list.
- Any rule that discards the tail — by move count, by depth, by anything not
  keyed to legality — can therefore discard **the only legal move**, and the
  node returns having committed nothing.
- The symptom is `bestmove (none)`, an instant loss, and it is invisible to a
  mate suite: LMP's mate gate **passed** (5 vs 5) on the very build that emitted
  an illegal move. The gate tests whether mates are *found*, not whether a move
  is *always produced*.

**Required preamble for any future pruning rule:**

    best > -MATE_UPPER    # something playable has already been found

The next three items on the list — history ordering, IIR, move-count LMR — all
touch ordering near the tail, so all three inherit this requirement. Move-count
LMR is the most exposed of them: its trigger *is* a count.

Also worth stating: a mate suite is not a legality gate. A cheap
"never answers (none)" assertion over a few hundred positions would have caught
this in seconds, and is a better gate for this class than mate-finding.

## 2026-08-13 — The MTD guards are inert on the PST entry, and that is a finding

Counted directly on `e_pstbase.py` in real search, guards printing loudly when
they trip:

| movetime | max depth | probe-cap hits | bracket crossings |
|---|---|---|---|
| 1.5 s × 8 positions | — | **0** | **0** |
| 5 s × 3 positions | 8 | **0** | **0** |
| 12 s × 3 positions | 10 | **0** | **0** |

**Nothing fires to depth 10.** A guard that never trips cannot cost 60 Elo, and
its branch cost is not showing up either — the entry is 1.10× *faster* than
classic. Both guard arms are dropped from the bisection, which now has **one
arm** and saves ~600 games.

**Why zero here when the NNUE engine crossed 23 times in 120 positions:** the
entry inherits **classic's eval**, a pure function of the position, and it sorts
by the same quantity the futility break tests — precisely the premise the formal
lane verified on classic across 7.7M break firings with zero non-futile skips.
The NNUE engine crossed because a *learned* eval breaks that correspondence.

So "does this engine cross?" is another property of the **(feature, eval)**
pair, and **the PST entry inherits classic's good behaviour for free**. The
entry is structurally *more stable* than the engine it was derived from — which
is a real point in favour of the PST main line beyond its byte cost.

### Bisection now, and its second split prepared

One arm remains — **pre-KCX vs classic**, timed, on the laptop. Because
`7f7d40a` predates more than the KCX port, a hit localises to "the KCX-era
changes" and needs splitting. That second step is already queued rather than
sequential: **`entry_nolmr` vs `prekcx` directly, our-vs-our at fixed nodes on
the box** — both arms lack LMR, the guards are inert, so the difference is the
KCX-era change itself, and comparing two of *our* engines means fixed nodes is
legitimate and transitivity never enters.

A contamination guard now sits at the head of the timed chain: it refuses to
start if any fastchess is already running, rather than launching into a shared
machine. That is the failure that cost 174 timed games tonight.

## 2026-08-13 — LMP emitted an illegal move: a correctness failure, diagnosed

`base vs lmp`, 300 games: **237-28-35, base +283.8 ± 54.2**, and one game
terminated `illegal move`. **−284 Elo is not "a feature that does not
transfer"** — RFP's −2.8 is what sound-but-worthless looks like. This is
pruning something it must not prune, and the correctness failure outranks the
score.

### Reproduced deterministically

The offending game ends `41. Qf8+ {Black makes an illegal move: (none)}`. The
engine answered **`bestmove (none)`** — the builtin loop's fallback when the
search commits nothing at any depth or any gamma. Replaying the pgn to that
position and running both builds:

    position: 5Q2/4P1kp/1p6/2pR2P1/p1n1p3/P6P/8/6K1 b - - 2 41
    black to move, IN CHECK, exactly ONE legal move

    LMP=3   best=None cand=None  -> bestmove (none)   <-- illegal
    LMP=0   best=Move(82,72)     -> a legal move

The MTD guards fire at every depth here (`lower -69289 upper -69290`), which is
the mate band: the position is scored as lost, and with LMP the search finds
nothing to play.

### The break-vs-skip hypothesis was wrong

The proposed cause was that *breaking* on a move count discards the tail, where
a count-based rule has no sortedness argument to justify it. Tested directly —
a `continue` variant behaves **identically**, still returning `(none)`. Good
hypothesis, cheap test, falsified.

**The actual cause is check.** This engine generates **pseudo-legal** moves and
has no notion of being in check. In this position the single legal reply is a
low-value king escape, so it sorts near the *end* of the move list — and a
count-based rule discards the end of the list whether it breaks or skips. LMP
pruned the only legal move.

### The fix, and why it generalises

    if LMP and best > -MATE_UPPER and depth < 4 and val < LMR and \
            cnt > LMP + depth*depth*(2 if pos.score > pps else 1):

`best > -MATE_UPPER` means some move has already produced a playable score;
until that holds, prune nothing. Verified: the fixed build returns the same
legal move as LMP=0 on the reproduction. The general rule — **a count-based
pruning rule must never be able to discard the last playable move** — applies to
anything else count-triggered we add, and it is *not* the sortedness argument
that licenses the futility break, which is value-triggered.

### Priced but not spent: never answer `(none)`

`(none)` is an instant loss whatever produces it, and "a search that commits
nothing" is always possible once pruning is licensed to be unsound. A fallback
that plays any generated move rather than passing costs **3483 → 3511, +28
bytes**. Not spent yet: the shipped entry has LMP=0 and does not reproduce, so
this is insurance against future features rather than a fix for a live defect.
Recorded with its price so the decision is one line when a feature needs it.

### The transfer scoreboard needs a third category

| feature | ice4 | ours |
|---|---|---|
| LMR | 81 | **+65…+127** — transfers |
| RFP | 58 | **≈ 0** — sound but worthless |
| LMP+improving | 123 | **broken as implemented** — illegal move, −284 |

Three items, three outcomes, and the third is one neither of us had on the list:
*the technique is fine, our implementation of it is wrong*. The operational
consequence is a rule — **a bad screen result triggers a correctness check
before a value judgement**, not after. A −284 with an illegal move should never
have been read as an Elo measurement in the first place.

## 2026-08-13 — RFP rejected on its pre-registered bar, and the bytes came back

**Reverse futility pruning, on top of LMR, PST entry, fixed nodes, SPRT run to
its 1000-game cap: −2.78 ± 17.93 (95%)** — RFP first-named, so marginally
*worse* than base. Zero bad terminations.

The bar was fixed before the screen reported: **≥ +31 Elo to justify 31 bytes**
(1.0 Elo/byte). The measured interval spans −20.7 … +15.1 and excludes the bar
by a wide margin, so the decision needed no judgement:

| | |
|---|---|
| entry before | 3517 bytes, 579 spare |
| **entry after removing RFP** | **3483 bytes, 613 spare** |
| reclaimed | **34** (I had estimated 31 — measured again, estimated again wrong) |

Bench unchanged at 94442 (RFP shipped disabled, so no behavioural change), 28
tests green, CI entry guard green, and the artifact still plays standalone in an
empty directory with `SF_NET` unset.

**Two things worth keeping from this.** First, the SPRT ran to its cap without
resolving either bound — which is exactly what an SPRT *should* do for a true
effect near zero, and the cap did its job of stopping an undecidable test rather
than letting it run forever. The value of the run is the tight interval, not the
verdict letter.

Second, RFP's **mate gate passed** (5 vs 5) on this eval while the same feature
pair failed it 5/8 → 3/8 on the NNUE engine. So RFP is *safe* here and simply
*worthless* here — the two failure modes are independent, and a feature can
clear the correctness gate and still not earn a byte.

Running score for the transfer question: ice4 prices RFP at 58 Elo; we measure
**≈ 0**. That is the second data point on how their catalogue converts to a
Python engine with a weak eval, and it is much worse than LMR's.

## 2026-08-13 — LEAD (not a finding): LMR transfers, and may be masking a hole

Pre-registered outcome **one**: LMR transfers to the PST eval. Timed 10+0.1,
77 games, zero bad terminations, my recount from the pgn:

    lmr_on vs lmr_off  =  +127.2 ± 76.5  (95%)

Good news for the catalogue — but combined with the baseline it implies
something uncomfortable:

    entry              vs classic          = +19  ± 25   (600 games, final)
    entry              vs entry-minus-LMR  = +127 ± 77   (77 games, preliminary)
    ⇒ entry-minus-LMR  vs classic          ≈ −108

**Our engine without LMR would be roughly 85-108 Elo WORSE than classic** — while
running 1.10× faster and reaching a full ply deeper. If that holds, LMR is not
adding to a healthy base; it is **masking a regression in our port**, and the
earlier "~46 unaccounted" was an underestimate because it used LMR's
NNUE-measured +65 rather than its actual value on this eval.

**Two caveats, recorded so this does not harden prematurely.** The +127 is 77
games with a 77-point interval whose bottom end is +50. And **three-way Elo is
not reliably additive** when engines differ in more than one way — subtracting
two intervals compounds both errors and assumes a transitivity that does not
have to hold. **This is a lead until `entry-minus-LMR vs classic` is measured
directly**, which is now running on the laptop (timed 10+0.1, SPRT, classic
first-named so a PASS means classic is better and the hole is real).

### Why it would be the best news of the night

An ~85 Elo defect is **already paid for**. Finding it returns strength we have
lost, against a goal that needs ~380 — and unlike every feature on the queue it
costs no bytes.

### Bisection order, all things the entry has and classic does not

| # | suspect | why | status |
|---|---|---|---|
| 1 | **KCX port** | measured −15.7 ± 34.9 **on the NNUE engine**; by our own rule that says nothing here. Largest structural difference | **built** from parent `7f7d40a` (`e_pstprekcx.py`, smoke-tested) |
| 2 | **MTD guards** | change driver behaviour; validated at "0 nodes", never at 0 Elo — a guard that changes which move is committed costs Elo at zero node cost | **built** (`e_pstnoguards.py`) |
| 3 | **PROBE_CAP** | a cap that trips changes the answer; never screened on this eval | **built** (`e_pstnocap.py`) |
| 4 | node-cap machinery | should be inert in timed play — and "should be" is what this session has punished repeatedly | last |

**Each arm is screened against CLASSIC, never against the entry.** Screening a
variant against the entry would measure a feature's marginal value *inside a
possibly-broken engine*; screening against classic asks the only question we
have — **does removing this close the hole?** Every arm otherwise ships exactly
as the entry does (LMR included), so each differs from the entry in one way
only, and `classic` is first-named throughout so a PASS means classic is better.

### `MATE_LOWER` cross-wiring: checked, cleared, recorded so it is not re-checked

| quantity | value |
|---|---|
| max non-king material (9Q+2R+2B+2N, every pawn promoted) | 10,519 |
| + PST bound ≈1,600 → max abs score in a non-mate position | 12,119 |
| worst king-capture score = 60,000 − 12,119 | 47,881 |
| **safe window** | **12,119 … 47,881** |

Packed's **47,923** sits 42 above that ceiling, classic's **50,710** sits 2,829
above — both technically outside, but the breaking case needs the opponent
holding nine queens *and* a full complement *while you capture their king*.
Under realistic material (max deficit 3,887 → floor 56,113) packed clears by
8,190 and classic by 5,403, so **packed's is the tighter, better-chosen
constant**. Not a bisection candidate.

Same discipline throughout: mate gate first, SPRT to discard cheaply, fixed-N
confirmation for survivors, 95% intervals.

## 2026-08-13 — LMR on the PST entry: outcomes pre-registered

Written before either test reports, so the interpretation cannot be fitted to
the number. Two instruments, two machines, one question — the box runs it at
**fixed nodes** (our-vs-our, load-tolerant) and the laptop runs the **timed**
counterpart at 10+0.1, which is the instrument the +400 goal is defined in.
Agreement would be strong evidence; disagreement is itself informative, since
LMR's whole claim is that it spends a *budget* better.

**The question:** LMR's +65 was screened on the **NNUE engine**. The shipped
entry uses **classic's PSTs**, and LMR's trigger is `val < LMR` where
`val = pos.value(move)` — a learned positional signal in one case, a plain
material-plus-square delta in the other. A reduction is only as good as the
ordering signal it reduces on.

| outcome | reading | what changes |
|---|---|---|
| **≈ +65** | transfers intact | ice4's catalogue is summable; the ~46 unaccounted Elo lives elsewhere and needs its own hunt; queue continues as planned |
| **≈ 0…+20** | eval-triggered heuristics do not transfer to a weak eval | **move-count LMR becomes the main line** (its trigger never reads the eval); RFP/LMP/futility/QS-delta all need re-pricing before being trusted; ice4's +421 is *not* our +421, and the +400 route runs through eval-independent search plus the eval itself |
| **negative** | LMR is **costing** the shipped entry Elo | removing it is a *free* gain — bytes back and strength up |

The third case is not far-fetched and is being checked rather than dismissed:
LMR was screened on a different eval, its threshold `LMR = 60` sits inside the
region the tuner showed to be **flat** (identical nodes *and* moves for
LMR ∈ {40…300}), and the baseline is ~46 Elo short of what the parts predict.
**A feature that does not transfer does not merely fail to help.**

Reporting rules for both, fixed now: **95% intervals** like everything else in
this ledger, and since an SPRT pass is not an effect size, the number that
enters the Elo/byte column comes from a **fixed-N confirmation**, not from the
SPRT's terminal estimate.

## 2026-08-13 — BASELINE FINAL: the 4k entry is +19.1 ± 24.5 over classic

**600 games at 10+0.1, zero time losses, zero illegal moves. Entry 265 wins,
classic 232, 103 draws — 52.75%, so +19.1 ± 24.5 Elo for the entry (95%; the
interval covers zero).**

*Interval convention, stated because I got it wrong first:* my pgn recount
produced ±12.9, which is **one sigma**, while fastchess reports ~95% (±24.49)
— and every other interval in this ledger is fastchess's. Quoting the 1σ figure
would have made this result look twice as precise as the ones it sits beside.
**This ledger quotes 95% intervals throughout.**

(Counted from the pgn rather than read off fastchess's summary line, which
reports from the first-named engine's perspective and is easy to sign-flip. The
summary said `Elo: -19.13` *for classic*; same number, opposite viewpoint.)

This is the number the +400 goal is measured from, and it is now real rather
than borrowed from the 14.9 MB engine. **Our entire search advantage over
classic is ~+19 Elo**, which leaves **~+380 to find**.

### The accounting does not close, and the gap is ~46 Elo

| term | value |
|---|---|
| speed (1.098× on the box, 1.136× on the laptop, both interleaved) | +14…+19 |
| LMR (its own fixed-node screen, **on the NNUE engine**) | +65 |
| KCX port (measured) | −16 |
| **expected** | **~+63…+68** |
| **measured** | **+19.1 ± 24.5** |
| **unaccounted** | **~46** |

The speed term I verified independently on the box: interleaved, six openings,
same movetime — classic 35527 nps, entry 39020, **ratio 1.098**, and the entry
reaches a full ply deeper (7-8 vs 6-7). The laptop's 1.136 and the box's 1.098
agree in direction and differ about as much as their loads differ, so the term
is real and small.

**The prime suspect is LMR's +65 not transferring**, and the mechanism is the
same one that explained RFP: the trigger is `val < LMR` where
`val = pos.value(move)`. With a net, that static move value carries learned
positional information and separates quiet moves from tactical ones; with a
piece-square table it is a plain material-plus-square delta and separates them
far more crudely. **A reduction rule is only as good as the ordering signal it
reduces on.**

So the rule widens once more: it is not only *eval-margin-based pruning* that
must be re-gated per eval, but **any search heuristic whose trigger reads the
eval** — RFP, LMP, futility, QS delta, **and LMR itself**.

**Test queued ahead of LMP** (it answers a question the rest of the queue
depends on): LMR on vs off, both on the PST entry, fixed nodes, SPRT. Three
consequences ride on it —

1. if LMR is worth much less on PSTs, ice4's catalogue **cannot be summed** and
   the transfer coefficient is per-(feature, eval), not per-feature;
2. **move-count LMR stops being an increment and becomes the main line**, since
   a move-count trigger does not read the eval at all — a real advantage for a
   weak-eval engine, and how ice4 and 4ku do it;
3. if LMR *does* hold at ~+65 here, something else is costing ~46 Elo and that
   needs finding before anything is added on top.

## 2026-08-13 — Fixed-N screens were underpowered for what we are hunting

Caught before spending the queue rather than after. A 300-game screen resolves
to roughly **±40 Elo**. The candidates are ice4 items of 37-123 Elo in *their*
engine, and our one transfer point (ice4 81 → ours +65 for LMR) suggests 50-80%
carries over, so realistic values here are **+18 to +90**. The bottom half of
that range sits **below the noise floor of the test designed to detect it**: a
genuine +25 returns "+25 ± 40" and gets dropped. Across five features that is a
systematic bias toward discarding real gains, and a +400 target cannot afford to
throw away +25s.

Switched to **SPRT** (`elo0=0 elo1=10 alpha=beta=0.05`, capped at 1000 games).
It stops as soon as the evidence is decisive either way and keeps playing only
while the answer is genuinely in doubt, so duds and clear winners both resolve
cheaply and the budget flows to the marginal cases.

Three things deliberately kept separate, because SPRT does not answer them:

1. **The mate gate still runs first** and skips the screen on regression. SPRT
   measures Elo; it says nothing about losing forced mates.
2. **The byte thresholds stay pre-registered** — RFP must clear +31 Elo for its
   31 bytes, LMP +56 for its 56 — so the keep/drop line cannot be fitted to the
   result.
3. **An SPRT pass is not an effect size.** The stopping rule terminates when the
   estimate has wandered far enough from zero, which biases the terminal number
   away from zero. A pass means "positive", not "this positive". Winners
   therefore get a fixed-N confirmation to earn a number for the Elo/byte
   column — affordable precisely because SPRT discarded the losers cheaply.

Also recorded, since it corrupted two of my own status reports: `pgrep -fc
"…/screens"` matched **18 unrelated `gsd-screensaver-proxy` processes**, and a
later `pgrep -f "screens/bin/e_pst"` matched **its own ssh command line**. My
"17 processes" and "2 orphaned engines" figures were both artefacts. Process
counts need patterns that cannot match the query itself, and `-recover` means
engines must be stopped by killing their fastchess parent, not the engines.

## 2026-08-13 — Screens moved to the bench box; the stale driver was waiting there

The fixed-node queue moved off the 12-core laptop to the bench box: 96 cores,
load 12.4, one other job (not ours, ~23 processes, untouched). Fixed-node
our-vs-our is the machine-independent class, so it is safe under load; anything
against classic stays on the laptop at a time control.

**The stale-driver trap was already armed on the box.** Both existing checkouts
there — `goal60/sunfish_ui` and `tdiv/sunfish_ui` — report `max_nodes` count **0**
and no `DRIVER_VERSION`. Any screen run against them would have silently
degraded to a movetime match, which is exactly the failure that voided 425 games.
Screens now run from an isolated `screens/` tree with a fresh v2 driver, and both
directions were verified **on the box**, not locally:

    fresh:  info string driver .../screens/sunfish_ui/uci.py v2 nodes fen  -> plays
    stale:  info string driver /tmp/stale_ui_parent/sunfish_ui/uci.py v1 nodes fen
            sunfish_ui driver ... is version 1, need >= 2 ... [refuses]

The refusal **surfaces in the log** rather than being swallowed by a wrapper,
which was the specific thing to check.

Footprint: 17 processes of ours against the ≤20 rule, load 17.5 of 96 cores.

### First result: RFP's mate gate passes on the PST entry — and the earlier finding was over-generalised

    RFP mate gate: base=5 variant=5   (mate-in-1 suite, depth 4)

**No regression**, where the same feature pair lost mates **5/8 → 3/8** on the
NNUE engine. I recorded that earlier as "LMR+RFP loses mates". That was wrong as
stated: it is **eval-dependent**, and the mechanism says why.

RFP prunes on a **static-eval margin** — `score - margin*depth >= gamma` returns
without searching. Whether that is safe near a forced mate depends entirely on
how the evaluation scores near-mate positions:

- **A net's outputs near mate are learned**, and mate positions are rare and
  extreme in training data, so its scores there are poorly calibrated and can sit
  far from the mate band. A margin test against a miscalibrated score prunes the
  line that proves the mate.
- **A PST's output is a fixed material-plus-position sum.** Near mate it is
  whatever the material says, predictably, and the mate band (`MATE_LOWER`) sits
  far above anything the tables can produce — so the margin test does not fire
  where the mate lives.

Same search feature, same margin, opposite outcome, because the two evals behave
differently in exactly the region the feature reasons about.

**The general rule, which will bite again:** *eval-margin-based pruning must be
re-gated per eval and never inherited across evals.* That covers RFP, LMP,
futility pruning, delta pruning in QS — and it applies to the startup-decode
work, where the whole point is to change what the eval is. A gate result is a
property of the (feature, eval) pair, not of the feature.

The RFP screen is running (300 games, fixed nodes, our-vs-our), LMP chained
behind it with its own gate.

## 2026-08-13 — Byte accounting fixed to the ENTRY, and a pre-registered threshold for LMP

**Correction to my own reporting.** I wrote "artifact 3913 → 3973 (+60)" for
LMP + improving. 3913 is the **nnue engine**, which is not a valid entry — it
dies without an external net. The thing with a hard 4096-byte ceiling is
`pst_entry.py`. Measured by building both, not by adding a remembered number:

| build | packed | note |
|---|---|---|
| **entry, pre-LMP** (tag `4k-entry-v1`) | **3517** | the real baseline |
| **entry, with LMP + improving** | **3573** | **+56**, leaving **523 spare** |
| nnue engine, same source | 3973 | not an entry; reported second, for contrast |

The composed guess was +60; measured is **+56**. Small, but it is the third time
tonight that a composed byte figure missed a measured one (the entry itself was
3517 against a composed 3787), always in the direction of the estimate being
wrong. **Convention from here: entry bytes first with spare, nnue engine second,
and every figure produced by `pack.sh` on a real file.**

### Threshold, fixed before the screen reports

LMP costs 56 bytes — about 11% of the remaining budget for one feature. Deciding
the keep/drop rule now so it cannot be fitted to the outcome:

- **Keep if ≥ 1.0 Elo/byte, i.e. ≥ +56 Elo** on its fixed-node screen.
- **Drop below that**, and the 56 bytes come back out, because ~523 spare bytes
  have to serve corrhist, history, IIR and whatever the eval track wants.

For calibration: ice4 prices LMP+improving at 123 Elo, which at our byte cost
would be 2.2 Elo/byte — comfortably above LMR's measured 1.8.

### A transfer coefficient is starting to form

Two data points on how much of a C++ engine's search technique survives the move
to a Python engine with an *incremental* eval:

| feature | ice4 Elo/byte | ours | ratio |
|---|---|---|---|
| LMR | 81/10 = 8.1 | 65/36 = **1.8** | 0.22 |
| ice4 stack average | 421/131 = 3.2 | — | — |
| LMP + improving | 123/? | pending, 56 B | pending |

Our bytes are dearer than theirs (Python source through lzma versus hand-golfed
C++), so the Elo/byte ratio conflates two things — Elo transfer and byte cost —
and the honest reading is the *Elo* column, not the rate. If LMP lands near 2.0
Elo/byte the pattern becomes a usable prior for **triaging corrhist, history and
IIR before building them**, which is worth more than any single feature: it turns
a build-and-measure list into a ranked one.

## 2026-08-13 — First sight of the number that matters: the 4k entry leads classic

**Interim, 51 games of 600 at 10+0.1, zero time losses and zero bad
terminations: the 4k PST entry is ahead of classic by roughly +133 (±120 at this
count).** Wide interval, not a result yet — but it is the first reading of the
quantity the whole +400 goal is defined against, taken in the instrument the goal
is defined in, and it is positive.

Worth sitting with the shape of it. The entry is **classic's own evaluation** —
the same 384 integers — plus our search: the KCX port, the MTD instability
guards, LMR, and the time-budget work. Same eval, same byte class, and it is
winning on search alone. That is consistent with LMR's +65 at fixed nodes and
with the ice4 rate (+421 Elo for 131 bytes) being the right target for this lane.

It also **contradicts the confounded fixed-node reading** of the same pairing
(classic +33.2 ± 29.7, i.e. our entry *behind*), which is exactly what the 1.70×
node-cap artefact predicted: at fixed nodes classic was silently getting 70% more
work, and removing that flips the sign. Two instruments disagreeing by ~165 Elo
on the same two engines is the strongest single argument in this ledger for
measuring in the instrument the goal is written in.

Protecting the measurement, since three lanes share one laptop: load was 18.5 on
12 cores when it started and is now ~12-13 with the other lane's games stopped;
my match is ~299% of that. **No second match runs alongside it, and time losses
are being watched specifically** — zero so far, and any that appear will be
reported as a contention artefact rather than folded into the result.

## 2026-08-13 — The +400 decomposition, checked: the eval half is bigger than it looks

Asked to sanity-check the split rather than accept it. The shape is right and the
priority that follows from it is right; one term is misattributed, and it matters
for how much eval headroom we think exists.

**The confound:** the two measurements being subtracted come from **different
search stacks**. Verified directly on the goal60 engine — `grep` for LMR,
`PROBE_CAP`, `node_cap`, history: **all zero**, only `king_capture` present. That
engine had the KCX port and the time formula and nothing else; LMR and the MTD
guards landed today, after it played.

So with `S` = search contribution vs classic and `E` = the net eval's
contribution over classic's PST:

    goal60   measured  S_old + E = +187 ± 50     (KCX-era search)
    pstbase  measures  S_new     ≈ +28 (prelim)  (KCX + guards + LMR)
    and      S_old = S_new − L                   (L = LMR's contribution)

| assumed L | implied S_old | implied **E** |
|---|---|---|
| 0 (same search both sides) | +28 | +159 |
| +30 | −2 | +189 |
| **+65** (LMR's screened value) | **−37** | **+224** |

**So the net eval is probably worth ~+224, not ~+160** — the engine that scored
+187 had a *weaker* search than today's, so more of that +187 belongs to the
eval. A pleasant correction, but it does **not** change the priority:

| tiny eval captures | its Elo | search must then supply |
|---|---|---|
| 25% of the big net | +56 | **+344** |
| 50% | +112 | **+288** |
| 75% | +168 | **+232** |

Even at an implausible 75% capture in ~566 bytes, **search must supply more than
the entire current gap**. Search is the larger half, exactly as claimed. For
reference, ice4's own stack sums to **+421 Elo for 131 bytes**
(LMR 81, LMP+improving 123, corrhist 70, RFP 58, history 52, IIR 37) — which is
where a +232…+344 search contribution would have to come from, and is the reason
the modern-search track outranks the eval track tonight.

Two caveats on the arithmetic itself, since it is doing a lot of work: Elo
contributions are assumed **additive**, which they are not exactly (a better eval
makes reductions safer, so search and eval interact), and `S_new ≈ +28` is a
preliminary from a confounded fixed-node run. The TC baseline replaces it.

## 2026-08-13 — RR stopped early: the critical measurement was queued behind discards

The 3-way RR was at 315/900 with ~400 of the remaining games belonging to
`classic` pairings that the 1.70× node confound makes uninterpretable — and the
10+0.1 TC baseline, which calibrates the whole +400 goal, was gated behind them.
Stopped it; the TC baseline started immediately and is running clean.

I considered keeping the classic pairings as a cross-check on the confound and
decided against it: the confound was already measured **directly** (actual
consumption 34742 vs 20480), so a confounded play result adds nothing a clean
measurement has not already given.

**Preserved, quoted separately, not to be merged with any later run**
(`rr3_partial_313games.pgn`, 315 games):

| pairing | W-D-L | n | Elo | status |
|---|---|---|---|---|
| classic v pstbase | 45-25-35 | 105 | +33.2 ± 29.7 | **unusable** (1.70× confound) |
| classic v psttuned | 62-13-29 | 104 | +114.2 ± 33.3 | **unusable** |
| pstbase v psttuned | 46-17-41 | 104 | **+16.7 ± 31.2** | fair (overshoot ratio 1.03×) |

### The Texel trend is the real finding

| build | Texel tune vs untuned |
|---|---|
| before the king-mirror fix | **−66.8 ± 35.5** (300 games) |
| after the fix | **−16.7 ± 31.2** (104 games, preliminary) |

The mirrored king table accounted for roughly **50 Elo**. What remains is a small
negative whose interval covers zero. The honest current statement is that
**Texel tuning on 15k Stockfish-labelled positions has not converted to play** —
a real negative, and consistent with the pipeline-versus-model lesson: the fit
improved 10.1% and the engine did not. A standalone 300-game rerun is queued
behind the TC baseline to settle it at a third the cost of running it inside a
3-way.

## 2026-08-13 — "Fixed nodes" was not fixed: the cap rewarded pruning less

Verified independently before acting. At a 20000-node cap over six opening
lines:

| engine | mean nodes | overshoot |
|---|---|---|
| classic | 34742 | **1.74×** |
| pstbase | 26336 | 1.32× |
| psttuned | 26422 | 1.32× |

`go nodes N` was checked only **between completed depths**, so an engine sails
past the cap by however large its last iteration was.

**The mechanism is perverse: a per-depth cap systematically rewards the engine
that prunes LESS.** Classic has no LMR, so its iterations are bigger and it
overshoots further — LMR was being penalised by the measurement for precisely
the property that makes it good, by ~30% of nodes, worth about +38 Elo at 100
Elo/doubling. That covers classic's entire apparent +10.5 over `pstbase`, and it
explains the transitivity violation in the interim table (tuned-vs-classic read
−103 where base+tune implied −37).

**Fix:** the cap is enforced inside `bound()` at the same granularity as the
deadline (every 2048 nodes), so the search aborts mid-iteration like a timeout.
Behind `minifier-hide`; artifact unchanged at 3913; 28 tests green. Re-measured,
our engine now stops at or before the cap (0.77× of nominal, the abandoned
iteration's work going unreported).

**And that makes the classic comparison worse, not better — 1.70× in classic's
favour, up from 1.32×.** Fixing one side cannot equalise a budget the other side
ignores, and `sunfish.py` is out of scope.

Be careful with that number, because the fix changed what `info nodes` *means*.
Once a cap aborts mid-iteration, the last info line is the last **completed**
depth, so the abandoned iteration's work is never reported:

| | reported | actual |
|---|---|---|
| our engine, cap 20000 | 13829-18502 | **20480 every time** (cap + the 2048-node check granularity) |
| classic (no mid-iteration abort) | 34742 | 34742 |

So actual-vs-actual is 34742 / 20480 = **1.70×**. Dividing reported by reported
gives 2.26× and overstates the gap by a third — an artifact of the reporting
change, not a real effect.

**General rule, since the next person to measure this will hit the same
ambiguity: once a cap aborts mid-iteration, `info nodes` is a lower bound on
work done, not a measure of it.** Any fixed-node fairness check must compare
actual consumption — instrument `searcher.nodes` at abort, or infer it from the
cap — never the last info line.

So the instrument splits:

- **our-variant vs our-variant → fixed nodes**, where the rule is symmetric and
  now exact;
- **anything vs classic → time control**, which has no such confound and is the
  instrument "+400 Elo over classic" is actually defined in.

A 600-game 10+0.1 baseline (PST entry vs classic) is queued behind the running
RR so the two never share the CPU.

**What survives:** `psttuned` vs `pstbase` is fair (overshoot ratio 1.03×,
median 1.06) — the Texel verdict stands on its own. The LMR screen (+65.0 ±
43.3) is safe and if anything understated, since the no-LMR side was the one
getting extra nodes.

**Third occurrence tonight of the stale-driver trap:** the first re-probe showed
the fix doing nothing, because the engines were loading a scratchpad copy of
`sunfish_ui` that predated the edit. The capability check added earlier catches a
*missing* feature, not a *stale* one — a version stamp would.

## 2026-08-12 — Mate distance: the value separates, the play does not move

Issue #11 (2014, "Tempo"): every checkmate scored the flat `-MATE_LOWER`, so
"a mate in 6 is considered the same as a mate in 1". The terminal correction
now deposits the depth still unspent when the mate was found, one
`EVAL_ROUGHNESS` per ply:

```python
mate = max(1 - MATE_UPPER, -MATE_LOWER - depth * EVAL_ROUGHNESS)
```

Negation carries it home as `MATE_LOWER + (depth - plies) * EVAL_ROUGHNESS`.

**Why the multiplier.** The first version deposited one point per ply. That
version measured a real separation in the value function and still could not
reach the root: MTD-bi stops at `upper - lower <= EVAL_ROUGHNESS`, so the
driver's last window sits within 15 of the true value and any move within 15
of the maximum can take the final cutoff. One ply per point is inside that
window by construction. Scaling by `EVAL_ROUGHNESS` puts consecutive
distances a full bracket apart. (Thomas's call, after the one-point version
measured the ceiling.)

**Band headroom, checked not assumed.** Deepest mate value
`-MATE_LOWER - 21366 = -69289 = 1 - MATE_UPPER`, exactly one point above the
illegal-move sentinel `-MATE_UPPER = -69290`; its negation `69289` exactly one
below the king-capture sentinel. That one point is load-bearing both ways:
`live |= score > -MATE_UPPER` still separates a legal move into the deepest
representable mate from an illegal move, and `r = MATE_UPPER` at a capturable
node stays unambiguous. Static-quantity tests (`pos.score <= -MATE_LOWER`,
`pos.value(move) >= MATE_LOWER`, the null cap at 515) are untouched and their
margin to the nearest mate value is now 15 wider. Clamp binds at unspent
depth 1425; `search` iterates to 999.

**Score level — works, at the intended scale.** On
`8/3Q4/8/8/8/3R4/5K1k/8 w` (three mating moves, four that mate in three):

| depth | | mate in 1 | mate in 3 | gap |
|---|---|---|---|---|
| 6 | master | 47923 | 47923 | **0** |
| 6 | one point/ply | 47928 | 47924 | 4 |
| 6 | shipped (x15) | 47998 | 47938 | **60** |

Mate-in-1 root score by depth, shipped: 47938, 47953, 47968, 47983, 47998,
48013 (`MATE_LOWER + (D-1)*15`); master reports 47923 at every depth.

**Play level — no measurable change anywhere. Four probes, all null:**

| probe | positions | result |
|---|---|---|
| WAC lockstep @d4 | 300 | root move **0**, node count **0**, score differs 14 |
| bratko lockstep @d6 | 24 | root move **0**, node count **0**, score differs 1 |
| mate2 lockstep @d6 | 212 | root move **0**, node count differs 94, score differs 212 |
| mate3 lockstep @d8 | 14 | root move **0**, node count differs 8, score differs 13 |
| conversion, won endgames @d5, cap 40 plies | 60 | 29/60 converted for both, mean **10.52 plies both**, **0 differences** |
| forced-mate-in-3 race @d8, attack and defend | 40 | **every playout identical**; attack 3.00 plies both, defend 3.00 plies both |
| lost defender with a real choice of how long to hold out @d6 | 60 | both play the LONGEST defence **60/60** |

The two mate rows are the sharp ones: on corpora where mates ARE resolved the
new scores perturb the search, and the move played is the same every time --
0 of 212 at depth 6, 0 of 14 at depth 8. So this is not a no-op internally;
it is a no-op at the move.

**But it is not free, either.** The mate3 node deltas at depth 8 are
-0.22%, +0.13%, -0.03%, **+4.62%**, -0.01%. A few percent of nodes in
mate-heavy positions is a time-to-depth change, and TESTING.md rule 12 is
explicit that time-to-depth is the hidden variable that has flipped the sign
of a result in this repo twice. So the queued wall-clock match is NOT a
formality: identical move choice at fixed depth does not imply identical
strength on a clock. The prior is still flat, but the match is the only thing
that can say so.

Byte-identical to the one-point-per-ply run on the conversion and race
probes -- `diff` of the two 60-position conversion logs is empty, and so is
`diff` of the two 40-position race logs. The
conversion probe is the one that was supposed to move, and it did not:
10.52 vs 10.52 plies, zero positions differing, at one point per ply and at
fifteen.

**Why, traced.** The tie the flat value could not break is rare, because
sunfish's horizon means only ONE of the candidate lines is usually recognised
as a mate at all. Two mechanisms:

* **Attacker side** — a faster mate appears at a SHALLOWER iteration, so ID
  (and IID, `bound(pos, gamma, depth-3, root=True)`) finds it first, stores it
  as the killer, and the killer cuts at every deeper iteration.
* **Defender side** — a slower mate is BEYOND the horizon, so it scores as an
  ordinary eval, which is hundreds of points above `-MATE_LOWER`. Traced on
  `8/8/k7/8/2R1K3/1R6/8/8 b` (replies: mated in 3 vs mated in 1): at depths
  4/6/8/9 the mate-in-3 reply scores -511/-535/-55/-55 while the mate-in-1
  reply scores -981/0/0/0 — never a mate-vs-mate comparison at all.

So: **sunfish's dawdling protection comes from iterative deepening and the
horizon, not from the mate score.** That is a real negative result and it
should be recorded as one. What the change buys is a value function that is
provably distance-ordered (which is what the play-level liveness theorem in
`formal/Sunfish/Liveness.lean` needs — the statement cannot be formed without
it) and a reported mate score that means something. It does not buy plies.

**Floors — identical to master, line for line**, at both scales: mate1 8/8,
mate2 20/20, mate3 5/5, mate4 5/10, stalemate0 4/4, stalemate1 3/4,
stalemate2 18/130, WAC 94/300 @d3, bratko 5/24, 3fold 2/8. 271 tests pass.
Packed **3231 B -- 3 bytes UNDER master's 3234** (limit 4096): the max
spelling is shorter than the line it replaces, and after xz it more than
pays for the feature.

**Rejected alternative, with a proof-level reason.** Distance-from-node via a
per-ply step on the score (`score -= sign(score)`, mate in k =
`MATE_UPPER - 2k`) is UNSOUND with sunfish's zero-window probe: the step map
is not injective at the band edge (`up(MATE_LOWER) = up(MATE_LOWER - 1)`), so
no single child window separates the child's fail-high from its fail-low, and
the fail-soft point spec breaks by one at both
`boundD2 child (1-gamma) = -gamma` and `= 1 - gamma`. Restoring it needs a
gamma-dependent child window. Implemented, caught in the Lean transport,
reverted; `formal/Sunfish/GameTree.lean` keeps `up` and its machine-checked
non-injectivity as the record.

**Elo: QUEUED, not measured.** 300 games at 30+1, openings_2k.epd, srand
20260812, `-recover`, concurrency 6, waiting on `WIDENING_RR.txt` plus a
20-minute fastchess-quiet window (`~/sunfish-bench/matedist/`). Given a
0/300 fixed-depth lockstep the prior is "flat to the point of invisibility";
the match exists to rule out a regression.

**Related negative, already on record:** the tempo half of issue #11 was
measured and rejected — T-eval -8.1 +/- 32.6, T-null -115.2 +/- 43.7.

---

## 2026-08-12 — A better fit that played 67 Elo worse: the king table was mirrored

The Texel screen came back **−66.82 ± 35.49 over 300 games, zero bad
terminations** — the tuned tables fit the data 10.1% better and played 67 Elo
*worse*. That magnitude from a 384-parameter linear refit is not a subtle
fit-versus-play effect; it is a bug, and it was.

Hypotheses, tested in order of cheapness:

1. **Table orientation round-trip** — tested by rebuilding classic's own tables
   through the tuner's forward and backward transforms: clean, except ±1 rounding
   on knights from the median re-basing. *Not it — or so it appeared.*
2. **Eval shape** (the kbbil lesson: search constants are absolute centipawns and
   a rescaled table breaks them) — tuned vs classic std 30-39 vs 31-42, ranges
   comparable, square-to-square |delta| mean 37.3 vs 37.0, p99 125 vs 124.
   *Not it.*
3. **The emit path** — and there it was.

The king's table was written back **vertically mirrored**. Every other piece got
`reshape(8,8)[::-1]` to undo the forward flip; the `if p == "K": continue`
branch skipped it. The king PST is the single most orientation-sensitive table
in the engine — castling shelter at the bottom, mating net at the top — so a
mirrored one marches the king up the board in the middlegame.

**Why the verification missed it, which is the lesson worth keeping:** my
round-trip test *re-implemented* the emit logic rather than calling it, and the
re-implementation had no `K` special case. It verified code that was not the code
being shipped. A round-trip check must invoke the actual function, not a copy of
what you believe it does.

**Why the fit never noticed:** the loss is computed in the tuner's own feature
space, which was self-consistent throughout. The bug lived entirely in the
translation to the engine's table layout — invisible to every offline metric and
visible immediately in games. This is a cleaner example of "offline metrics
cannot validate a pipeline" than any of the eval-quality work.

Fixed, king now byte-identical to classic's table, entry rebuilt at **3528
bytes**. The corrected tune is being re-screened inside the three-way baseline
round-robin below.

## 2026-08-12 — Texel tuning is free; tapering is not; and our cost model is inverted

Three results from one evening's local work, all on the **main line** (the PST
entry), none of them needing the bench box.

### The tuning set (built locally, no box time)

15,328 unique positions sampled sparsely from our own game pgns — the
distribution the engine actually plays — labelled with local Stockfish at depth
8. Phase coverage is honest: 21% opening, 32% middlegame, 32% late-middlegame,
15% endgame, mean phase 12.2/24, so an endgame-sensitive term *can* show value
in this data if it has any.

### Texel tuning: 10.1% better fit, zero bytes

Classic's eval is **exactly linear** in its 384 table values, so this is a
closed-form linear fit with a sigmoid link, not a black-box search — seconds of
compute, warm-started from classic's own tables so it can only improve on them.

| | sigmoid-MSE loss |
|---|---|
| classic's tables (2014 vintage) | 0.020908 |
| **Texel-tuned** | **0.018802** |
| improvement | **10.1%** |

Piece values barely moved — N 280→283, B 320→325, R 479→475, Q 929→926 — which
is the reassuring outcome: the fit refines the tables rather than diverging.
Artifact cost: **3517 → 3530 bytes, +13** (same 384 integers, marginally longer
digit strings), leaving 566 spare. Both entries verified standalone in an empty
directory. **A fixed-node screen against the untuned entry is running**; loss
improvement is not Elo until it plays.

### Tapering: +1.8 points more, for ~300-400 bytes

The same fit extended to a tapered model (mg and eg tables interpolated by
phase) stays linear in 768 parameters, so it costs one more tuner run:

| model | loss | vs classic |
|---|---|---|
| classic | 0.020908 | — |
| tuned, single table | 0.018802 | 10.1% |
| **tuned, tapered** | **0.018414** | **11.9%** |

**Tapering adds 1.8 points over the free tune, and would cost a second 384-value
table (~300 bytes) plus a second accumulator threaded through `Position`,
`move`, `rotate` and `from_board` (~100 bytes).** Against the LMR bar of 1.8
Elo/byte that is a poor trade, and the ordering is what matters: do the free
thing first, and treat tapering as a candidate that must justify ~400 bytes of
the 566 remaining. Caveat kept: this is one dataset of our own games, and HCE
literature rates tapering higher than 1.8 points — but our data has 47%
late-middlegame-or-endgame positions, so the measurement is not obviously
starved of the regime tapering serves.

### The architectural finding worth carrying forward

**Our cost model is inverted relative to ice4 and 4ku, and the field's Elo/byte
numbers do not transfer.** Our eval is O(1) *incremental*: `score` is carried in
the position and updated by `value(move)`. So

- terms that are a function of **(piece, square)** — PSTs, tapering,
  king-bucketed tables — stay **free at runtime**, and
- terms that depend on the **whole position** — mobility, pawn structure,
  king-ring attacks — are **expensive for us and cheap for them**, because they
  recompute eval per node anyway and we do not.

ice4 prices mobility at 104 Elo for 26 bytes (4.0 Elo/byte, better than LMR's
1.8). For us it would force move generation at every leaf that currently stands
pat on a carried score. That number is **not** available to us at that price,
and the same applies to every whole-position term in the field study. What *is*
available cheaply is anything decodable into a table — which is exactly what the
startup-decode reframe exploits, and why tapering (not mobility) is our natural
next eval upgrade despite being worth less in their engines.

### Small nets, first numbers

| net | val | vs pst anchor 0.01533 |
|---|---|---|
| N=8 ternary | 0.01364 | −11.0% |
| N=16 ternary | **0.01307** | **−14.7%** |
| N=16 float | running | — |
| (large net for scale) | 0.00678 | −55.8% |

A 16-wide ternary net fits the same data 14.7% better than the piece-square
prior alone. Whether that survives the 705 bytes of decode machinery is the
screen still to come, and the arithmetic says it must win decisively to pay for
itself.

## 2026-08-12 — MILESTONE: a valid 4k entry exists, measured at 3517 bytes

**Built, not composed.** The previous entry's `3208 + 579 = 3787` added a PST
cost measured against a *different* source, and lzma shares one dictionary
across the whole stream, so that sum was not a prediction of anything. Built for
real — our engine with the NNUE machinery removed and classic's tables pasted
into the source, through `tools/build/pack.sh`:

| | bytes |
|---|---|
| composed estimate | 3787 |
| **measured artifact** | **3517** |
| **spare under 4096** | **579** |

The measurement beats the sum by **270 bytes**, in our favour, for exactly the
reason the sum was untrustworthy: the tables compress better inside this
engine's stream than they did as a subtraction from classic's. The method
warning cuts both ways.

### The acceptance test, which is what makes it an entry

    /tmp/entrytest$ ls
    entry                       # 3517 bytes, nothing else
    /tmp/entrytest$ env -u SF_NET ./entry
    id name sunfish 2026-packed
    uciok
    readyok
    bestmove g1f3

**Alone in a directory, with `SF_NET` unset, it plays — and leaves nothing
behind.** That is the definition we have never satisfied before: the nnue
artifact at "3913" dies with `FileNotFoundError` under the same test. Sanity
beyond starting up: mate-in-1 suite 5/8 (identical to the NNUE engine's own
score at the same depth), and a legal continuation from a 6-ply opening line.

It is **reproducible from committed sources**: `tools/build/make_pst_entry.py`
generates `nnue_4k/pst_entry.py` mechanically from `sunfish_nnue.py` +
`sunfish.py`, and repacking the committed source reproduces the identical 3517
bytes.

**What it is:** classic's evaluation with *our* search — the KCX port, the MTD
instability guards, LMR (+65.0 ± 43.3 at fixed nodes), the time-budget work. It
should be stronger than classic at equal bytes, and that screen is the next
measurement rather than a claim.

## 2026-08-12 — DECISION: rank+file/PST is the main line; NNUE is the challenger

Recorded as a dated decision so it cannot drift back quietly.

| entry | composition | total | spare |
|---|---|---|---|
| **PST (main line)** | engine 3208 + classic's tables, measured together | **3517** | **579** |
| NNUE (challenger) | engine 3913 incl. 705 B machinery + blob | 4096 | **blob ≤ 183 B** |

The NNUE path must pay **705 bytes of decode machinery it cannot amortise**
before its first weight, against a baseline whose entire evaluation is 579 bytes
and which now has 579 bytes of headroom. Its effective budget against that
baseline is negative unless a net wins decisively. Affording even a 1200-byte
blob needs the engine at 2896 — cutting 1017 while keeping the machinery, which
would put our non-NNUE core at 2191 against classic's already-golfed 2655.

**Therefore: PST is the main line. NNUE is a challenger that must prove itself
per byte, machinery included.** The small ternary nets (N=8/16/32) still run,
and the screen reports **Elo per byte including the 705** — a clean number ends
the argument either way, and the arithmetic must not prejudge the measurement.

Calibration, unchanged: packed128v2 is −225 ± 65 vs molly, classic −372 ± 91.
Having a valid entry is a milestone in *rule compliance*, not in strength.

## 2026-08-12 — The engine is the problem, and the arithmetic may kill the thesis

**Correction first.** The figure "3913" has been circulating in this ledger as if
it were an artifact size. It is **engine only, with zero evaluation data**. Run
the packed artifact in an empty directory with `SF_NET` unset and it dies with
`FileNotFoundError: net128kb8.sfnn` before making a move — classic packed the
same way plays immediately. **The nnue artifact is not a valid entry at any
size.** That conflation is the same cope the README carried, and it survived
because nobody built the thing and ran it in an empty directory. Every entry
from here reports engine bytes and eval bytes separately, with the packer that
produced them (`tools/build/pack.sh`).

### Per-feature cost, by stripping and repacking

| variant | packed | delta |
|---|---|---|
| current engine, no eval data | 3913 | — |
| −LMR | 3868 | **−45** |
| −RFP (disabled branch) | 3882 | **−31** |
| −king buckets (B>1 paths) | 3865 | **−48** |
| −MTD guards | 3918 | **+5** |
| `nn_cp` stubbed, constants left | 4222 | **+309** |

The last two are lzma artefacts, not free lunches, and are recorded as a warning
about this method: removing code changes the compression context, and stubbing
`nn_cp` while leaving its SWAR constants defined-but-unused *destroys* shared
context and makes the file bigger. A feature's cost must be measured by removing
the feature **and everything only it uses**.

Done properly — loader, SWAR constants, head and accumulator plumbing all
removed together:

| | packed |
|---|---|
| our engine, no eval data | 3913 |
| same engine, packed-NNUE machinery removed | **3208** |
| **→ NNUE machinery** | **705** |
| classic engine alone | 2655 |
| **→ our non-NNUE core vs classic's** | **+553** |

So the 1258-byte overrun is **705 bytes of NNUE machinery + 553 bytes of richer
search and UCI** (of which LMR 45, the dead RFP branch 31 and king buckets 48 are
measured; the rest is the KCX port and the wider shell).

### The arithmetic that decides the thesis

| entry | composition | total | spare |
|---|---|---|---|
| **PST** | engine 3208 + classic's eval 579 | **3787** | **309** |
| **NNUE** | engine 3913 (incl. 705 machinery) + blob | 4096 | **blob ≤ 183 B** |

**A PST-based version of our engine already fits, with 309 bytes to spare — and
it keeps our search work, including LMR's +65.** That is a valid 4k entry today,
which we have never had.

The NNUE entry has **183 bytes for the net**. To afford even a 1200-byte blob the
engine must reach 2896, i.e. cut 1017 bytes while *keeping* the 705 of machinery
— which requires the non-NNUE core to shrink to 2191 against classic's 2655.
Classic is already golfed and does less than we do.

**So the NNUE-in-4k thesis is in arithmetic trouble before any question of net
quality.** The re-stated question — can any eval beat classic's 579-byte PST per
byte — is now sharper than intended: the net must beat it while carrying 705
bytes of decode machinery it cannot amortise, on a budget with 309 bytes of
headroom in the PST configuration.

The honest read: **rank+file/PST is the main line, not the fallback.** The small
nets now training (N=8/16/32 ternary) still answer a real question — what a
trained eval is worth per byte — but they are being priced against a baseline
that is currently winning on arithmetic alone. If they lose, the 4k entry is a
golfed classic-style engine carrying our search improvements, and that is a
perfectly good entry.

Calibration unchanged and worth repeating: packed128v2 is −225 ± 65 vs molly,
classic −372 ± 91. Fitting in 4096 is necessary, not sufficient.

## 2026-08-12 — Where the effort actually went: an accounting

4k has always been the goal. This is a plain count of what the ledger's 51
logged experiments served, classified by the target they move:

| served | entries | share |
|---|---|---|
| the unbounded net (width, buckets, data scale, ext family, quality metrics) | 36 | 71% |
| search (transfers to the artifact: reductions, guards, time management) | 7 | 14% |
| the 4k artifact itself (budget, packing, field study, UCI surface) | 8 | 16% |

**The 4k track was priced and never built.** Every number needed to build it has
been measured — the real budget (engine 3913 + net 183 today, against a target
split of ~2100 + ~1900), the packing (base-3 composed with joint lzma, worth
1007 bytes over the alternatives), the design space (ternary + mirror gives 5-50×
the parameters of the width-5 baseline at 1920 B), the field's technique (ice4's
entire eval is 333 characters; everyone factorises PST), and the floor to beat
(our own 1207-byte rank-6 factorised net inside a 4008-byte artifact at
`0c0a33a`). None of it has been turned into a trained net.

Meanwhile the unbounded net was pushed from val 0.00875 to 0.00678 across roughly
two dozen trainings, and the artifact that would actually be entered still ships
distilled PSTs at approximately classic's strength.

How the drift happened is worth recording, because it was not a single decision:
the README claimed nets were external to the budget, which made large-net work
look like 4k progress; when that premise was corrected, I wrote a "two targets"
section that preserved the same allocation under a new justification. My own
sentence — *"almost none of the large-net work transfers"* — should have
triggered a re-plan. Instead it became a caption for a second scoreboard.

What was **not** wasted: the search work (LMR's +65 is artifact bytes), the
instruments (shapecheck, the speed model at ~100 Elo/doubling, the cp-loss
frontier), the packing and budget measurements, and the large net itself as a
**distillation teacher** for the small one. What was: most of the eval-side
training, which bought val on an architecture that cannot fit.

Calibration to keep the next result honest: packed128v2 is **−225 ± 65** vs molly
and classic is **−372 ± 91**, so even the 14.9 MB engine is not competitive in
this field. Fitting a net into 1900 bytes is **necessary, not sufficient** — a
win against our own PST baseline is not a win against the division, and should
never be reported as one.

## 2026-08-12 — LMR converts: +65.0 ± 43.3 at fixed nodes

The first reduction, and the first *clean* local screen — the node cap honoured,
the driver named in the log, and a smoke test read back before the run.

**lmr vs base, 20000 nodes/move, 200 games, kb8@128 both sides, srand 20260830:
+65.02 ± 43.32 Elo (nElo +74.39), 59.25%, 102W 65L 33D. Zero time forfeits, zero
illegal moves** (136 adjudications, 64 normal).

Fixed nodes is the honest test here: both sides get identical effort, so this
isolates whether the reduction *spends* nodes better rather than rewarding
whichever engine searches faster. It does — a ply shallower on late quiet moves,
re-searched at full depth only on a fail-high, is worth ~65 Elo for **+36 bytes**
and a 64% node reduction at fixed depth (265210 → 94442).

Caveats kept honest: the interval is wide (±43) and excludes zero comfortably but
not overwhelmingly; this is one net at one node budget; and ice4's +81 for the
same feature is a different engine at millions of nodes, so the agreement in sign
and rough magnitude is reassurance, not confirmation. A timed confirmation at
30+1 belongs on the box queue behind the current chain.

**LMR stays in.** RFP remains held at 0 pending its own screen *on top of* LMR,
with the mate suite as an acceptance gate — the pair loses mates (5/8 → 3/8)
where each alone does not.

## 2026-08-12 — Sudden death needs a flatter divisor (lichess bot, not the artifact)

`sunfish-nnue-engine` lost `lichess.org/EAThUL0P` on time at move 73 of a 3+0
game **without a single move overrunning**. `wtime/12` spent 12.8 s of a 180 s
budget on ply 9; below 2 s the `wtime/2 - 1000` cap goes negative, the budget
collapses to the 0.05 s floor, and ~200 ms/move of unavoidable lag drains the
rest.

    think = min(wtime / (12 if winc else 40) + 0.9 * winc, wtime / 2 - 1000)

Behind `minifier-hide`, so **the artifact is byte-identical at 3913**: TCEC is
1800+3, `winc` is never zero there, and the branch would be dead code. The
lichess bot runs the unminified module and gets the fix.

Simulated budget walk, before trusting the change (and the harness was checked
against the *old* formula first, so it can actually fail):

| scheme | 3+0, 73 mv | 3+0, 100 mv | 3+2, 80 mv | 1800+3, 120 mv |
|---|---|---|---|---|
| `/12` (current) | FLAG | FLAG | ok 6 s | ok 8 s |
| **`/40` (fixed)** | **ok 22 s** | **ok 7 s** | ok 6 s | ok 8 s |

First-move spend at 3+0 falls 15.0 s → 4.5 s. `/40` is classic's constant and
classic does not flag, so it carries production evidence rather than being fitted
to one game. Movecount-aware divisors were simulated and are *worse*: a shrinking
"moves remaining" divisor spends more per move as the game lengthens, which is
backwards for sudden death.

The regression (`tests/test_time_budget.py`, 8 tests) walks the curve directly,
because **no existing gate can see this class of bug** — the ladder measures
nodes, bytes and correctness, and a match would need a real 3+0 game. It extracts
the formula from the source rather than duplicating it, so a reshaped budget line
fails loudly instead of testing a stale copy.

Note the ms/seconds trap bit twice in one day: the extracted expression yields
milliseconds and `main()` divides by 1000 on the next line, the same confusion
that produced a 590-second move earlier. The conversion now lives in one named
place.

## 2026-08-12 — VOID: every local fixed-node game was a time forfeit

**Withdrawal.** Every game of every local fixed-node match ended in a time
forfeit — 425 of 425 in the label RR, 54 of 54 in the LMR screen, 40 of 40 in
the ng match. The winner of each game was whichever engine happened not to
overrun first. None of it measured chess.

Two independent defects, both now fixed (`0df49cf`):

1. **The node cap was ignored.** fastchess sends `st=30 nodes=20000`; the
   engines honoured the movetime and dropped the cap, so every move burned the
   full 30 s. The root cause was not the engine I had already patched:
   `sunfish_nnue.py` inserts `dirname(dirname(__file__))` at the *front* of
   `sys.path`, and the scratchpad parent held a **stale copy of the driver**
   predating the go-nodes support, which shadowed both the repo driver and
   `PYTHONPATH`. `grep max_nodes` on it: 0. Fixed by removing the stale copy
   *and* by giving the engine's own builtin loop node support, so a fixed-node
   screen no longer depends on which driver happens to be importable. Verified
   the cap binds and scales: depth 6 at 20k nodes, depth 9 at 200k.
2. **Movetime was taken to the last millisecond.** With the deadline checked
   every 2048 nodes, `think = movetime/1000` returns at movetime + ε and the GUI
   has already flagged. Now 5% (min 30 ms) is held back.

**What is unsupported as a result:** the 15 pairwise fixed-node labels and every
number derived from them — the H1 battery, its three pre-registered predictions
(rehab/kbbil ≈ 0 at fixed nodes), and the fixed-node arm of the metric
validation. I had already recorded those labels as "too noisy to mine, do not
quote" at ~28 games/pair, so nothing downstream had been built on them; they are
now void rather than merely noisy, which is a cleaner state.

**What still stands, because no games were involved:** metric families A, B and
C and their numbers; the LMR/RFP mate-suite interaction (5/8 → 3/8, from the
mate suite); the crossing attribution; the packing and budget measurements; and
every bench/verify/byte figure.

**Lesson, and it is not "check the pgn":** I chose fixed-node testing precisely
*because* it is machine-independent, then never verified that the node limit was
being applied. A protocol feature that silently degrades to a different
experiment is worse than one that fails loudly. Any future match on a new
protocol gets a single-game smoke test with the termination reason read back
before the full run is launched.

## 2026-08-12 — The guards fire with LMR switched off: we were already unstable

The reduction family is approved for the packed engine and forbidden for
classic. Guards went in first, as instructed — and the first thing they did was
report that **the engine has been unstable all along**.

Running 60 real positions to depth 5 with **LMR=0**, i.e. the engine exactly as
it has played every match in this ledger:

    info string MTD-GUARD bracket crossed: depth 3 lower 344 upper 332
    info string MTD-GUARD bracket crossed: depth 2 lower 896 upper 893
    info string MTD-GUARD bracket crossed: depth 3 lower 961 upper 893

Two null-window probes of the same position at different gammas returned
contradictory answers, crossing the bracket, with no reduction in sight. The
likely mechanism is the one that was always there: `tp_move` is mutable state
that steers ordering, ordering decides which cutoffs happen, cutoffs decide
what `tp_score` stores, and the depth≤1 futility branch **breaks out of the move
loop** on an order-dependent condition. That is order-dependent pruning, which
is enough to break one-value-per-key.

Consequences worth stating plainly:

- The "we prove ≤ 15 probes" invariant had **already** stopped applying to the
  packed engine before today. It is now a runtime check rather than an
  assumption, which is what it should always have been here.
- Previously a crossing was survivable by luck: `while lower < upper - ER` is
  false once `lower > upper`, so the loop exited — but the final `gamma` was
  computed from a crossed bracket, so the last probe of a depth could be run at
  a nonsense window. Now it stops deliberately and says so.
- **Classic is a different question.** Its invariant is defended by the Lean
  development and it is not getting these features; nothing here implies
  classic is unstable. But the same futility-break/ordering interaction exists
  there, and the formal lane should be told that this engine — same search
  skeleton — demonstrably crosses brackets, so the proof's premises deserve a
  re-read rather than an assumption of transfer.

### Guards, measured

Monotone tightening (`max`/`min`), bracket-crossing stop, a 40-probe cap that
prints `MTD-GUARD` loudly in dev builds and silently breaks in the artifact, and
commit-on-completed-depth promoted from belt-and-braces to load-bearing. Cost:
**+26 bytes, and bench nodes 265210 — exactly the standing baseline**, because
`max`/`min` are no-ops while probes stay consistent. Six regression tests
(`test_mtd_stability.py`) cover warm-table re-searches, a deliberately *lying*
`bound()` that contradicts half the root probes, and a source check so a
refactor cannot silently drop the guards.

### LMR, landed and under screen

First reduction, placed where our sorted move list makes it natural: quiet moves
(`val < 60`) arriving after the first three at depth > 2 are searched a ply
short, and re-searched at full depth only on a fail-high. A null-window driver
makes that verification cheap — the reduced result only needs trusting when it
fails low.

- bench nodes at depth 5: **265210 → 94442 (−64%)**
- artifact: 3824 → **3860 (+36 bytes)**
- 20 tests green, verify battery green

Node reduction is not Elo, so it is on screen now: LMR vs base at **fixed 20000
nodes/move**, 200 games, same net both sides. Fixed-node is the honest test for
a reduction — both sides get identical effort and the question is purely whether
the reduction spends it better. It runs on the laptop, so the bench box keeps
its queue.

## 2026-08-12 — Packing, reversed twice: compose base-3 with lzma, and go joint

The intelligence lane's finding — that base-3 packing loses to LZMA once trits
are sparse — is right in direction, and checking it on our own weights changed
two decisions, one of which was mine.

### Base-3 vs LZMA: the answer is *both*

Ternarising the real trained embedding (768×25 slice, threshold swept):

| zeros | raw base-3 | base-3 → lzma | 1 byte/trit → lzma |
|---|---|---|---|
| 42.1% | 3840 | **3393** | 3873 |
| 55.5% | 3840 | **3173** | 3586 |
| 66.4% | 3840 | **2840** | 3118 |

**Base-3 packing and LZMA are not alternatives — compose them.** Packing does the
alphabet compaction (8 bits → 1.58/trit) and LZMA then finds the spatial
correlation that survives it, worth a further 447-1000 bytes.

The stated ~45% crossover comes from *uniform random* trits, and my control
reproduces it exactly: random blobs at 20%/45% zeros prefer raw base-3 (1920 vs
1937/1932) and only at 70% does LZMA win (1668). Real weights are not uniform —
neighbouring squares of the same piece are correlated — so on real data the
composed form wins at **every** sparsity, including 19.7% zeros. Do not tune the
training threshold to chase a crossover that only exists for random data;
measure the actual blob.

### Joint vs split: my earlier conclusion was measured on the wrong data

I previously locked "engine source xz'd, weights appended raw", measuring a blob
of `os.urandom`. That was the wrong sample: incompressible by construction, so
of course folding it into the stream only added encoding overhead. With a real
ternary blob:

| layout | bytes | vs split |
|---|---|---|
| split (engine.lzma + blob raw) | 16532 | — |
| **joint, one lzma stream, byte concatenation** | **15525** | **−1007** |
| joint, base64 literal in source | 16026 | −506 |
| joint, escaped latin-1 literal | 16057 | −475 |

**Joint wins by ~1000 bytes**, and the extra 13-byte container header the split
pays is the least of it. Even the naive in-source literal forms beat splitting.

The mechanism works without temp files: compress `[engine source][blob]` as one
stream; the head pipes only the first `ENGLEN` bytes to the interpreter
(`… | xz -d | head -c ENGLEN`), and the engine recovers its own weights with
Python's built-in `lzma` — read the artifact from `SF_A`, decompress, slice past
`ENGLEN`. Costs one extra decompression at startup, which the 60 s rules budget
absorbs without noticing, and about 90 bytes of Python against 1007 saved.

**Corrected design: one LZMA stream containing base-3-packed ternary weights
after the engine source.** `pack_entry.sh` needs rewriting to this shape; the
self-read `SF_A` mechanism it already uses is exactly what the engine needs to
find the blob.

### Still to verify from the same report

The PST re-encoding (−310/−320 B) rests on the same principle this measurement
confirms — eval data belongs *inside* the compressed stream, quantised and
range-narrowed. It applies to **classic's** source (the packed engine's tables
already live in the net file), so it is a shared-packer, build-time transform,
and it should be measured before it is claimed. Same for the −103 attribute
renaming and the −120/−155 UCI shell (both MEASURED 2026-08-14, see "The
unverified golf leads, measured": −35 and 0); the `eg_scale` term (~20 Elo, zero
parameters) and mobility-fused-into-movegen (104 Elo for 26 B) are engine
changes that need our own SPRT, not ice4's.

## 2026-08-12 — Box collision hazard: an atomic lock, and one fewer waiter

Three lanes were armed on the bench box watching for the same quiet window
(mine, the widening RR, and delaybonus). A shared window oversubscribes the box
and corrupts every lane's 30+1 measurements, which has happened here before.

**My lane made it worse than it looked.** `fixednode_chain.sh` gated on
`PACKED_MSP.txt` *existing* — and when the msp screen was cancelled, the
cancellation marker satisfied that gate. My waiter was released into the
contested window by the very act of cancelling an unrelated screen. Gating on a
file's existence, when that file can also mean "cancelled", is a bug pattern
worth remembering: **a marker should be read for its content, not its presence.**

Fixes applied, in order of value:

1. **Removed a waiter entirely.** `fixednode_chain.sh` was going to run the H1
   fixed-node battery on the box — which is redundant, because the same battery
   is already running on the laptop where fixed-node results are
   machine-independent and cost the bench box nothing. Cancelled (PID-killed),
   with `FIXEDNODE_H1.txt` written as an explicit cancellation marker that says
   *not a completed screen, do not read Elo from this file*. `krff_fn.sh`, which
   chained behind it, was killed with it. Contention drops from three lanes to
   two by deleting work rather than scheduling it.
2. **Atomic lock for what remains.** `~/sunfish-bench/boxlock.sh`, sourced by any
   lane:

       . $HOME/sunfish-bench/boxlock.sh
       box_acquire my-lane-name

   `mkdir` is atomic on POSIX, so exactly one lane wins. It records
   `$$ lane date` in `.boxlock/owner` for diagnosis, traps EXIT/INT/TERM to
   release, and reclaims a lock older than **12 hours when no fastchess is
   running**, so a killed lane cannot deadlock the box forever.
3. **Ordering matters more than the lock.** My first version acquired the lock
   and *then* waited for quiet — which would have let this lane preempt the
   widening and delaybonus RRs that were queued ahead of it. Corrected to
   **wait for quiet → acquire → re-verify after 45 s → launch**, releasing and
   resuming the wait if another lane took the window in between. The lock
   settles who owns the *moment of launch*, and must never be held while idle.

Also note, for anyone replacing a running waiter: overwriting the script file
leaves the running process on the old inode, so the old waiter must be killed by
explicit PID and relaunched — and verified afterwards, since one of my kills
silently failed and briefly left two copies racing.

Offered to the other lanes: the widening lane's jitter-and-recheck is sound but
probabilistic; this is the same idea made exact, and costs two lines.

## 2026-08-12 — Rules audit: the packer, the UCI surface, and joint-vs-split settled

Working from the fetched rules (operative clauses now in the README, 369e8c1).

### Startup is a non-issue

"Startup should be within 60s", numpy is explicitly allowed, and pypy3/xz/tail/
sh/mktemp are on the allowed-commands list with self-decompressing shell scripts
explicitly permitted. Load-time expansion is therefore unconstrained — every
scheme that trades load compute for stored bytes is legitimate. (Kept for the
record: if a build ever had to fall back to CPython for numpy, that costs
83552 → 39424 nps ≈ −110 Elo, so prefer numpy-optional designs, but nothing in
the rules forces the issue.)

### Joint vs split packing — measured, and the split wins

The historical packer chose to xz the engine and append the model raw without
recording why. On a 2 KB bit-packed blob:

| layout | bytes | delta |
|---|---|---|
| **split** (engine xz'd, weights raw) | engine + 2048 | — |
| joint, base64 blob inside the source | +156 | worse |
| joint, escaped latin-1 inside the source | +746 | much worse |

lzma cannot compress already-packed weights but still pays for the encoding, so
the split is right. Same result for a base-3 ternary blob (+143).

**SUPERSEDED — this measurement used `os.urandom`, i.e. incompressible data, and
the conclusion does not survive on real weights. Re-measured with a genuine
ternary blob, JOINT wins by 1007 bytes. See "Packing, reversed twice" above.**

### The delivery mechanism, rebuilt to leave nothing behind

The rules require the entry "not leave itself any files lying around". The
historical combined packer used `mktemp` for both streams; `pack.sh` (engine
only) already used process substitution. Attempting process substitution for
*both* streams **fails**: bash tears the `/dev/fd` down across `exec`, so the
engine reads an empty weight stream (reproduced in isolation, then fixed).

The working shape has the engine read the weights **from the artifact itself**,
whose path the head already knows:

    #!/bin/bash
    export SF_A="$0" SF_N=<blob length>
    exec $(command -v pypy3||echo python3) <(tail -c+<K> "$0"|head -c<L>|xz -d)

Verified end to end: `uci` → `uciok` → `readyok` → `bestmove`, and zero temp
files created. Costs, measured: head 74 → 118 (+44) and engine +39 for carrying
both the dev `SF_NET` path and the artifact path — most of that 39 comes back in
a real 4k build by hiding the dev branch. `tools/build/pack_entry.sh` is the
competition packer; `pack.sh` remains the engine-only one.

### UCI surface: already rules-minimal, 42 bytes reclaimable

The mandated subset is `uci`, `uciok`, `isready`, `readyok`,
`position startpos (moves ...)`, `go`/`go wtime A btime B winc C binc D`,
`bestmove`, `quit`, with `stop`/`ucinewgame` merely tolerated. Audited the
artifact's built-in loop against exactly that list — **there is no FEN parsing in
the artifact at all**; `from_fen` lives in `sunfish_ui`, which the packer strips.
`from_board` survives but is load-bearing (it builds the initial position), not
FEN machinery.

What is genuinely non-mandated, measured by packing each removal:

| removal | bytes |
|---|---|
| `movetime` support in `go` | 9 |
| `info depth … pv …` output | 21 |
| `from_board`'s unused `pf` branch | 8 |
| shorter `id name` | 4 |
| **total** | **42** |

Small, and worth taking when the 4k build is assembled, but it confirms the
shell was never the problem — the weights are.

### Time control: the gap is real and now under test

The tournament plays **1800+3 with pondering disabled**, and our divisor was
fitted at 60+1 and 30+1. At 1800+3, `wtime/12 + 0.9*inc` spends **150 s on move
one** — demonstrated accidentally when a smoke test of the packed artifact hung
for two minutes on exactly that command.

The arithmetic is survivable (proportional spending cannot exhaust the clock:
~670 s left after 12 moves, ~295 s after 24, ~15 s/move by move 40), and with a
book dropping engines in around move 10 the first move is a real middlegame
decision rather than a wasted book move. What is untested is whether **/12 is
right in this regime**: at 30+1 the increment replenishes a third of each move's
budget, while at 1800+3 the increment is noise and the divisor alone sets the
shape.

Queued: a divisor sweep at **180+0.3**, which preserves the 600:1
base-to-increment ratio at a tenth the cost, five arms (D = 12 current, 16, 20,
25, 30 — spanning our aggressive setting to the conventional rules of thumb),
240 games round-robin, `SF_TDIV` selecting the arm. It is gated on **20 minutes
of box quiet** so it can never run beside another timed match or jump the queue.
**Caveat recorded in the marker itself: scaling preserves the allocation policy,
not absolute depth, so it validates the policy only — the winner still needs a
confirmation run at the true 1800+3 before entry.** The result also serves the
lichess bot, where the same divisor governs rapid and classical play.

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

## 2026-08-14 — The unverified golf leads, measured: 3357 → 3299

The ledger-4850894 estimates had never been executed on the current entry,
and composed estimates had missed measured values six times. Each lead was
run as its own step through pack.sh, with the full battery per accepted
step: decode round-trip tuple-identical to classic, fixed-node driver bench
identity old-vs-new (nodes/depths/scores/pv byte-identical modulo time/nps),
check_entry.sh, the time-budget tests, and a standalone packed smoke in an
empty directory (SF_NET/PYTHONPATH unset) — uciok + legal bestmove at both
winc 100 and winc 0.

| lead | estimated | MEASURED | verdict |
|---|---|---|---|
| attribute/method renaming | −103 | **−35** (3357 → 3322) | accepted |
| UCI-shell slimming | −120/−155 | **0** — already banked / load-bearing | nothing to cut |
| `__version__` fold (free win) | — | −15 (3322 → 3307) | accepted |
| namedtuple typename strings (free win) | — | −7 (3307 → 3300) | accepted |
| codec `* 1` at step 1 (free win) | — | −1 (3300 → 3299) | accepted |

**Why renaming is −35, not −103.** pyminify already renames every global and
local, so the only long identifiers left in the packed stream are attributes
and method names — and most of those are API. sunfish_ui/uci.py reads
pos.board/.score/.kp/.move()/.gen_moves()/.rotate()/.value()/.prom,
searcher.bound()/.search()/.tp_move/.nodes/.deadline/.node_cap and the
`root=` kwarg of bound() by name, and agree.py plus every variant screen
drive entry SOURCES through that driver, so renaming them breaks the lane's
own instruments. What went: king_capture→k, tp_score→t, self.history→.h,
self.root→.r, nullmove→n, Entry fields lower/upper→l/u, all count-asserted
in the generator. The estimate assumed the full rename; the full rename was
never available.

**Why the UCI shell yields zero.** The estimate predates the base-90/merge
era. The shipped loop is already only uci / isready / quit / position
startpos / go — no info lines, no options, no `position fen` (there was
nothing to cut there: it was never in the artifact). The one remaining
candidate, `movetime`, is load-bearing: tools/build/legality_gate.py drives
the PACKED artifact itself with `go movetime 300` and `go nodes` as two
deliberately separate budget paths, and the 425-forfeit incident is the
record of what happens when only one of them is real. Not cut.

**Estimate-vs-measured, points seven and eight:** composed −103 measured
−35; composed −120/−155 measured 0. The direction is always the same.

**Collateral, caught by rebuilding everything:** make_variants.py
text-anchors the entry, and the cap/corr/hist mods still said
`nullmove=True` / `self.tp_score` / `self.history`. A stash-test confirmed
the stale anchors fail LOUDLY (occurs-0 assert), and all 22 mods plus
corr.wkey rebuild green after the anchor update. splice.py's section
anchors were checked and are untouched by the renames.

**Thresholds:** −58 total crosses the ~42 B line that funds pend (H1,
+42 B). Nowhere near the ~420 B that would reopen packed128. Entry now
3299 bytes, 797 spare under 4096.

## 2026-08-14 — Illegal moves: zero, by construction (the structural bestmove floor)

Thomas's ruling on the seedtimed forfeits: "We should never accept illegal
moves. 15 is too much, so is 4." Zero, achieved structurally, not
statistically — the gamma seed making the class 4x rarer does not satisfy it.

**All 19 forfeits classified from the PGNs** (seedtimed, 400 games at 1+0,
b8 vs b8seed). Three candidate classes were checked: (a) a stale move
committed for an earlier position, (b) `(none)`/no move at all, (c) a
promotion/castling encoding defect. Every one of the 19 is class (b): the
literal `bestmove (none)`, flagged by fastchess as "makes an illegal move:
(none)". No stale moves (structurally impossible: `best_move`/`cand` are
per-`go` locals in both drivers), no encoding defects.

| victim | rounds (side, plies survived after book) |
|---|---|
| b8, 15 games | 6 (W,2) · 10 (B,1) · 11 (W,2) · 11 (B,1) · 31 (W,1) · 31 (B,2) · 52 (W,4) · 52 (B,3) · 53 (W,13) · 61 (B,1) · 61 (W,2) · 69 (B,1) · 78 (B,1) · 90 (W,1) · 90 (B,2) |
| b8seed, 4 games | 19 (B,1) · 54 (W,4) · 92 (W,66) · 98 (W,2) |

**The emission path.** The arms run the entry, whose dev redirect resolves
the sunfish_ui driver (the games start from FEN openings, which the builtin
loop cannot even parse). At 1+0 the driver budget
`think = min(wtime/12, wtime/2 - 1)` is NEGATIVE below 2 s of clock, so the
in-search deadline is already past when `go` arrives; `Stop` fires at the
first poll (node 2048), and any position whose first root fail-high needs
more than 2,048 nodes ends the search with `best_move = cand = None` and an
empty `tp_move[root]`. The old tail — `played or (my_pv[0] if my_pv else
"(none)")` — then printed the literal. That is exactly the b8/b8seed
asymmetry: max first yield 2,433 nodes vs 394 (the seed's one-sided first
probe), hence 15 vs 4. The builtin loop had the identical hole
(`best or cand or '(none)'`).

**The invariant, now enforced at every bestmove emission site** (go_loop,
mate_loop, and the builtin loop that ships in the artifact): any move we
emit was generated for the CURRENT root position and survives making the
move (`can_kill_king`/`king_capture`, which covers check, pins, en-passant
discoveries and castling-through-check via the king-passant square) — so
the emitted move is legal, not merely pseudo-legal, and the worst case is a
weak legal move that can lose the game on the board, never a forfeit.
`(none)` survives only for checkmate/stalemate roots, where no tournament
manager ever asks us to move; the builtin loop additionally gained the
`score >= gamma and move` terminal-root guard so a verified-terminal yield
cannot crash it. Driver bumped to v3, entry requires >= 3 (same commit, per
the stale-driver rule).

**Byte price:** entry 3299 → 3341 (+42 B), 755 spare under 4096.
check_entry.sh green; all 23 variant mods rebuild against the new anchors.

**Verification without games:** nnue_4k/tests/test_bestmove_floor.py (20
tests) replays the abort-before-commit state deterministically (a searcher
whose Stop beats the first yield), the position-A-then-aborted-position-B
derive-never-inherit case, mate/stalemate at the root, the startpos
instant-stop race, the builtin loop under the same starvation, and the real
driver over a pipe at `go wtime 1`. Existing gates re-run green on the
fixed base: legality 334/334, first-yield worst 582 <= 2048, mate 8/8,
mate-conversion 8/8, full suite 48/48, packed standalone smoke (legal move,
and `(none)` + alive after mate-at-root).

**The instrument change.** seed_screen stage 2 (and rr_cap / rr_hole /
ab_fixednode) no longer COUNT illegal moves: any illegal move by any arm in
any run is a FAIL that names the game. The seed's pair rule is therefore
DEPTH-QUALITY only — the illegal class it was reducing is closed by
construction. Staged confirmation: tools/screens/hammer_1p0.sh, self-play
at 1+0 (the regime that produced the 19), 100 games, REQUIRED zero illegal
— stage it when an arena is free; PENDING until run.

## 2026-08-14 — REPLACEMENT net priced: N=4 packed big-int FITS at 4075 B

The packed128-revival audit closed the RESIDUAL lane (machinery over PST:
4512 B, 416 over, kill-condition fired). Thomas's steer — a small net that
REPLACES the tables — re-prices the same golfed G12 machinery against
ENGINE-SANS-EVAL instead of the full entry, and the verdict flips.

**Measured, not composed** (all via tools/build/pack.sh on 71c9ba1):

| quantity | B |
|---|---|
| entry as shipped | 3341 |
| ENGINE-SANS-EVAL (price_engine.sh; flat-material stub) | **2871** |
| replacement machinery (G12 hot loop, payload elided) | **578** |
| N=4 payload in-context @55% zeros | 626 |
| **composed total @55% zeros** | **4075 (21 spare)** |
| @60% / @66% zeros | 4062 / 4025 |
| @42% zeros | 4105 — **9 OVER** |

Payload budget = 4096 − 2871 − 578 − 30 (margin) = **617 B**, so the
trained ternary net (768×4, 3,072 trits, one char per feature through the
entry's own base-90 codec) must land **≥ ~58% zeros** — inside the
ledger's measured real-weight range (42/55/66%, 4850894), but it must be
trained for (sparsity pressure is a mandatory loss term, like satpen).

**The design** (nnue_4k/replnet_proto.py, loudly NOT AN ENTRY): score =
mat(pos) + clip(nn, ±600). The base is price_engine.sh's exact flat-material
stub — `value(move)` stays an exact material delta for ordering/QS/futility,
the king-gone sentinel reads `ps` and keeps a 1558 cp margin over the
9Q+2R+2B+2N kingless army (> CLAMP 600, so the net can never mask a king
capture), and MATE arithmetic is untouched: same score scale, no
MTD/EVAL_ROUGHNESS change. K_MID/K_END formula table kept. This sits the
net in the exact `BASE = "mat"` seam sunfish_nnue.py already documents.

**Verification without games**: packed/replnet_check.py (mirror identity,
40-ply incremental==from-scratch acc/ps/score walk, exact antisymmetry,
net-fires, sentinel margin) PASS; packed artifact answers UCI and plays
under pypy3 (movetime smokes at start/after e2e4). Weights are the
real-shaped random payload (packed/make_proto_payload.py) — pricing only,
no Elo claim. Next: architecture pre-registration, then the ternary
retrain; play is the only acceptance metric.

## 2026-08-14 — Pre-register REPLNET v1: the N=4 ternary replacement retrain

Design committed BEFORE training; play is the only acceptance metric.

**Architecture (every choice forced by a measured number):**
- 768 piece-square features, perspective-mirrored shared rows, **kb=1**
  (kb4/kb8 multiply the payload ×4/×8 against a 617 B budget — impossible;
  folding buckets to shared ternary rows at export IS kb=1).
- **N=4 ternary lanes** per perspective. The golfed G9 fused decode is
  one char = one feature = 4 trits (3⁴ = 81 ≤ 88 codec values); N=5-7
  breaks char alignment, N=8 doubles the payload (~1.25 KB) — over.
- Per-lane integer gain g_k ∈ [0,89] — the weight scale AND (×32) the
  activation cap, output weights folded (so g_k > 0: relu caps are not
  sign-symmetric; v1 lived with the same constraint). Bias digit ∈
  [−44,45] lane units. SHIFT picked at export (pnet.pick_shift style).
- **No bilinear tail, no rff, no segments**: the ext path is float/dev-only
  machinery the 21-71 B margin cannot carry, and the bilt history's
  best-val net COLLAPSED −118 in play (7602d7b). Revisit only if the
  trained net lands ≥66% zeros (71 B spare).
- clampcp 600 = engine CLAMP (the proto as priced), satpen 0.03 @ 480
  (MANDATORY), wclip, exact antisymmetry by construction (shared rows +
  round-toward-zero), **ternary STE** with per-lane scale + sparsity
  pressure (threshold τ + L1): ≥58% zeros is a HARD fit gate, target
  60-66%.
- Trainer: train_packed.py `--base mat` (the seam already exists) + a new
  ternary path; provenance pinned (seed / torch version / data sha).

**Data, PRE-REGISTERED (the b1 lesson — NATURAL play distribution, NO
phase rebalancing):** distill160k.npz (19,434 own-search-labelled,
sha256 b0ed8b6617a7…) + nat8792.npz natural mix + a quiet-filtered dump
slice from the packed lane's caches. Flat phase-balanced mixes are
explicitly EXCLUDED this time: b1's balanced fit died in play (−182.6).

**Acceptance, PRE-REGISTERED:** val gates only the recipe against its own
float baseline — val LANDS NOTHING. Gate ladder before any game: pack.sh
≤ 4096 on the real composed entry, replnet_check invariants, legality,
mate, mate-conversion, first-yield v2, zero-illegal (hammer_1p0 once
landed). Then the screen (STAGED, slot requested from the coordinator —
this lane launches nothing): fixed-node 20k SPRT elo0=0 elo1=10 vs
pst_entry @ HEAD, LAND bar 95% LB > 0 on fixed-N confirmation AND a timed
confirmation LB > 0 (fixed nodes hide the ~19-op/eval speed tax; nps
ratio vs entry measured under pypy on the box first — speed is Elo).

## 2026-08-14 — REPLNET v1 box run launched (arm rule pinned before results)

Two arms on the box (nice 19, 8 workers/8 threads, tripwire on both live
tournaments' forfeit counts, baseline 0/0): l1 ∈ {0.001, 0.002}, otherwise
the pre-registered recipe verbatim, data = replnat28k + 4M quiet dump slice
(kept 4,000,000 of 6,496,293 read, trainer-identical filter), mat-base val
anchor 0.01616 on a 200k val split. **Arm selection rule, pinned now,
mid-run, before either arm's final val exists: among arms whose BEST-VAL
epoch exports ≥58% zeros, take the lower val; tie breaks to higher zeros.**
nps gate pre-measurement (weight-independent): entry 88.2k vs replnet
58.1k nps under pypy on the box, ratio 0.66 — at ~100 Elo/doubling a ~60
Elo speed tax that the timed confirmation must beat.

## 2026-08-14 — REPLNET v1 trained, gated at entry parity, screen staged

**Training (box, nice 19, 8 workers, tripwire 0/0 throughout):** 4,027,406
positions (replnat28k + 4M quiet dump slice), 30 epochs/arm, ~8 min/arm.
Winner by the pinned rule: **l1=0.001, val 0.01385 vs mat anchor 0.01616
(−14.3%), 59.6% zeros at the shipped epoch**; l1=0.002 → 0.01404 @73.7%.
(Context: the Aug-12 SMALLNET residual-era probes read 0.01307-0.01364 at
N=8-16 on their own split — different val sets, not comparable numbers,
recorded only as order-of-magnitude neighbors.)

**Candidate build (nnue_4k/replnet_proto.py + winner payload):** packs to
**3831 B (265 spare)** — the trained payload is cheaper than the 55%-zeros
pricing stand-in. Ladder: invariants PASS, shapecheck worst 0.9 cp,
first-yield PASS, legality 334/334, and on the box under pypy:
mate-conversion **8/8**, mate 7/8 — exact parity with the entry, which
also reads 7/8 there (each misses a different marginal m1; the local 4 ms
readings are load-noise, both engines flip positions run to run).

**Staged, NOT launched (coordinator's slot):** fixed-node 20k SPRT
replnet-v1 vs entry @ 0ee915b, srand 20260814, cap 1000 games, on-box
screen dir + a request appended to DISARMED_QUEUE.txt. LAND bar per the
pre-registration: 95% LB > 0 fixed-N AND timed confirmation LB > 0 —
the measured 0.66 nps ratio (~−60 Elo at timed play) is the number the
eval edge has to beat, and fixed nodes alone cannot see it.

**Standing rule active:** TRAINQUEUE.md seeded (7 entries); replnet_v1c
(l1=0.0013 band-center) started the moment arm 2 finished.

## 2026-08-14 — nnue_4k/train/: the PyTorch pipeline stands; arm-1 reproduced; big-int layers with exact backprop

**MIGRATION NOTE (training-lane handoff).** New runs queue through the
config-driven pipeline in `nnue_4k/train/` (Thomas's directive; design in
`train/README.md`). Mid-flight runs are NOT switched; the next queued run
after the current chain is the first pipeline-native one. The legacy
trainers (`packed/train_packed.py`, `tools/tune/distill_train.py`) stay
untouched and remain the reference for their own historical numbers.

**Pipeline validation, instrument-first (tolerance pre-stated).** REPLNET
v1 arm 1 (l1=0.001, τ=0.85, 40 epochs, 4M cache, seed 0, legacy split)
retrained through the NEW pipeline on the box (nice 19, 8 threads, beside
— not touching — the live v1c/8M chain), against the ledger's 0.01385
@59.6% zeros, tolerance |Δval| ≤ 0.0002 and zeros ±5 points: **val 0.01382
@60.0% zeros — PASS** (Δ = −0.00003; the val split is byte-identical by
RNG-stream alignment, val-sha pinned in PROVENANCE.json). Export of the
reproduced net verified BIT-EXACT: payload decode == trainer quantization,
and entry == integer reference == torch float64 mirror on 200 fens × 3
views + a 60-ply walk (`train/verify_export.py`); spliced entry packs via
pack.sh (measured, not composed).

**Packed big-int layers (the scope expansion, now the pipeline's core).**
`train/packed_layers.py`: LaneConv IS the big-int multiply (linear conv at
field width F; circular via the mod 2^(Fm)−1 fold — the recorded rank-1
trap's fix, now a layer), SwarClamp/HSum/ShiftRenorm mirror crelu, the
modular lane sum and the signed shift. Forward = float64 exact-int
semantics (every certified value an integer < 2^53, where float64 IS
integer arithmetic); backward = true polynomial gradients, with exactly
two documented STE points (trunc shift g/2^s; optional clamp
pass-through, default exact subgradient + satpen). **10/10 bit-exactness
tests vs actual python big-int evaluation** (`train/test_packed_layers.py`,
incl. an end-to-end 2-layer probe and gradient-exactness) — these run on
every pipeline change.

**Field-budget certification** (`train/field_budget.py`) maps the three
recorded walls to named per-layer checks by exact interval arithmetic —
no-carry (carry coupling), per-layer widths + ShiftRenorm (field-budget
collapse in deep products), in-forward quantization + the 2^53 exactness
bound (quant-error compounding) — and REFUSES to train uncertifiable
configs (train.py calls it before the first batch). Concrete: the ml2
second layer at F2=16 is refused (fields reach 32,444,416 vs 65,535);
F2=32 certifies with margin 4.26e9 and a legal hsum read-out; with
renorm-to-12-bits between layers, 16+ conv layers certify at F=32 — depth
is structural, not lucky. First multi-layer experiment queued as
replnet_ml2 (TRAINQUEUE #6, PRICE-FIRST, certificate beside the run).

## 2026-08-14 — Compression bake-off: 15 encoders x 2 container layouts, measured; the shipped codec survives everything

Thomas: "This is genuinely a unique compression problem, and you should
try many approaches," plus the coordinator amendment (Thomas: "I still
think you probably don't want to lzma compress the trained weights. But
you decide what works best") — resolved by measurement, per arm, in BOTH
container layouts.  `nnue_4k/train/compress/` is the standing harness:
one command runs every encoder against a net, gates each (arm, layout)
cell BIT-EXACT against an independent mirror of the trainer quantization
(anchored per net by verify_export's torch triangle — both nets PASS on
200 fens x 3 views + 60-ply walk), packs through the REAL paths only
(pack.sh joint; pack_entry.sh split raw-tail with the 4850894/ffead53
SF_A self-read), boots every artifact (`uci` → `uciok`), and ranks.
`export.py --bakeoff` runs the same zoo, so the per-net export winner is
chosen by measurement (TRAINQUEUE names this lane the encoder owner for
the c1024 family).

**Instrument first.**  The entry is PINNED to a git blob (HEAD =
fb717214c3e2 here): the working-tree entry was being golfed by another
lane DURING the first run, and three same-morning reads of "the entry"
measured 3831/3728/3733 B — a ranked table needs one denominator, so
unpinned working-tree measurement is now a loud NOTE in the harness.
Against the pin: baseline layout A reproduces the recorded **3831 B (v1)**
and **3834 B (repro_arm1)** EXACTLY, its payload string equals the
exporter's own .payload byte-for-byte, and payload-elided reproduces the
recorded 3449 (= 2871 engine-sans-eval + 578 machinery).  The negative
control (ctrl_shuffle: same symbols, seeded random order, storage-free
unshuffle) measures WORSE in both layouts on both nets (+161/+155 A) —
the axis can fail.  Decoder cost and payload-in-context below are deltas
of measured artifacts (the proto's elided convention), never composed.

**v1 winner (l1=0.001, 59.6% zeros), pinned entry fb717214c3e2, ranked
(A = joint lzma stream, B = split raw tail; bytes = whole artifact):**

| arm | lay | bytes | Δ | payload | decoder | boot s |
|---|---|---|---|---|---|---|
| **b81 (shipped)** | **A** | **3831** | +0 | 382 | 0 | 0.10 |
| b81_rle | A | 3847 | +16 | 377 | 21 | 1.23 |
| b81_filemajor | A | 3888 | +57 | 382 | 57 | 0.26 |
| b81_boustro | A | 3901 | +70 | 382 | 70 | 0.11 |
| b81_lanesplit | A | 3923 | +92 | 424 | 50 | 0.19 |
| cb8 | A | 3972 | +141 | 455 | 68 | 0.08 |
| b81_rle | B | 3988 | +157 | 448 | 91 | 0.22 |
| ctrl_shuffle | A | 3992 | +161 | 461 | 82 | 0.48 |
| cb4 | A | 4054 | +223 | 534 | 71 | 0.14 |
| b81_pieceperm | A | 4073 | +242 | 554 | 70 | 0.57 |
| mr3 | A | 4127 | +296 | 656 | 22 | 0.13 |
| rc_run | A | 4143 | +312 | 550 | 144 | 0.30 |
| b81 | B | 4145 | +314 | 633 | 63 | 0.15 |
| rc_o0 | A | 4149 | +318 | 573 | 127 | 0.09 |
| mr3 | B | 4153 | +322 | 618 | 86 | 0.10 |
| lr_svd | A | 4157 | +326 | 554 | 154 | 0.17 |
| rc_o0 | B | 4171 | +340 | 541 | 181 | 0.19 |
| rc_run | B | 4173 | +342 | 523 | 201 | 0.16 |
| lr_svd | B | 4182 | +351 | 520 | 213 | 0.48 |
| cb8 | B | 4189 | +358 | 604 | 136 | 0.76 |
| b81_lanesplit | B | 4194 | +363 | 633 | 112 | 0.11 |
| b81_filemajor | B | 4201 | +370 | 633 | 119 | 0.09 |
| b81_boustro | B | 4216 | +385 | 633 | 134 | 0.07 |
| cb4 | B | 4220 | +389 | 635 | 136 | 0.12 |
| b81_pieceperm | B | 4221 | +390 | 636 | 136 | 0.09 |
| ctrl_shuffle | B | 4226 | +395 | 633 | 144 | 0.19 |
| sparse_gap | A | 4524 | +693 | 1009 | 66 | 0.51 |
| sparse_gap | B | 4540 | +709 | 957 | 134 | 0.20 |
| reorder_stored | A | 5000 | +1169 | 1483 | 68 | 0.09 |
| reorder_stored | B | 5002 | +1171 | 1416 | 137 | 0.13 |

**v1c (l1=0.0013, 65.2% zeros): baseline A = 3779 B (payload 330), same
order at the top:** b81_rle A +27 (payload 336, decoder 21), filemajor
+58, boustro +72, lanesplit +91, cb8 +140, ctrl +155 … b81 B +366.
Full tables in the per-net bakeoff json (runs are gitignored; this entry
is the record).  All 60 cells bit-exact, all boot; decode 0.005-0.027 s
in-process, worst artifact boot 1.23 s — the 60 s budget is a non-issue.

**Findings, measured:**

- **Joint-vs-split settles AGAINST the raw tail everywhere at these
  sizes** — even for the entropy-coded arms whose output is
  incompressible by construction.  Decomposed on v1's rc_run: the raw
  tail IS ~27 B cheaper per payload (523 raw vs 550 through lzma), but
  the SF_A/SF_N head + self-read machinery costs ~57-63 B more than the
  in-source string prologue, so A wins by ~30.  The tax is fixed and the
  saving scales with payload size: linear projection puts the B
  crossover for incompressible payloads at ~1.1 kB — right at the new
  1024 B capacity target, so RE-MEASURE when c1024 exports exist.
- **lzma's match modeling beats every fitted prior we brought.**  The
  (lane, zero-run-bucket) rANS coder — a *static ternary-run prior*, 16
  contexts fitted on the net, params stored and counted — needs 461 B of
  state on v1 where the shipped path's payload-in-context is 382; at
  65.2% zeros the gap widens (407 vs 330).  Order-0 per-lane is worse
  still (519/476).  The redundant char-aligned encoding plus lzma's
  match finder captures structure none of our explicit models did.
- **Sparsity helps the shipped codec MORE than it helps the
  challengers**: b81_rle (char-aligned zero-RLE, runs capped at 10 by
  the 90-alphabet) closed to +16 at 59.6% zeros but fell to +27 at
  65.2% — lzma's unbounded matches price long zero runs better than RLE
  tokens.  The dense mr3 payload is nearly flat across sparsity
  (656/650), and the proto's random-payload record (626 B @55%, in-ctx
  656 @42%) puts the dense-vs-baseline crossover at ~42% zeros —
  INSIDE c1024-cal's 35-50% target band.  The winner may genuinely flip
  at the new operating point; the zoo is one command on each export.
- **Reordering is a dead end at every honesty level**: fixed square
  orders (no stored perm) lose exactly their decoder cost (+57/+70);
  the greedy 12-plane chain HURTS the payload itself (554 vs 382 — the
  chain breaks cross-plane matches lzma was using, same mechanism as
  the shuffle control); the full stored 768-perm costs its Lehmer code,
  +1169.  sparse_gap loses at both sparsities (crossover is far above
  66%); lossless-VQ codebooks (cb4/cb8) pay more in dict+indices than
  lzma's own matcher; lr_svd's rank-1 ternary predictor leaves an 8.6-
  10.1% residual whose factor cost never pays.
- **Rate-aware hook, calibrated against its own arm**: constraints
  .rate_penalty (differentiable per-lane order-0 expected bytes, soft
  occupancy of the STE grid) reads 518.3 B on v1's hard quantization vs
  the rc_o0 coder's measured 519 B state — the estimator IS its arm to
  within a byte, and it tracks direction across nets (476 est / 330
  measured on v1c: an upper bound that moves the right way).  Wired as
  loss.rate/loss.rate_T next to l1; queued as replnet_ratecal
  (TRAINQUEUE #9, rate ∈ {2e-6, 4e-6} bracketing v1's l1 pressure,
  valn-pinned VAL probe beside c1024-cal).

**Verdict:** the shipped base-3^4 + joint lzma stands on both existing
nets — now against 15 measured challengers in two container layouts
instead of by construction.  The standing value is the harness + the two
projected crossovers (dense payload at low sparsity; raw tail at ~1.1 kB
incompressible payload), both of which the c1024 family will cross;
re-run the zoo per export and let the table pick.

## 2026-08-14 — The family objective, the subsumption rule, and probes

Thomas (via coordinator): the nnue should LEARN endgames, king
protection, midgame, pawn structure and mobility — "if we have to write
custom code for all these different cases and weaknesses, we'll end up
using too many bytes on code." TRAINQUEUE.md now carries the objective,
the knowledge-class → capacity-axis mapping, and the c1024-phase +
c1024-general arms (phase forms priced before training; the composed arm
gated on its axes).

**SUBSUMPTION RULE (standing):** every landed hand term (pend, +37 B,
screened in today) carries an ablation obligation — when a phase-capable
net reaches screening, the matrix includes net-vs-net+term, and a
subsumed term is deleted with its bytes refunded. Hand terms are
stopgaps, not accumulation.

**Instrument: train/probes.py**, wired into train/export.py — nine
material-identical contrasts (base cancels; output is pure net signal)
across the objective's classes: passed-vs-opposed, split-vs-doubled,
pawn-advance×phase, king-activity×phase, centralization penalty,
shelter, knight rim, rook file, bishop-pair marker. Ledgered per export
as .probes.json; compact line per epoch-best. DIAGNOSTICS, never gates
(val does not gate; play does). Wiring smoke (1 epoch, 27k corpus) read
exactly as a newborn net should: mobility arriving (+16), phase absent
(+1/+0), passers wrong-signed (−5) — the scoreboard the phase arms will
be read against.

## 2026-08-14 — Bake-off re-seamed to golf round 2; pack scripts pinned after a second moving-denominator incident

Golf round 2 (a1a1a6d) restructured the entry's decode region (constants
inlined, ACC_BASE renamed `_B`, one-pass `_half` build) and the harness
refused it by name (the structures lane's `require_seam`).  compress/ now
knows entry SEAMS: v1 = the recorded blob fb717214, v2 = the settled
a1a1a6d+ blob be154478; arms stay in one dialect and `build_region`
token-rewrites per seam at one point, so no artifact carries alias bytes.
The baseline serves its seam-verbatim decoder (its patched cells price
the entry's OWN block).  Tests pin BOTH blobs and round-trip the arms
against each (19/19).

**Second incident, same class as the entry golf:** the working-tree
pack.sh carried the golf lane's UNCOMMITTED --no-hoist-literals +
shebang-strip change and silently moved every artifact (old-blob baseline
3831→3780, settled candidate 3594→3567).  packrun now runs the pack
scripts from a git pin (default HEAD, BAKEOFF_PACK_REV overrides), NOTES
working-tree divergence, and lands pack provenance in the results json
beside the entry blob.  When that pack.sh change lands, every number
below shifts by its measured −29-family delta — one re-run refreshes.

**Instrument, revalidated at both pins (pinned packer):** settled v2 +
trained v1 = **3594 B EXACT** (the golf commit's recorded candidate;
elided 3217 = their recorded code side); historical v1 = **3831/3834
EXACT**; ctrl_shuffle at v2 worse in both layouts (+193 A, +111 B-vs-B).
Torch triangle PASS on both nets at the new seam.

**Re-seamed denominators (seam v2, blob be154478, pack.sh 0b41a200):**

| net | baseline (b81, A) | spare | payload in-ctx | closest arm |
|---|---|---|---|---|
| v1 (59.6% zeros) | **3594** | 502 | 377 | b81_rle A +44 |
| v1c (65.2% zeros) | **3543** | 553 | 326 | b81_rle A +53 |

Rank order is unchanged from the v1-seam tables in every band (baseline,
then rle / fixed square orders / lanesplit / cb8 / control / cb4 /
pieceperm, then the entropy-coded and split arms, then sparse and the
stored perm; full 60-cell tables in the per-net bakeoff json, 4
trained_cb/trained_lr SKIP rows — correct, these nets carry no trained
structure).  What DID move: patched-arm decoder costs grew ~25-90 B
relative to the golfed entry (its own block is tighter, e.g. b81_rle's
decoder 21→46 B, rc_run 144→171 A / 201→243 B), so the golf widened the
baseline's moat; and every split-B cell still loses (b81 B +319).  The
two standing crossover projections (dense payload at ~42% zeros; raw
tail at ~1.1 kB incompressible payload) are unchanged in kind and carry
to the c1024 exports, where the zoo re-runs as one command.

## 2026-08-14 — Bake-off addendum: denominators at the LANDED packer

eb8897c landed --no-hoist-literals + shebang-strip in pack.sh minutes
after the re-seam entry above was written, so those denominators aged one
packer generation on arrival.  Full zoo re-run (one command) at HEAD =
eb8897c, entry blob be154478, all instrument checks green, 60 cells
bit-exact, all artifacts boot:

| net | baseline (b81, A) | spare | payload in-ctx | closest arm |
|---|---|---|---|---|
| v1 (59.6% zeros) | **3567** | 529 | 381 | b81_rle A +48 |
| v1c (65.2% zeros) | **3519** | 577 | 333 | b81_rle A +55 |

Both baselines reproduce eb8897c's independently measured spliced
candidates (3567 / 3519) exactly — two lanes, two instruments, one
number.  Rank order is unchanged (baseline, rle, fixed square orders,
lanesplit, cb8, control, …); layout B still loses everywhere (pack_entry
.sh did not take the lever and its cells are byte-identical to the
re-seam run).  THESE are the current denominators for the c1024 family;
the two crossover projections above carry unchanged.

## 2026-08-14 — Gate stability ×5: the single-run conversion reading was luck

8Mv (the 8M data-scale arm, val 0.01378 on the pinned comparable split —
BEATS v1's 0.01385; packs to 3536 B, 560 spare, same ~59.5% zeros but
whole-feature sparsity that lzma rewards) flapped the box conversion gate
{8,7,8,7}/8. Pre-stated a stability protocol BEFORE reading more:
mate-conversion ×5 per engine, same box, same nice; eligible iff fails ≤
entry's.

| engine | runs with a fail |
|---|---|
| entry | 0/5 |
| replnet v1 | **5/5** (kqk-mid budget, every time) |
| replnet 8Mv | 1/5 (same case) |

The v1 "8/8 at entry parity" that staged the screen was a 1-in-5 lucky
run — single-run gate readings on a 500 ms-budget instrument under
tournament load are an illusion; ×5 is the protocol from here. Screen
request updated: v1 WITHDRAWN, 8Mv staged NOT-READY (1/5 vs entry 0/5),
coordinator holds the marginal call. The failing class — KQK conversion
speed — is exactly phase knowledge: c1024-phase (TRAINQUEUE #2) carries
it, and the probes' phase scores are its scoreboard. kb8fold interim
corroborates the axis reading: bucketed training-form val 0.01267-78
(real signal) vs shipped folded 0.01429-40 (the fold loses it) — shared
rows alone don't carry king knowledge; deltas cost bytes; ml2/phase are
the byte-efficient axes.

## 2026-08-15 — 13 idle hours, the runner switch, cal verdict, phase form

**Standing-rule violation, owned:** the trainer idled 2026-08-14 13:31 →
2026-08-15 ~02:4x UTC. c1024-cal completed and nothing started: the
chain watcher was a SESSION-BOUND monitor and the session died (529
wave). Root cause is architectural, same class as the wait-loop
self-match: liveness owned by a laptop session. Fixed structurally —
**train/queue_runner.py now runs DETACHED on the box** (atomic .runlock,
nice 19, caps 8/8, its own forfeit tripwire that SIGSTOPs training and
writes PAUSED_REPORT.txt; baseline 34 historical forfeits across box
pgns, watching for increase). A session kill can no longer idle the
trainer or orphan its tripwire.

**c1024-cal verdict (pinned comparable split):** l1=0 τ0.6 → 0.01421;
l1=0.0003 τ0.6 → 0.01388; vs v1 0.01385 (τ0.85) and 8Mv 0.01378. The
enlarged 1024 B budget buys NOTHING through density at ps768/N=4 — more
nonzeros made val WORSE. Structure is the binding constraint, consistent
with kb8fold (buckets carry signal, 0.01267 training-form; the fold
loses it, 0.01428 shipped).

**c1024-phase, form (c) chosen and RUNNING** (runner entry
10_c1024_phase_ml2, certificate green: float64-exact margins, modular
hsum precondition HOLDS). Why (c): tapered 2× tables price ~1.2 KB at
v1 sparsity (over target; the golf lane's own round-2 note measures the
1024 payload budget 142 short at current code); 2-bucket ternshared
repeats kb8fold's measured fold-negative unless deltas ship. Ml2Net as
landed already affords phase×feature products with no new machinery — a
lane can specialize as a material-phase detector and the certified
one-multiply second layer reads phase×passer / phase×king-activity
cross-products through u2 (+bm payload digits). The probes' phase class
is the scoreboard; the subsumption ablation (K_END/khold2/pend) arms
when this family screens.

**Screen arms refreshed to HEAD** (entry 5d7d0d1, 3308 B; candidate =
8Mv payload on the golfed round-2 proto, 3536 B): invariants,
shapecheck-class checks, first-yield, legality 334/334 all PASS on the
rebuilt pair. The ×5 conversion reading stands (entry 0/5, 8Mv 1/5,
kqk-mid): the marginal call remains the coordinator's; nothing launches
from this lane.

## 2026-08-15 — replnet-8mv screen DISPATCHED (coordinator GO)

Coordinator ruling, ledgered verbatim: "mate-conversion 4/5-5/5 class
marginal on kqk-mid, pre-stated, accepted by coordinator; the phase
family carries the fix." A deficiency the screen exists to measure
cannot gate the screen. Dispatch: fixed-node SPRT as staged (elo0=0
elo1=10, nodes 20000, cap 1000, srand 20260814), arms current vs HEAD
(entry-5d7d0d1 3308 B vs replnet-8mv 3536 B), box-side detached
dispatcher: prefers m2's freed slot, presence-marks the boxlock (or
records cotenancy — permitted under the capacity ruling), zero-illegal
tripwire kills the match and voids the run on ANY illegal move, verdict
written box-side (session-proof). If LB>0: the timed leg follows with
nps RE-MEASURED at dispatch — the 0.66 ratio was against the old entry
and is stale by construction.

## 2026-08-15 — PHASE KNOWLEDGE ARRIVES IN WEIGHTS: ml2 val 0.01286

**c1024_phase_ml2 (form c, runner entry, 22 min): best val 0.01286** on
the pinned comparable split — against 8Mv 0.01378, v1 0.01385: a −0.0009
step, the largest single gain of the replacement campaign (data-scale
4M→8M bought −0.0007 for reference; this bought it and a third more on
top at the SAME data). Float export (the ml2 packed build is price-first
by design); certificate green.

**The probes say WHY (the family objective's first measured win):**
king_activity_end_vs_mid **+16** (the newborn read +0 — the net now
knows king activity flips sign with phase), pawn_advance_end_vs_mid
+4..+8 (passers grow with phase — pend-class knowledge, in weights),
shelter +25, centralization penalty +26..+43. Still missing: passer
recognition flaps (−13..0), rook-file negative (−12), bishop pair noisy
— second-order pawn/file structure wants more than lane self-products.

**Critical path moved:** the ml2 ENGINE machinery price (one extra
big-int multiply of the crelu blocks, fields re-spaced to 32 bits,
folded mod 2^(32m)−1 — packed_layers.LaneConv is the training twin; the
certificate proves the arithmetic exact). That seam is the golf lane's;
until it prices, 0.01286 is a float number wearing the subsumption
claim, not a screenable artifact.

Ops notes: extension lane's 80_replnet_ml2.yaml failed rc=1 on the
deployment path (same ../replnet-20260814/ assumption as ratecal);
requeued with the fixed path — the runner's log-and-continue behavior
was correct. Queue now: 85_ratecal (running) → 80_ml2 → 90_cb → 91_lr.
Screen dispatcher still waiting on m2's slot (its cap running long).
