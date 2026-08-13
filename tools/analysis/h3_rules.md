# H3 disambiguation rules (pre-registered)

Frozen BEFORE computing, 2026-08-14, follow-up to `loss_rules.md`. Question: in the 34
below-median-depth (d:LOW) windows preceding the entry's middlegame collapses (pyleague,
300+0), is the depth crater (a) QSEARCH/TREE EXPLOSION (same wall budget buying fewer
plies) or (b) TM COMMIT/ALLOCATION (normal search, badly allocated budget)? Log parsing
only; no engines. `tools/analysis/h3_log_mining.py` implements this file and refuses to
print a verdict if the instrument-sanity control fails.

## Telemetry ledger (checked before registration)

| source | telemetry | usable? |
|---|---|---|
| laptop pyleague `fastchess.log` (snapshot 2026-08-14 ~01:15, 134 MB, ladder mid-run, copied once) | full UCI comm. ENTRY: `info depth/score/pv` only — NO nodes, and output arrives in one buffered burst (all info timestamps within ~0.2 ms), so per-depth wall times are unrecoverable. Per-move wall time (go->bestmove receipt) and exact `go wtime` ARE recoverable. CLASSIC: full `depth/time/nodes/nps` per line, real per-line timestamps | YES (primary) |
| the bench box `eval-c1-20260813/{c1,c2,d1}screen.log` | driver stdout; engine info lines echoed ONLY on PV warnings (5.5k of ~35k moves, biased sample) | NO — the planned fixed-node nodes-per-depth test is impossible from existing logs |
| the bench box `elo-noiid/stdout.log`, `elo-fresh-king-score/match.log` | driver progress lines only, no UCI | NO |

Consequence, stated up front: nodes are unobservable for the entry everywhere. (a) can only
be detected via STOP-MODE and BUDGET-RESIDUAL signatures plus a policy-artifact test; the
definitive nodes-level answer stays with the engine re-analysis probe.

## Entry TM policy (from source, `nnue_4k/pst_entry.py` go handler)

`think = min(wtime/12 + 0.9*winc, wtime/2 - 1000)` ms (the `/40` sudden-death divisor at
line 789 sits in a minifier-hide block and is NOT in the artifact; observed ladder times
~25 s on move 9 at 300 s confirm /12 is live). For wtime >= 5000 and winc=0 this is exactly
`wtime/12`. Search soft-breaks at the first MTD probe boundary after `0.8*think` and
hard-aborts (Stop, checked every 2048 nodes) at `think`. Therefore per-move F =
t_move/(wtime/12) lies in ~[0.8, 1.05]:
- F >= 0.95 = HARD-ABORT (the running iteration could not fit — expensive tree),
- F < 0.95 = SOFT-BREAK (loop chose to stop with ~20% budget unused — normal completion).

## Move sets (entry own moves, matched log<->PGN by exact UCI move sequence; middlegame
filter: fullmove >= 15, npm > 12, wtime >= 5000 ms; PGN snapshot = the taxonomy snapshot)

- W: the 5 own moves strictly before the swing in each of the 34 d:LOW MIDDLEGAME
  SELF-DETECTED losses (game list and swing moves come from re-running loss_mining on the
  same snapshot).
- B: other pre-swing middlegame own moves of those same 34 games (outside W).
- SM (selection-matched control): non-loss entry games, pseudo-swing at fullmove 32 (the
  loss median); keep games whose pseudo-window is d:LOW by the same rule (mean of 5 own
  depths vs game median, delta <= -1.0); their windows.
- PC (positive control, unselected): pseudo-windows at fullmove 32 in ALL non-loss entry
  games with >= 5 scored middlegame own moves before it.

## Statistics (all medians; computed once, no peeking-and-tuning)

- P0 policy gate: >= 80% of all matched entry moves (wtime >= 5000) must have
  F in [0.75, 1.10], else verdict = LOGS-INSUFFICIENT (unknown build/policy).
- R0 policy-artifact rate: r_loss = 34/94 (share of mid-SD losses with d:LOW window, from
  the taxonomy) vs r_nonloss = share of PC-eligible games whose pseudo-window at 32 is
  d:LOW (robustness: also at 28 and 36; use the 32 value for the verdict).
- H hard-abort rate: P(F >= 0.95) in W, SM, B, PC.
- D0 budget-depth residual: OLS fit depth = a + b*ln(think) on PC-eligible non-loss
  middlegame moves; gap = median residual(W) - median residual(SM).
- G co-tenancy guard: classic nps (its own info lines) bucketed per 5 wall-clock minutes;
  a W move is confounded if its bucket median < 0.8x classic's global median nps. More
  than 1/3 confounded => LOGS-INSUFFICIENT.
- Instrument sanity (positive control): |H(PC) - H(B)| < 0.15 required — windows from
  games with NO collapse must classify as baseline, else no verdict is printed.

## Verdict rules (all four outcomes acceptable)

- (b) TM-ALLOCATION: r_nonloss >= 0.75 * r_loss AND H(W) - H(SM) < 0.15 AND gap > -0.5 ply.
- (a) EXPLOSION: [H(W) - H(SM) >= 0.15 OR gap <= -0.5 ply] AND r_nonloss < 0.75 * r_loss.
- MIXED: an (a) criterion fires AND r_nonloss >= 0.75 * r_loss.
- LOGS-INSUFFICIENT: P0 fails, or G trips, or fewer than 20 matched W moves.

## What the logs cannot show

No node counts for the entry, no per-depth times (buffered flush): a bushy tree that still
fits inside soft-break budgets is invisible to F and lands on D0/R0 alone; nps drift of the
entry itself (vs classic's, used by G) is unmeasured. The engine re-analysis probe
(LOSS_TAXONOMY.md) remains the definitive (a)-vs-(b) instrument; this analysis picks which
screen to run FIRST, not the final mechanism.
