# Loss-mining rules (pre-registered)

Frozen BEFORE bulk counting, 2026-08-14. `tools/analysis/loss_mining.py` implements exactly
these rules and must reproduce the positive controls below before it is allowed to emit a
taxonomy. PGN parsing only — no engine processes. Changing any threshold after the controls
run requires re-registering (new commit, note the change and why).

## Unit

One LOSS by a tracked engine: a decisive game (`1-0`/`0-1`) where the tracked engine is on
the losing side. Draws and unfinished (`*`) games are excluded from the taxonomy (counted in
the corpus ledger only).

## Engine-name mapping (ledgered from PGN `Engine*Name` tags)

| PGN player name | EngineName tag | identity |
|---|---|---|
| `sunfish4k` (pyleague), `base` (eval-c1 dir) | `sunfish 2026-packed` | THE ENTRY (nnue_4k/pst_entry.py lineage) |
| `c1`, `c2`, `d1`, `aa` (eval-c1 dir) | `sunfish 2026-packed` | entry eval-variants; `d1` is the known-bad eval |
| `classic` (pyleague), `master-*`, `master` (elo-* / gauntlet dirs) | `sunfish 2026` | sunfish-classic lineage |
| `d-house`, `numbfish`, `neurofish` (pyleague) | own names | python-league field opponents |

All `elo-*-20260813` and `*-gauntlet-20260812` bench-box corpora are classic-vs-classic
(EngineName `sunfish 2026` on both sides): control material, no entry games.

## Eval/depth extraction

fastchess move comments `{<score>/<depth> <time>s}`. Score `+0.34` is centipawns*0.01 from
the MOVER's own perspective; `+M5`/`-M3` are mates (mapped to ±50000 cp); sunfish-family
mate scores appear as `|score| >= 300.00` (e.g. `479.23`) and any |cp| >= 30000 is treated
as mate-magnitude. `book` comments carry no eval. A loss with fewer than 4 scored own moves
is classified `UNSCORED` and excluded from swing/depth cells (kept in totals).

Let e_1..e_k be the loser's own-move evals in cp (own perspective), d_i the depth recorded
with e_i.

## (b) Decisive swing — swing-class

- Candidate swing at own move i (i >= 2): `e_i - e_(i-1) <= -150` cp.
- RECOVERY HYSTERESIS: the candidate is voided if any later own eval `e_j > e_(i-1) - 50`
  (j > i), i.e. the eval later gets back within 50 cp of the pre-drop level. (Hand-check C2
  motivates this: a one-move `-15.11` spike at 28.Rd5 reverts next move; the real collapse
  is 39.Nd5.)
- Decisive swing = FIRST non-voided candidate. Its move number = swing move; the swing
  position = board before the loser plays that move.
- MATE-ANNOUNCEMENT RULE: if the first non-voided candidate has mate-magnitude `e_i` AND
  `e_(i-1) <= -100`, it is not a swing — the game had already slid away (hand-check C4).
- Class `SELF-DETECTED`: a decisive swing exists (the loser's own eval saw the drop
  immediately after its own move pair).
- Class `CREEPING`: no decisive swing (positional grind, or slide ending in a mate
  announcement per the rule above). Anchor move for phase/depth = first own move with
  `e_i <= -100` and no later own eval `> -100`; if none exists, the last scored own move.

## (a) Phase at the swing/anchor

Replay moves textually with python-chess (a move parser, not an engine). At the swing (or
creeping-anchor) position, npm = non-pawn material of BOTH sides, N=B=3, R=5, Q=9 (max 62).

- `ENDGAME`: npm <= 12
- `OPENING`: fullmove <= 14 AND npm >= 24
- `MIDDLEGAME`: otherwise

## (c) Depth signal (time-pressure proxy)

window = recorded depths of the 5 own moves strictly before the swing/anchor move (>= 2
required, else `UNKNOWN`). delta = mean(window) - median(all scored own-move depths in the
game).

- `LOW` (time-pressure-like): delta <= -1.0
- `HIGH`: delta >= +1.0
- `NORMAL`: otherwise

In fixed-node corpora (`TimeControl "-"`, the eval-c1 dir) this proxies position complexity,
not clock pressure — interpreted as such, never as time trouble.

## (d) Termination class

From the `Termination` tag plus the final move comment:
- `MATE`: final comment contains "mates"
- `TIME`: Termination "time forfeit" or final comment contains "on time"
- `ABANDONED`: Termination "abandoned"
- `ADJUD`: Termination "adjudication" (score/draw-rule adjudication) otherwise
- `OTHER`: anything else

## Secondary, pre-registered: OPTIMISM (eval-disagreement proxy)

For own move i with eval e_i and the opponent's next recorded eval o_i (opponent
perspective): disagreement D_i = e_i + o_i (cp). Both accurate => D ~ 0; D >> 0 means the
loser is more optimistic than its opponent. OPTIMISM(loss) = mean of D_i over the 6 own
moves strictly before the swing/anchor (mate-magnitude values excluded; >= 2 pairs required,
else null). Calibration logic: d1 (known-bad eval) losses should show large positive
OPTIMISM; classic-vs-classic control should sit near 0. The entry's position between them is
the PGN-visible eval-wrongness signal.

## Positive controls (hand-labeled before the classifier ran)

Keys are (corpus, 1-based game index in the snapshot, White, Black, Result). Snapshots are
fixed copies; pyleague is an append-only file so early indices are stable.

| # | corpus | idx | White | Black | Result | loser | class | swing/anchor move (side) | phase | depth | term |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | pyleague | 1 | d-house | sunfish4k | 1-0 | sunfish4k | SELF-DETECTED | 17 (B) | MIDDLEGAME | NORMAL | MATE |
| C2 | pyleague | 2 | sunfish4k | d-house | 0-1 | sunfish4k | SELF-DETECTED | 39 (W) | ENDGAME | HIGH | ADJUD |
| C3 | pyleague | 4 | sunfish4k | classic | 0-1 | sunfish4k | SELF-DETECTED | 42 (W) | ENDGAME | NORMAL | ADJUD |
| C4 | elo-noiid | 2 | master-697d69a | noiid-53b35eb | 0-1 | master-697d69a | CREEPING | 18 (W) | MIDDLEGAME | NORMAL | MATE |
| C5 | d1screen | 4 | base | d1 | 1-0 | d1 | SELF-DETECTED | 29 (B) | MIDDLEGAME | NORMAL | ADJUD |

Hand-reasoning kept short: C1 12...Qxb2 drop is voided by recovery (+0.61 > +0.32-0.50);
decisive is 16...Qa3 -0.95 -> 17...Kf8 -3.95. C2 28.Rd5 spike voided by hysteresis;
decisive 38.bxa4 -1.68 -> 39.Nd5 -3.76 in a B-vs-N pawn endgame (npm 6). C3 41.h3 -0.46 ->
42.bxc3 -4.78, bishop endgame (npm 6). C4 slides -0.01 -> -1.11 -> -3.24 with no 150 cp
step; first qualifying drop is the mate announcement from -3.24 => CREEPING, anchor 18.Bg3
(first eval <= -1.00 that never recovers). C5 28...Nxa1 +3.95 -> 29...Rxf7 +1.66 (drop
-229, never back above +3.45), middlegame npm ~45.

## What PGN-only analysis cannot see (honest limits)

- It cannot separate "the eval was WRONG" from "the search was too shallow to see the
  refutation": both look like a SELF-DETECTED swing. OPTIMISM is a proxy (sustained
  disagreement with a stronger opponent's eval leans eval-wrongness), calibrated by the d1
  corpus, but the clean disambiguation needs engine re-analysis of the swing positions
  (fixed-depth probe: does the entry's OWN eval at depth 1 already mis-score the post-swing
  position, or does deeper search fix the move choice?). That follow-up runs in a bench-box
  slot later; its spec ships in nnue_4k/LOSS_TAXONOMY.md. Nothing is launched now.
- Recorded evals are the engine's claims mid-search (aspiration artifacts, hash effects);
  single-move spikes are handled by hysteresis but systematic reporting quirks are not.
- Adjudicated games hide how the losing side would actually have been finished off.
