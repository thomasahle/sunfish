<!--
Strength claims need numbers. Functional tests say the engine is CORRECT;
they say nothing about whether it is STRONGER. Full methodology and the
reasoning behind each rule: docs/TESTING.md
-->

## What this changes

<!-- One paragraph. What moved, and why. -->

## Strength measurement

<!--
REQUIRED for any change to search, evaluation, tables, or time management.
For docs / tooling / test-only changes, delete this section and say which.
-->

| | |
|---|---|
| Base | `<commit or branch>` |
| Test | `<commit or branch>` |
| Time control | `<e.g. 30+1>` |
| Book | `<name>`, `<N>` positions |
| Games | `<W>-<L>-<D>`, `<total>` |
| **Elo** | **`+X ± Y`** (95%) |
| nElo | `+X ± Y` |
| SPRT | `elo0=0 elo1=10 alpha=0.05 beta=0.05` → accepted / rejected / undecided at cap |
| Time losses | `0` |
| Illegal moves / `(none)` | `0` |

<details>
<summary>Raw fastchess output</summary>

```
<paste the final result block>
```
</details>

### Measurement checklist

- [ ] **Both engines frozen** with `git archive`, including `sunfish_ui/` — no shared files
- [ ] **Machine otherwise idle** for the whole match (a timed match measures thinking time; contention corrupts it)
- [ ] **Book** has at least as many positions as the match has rounds, run with `-games 2` for colour-swapped pairs
- [ ] **`-recover` passed**, and the finished game count verified **against the PGN**, not the log
- [ ] **Zero time losses and zero illegal moves** — if not, the run measures a bug, not a change
- [ ] Intervals quoted at **95%**, not 1σ
- [ ] The **accept/reject bar was written down before the result was read**

### Things that quietly invalidate a result

- **Fixed nodes is only valid between engines that honour the node cap the same way.** The cap is checked between completed depths, so an engine that prunes less overshoots further and silently gets more search. Against an engine without a mid-search cap, use a **time control**.
- **An SPRT pass is not an effect size.** Stopping happens when the estimate wanders far enough, so the reported Elo is biased away from zero. If you need a number, confirm the winner at fixed N.
- **A mate suite is not a legality gate.** "Finds mates" and "always returns a legal move" are different questions; a feature can pass the first while emitting `bestmove (none)`.
- **Offline metrics validate models, never pipelines.** A better loss can ship a worse engine if the emit path is wrong.
- **Elo is not additive across engines** that differ in more than one way. Measure the pair you care about rather than subtracting two other results.

## Correctness

- [ ] `pytest` green
- [ ] `python3 formal/scripts/model_audit.py` green — if a modelled region changed, the Lean model and README were re-audited **in the same commit**
- [ ] `lake build` green (if `formal/` changed), zero `sorry`

## Size — packed / 4k entry only

<!-- Delete unless this touches the packed engine or the 4k entry. -->

| | bytes | spare |
|---|---|---|
| entry before | | |
| entry after | | |

- [ ] Byte counts come from **`tools/build/pack.sh` on a real file** — never a sum of separately measured parts
- [ ] Artifact runs **alone in an empty directory** with `SF_NET` unset, and leaves nothing behind
- [ ] `tools/build/check_entry.sh` green (catches source drift that leaves the byte count unchanged)
- [ ] Elo per byte stated, and the keep/drop bar pre-registered
