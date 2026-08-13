<!--
Strength claims need numbers. Functional tests say the engine is CORRECT;
they say nothing about whether it is STRONGER. Methodology: docs/TESTING.md
-->

## What this changes

<!-- One paragraph. What moved, and why. -->

## Strength

<!--
REQUIRED for any change to search, evaluation, tables, or time management.
For docs / tooling / test-only changes, delete this section and say which.
-->

```
LLR: <x> (<lower>,<upper>) <elo0,elo1>
Total: <N> W: <w> L: <l> D: <d>
Ptnml(0-2): <ll>, <ld>, <dd+wl>, <wd>, <ww>
Elo: <+x.xx +/- y.yy> (95%)
```

<!-- fastchess prints all of this; run with `-repeat -games 2` for Ptnml. -->

- [ ] Both engines frozen with `git archive`, including `sunfish_ui/`
- [ ] Machine otherwise idle for the whole match — contention corrupts a timed result
- [ ] Time control stated and appropriate: **30+1 minimum** for a decision
- [ ] Book has at least as many positions as rounds, run `-repeat -games 2`
- [ ] `-recover` passed, game count verified **against the PGN**, not the log
- [ ] **Zero time losses, zero illegal moves** — otherwise the run measures a bug
- [ ] Bounds written down **before** the result was read

## Size

<!-- Both engines, both numbers. Measured, never estimated. -->

|  | classic lines | classic bytes | nnue-4k lines | nnue-4k bytes |
|---|---|---|---|---|
| before | 138 | 3234 | 213 | 3882 |
| after |  |  |  |  |

```sh
bash tools/build/clean.sh sunfish.py | wc -l          # minified lines
bash tools/build/pack.sh  sunfish.py /tmp/out         # packed bytes
bash tools/build/clean.sh nnue_4k/sunfish_nnue.py | wc -l
bash tools/build/pack.sh  nnue_4k/sunfish_nnue.py /tmp/out
```

- [ ] Numbers from the commands above on a real file — never a sum of parts
- [ ] 4k entry (if touched) runs **alone in an empty dir** with `SF_NET` unset, and `tools/build/check_entry.sh` is green

## Correctness

- [ ] `pytest` green
- [ ] `python3 formal/scripts/model_audit.py` green — if a modelled region changed, the Lean model and README were re-audited **in the same commit**
- [ ] `lake build` green (if `formal/` changed), zero `sorry`

<details>
<summary>Things that quietly invalidate a result</summary>

- **Fixed nodes only compares engines that honour the cap the same way.** It is checked between completed depths, so the engine that prunes less overshoots further and silently gets more search. Against an engine without a mid-search cap, use a time control.
- **An SPRT pass is not an effect size.** Stopping happens when the estimate wanders far enough, so the reported Elo is biased away from zero. Confirm a winner at fixed N.
- **A mate suite is not a legality gate.** "Finds mates" and "always returns a legal move" are different questions.
- **Offline metrics validate models, never pipelines.** A better loss can ship a worse engine if the emit path is wrong.
- **Elo is not additive** across engines that differ in more than one way. Measure the pair you care about.
</details>
