"""Where does the teacher's value stop moving? Read the sweep and choose N.

"Label with our own search" is not a spec until the node budget is one. Too
low and the teacher is the student -- distillation from a frontier the engine
already reaches teaches it nothing. Too high and we pay for depth the 384
parameters cannot represent anyway, and buy tactical noise with it.

The choice is made on two curves, not on one:

  * SUCCESSIVE deltas -- how much does the label move when the budget
    doubles-and-then-some? This needs no reference and cannot be gamed by
    calling the deepest run "truth".
  * agreement with the deepest budget measured, as a cross-check.

Reported alongside the cost, because a budget we cannot afford to run over the
whole set is not a candidate.

usage: distill_converge.py OUT_DIR_OR_FILES...
"""
import glob
import json
import sys

import numpy as np

files = []
for a in sys.argv[1:]:
    files += sorted(glob.glob(a + "/*.jsonl")) if not a.endswith(".jsonl") else [a]
recs, meta = [], None
for f in files:
    for ln in open(f):
        o = json.loads(ln)
        if "meta" in o: meta = o["meta"]; continue
        recs.append(o)
keys = [k for k in recs[0] if k.startswith("n")]
Ns = sorted(int(k[1:]) for k in keys)
print("teacher %s  sha %s  interpreter %s"
      % (meta["teacher_version"], meta["teacher_sha256"][:12], meta["interpreter"]))
print("%d positions, budgets %s\n" % (len(recs), Ns))

cp = {N: np.array([r["n%d" % N]["cp"] if r["n%d" % N]["cp"] is not None else np.nan
                   for r in recs], dtype=float) for N in Ns}
dep = {N: np.array([r["n%d" % N]["depth"] for r in recs], dtype=float) for N in Ns}
used = {N: np.array([r["n%d" % N]["nodes"] for r in recs], dtype=float) for N in Ns}
fy = np.array([r["n%d" % Ns[-1]]["first_yield"] or -1 for r in recs], dtype=float)
ok = ~np.isnan(cp[Ns[0]])
for N in Ns: ok &= ~np.isnan(cp[N])
print("labelled at every budget: %d/%d" % (ok.sum(), len(recs)))

ref = cp[Ns[-1]][ok]
print("\n%-9s %-6s %-9s %-9s %-8s %-9s %-8s" %
      ("budget", "depth", "med|d| vs", "mean|d|", ">25cp", "r(deepest)", "nodes"))
print("%-9s %-6s %-9s %-9s %-8s %-9s %-8s" %
      ("", "mean", "previous", "vs prev", "vs prev", "", "actual"))
prev = None
for N in Ns:
    v = cp[N][ok]
    if prev is None:
        md = mn = fr = float("nan")
    else:
        dd = np.abs(v - prev)
        md, mn, fr = np.median(dd), dd.mean(), 100 * (dd > 25).mean()
    r = np.corrcoef(v, ref)[0, 1]
    print("%-9d %-6.2f %-9.1f %-9.1f %-8s %-9.4f %-8.0f"
          % (N, dep[N][ok].mean(), md, mn, "%.1f%%" % fr if fr == fr else "-", r, used[N][ok].mean()))
    prev = v

print("\nSpread of the label itself: sd %.0f cp, |cp|>500 on %.1f%% of positions"
      % (ref.std(), 100 * (np.abs(ref) > 500).mean()))
print("First yield at the deepest budget: median %d, max %d nodes"
      % (np.median(fy[fy >= 0]), fy.max()))
