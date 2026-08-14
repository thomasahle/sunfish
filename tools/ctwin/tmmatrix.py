#!/usr/bin/env python3
"""Run a set of virtual-clock arms against one baseline and rank the results.

Because nothing in the surrogate reads a wall clock, cells can run CONCURRENTLY
without contaminating each other -- the thing that makes a real TM matrix
expensive (every arm needs the box to itself) does not apply here.  That is
the whole reason a matrix is affordable at all.

Cells are `manager[:knob=v...]` specs, one per --arm, each played against
--baseline at every --tc.  Dominated cells are pruned in the report rather
than in the run: a cell that loses on Elo AND spends more AND flags more is
not interesting, but it still cost nothing to measure and its number is
printed.

RANKING IS NOT DECIDING.  The output of this script is an ORDER, and the
recommendation at the bottom is the single composite that earns the one real
wall-clock match (docs/TESTING.md rule 15).
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_cell(arm, baseline, tc, rounds, overhead, elo0, elo1, seed, outdir):
    tag = "%s__vs__%s__%s" % (arm.replace(":", "_").replace("=", ""),
                              baseline.replace(":", "_").replace("=", ""),
                              tc.replace("+", "p"))
    path = os.path.join(outdir, tag + ".json")
    if os.path.exists(path):
        return arm, tc, json.load(open(path))
    cmd = [sys.executable, os.path.join(HERE, "vmatch.py"),
           "--arm-a", arm, "--arm-b", baseline, "--tc", tc,
           "--rounds", str(rounds), "--overhead", str(overhead),
           "--elo0", str(elo0), "--elo1", str(elo1), "--seed", str(seed),
           "--json", path, "--quiet"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("cell %s @ %s FAILED\n%s\n%s" % (arm, tc, p.stdout, p.stderr))
    return arm, tc, json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tc", action="append", required=True)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--overhead", type=float, default=0.05)
    ap.add_argument("--elo0", type=float, default=-10.0)
    ap.add_argument("--elo1", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--outdir", default=os.path.join(HERE, "matrix"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cells = [(a, tc) for a in args.arm for tc in args.tc]
    results = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run_cell, a, args.baseline, tc, args.rounds,
                          args.overhead, args.elo0, args.elo1, args.seed,
                          args.outdir) for a, tc in cells]
        for f in concurrent.futures.as_completed(futs):
            arm, tc, out = f.result()
            results[(arm, tc)] = out
            print("done %-28s %-8s elo %+7.1f [%+.0f,%+.0f] %d games"
                  % (arm, tc, out["elo"], out["ci"][0], out["ci"][1],
                     out["games"]), flush=True)

    print("\nMATRIX vs %s   (virtual clock, overhead %.3fs, %d rounds/cell)"
          % (args.baseline, args.overhead, args.rounds))
    print("%-26s %-8s %8s %-18s %6s %7s %7s %7s %6s"
          % ("arm", "tc", "elo", "95% CI", "games", "medspd", "base", "blind%",
             "flags"))
    rows = []
    for (arm, tc), o in sorted(results.items(), key=lambda kv: -kv[1]["elo"]):
        me, base = o["arms"][arm], o["arms"][args.baseline]
        rows.append({
            "arm": arm, "tc": tc, "elo": o["elo"], "ci": o["ci"],
            "games": o["games"], "median_spend": me["median_spend"],
            "baseline_spend": base["median_spend"],
            "blind_pct": 100.0 * me["blind"] / max(me["moves"], 1),
            "flags": me["flags"], "baseline_flags": base["flags"],
            "verdict": o["verdict"],
        })
        print("%-26s %-8s %+8.1f [%+6.0f,%+6.0f] %6d %7.3f %7.3f %7.1f %6d"
              % (arm, tc, o["elo"], o["ci"][0], o["ci"][1], o["games"],
                 me["median_spend"], base["median_spend"],
                 rows[-1]["blind_pct"], me["flags"]))

    # Dominated: something else at the same TC is better on Elo AND does not
    # flag more.  Reported, never silently dropped.
    dom = []
    for r in rows:
        if any(o["tc"] == r["tc"] and o["elo"] > r["elo"]
               and o["flags"] <= r["flags"] and o["arm"] != r["arm"]
               for o in rows):
            dom.append("%s @ %s" % (r["arm"], r["tc"]))
    if dom:
        print("\nDOMINATED (beaten on Elo without flagging less): %s"
              % ", ".join(dom))
    json.dump(rows, open(os.path.join(args.outdir, "ranking.json"), "w"), indent=1)
    print("\nRANKING IS NOT A VERDICT: the top composite earns ONE real-clock "
          "match plus the 1+0 hammer (docs/TESTING.md rule 15).")


if __name__ == "__main__":
    main()
