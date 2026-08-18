#!/usr/bin/env python3
"""Reject C-twin policies that lose Sunfish's eventual-mate guarantees."""

import json
import pathlib
import re
import shlex
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INFO = re.compile(r"info depth (\d+) .* score (-?\d+)")
SUITES = (("mate1.fen", 1, 8, 8),
          ("mate2_eventual.fen", 2, 5, 5),
          ("mate3_eventual.fen", 3, 2, 2))


def mate_depth(options, moves):
    """Uniform depth bound for this policy's mate-in-moves gate."""
    k = 2 * moves - 1
    null_limit = options.get("NULL_LIMIT", 750)

    # Each real edge spends its ply and possibly one intrinsic-LMR unit.
    cost = 1 + bool(null_limit and options.get("LMR", 75) != -70000)

    # D >= C*(k-1)+A keeps the last attacker beyond the cap horizon;
    # D >= C*k+1 leaves its terminal child at positive depth.
    cap_depth = max(options.get("FUT_MAX", 1),
        options.get("FUT_CAP_DEPTH", 3) if options.get("FUT_CAP", 1) else 0)
    attacker = cap_depth + 1
    depth = max(cost * (k - 1) + attacker, cost * k + 1)

    # D >= C*(k-2)+B keeps the last defender beyond the null horizon.
    # If the shallow-null interval is empty, the ordinary positive-depth
    # fold is already sufficient; otherwise it ends at LMR_MIN_DEPTH.
    if k > 1:
        lmr_depth = options.get("LMR_MIN_DEPTH", 6)
        null_depth = options.get("NULL_MIN_DEPTH", 2)
        defender = lmr_depth if null_limit and null_depth + 1 < lmr_depth else 1
        depth = max(depth, cost * (k - 2) + defender)
    return depth


def wait_for(process, prefix):
    lines = []
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("engine stopped during policy gate")
        lines.append(line.rstrip())
        if line.startswith(prefix):
            return lines


def mate_floor(argv, name, depth, limit):
    path = ROOT / "tests" / "files" / name
    positions = [line.strip() for line in path.read_text().splitlines() if line.strip()][:limit]
    process = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    found = 0
    try:
        for fen in positions:
            process.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
            process.stdin.flush()
            wait_for(process, "ok")
            lines = wait_for(process, "done")
            scores = [int(match.group(2)) for line in lines
                      if (match := INFO.match(line)) and int(match.group(1)) == depth]
            found += bool(scores and scores[-1] > 10000)
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
    return found, len(positions)


def main():
    request = json.load(sys.stdin)
    options = request["options"]
    if not options.get("MATE_DIST", 1) or not options.get("EVAL_ROUGHNESS", 15):
        print("mate-distance:disabled")
        return 1
    try:
        suites = [(name, mate_depth(options, moves), limit, floor)
                  for name, moves, limit, floor in SUITES]
    except ValueError as error:
        print(f"mate-depth:{error}")
        return 1
    if "--horizon-only" in sys.argv:
        print(" ".join(f"{name}:depth={depth}" for name, depth, _, _ in suites))
        return 0
    argv = [request["engine"], *shlex.split(request["engine_args"])]
    argv += [f"{name}={value}" for name, value in sorted(options.items())]
    results = {name: mate_floor(argv, name, depth, limit)
               for name, depth, limit, floor in suites}
    print(" ".join(f"{name}:{found}/{total}" for name, (found, total) in results.items()))
    return int(any(results[name][0] < floor for name, depth, limit, floor in suites))


if __name__ == "__main__":
    sys.exit(main())
