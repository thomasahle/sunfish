#!/usr/bin/env python3
"""Reject C-twin policies that lose Sunfish's deterministic mate floors."""

import json
import pathlib
import re
import shlex
import subprocess
import sys


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INFO = re.compile(r"info depth (\d+) .* score (-?\d+)")
SUITES = (("mate2.fen", 6, 20), ("mate3.fen", 8, 5))


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
    argv = [request["engine"], *shlex.split(request["engine_args"])]
    argv += [f"{name}={value}" for name, value in sorted(options.items())]
    results = {name: mate_floor(argv, name, depth, limit)
               for name, depth, limit in SUITES}
    print(" ".join(f"{name}:{found}/{total}" for name, (found, total) in results.items()))
    return int(any(found < total for found, total in results.values()))


if __name__ == "__main__":
    sys.exit(main())
