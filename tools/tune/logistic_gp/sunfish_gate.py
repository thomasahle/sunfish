#!/usr/bin/env python3
"""Reject C-twin policies that lose Sunfish's eventual-mate guarantees."""

import argparse
import json
import os
import pathlib
import re
import selectors
import shlex
import subprocess
import sys
import time


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INFO = re.compile(r"info depth (\d+) .* score (-?\d+)")
DONE = re.compile(r"done nodes (\d+) gen (\d+)")
SUITES = (("mate1.fen", 1, 8, 8),
          ("mate2_eventual.fen", 2, 5, 5),
          ("mate3_eventual.fen", 3, 2, 2))


def mate_depth(options, moves):
    """Uniform depth bound for this policy's mate-in-moves gate."""
    k = 2 * moves - 1
    null_limit = options.get("NULL_LIMIT", 750)
    null_depth = options.get("NULL_MIN_DEPTH", 2)
    fuel_depth = options.get("FUEL_MIN_DEPTH", 6)
    fuel = options.get("FUEL_NULL", 1) if null_limit and fuel_depth < 99 else 0
    if null_limit and not fuel and null_depth < 99:
        raise ValueError("unbounded-classical-null")

    # Each real edge spends its ply, the hot-node fuel, and possibly LMR.
    lmr = options.get("LMR_RED", 1)
    if (options.get("LMR", 75) == -70000
            or options.get("LMR_MIN_DEPTH", 6) == 99
            or not options.get("LMR_LIMIT", 750)):
        lmr = 0
    cost = 1 + fuel + lmr

    # D >= C*(k-1)+A keeps the last attacker beyond the cap horizon;
    # D >= C*k+1 leaves its terminal child at positive depth.
    cap_depth = max(options.get("FUT_MAX", 1),
        options.get("FUT_CAP_DEPTH", 3) if options.get("FUT_CAP", 1) else 0)
    attacker = cap_depth + 1
    depth = max(cost * (k - 1) + attacker, cost * k + 1)

    # D >= C*(k-2)+B keeps the last defender beyond the null horizon.
    # If the shallow-null interval is empty, the ordinary positive-depth
    # fold is already sufficient; otherwise it ends at FUEL_MIN_DEPTH.
    if k > 1:
        defender = fuel_depth if null_limit and null_depth + 1 < fuel_depth else 1
        if defender == 99:
            raise ValueError("null-transition-disabled")
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
            process.wait()
    return found, len(positions)


def node_count(argv, options, positions, depth, timeout):
    process = subprocess.Popen(
        [*argv, *(f"{name}={value}" for name, value in sorted(options.items()))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=0)
    nodes = gens = 0
    deadline = time.monotonic() + timeout
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    buffer = b""
    try:
        for fen in positions:
            process.stdin.write(
                f"reset\nposition fen {fen}\ngo depth {depth}\n".encode())
            process.stdin.flush()
            while True:
                while b"\n" in buffer:
                    raw, buffer = buffer.split(b"\n", 1)
                    if match := DONE.match(raw.decode(errors="replace")):
                        nodes += int(match.group(1))
                        gens += int(match.group(2))
                        break
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError(f"node gate exceeded {timeout:g}s")
                    chunk = os.read(process.stdout.fileno(), 1 << 16)
                    if not chunk:
                        raise RuntimeError("engine stopped during node gate")
                    buffer += chunk
                    continue
                break
        process.stdin.write(b"quit\n")
        process.stdin.flush()
        process.wait(timeout=5)
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    return nodes, gens


def node_ratio(request, options, book, depth, count, timeout):
    lines = [line.split(";")[0].strip() for line in pathlib.Path(book).read_text().splitlines()]
    positions = [line for line in lines if line][:count]
    if len(positions) < count:
        raise ValueError(f"node book has only {len(positions)} positions")
    argv = [request["engine"], *shlex.split(request["engine_args"])]
    baseline = node_count(
        argv, request.get("baseline_options", {}), positions, depth, timeout)
    candidate = node_count(argv, options, positions, depth, timeout)
    return candidate[0] / baseline[0], baseline, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon-only", action="store_true")
    parser.add_argument("--node-factor", type=float)
    parser.add_argument("--node-book")
    parser.add_argument("--node-depth", type=int, default=8)
    parser.add_argument("--node-positions", type=int, default=4)
    parser.add_argument("--node-timeout", type=float, default=8)
    args = parser.parse_args()
    if args.node_factor is not None and min(
            args.node_factor, args.node_depth, args.node_positions,
            args.node_timeout) <= 0:
        parser.error("node limits must be positive")
    if args.node_factor is not None and not args.node_book:
        parser.error("--node-factor requires --node-book")
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
    if args.horizon_only:
        print(" ".join(f"{name}:depth={depth}" for name, depth, _, _ in suites))
    else:
        argv = [request["engine"], *shlex.split(request["engine_args"])]
        argv += [f"{name}={value}" for name, value in sorted(options.items())]
        results = {name: mate_floor(argv, name, depth, limit)
                   for name, depth, limit, floor in suites}
        print(" ".join(f"{name}:{found}/{total}" for name, (found, total) in results.items()))
        if any(results[name][0] < floor for name, depth, limit, floor in suites):
            return 1
    if args.node_factor is not None:
        try:
            ratio, baseline, candidate = node_ratio(
                request, options, args.node_book, args.node_depth,
                args.node_positions, args.node_timeout)
        except (TimeoutError, ValueError) as error:
            print(f"node-gate:{error}")
            return 1 if isinstance(error, TimeoutError) else 2
        print(f"nodes:{candidate[0]}/{baseline[0]} ratio={ratio:.3f}")
        if ratio > args.node_factor:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
