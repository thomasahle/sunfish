#!/usr/bin/env python3
"""Run one CLOP sample as one UCI game through fastchess.

CLOP invokes this program once per game as

    clop_fastchess.py [fixed arguments] PROCESSOR SEED NAME VALUE ...

Use ``Replications 2`` in the CLOP experiment: consecutive seeds then use the
same opening with reversed engine order, forming a color-swapped pair.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys


SCORE = re.compile(
    r"Score of (candidate|baseline) vs (candidate|baseline):\s+"
    r"(\d+)\s+-\s+(\d+)\s+-\s+(\d+)")


def engine(command, name, arguments, options):
    command = shutil.which(command) or command
    result = ["-engine", f"cmd={pathlib.Path(command).resolve()}", f"name={name}"]
    if arguments:
        result.append(f"args={arguments}")
    result += [f"option.{key}={value}" for key, value in sorted(options.items())]
    return result


def parse_options(values):
    if len(values) % 2:
        raise ValueError("CLOP supplied an option name without a value")
    return dict(zip(values[::2], values[1::2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastchess", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--baseline-engine")
    parser.add_argument("--engine-args", default="")
    parser.add_argument("--baseline-args")
    parser.add_argument("--baseline-option", action="append", default=[])
    parser.add_argument("--fixed-option", action="append", default=[])
    parser.add_argument("--openings", required=True)
    parser.add_argument("--opening-count", type=int, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--tc", default="3+0.1")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("processor")
    parser.add_argument("seed", type=int)
    parser.add_argument("options", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # An invalid result asks CLOP to stop after every lower-numbered game has
    # finished. It may leave harmless sample-only records at the end of .dat.
    if args.max_games and args.seed >= args.max_games:
        return 2

    tuned = dict(item.split("=", 1) for item in args.fixed_option)
    tuned.update(parse_options(args.options))
    baseline = dict(item.split("=", 1) for item in args.baseline_option)
    baseline_engine = args.baseline_engine or args.engine
    baseline_args = args.engine_args if args.baseline_args is None else args.baseline_args
    opening = (args.start - 1 + args.seed // 2) % args.opening_count + 1

    candidate = engine(args.engine, "candidate", args.engine_args, tuned)
    opponent = engine(baseline_engine, "baseline", baseline_args, baseline)
    engines = candidate + opponent if args.seed % 2 == 0 else opponent + candidate
    command = [
        args.fastchess, *engines,
        "-each", "proto=uci", f"tc={args.tc}",
        "-openings", f"file={pathlib.Path(args.openings).resolve()}", "format=epd",
        "order=sequential", f"start={opening}",
        "-rounds", "1", "-games", "1", "-concurrency", "1", "-recover",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=4", "score=500",
        "-output", "format=cutechess", "-scoreinterval", "1", "-ratinginterval", "0",
    ]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    matches = SCORE.findall(process.stdout)
    if process.returncode or not matches:
        print(process.stdout, file=sys.stderr)
        return 1
    first, _, wins, losses, draws = matches[-1]
    wins, losses, draws = map(int, (wins, losses, draws))
    if first == "baseline":
        wins, losses = losses, wins
    if wins + losses + draws != 1:
        print(f"fastchess completed {wins + losses + draws} games", file=sys.stderr)
        return 1
    print("W" if wins else "L" if losses else "D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
