#!/usr/bin/env python3
"""Run one CLOP sample as one UCI game through fastchess.

CLOP invokes this program once per game as

    clop_fastchess.py [fixed arguments] PROCESSOR SEED NAME VALUE ...

Use ``Replications 2`` in the CLOP experiment: consecutive seeds then use the
same opening with reversed engine order, forming a color-swapped pair.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import pentanomial  # noqa: E402


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


def digest(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def command_identity(command, arguments):
    executable = shutil.which(command) or command
    files = [path for path in [executable, *shlex.split(arguments)]
             if pathlib.Path(path).is_file()]
    return {
        "command": str(pathlib.Path(executable).resolve()), "arguments": arguments,
        "files": [(str(pathlib.Path(path).resolve()), digest(path)) for path in files],
    }


def result(output):
    pentanomial.reject_failures(output)
    matches = SCORE.findall(output)
    if not matches:
        return None
    first, _, wins, losses, draws = matches[-1]
    wins, losses, draws = map(int, (wins, losses, draws))
    if first == "baseline":
        wins, losses = losses, wins
    return (wins, losses, draws) if wins + losses + draws == 1 else None


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
    parser.add_argument("--cache", default="clop-match-cache")
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
    study = {
        "version": 1, "runner": digest(__file__),
        "fastchess": command_identity(args.fastchess, ""),
        "candidate": command_identity(args.engine, args.engine_args),
        "baseline": command_identity(baseline_engine, baseline_args),
        "openings": (str(pathlib.Path(args.openings).resolve()), digest(args.openings)),
        "opening_count": args.opening_count, "start": args.start, "tc": args.tc,
        "max_games": args.max_games, "seed": args.seed, "tuned": tuned, "baseline_options": baseline,
    }
    identity = hashlib.sha256(
        json.dumps(study, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"seed-{args.seed:09d}-{identity}.log"
    output = path.read_text(errors="replace") if path.exists() else ""
    game = result(output) if output.startswith(f"clop-match-identity {identity}\n") else None
    if game is None:
        with path.open("w") as log:
            log.write(f"clop-match-identity {identity}\n")
            log.flush()
            os.fsync(log.fileno())
        with path.open("a", buffering=1) as log:
            process = subprocess.run(command, text=True, stdout=log, stderr=subprocess.STDOUT)
            log.flush()
            os.fsync(log.fileno())
        output = path.read_text(errors="replace")
        game = result(output)
        if process.returncode or game is None:
            print(output, file=sys.stderr)
            return 1
    wins, losses, draws = game
    print("W" if wins else "L" if losses else "D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
