#!/usr/bin/env python3
"""Measure recovery-start candidates against a fixed UCI baseline."""

import argparse
import asyncio
import json
import math
import pathlib
import re
import shutil


SCORE = re.compile(r"Score of candidate vs baseline:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)")


def engine(command, name, arguments, options):
    executable = shutil.which(command) or command
    result = ["-engine", f"cmd={pathlib.Path(executable).resolve()}", f"name={name}"]
    if arguments:
        result.append(f"args={arguments}")
    result += [f"option.{key}={value}" for key, value in sorted(options.items())]
    return result


def elo(wins, losses, draws):
    # A half-game prior keeps tiny screens finite without changing their rank much.
    score = (wins + draws / 2 + .5) / (wins + losses + draws + 1)
    return 400 * math.log10(score / (1 - score))


def result(number, options, output, games):
    matches = SCORE.findall(output.decode(errors="replace"))
    if not matches:
        return None
    wins, losses, draws = map(int, matches[-1])
    if wins + losses + draws != games:
        return None
    return {"number": number, "wins": wins, "losses": losses, "draws": draws,
            "elo": elo(wins, losses, draws), "options": options}


def save(path, results):
    target = pathlib.Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(sorted(results, key=lambda r: r["number"]), indent=2) + "\n")
    temporary.replace(target)


async def play(args, number, options, semaphore):
    async with semaphore:
        command = [
            args.fastchess,
            *engine(args.engine, "candidate", args.engine_args, options),
            *engine(args.baseline_engine or args.engine, "baseline",
                    args.baseline_args if args.baseline_args is not None else args.engine_args, {}),
            "-each", "proto=uci", f"tc={args.tc}",
            "-openings", f"file={pathlib.Path(args.openings).resolve()}", "format=epd",
            "order=sequential", f"start={args.start}",
            "-rounds", str(args.pairs), "-games", "2", "-repeat", "-concurrency", "1", "-recover",
            "-draw", "movenumber=40", "movecount=8", "score=10",
            "-resign", "movecount=4", "score=500",
            "-output", "format=cutechess", "-scoreinterval", "1", "-ratinginterval", "0",
        ]
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output = await process.stdout.read()
        status = await process.wait()
        pathlib.Path(args.logs, f"start-{number:02d}.log").write_bytes(output)
        match = result(number, options, output, 2 * args.pairs)
        if status or match is None:
            raise RuntimeError(f"start {number} failed with status {status}")
        return match


async def run(args):
    pathlib.Path(args.logs).mkdir(parents=True, exist_ok=True)
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    semaphore = asyncio.Semaphore(args.slots)
    results = []
    pending = []
    selected = set(args.number) if args.number else set(range(len(manifest["starts"])))
    for number, options in enumerate(manifest["starts"]):
        if number not in selected:
            continue
        log = pathlib.Path(args.logs, f"start-{number:02d}.log")
        cached = result(number, options, log.read_bytes(), 2 * args.pairs) if log.exists() else None
        if cached is None:
            pending.append((number, options))
        else:
            results.append(cached)
            print(f"start {number:02d}: {cached['elo']:+.1f} Elo (cached)", flush=True)
    save(args.output, results)
    tasks = [play(args, number, options, semaphore) for number, options in pending]
    for task in asyncio.as_completed(tasks):
        match = await task
        results.append(match)
        save(args.output, results)
        print(f"start {match['number']:02d}: {match['elo']:+.1f} Elo "
              f"({match['wins']}-{match['losses']}-{match['draws']})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--fastchess", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--baseline-engine")
    parser.add_argument("--engine-args", default="")
    parser.add_argument("--baseline-args")
    parser.add_argument("--openings", required=True)
    parser.add_argument("--tc", default="3+0.1")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--number", type=int, action="append",
                        help="screen only this zero-based manifest entry (repeatable)")
    args = parser.parse_args()
    if min(args.pairs, args.slots, args.start) < 1:
        parser.error("pairs, slots, and start must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
