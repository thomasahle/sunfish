#!/usr/bin/env python3
"""Measure optimizer recommendations on one independent opening set."""

import argparse
import asyncio
import hashlib
import json
import math
import pathlib
import re
import shlex
import shutil
import subprocess
import sys


sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pentanomial  # noqa: E402
import locking  # noqa: E402


ELO = re.compile(r"Elo difference:\s+([^,\s]+)\s+\+/-\s+([^,\s]+)")
UCI_OPTION = re.compile(r"^option name (.+?) type ")


def engine(command, name, arguments, options):
    executable = shutil.which(command) or command
    result = ["-engine", f"cmd={pathlib.Path(executable).resolve()}", f"name={name}"]
    if arguments:
        result.append(f"args={arguments}")
    result += [f"option.{key}={value}" for key, value in sorted(options.items())]
    return result


def identifier(options):
    payload = json.dumps(options, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def digest(path):
    path = pathlib.Path(path).resolve()
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return {"path": str(path), "sha256": result.hexdigest()}


def slice_digest(path, start, pairs):
    lines = [line for line in pathlib.Path(path).read_bytes().splitlines(keepends=True)
             if line.strip()]
    selected = lines[start - 1:start - 1 + pairs]
    if len(selected) != pairs:
        raise RuntimeError("validation opening slice is incomplete")
    return hashlib.sha256(b"".join(selected)).hexdigest()


def command_identity(command, arguments):
    executable = shutil.which(command) or command
    return {
        "command": str(pathlib.Path(executable).resolve()), "arguments": arguments,
        "files": [digest(path) for path in [executable, *shlex.split(arguments)]
                  if pathlib.Path(path).is_file()],
    }


def validate_options(command, arguments, required):
    process = subprocess.run(
        [command, *shlex.split(arguments)], input="uci\nquit\n", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    advertised = {match.group(1) for line in process.stdout.splitlines()
                  if (match := UCI_OPTION.match(line))}
    missing = sorted(set(required) - advertised)
    if process.returncode or "uciok" not in process.stdout or missing:
        raise RuntimeError(f"cannot validate UCI options: {', '.join(missing)}")


def result(results, rating=None):
    counts, (wins, losses, draws), pair_scores = pentanomial.summarize(results)
    mean, _, interval = pentanomial.posterior(counts)

    def to_elo(score):
        score = min(max(score, 1e-9), 1 - 1e-9)
        return 400 * math.log10(score / (1 - score))

    posterior_elo = to_elo(1 - mean)
    posterior_interval = [to_elo(1 - interval[1]), to_elo(1 - interval[0])]
    fast_elo, fast_error = rating or (math.nan, math.nan)
    elo = fast_elo if math.isfinite(fast_elo) else posterior_elo
    error = (fast_error if math.isfinite(fast_error)
             else max(elo - posterior_interval[0], posterior_interval[1] - elo))
    return {
        "wins": wins, "losses": losses, "draws": draws,
        "pentanomial": counts, "pair_scores": pair_scores,
        "elo": elo, "error": error,
        "fastchess_elo": fast_elo if math.isfinite(fast_elo) else None,
        "fastchess_error": fast_error if math.isfinite(fast_error) else None,
        "posterior_elo": posterior_elo, "posterior_interval": posterior_interval,
    }


def parse(output, pairs):
    text = output.decode(errors="replace")
    try:
        results, _ = pentanomial.game_results(text, subject="candidate")
    except ValueError:
        return None
    if len(results) != 2 * pairs:
        return None
    ratings = ELO.findall(text)
    rating = tuple(map(float, ratings[-1])) if ratings else None
    return result(results, rating)


def recover(log):
    """Read every complete pair from successive attempts at one match."""
    paths = [log, *sorted(log.parent.glob(f"{log.stem}.part-*{log.suffix}"))]
    results = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        try:
            segment, _ = pentanomial.game_results(
                text, partial=True, subject="candidate")
        except ValueError:
            continue
        results.extend(segment)
    return results, paths


def save(path, payload):
    target = pathlib.Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


async def play(args, config, semaphore):
    async with semaphore:
        log = pathlib.Path(args.logs, f"configuration-{config['id']}.log")
        cached = parse(log.read_bytes(), args.pairs) if log.exists() else None
        if cached is not None:
            return config["id"], cached
        previous, paths = recover(log)
        completed = len(previous) // 2
        if completed > args.pairs:
            raise RuntimeError(f"validation {config['id']} has too many recovered pairs")
        if completed == args.pairs:
            return config["id"], result(previous)
        current = log if not log.exists() else log.with_name(
            f"{log.stem}.part-{len(paths):02d}{log.suffix}")
        remaining = args.pairs - completed
        command = [
            args.fastchess,
            *engine(args.engine, "candidate", args.engine_args, config["options"]),
            *engine(args.baseline_engine or args.engine, "baseline",
                    args.baseline_args if args.baseline_args is not None else args.engine_args,
                    dict(item.split("=", 1) for item in args.baseline_option)),
            "-each", "proto=uci", f"tc={args.tc}",
            "-openings", f"file={pathlib.Path(args.openings).resolve()}",
            f"format={args.opening_format}",
            "order=sequential", f"start={args.start + completed}",
            "-rounds", str(remaining), "-games", "2", "-repeat",
            "-concurrency", "1", "-recover",
            "-draw", "movenumber=40", "movecount=8", "score=10",
            "-resign", "movecount=4", "score=500",
            "-report", "penta=true", "-output", "format=cutechess",
            "-scoreinterval", "10", "-ratinginterval", "10",
        ]
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output = bytearray()
        with current.open("wb") as target:
            while line := await process.stdout.readline():
                output += line
                target.write(line)
                if b"Elo difference:" in line:
                    target.flush()
                    print(f"{config['id']}: {line.decode().strip()}", flush=True)
        status = await process.wait()
        current_result = parse(output, remaining)
        if status or current_result is None:
            raise RuntimeError(f"validation {config['id']} failed with status {status}")
        combined = previous + pentanomial.game_results(
            output.decode(errors="replace"), subject="candidate")[0]
        return config["id"], result(combined) if previous else current_result


async def validate(args):
    with locking.exclusive(args.output):
        await validate_locked(args)


async def validate_locked(args):
    pathlib.Path(args.logs).mkdir(parents=True, exist_ok=True)
    records = json.loads(pathlib.Path(args.recommendations).read_text())
    for record in records:
        record["validation"] = identifier(record["options"])
    configurations = {
        identifier(record["options"]): record["options"] for record in records
    }
    baseline_engine = args.baseline_engine or args.engine
    baseline_args = args.baseline_args if args.baseline_args is not None else args.engine_args
    baseline_options = dict(item.split("=", 1) for item in args.baseline_option)
    opening_slice = slice_digest(args.openings, args.start, args.pairs)
    if args.opening_slice_sha256 and opening_slice != args.opening_slice_sha256:
        raise RuntimeError("validation opening-slice hash does not match")
    validate_options(args.engine, args.engine_args,
                     set().union(*(options.keys() for options in configurations.values())))
    validate_options(baseline_engine, baseline_args, baseline_options)
    payload = {
        "protocol": {
            "tc": args.tc, "pairs": args.pairs, "start": args.start,
            "opening_slice_sha256": opening_slice,
            "opening_format": args.opening_format,
            "fastchess": digest(shutil.which(args.fastchess) or args.fastchess),
            "recommendations": digest(args.recommendations),
            "candidate": command_identity(args.engine, args.engine_args),
            "baseline": command_identity(baseline_engine, baseline_args),
            "baseline_options": baseline_options, "openings": digest(args.openings),
        },
        "matches": {}, "records": records,
    }
    output = pathlib.Path(args.output)
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["protocol"] != payload["protocol"] or previous["records"] != records:
            raise RuntimeError("validation output belongs to a different study")
        payload = previous
    pending = [
        {"id": key, "options": options}
        for key, options in configurations.items() if key not in payload["matches"]
    ]
    semaphore = asyncio.Semaphore(args.slots)
    tasks = [play(args, config, semaphore) for config in pending]
    errors = []
    for task in asyncio.as_completed(tasks):
        try:
            key, match = await task
        except Exception as error:
            errors.append(str(error))
            print(error, file=sys.stderr, flush=True)
            continue
        payload["matches"][key] = match
        save(output, payload)
        print(f"{key}: {match['elo']:+.2f} +/- {match['error']:.2f} Elo ",
              f"({match['wins']}-{match['losses']}-{match['draws']})", flush=True)
    save(output, payload)
    if errors:
        raise RuntimeError(f"{len(errors)} validation match(es) failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recommendations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--fastchess", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--baseline-engine")
    parser.add_argument("--engine-args", default="")
    parser.add_argument("--baseline-args")
    parser.add_argument("--baseline-option", action="append", default=[])
    parser.add_argument("--openings", required=True)
    parser.add_argument("--opening-format", choices=("epd", "pgn"), default="epd")
    parser.add_argument("--opening-slice-sha256")
    parser.add_argument("--tc", default="3+0.1")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--start", type=int, default=1201)
    args = parser.parse_args()
    if min(args.pairs, args.slots, args.start) < 1:
        parser.error("pairs, slots, and start must be positive")
    asyncio.run(validate(args))


if __name__ == "__main__":
    main()
