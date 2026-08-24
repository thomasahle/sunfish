#!/usr/bin/env python3
"""Measure each non-master panel member against master on common pairs."""

import argparse
import hashlib
import json
import math
import pathlib
import shlex
import subprocess
import sys


sys.path.insert(0, str(pathlib.Path(__file__).parent))
import opponent_panel  # noqa: E402
import pentanomial  # noqa: E402


def file_identity(path):
    path = pathlib.Path(path).resolve()
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return {"path": str(path), "sha256": result.hexdigest()}


def identities(args, panel):
    engines = []
    for member in panel:
        paths = [member["engine"], *shlex.split(member.get("args", "")),
                 *member.get("identity_files", [])]
        engines.append({
            "name": member["name"], "options": member.get("options", {}),
            "files": [file_identity(path) for path in paths if pathlib.Path(path).is_file()],
        })
    return {
        "fastchess": file_identity(args.fastchess), "panel": file_identity(args.panel),
        "openings": file_identity(args.openings), "runner": file_identity(__file__),
        "python": file_identity(sys.executable), "engines": engines,
    }


def elo(score):
    return 400 * math.log10(score / (1 - score)) if 0 < score < 1 else None


def engine(member, name):
    result = ["-engine", f"cmd={member['engine']}", f"name={name}"]
    if member.get("args"):
        result.append(f"args={member['args']}")
    options = {} if member.get("options") == "default" else member.get("options", {})
    result += [f"option.{key}={value}" for key, value in sorted(options.items())]
    return result


def calibrate(args):
    panel = json.loads(pathlib.Path(args.panel).read_text())
    masters = [member for member in panel if member["name"] == "master"]
    if len(masters) != 1:
        raise RuntimeError("calibration panel needs exactly one master")
    master = masters[0]
    logs = pathlib.Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    results, commands = [], {}
    for member in panel:
        if member is master:
            continue
        command = [
            args.fastchess,
            *engine(member, "opponent"), *engine(master, "master"),
            "-each", "proto=uci", f"tc={args.tc}",
            "-openings", f"file={pathlib.Path(args.openings).resolve()}",
            "format=epd", "order=sequential", f"start={args.start}",
            "-rounds", str(args.pairs), "-games", "2", "-repeat",
            "-concurrency", str(args.concurrency),
            "-draw", "movenumber=40", "movecount=8", "score=10",
            "-resign", "movecount=4", "score=500",
            "-output", "format=cutechess", "-scoreinterval", "1", "-ratinginterval", "0",
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        commands[member["name"]] = command
        output = process.stdout.decode(errors="replace")
        pathlib.Path(logs, f"{member['name']}.log").write_bytes(process.stdout)
        failure = opponent_panel.failure(output)
        if process.returncode or failure:
            marker = failure or f"status {process.returncode}"
            raise RuntimeError(f"{member['name']} calibration failed: {marker}")
        try:
            games, _ = pentanomial.game_results(output, subject="opponent")
            counts, (wins, losses, draws), scores = pentanomial.summarize(games)
        except ValueError as error:
            raise RuntimeError(f"{member['name']} calibration failed: {error}") from error
        if wins + losses + draws != 2 * args.pairs:
            raise RuntimeError(f"{member['name']} completed {wins + losses + draws} games")
        score = (wins + draws / 2) / (2 * args.pairs)
        loss_mean, loss_sd, loss_interval = pentanomial.posterior(counts)
        posterior_score = 1 - loss_mean
        score_interval = [1 - loss_interval[1], 1 - loss_interval[0]]
        results.append({
            "opponent": member["name"], "master": master["name"],
            "wins": wins, "losses": losses, "draws": draws,
            "pairs": args.pairs, "games": 2 * args.pairs,
            "tc": args.tc, "start": args.start,
            "pentanomial": counts, "pair_scores": scores,
            "score": score, "posterior_score": posterior_score,
            "posterior_score_sd": loss_sd, "posterior_score_95": score_interval,
            "rough_elo": elo(score), "posterior_elo": elo(posterior_score),
            "posterior_elo_95": [elo(bound) for bound in score_interval],
            "saturated": not .05 <= score <= .95,
        })
    payload = {
        "identities": identities(args, panel), "commands": commands,
        "results": results,
    }
    pathlib.Path(logs, "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastchess", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--tc", default="3+0.1")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if min(args.pairs, args.start, args.concurrency) < 1:
        parser.error("pairs, start, and concurrency must be positive")
    calibrate(args)


if __name__ == "__main__":
    main()
