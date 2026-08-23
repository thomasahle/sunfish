#!/usr/bin/env python3
"""Build and lock the C twin plus a stationary external-opponent panel."""

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shlex
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).parents[2]
STOCKFISH_RELEASE = "Stockfish 15"
STOCKFISH_REVISION = "e6e324eb28fd49c1fc44b3b65784f85a773ec61c"
SOURCE_FILES = (
    "sunfish.py", "LICENSE.md", "tools/ctwin/sunfish.c", "tools/ctwin/gen_tables.py",
)


def sha256(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def run(*command, cwd=None, text=True):
    return subprocess.check_output(command, cwd=cwd, text=text).strip()


def write_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy(source, target, executable=False):
    target = pathlib.Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o755 if executable else 0o644)


def artifact(root, path):
    path = pathlib.Path(path)
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


def tree_artifact(root, path):
    path = pathlib.Path(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    result = hashlib.sha256()
    for item in files:
        result.update(str(item.relative_to(path)).encode() + b"\0")
        result.update(sha256(item).encode() + b"\0")
        result.update(oct(item.stat().st_mode & 0o777).encode() + b"\n")
    return {
        "path": str(path.relative_to(root)), "sha256": result.hexdigest(),
        "files": len(files),
    }


def clean_remote(url):
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:").removesuffix(".git")
    return url.removesuffix(".git")


def committed_source(root, revision, output):
    records = []
    for relative in SOURCE_FILES:
        data = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=root)
        source = root / relative
        if source.read_bytes() != data:
            raise RuntimeError(f"{relative} differs from {revision}")
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o644)
        records.append(artifact(output, target))
    return records


def copy_repository(source, output, artifact_root):
    if run("git", "status", "--porcelain", cwd=source):
        raise RuntimeError(f"opponent checkout is dirty: {source}")
    revision = run("git", "rev-parse", "HEAD", cwd=source)
    files = run("git", "ls-files", "-z", cwd=source, text=False).split(b"\0")
    for encoded in filter(None, files):
        relative = pathlib.Path(os.fsdecode(encoded))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe tracked path: {relative}")
        target = output / relative
        copy(source / relative, target, os.access(source / relative, os.X_OK))
    return {
        "source": clean_remote(run("git", "remote", "get-url", "origin", cwd=source)),
        "revision": revision, "tree": tree_artifact(artifact_root, output),
    }


def relative_member(name, weight, engine, arguments, options, source, revision, identity_files):
    return {
        "name": name, "weight": weight, "engine": engine, "args": arguments,
        "options": options, "source": source, "revision": revision,
        "license": "GPL-3.0-only", "identity_files": identity_files,
    }


def absolute_member(root, member):
    result = member.copy()
    result["engine"] = str((root / result["engine"]).resolve())
    if result["args"]:
        result["args"] = str((root / result["args"]).resolve())
    result["identity_files"] = [
        str(pathlib.Path(path) if pathlib.Path(path).is_absolute() else (root / path).resolve())
        for path in result["identity_files"]
    ]
    return result


def calibration_plan():
    return {
        "tc": "3+0.1", "pairs": 50, "games_per_opponent": 100,
        "start": 1, "common_openings": "openings.epd", "reject_engine_failures": True,
        "command": [
            "python", "runner/calibrate_panel.py", "--fastchess", "runner/fastchess",
            "--panel", "panel.json", "--openings", "openings.epd",
            "--logs", "calibration", "--tc", "3+0.1", "--pairs", "50", "--start", "1",
        ],
    }


def validate_calibration(record, opponents):
    plan = calibration_plan()
    results = record.get("results", [])
    if any(record.get(key) != value for key, value in plan.items()):
        raise RuntimeError("calibration record does not match the frozen plan")
    if record.get("status") != "complete" or [result.get("opponent") for result in results] != opponents:
        raise RuntimeError("calibration results do not cover the frozen opponents")
    if any(result.get("games") != plan["games_per_opponent"] or
           result.get("pairs") != plan["pairs"] or result.get("tc") != plan["tc"]
           for result in results):
        raise RuntimeError("calibration results are incomplete or use the wrong time control")
    if any(result.get("saturated") is not False for result in results):
        raise RuntimeError("calibration opponent is saturated")
    if [log.get("path") for log in record.get("logs", [])] != [
            f"calibration/{opponent}.log" for opponent in opponents]:
        raise RuntimeError("calibration record does not cover the frozen logs")
    paths = [*record.get("command", []), *(log.get("path", "") for log in record.get("logs", []))]
    if any(pathlib.Path(path).is_absolute() for path in paths):
        raise RuntimeError("calibration record contains an absolute host path")


def complete_calibration(output, members):
    """Normalize completed calibration evidence without retaining host paths."""
    output = pathlib.Path(output)
    directory = output / "calibration"
    raw = directory / "results.json"
    if not raw.is_file():
        raise RuntimeError("calibration did not produce results.json")
    payload = json.loads(raw.read_text())
    results = [{key: value for key, value in result.items() if key != "pair_scores"}
               for result in payload.get("results", [])]
    opponents = [member["name"] for member in members if member["name"] != "master"]
    plan = calibration_plan()
    record = plan | {"status": "complete", "results": results}
    logs = []
    for opponent in opponents:
        log = directory / f"{opponent}.log"
        if not log.is_file():
            raise RuntimeError(f"calibration log is missing: {opponent}")
        logs.append(artifact(output, log))
    record["logs"] = logs
    validate_calibration(record, opponents)
    raw.unlink()
    write_json(output / "calibration.json", record)
    return record, [directory / f"{opponent}.log" for opponent in opponents]


def run_calibration(output, python, members):
    plan = calibration_plan()
    command = [str(python), *(output / part if "/" in part else part for part in plan["command"][1:])]
    subprocess.run([str(part) for part in command], cwd=output, check=True)
    return complete_calibration(output, members)


def freeze(args):
    root = pathlib.Path(args.source_root).resolve()
    output = pathlib.Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"freeze directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    revision = args.source_commit or run("git", "rev-parse", "HEAD", cwd=root)
    sources = committed_source(root, revision, output)

    tables = output / "tables.txt"
    with tables.open("wb") as target:
        subprocess.run(
            [sys.executable, output / "source/tools/ctwin/gen_tables.py"],
            cwd=output / "source/tools/ctwin", stdout=target, check=True)
    compiler = pathlib.Path(shutil.which(args.cc) or args.cc).resolve()
    python = pathlib.Path(shutil.which(args.python) or args.python).resolve()
    flags = shlex.split(args.cflags)
    binary = output / "sunfish_c"
    rebuild = output / "sunfish_c.rebuild"
    command = [compiler, *flags, "-o", binary, output / "source/tools/ctwin/sunfish.c"]
    subprocess.run(command, check=True)
    command[command.index(binary)] = rebuild
    subprocess.run(command, check=True)
    if binary.read_bytes() != rebuild.read_bytes():
        raise RuntimeError("the C compiler did not reproduce the same binary")
    rebuild.unlink()

    opponents = output / "opponents"
    stockfish = opponents / "stockfish15"
    stockfish_license = opponents / "stockfish-Copying.txt"
    copy(args.stockfish, stockfish, True)
    copy(args.stockfish_license, stockfish_license)
    stockfish_lock = opponents / "stockfish.lock.json"
    write_json(stockfish_lock, {
        "source": "https://github.com/official-stockfish/Stockfish",
        "release": STOCKFISH_RELEASE, "revision": STOCKFISH_REVISION,
        "binary": artifact(output, stockfish),
        "license": artifact(output, stockfish_license),
    })

    chessidle_root = opponents / "chessidle"
    chessidle = copy_repository(
        pathlib.Path(args.chessidle_source).resolve(), chessidle_root, output)
    chessidle_lock = opponents / "chessidle.lock.json"
    write_json(chessidle_lock, chessidle)
    chessidle_wrapper = opponents / "chessidle.sh"
    chessidle_wrapper.write_text(
        '#!/bin/sh\nHERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        'cd "$HERE/chessidle" || exit\n'
        f'PYTHONDONTWRITEBYTECODE=1 exec {shlex.quote(str(python))} -m chessidle\n')
    chessidle_wrapper.chmod(0o755)

    runner = output / "runner"
    fastchess = runner / "fastchess"
    calibrator = runner / "calibrate_panel.py"
    panel_helper = runner / "opponent_panel.py"
    penta_helper = runner / "pentanomial.py"
    copy(args.fastchess, fastchess, True)
    copy(args.calibrator, calibrator, True)
    copy(args.opponent_panel, panel_helper)
    copy(args.pentanomial, penta_helper)
    openings = output / "openings.epd"
    copy(args.openings, openings)

    source = args.source_url or clean_remote(run("git", "remote", "get-url", "origin", cwd=root))
    members = [
        relative_member("master", 2, "sunfish_c", "tables.txt", "default", source, revision,
                        ["source/LICENSE.md"]),
        relative_member("stockfish-1800", 1, "opponents/stockfish15", "", {
            "UCI_LimitStrength": "true", "UCI_Elo": 1800, "Threads": 1, "Hash": 16,
        }, "https://github.com/official-stockfish/Stockfish", STOCKFISH_REVISION, [
            "opponents/stockfish.lock.json", "opponents/stockfish-Copying.txt",
        ]),
        relative_member("chessidle", 1, "opponents/chessidle.sh", "", {
            "Threads": 1, "Hash": 16,
        }, chessidle["source"], chessidle["revision"], [
            "opponents/chessidle.lock.json", "opponents/chessidle/LICENSE", str(python),
        ]),
    ]
    panel = output / "panel.json"
    write_json(panel, [absolute_member(output, member) for member in members])

    calibration, calibration_logs = run_calibration(output, python, members)
    calibration_file = output / "calibration.json"

    paths = [
        binary, tables, stockfish, stockfish_license, stockfish_lock,
        chessidle_wrapper, chessidle_lock, chessidle_root / "LICENSE", panel,
        fastchess, calibrator, panel_helper, penta_helper, openings,
        calibration_file, *calibration_logs,
    ]
    manifest = {
        "schema": "sunfish-panel-freeze-v2", "source": source, "revision": revision,
        "source_files": sources, "artifacts": [artifact(output, path) for path in paths],
        "opponent_source_trees": [chessidle["tree"]],
        "compiler": {
            "path": str(compiler), "sha256": sha256(compiler), "flags": flags,
            "version": run(compiler, "--version").splitlines()[0],
            "platform": platform.platform(), "machine": platform.machine(),
        },
        "runtime": {
            "path": str(python), "sha256": sha256(python),
            "version": run(python, "--version"),
        },
        "panel": members, "weights": {member["name"]: member["weight"] for member in members},
        "calibration": calibration,
        "freeze_script_sha256": sha256(__file__),
    }
    write_json(output / "manifest.json", manifest)
    verify(output / "manifest.json")
    return manifest


def verify(path):
    path = pathlib.Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "sunfish-panel-freeze-v2":
        raise RuntimeError("unknown freeze manifest schema")
    records = [*manifest["source_files"], *manifest["artifacts"]]
    for record in records:
        target = path.parent / record["path"]
        if not target.is_file() or sha256(target) != record["sha256"]:
            raise RuntimeError(f"frozen artifact changed: {record['path']}")
    for record in manifest["opponent_source_trees"]:
        target = path.parent / record["path"]
        if not target.is_dir() or tree_artifact(path.parent, target) != record:
            raise RuntimeError(f"frozen source tree changed: {record['path']}")
    runtime = manifest["runtime"]
    if not pathlib.Path(runtime["path"]).is_file() or sha256(runtime["path"]) != runtime["sha256"]:
        raise RuntimeError("frozen Python runtime changed")
    calibration = json.loads((path.parent / "calibration.json").read_text())
    if calibration != manifest.get("calibration") or calibration.get("status") != "complete":
        raise RuntimeError("frozen calibration record does not match the manifest")
    opponents = [member["name"] for member in manifest["panel"] if member["name"] != "master"]
    validate_calibration(calibration, opponents)
    if any(record not in manifest["artifacts"] for record in calibration.get("logs", [])):
        raise RuntimeError("frozen calibration logs are not covered by the manifest")
    if (path.parent / "calibration/results.json").exists():
        raise RuntimeError("raw calibration results were not normalized")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify")
    parser.add_argument("--output")
    parser.add_argument("--source-root", default=ROOT)
    parser.add_argument("--source-commit")
    parser.add_argument("--source-url")
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--cflags", default="-O3 -march=native -Wall -Wextra")
    parser.add_argument("--stockfish")
    parser.add_argument("--stockfish-license")
    parser.add_argument("--chessidle-source")
    parser.add_argument("--fastchess")
    parser.add_argument("--openings")
    parser.add_argument("--calibrator", default=pathlib.Path(__file__).with_name("calibrate_panel.py"))
    parser.add_argument("--opponent-panel", default=pathlib.Path(__file__).with_name("opponent_panel.py"))
    parser.add_argument("--pentanomial", default=pathlib.Path(__file__).with_name("pentanomial.py"))
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    elif all((args.output, args.stockfish, args.stockfish_license, args.chessidle_source,
              args.fastchess, args.openings)):
        freeze(args)
    else:
        parser.error("freeze needs output, opponent, fastchess, and opening paths")


if __name__ == "__main__":
    main()
