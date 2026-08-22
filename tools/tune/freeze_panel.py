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
        "revision": "Stockfish 15", "binary": artifact(output, stockfish),
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
        }, "https://github.com/official-stockfish/Stockfish", "Stockfish 15", [
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

    calibration = output / "calibration.json"
    write_json(calibration, {
        "status": "pending", "tc": "3+0.1", "pairs": 50, "games_per_opponent": 100,
        "start": 1, "common_openings": "openings.epd", "reject_engine_failures": True,
        "command": [
            str(python), "runner/calibrate_panel.py", "--fastchess", "runner/fastchess",
            "--panel", "panel.json", "--openings", "openings.epd",
            "--logs", "calibration", "--tc", "3+0.1", "--pairs", "50", "--start", "1",
        ],
    })

    paths = [
        binary, tables, stockfish, stockfish_license, stockfish_lock,
        chessidle_wrapper, chessidle_lock, chessidle_root / "LICENSE", panel,
        fastchess, calibrator, panel_helper, penta_helper, openings, calibration,
    ]
    manifest = {
        "schema": "sunfish-panel-freeze-v1", "source": source, "revision": revision,
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
        "freeze_script_sha256": sha256(__file__),
    }
    write_json(output / "manifest.json", manifest)
    verify(output / "manifest.json")
    return manifest


def verify(path):
    path = pathlib.Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "sunfish-panel-freeze-v1":
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
