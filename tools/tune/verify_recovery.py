#!/usr/bin/env python3
"""Audit the frozen five-tuner benchmark and its corrected recovery starts."""

import argparse
import hashlib
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).parents[2]
DEFAULT_MANIFEST = pathlib.Path(__file__).with_name("recovery_benchmark.json")


def digest(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def centered(spec, point):
    result = json.loads(json.dumps(spec))
    for parameter in result["parameters"]:
        start = point[parameter["name"]]
        radius = min(start - parameter["min"], parameter["max"] - start)
        if parameter["type"] == "integer":
            radius = int(radius)
        if radius <= 0 or abs(parameter["default"] - start) > radius:
            raise ValueError(f"cannot center {parameter['name']} at {start}")
        parameter["min"], parameter["default"], parameter["max"] = (
            start - radius, start, start + radius)
    return result


def require(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected}, got {actual}")


def audit(manifest_path=DEFAULT_MANIFEST, root=ROOT):
    root = pathlib.Path(root)
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    space_record = manifest["space"]
    space_path = root / space_record["path"]
    space = json.loads(space_path.read_text())
    require(digest(space_path), space_record["sha256"], "parameter-space bytes")
    require(canonical_digest(space), space_record["canonical_sha256"], "parameter space")
    names = [parameter["name"] for parameter in space["parameters"]]
    require("FUT_MAX" in names, False, "dead FUT_MAX axis present")
    require("NULL_CAP_MARGIN" in names, True, "NULL_CAP_MARGIN axis missing")
    reference = manifest["reference"]
    require(canonical_digest(reference["options"]), reference["canonical_sha256"], "reference")
    for start in manifest["starts"]:
        label = f"start {start['source_index']}"
        require(start["options"]["NULL_CAP_MARGIN"], start["options"]["EVAL_ROUGHNESS"], label)
        require(canonical_digest(start["options"]), start["canonical_sha256"], label)
        generated = centered(space, start["options"])
        require(canonical_digest(generated), start["centered_space_sha256"], f"{label} space")
        rendered = (json.dumps(generated, indent=2) + "\n").encode()
        require(hashlib.sha256(rendered).hexdigest(), start["materialized_sha256"], label)
    paths = dict(manifest["artifacts"]["project"])
    for method in manifest["methods"].values():
        paths.update(method.get("files", {}))
    for name, expected in paths.items():
        require(digest(root / name), expected, name)
    for name in ("chess_tuning_tools", "rbfopt"):
        path = root / "tools" / "tune" / name / "Dockerfile"
        require(digest(path), manifest["methods"][name]["dockerfile_sha256"], str(path))
    budget = manifest["budget"]["games"]
    methods = manifest["methods"]
    clocks = {
        "logistic GP": 2 * methods["logistic_gp"]["settings"]["total_batches"],
        "CTT": (2 * methods["chess_tuning_tools"]["settings"]["iterations"]
                * methods["chess_tuning_tools"]["settings"]["rounds"]),
        "RBFOpt": methods["rbfopt"]["settings"]["games"],
        "SPSA": (2 * methods["spsa"]["settings"]["iterations"]
                 * methods["spsa"]["settings"]["pairs_per_step"]),
        "CLOP": methods["clop"]["settings"]["max_games"],
    }
    require(2 * manifest["budget"]["paired_observations"], budget, "budget")
    for name, games in clocks.items():
        require(games, budget, f"{name} game clock")
    return manifest, space


def trace(engine, tables, openings, options, count=100, depth=8):
    positions = [line.strip() for line in pathlib.Path(openings).read_text().splitlines()
                 if line.strip()][:count]
    if len(positions) != count:
        raise RuntimeError(f"training book has only {len(positions)} positions")
    commands = [f"setoption name {name} value {value}" for name, value in options.items()]
    commands.append("ucinewgame")
    for position in positions:
        commands += [f"position fen {position}", f"go depth {depth}"]
    commands.append("quit")
    process = subprocess.run(
        [str(engine), str(tables)], input="\n".join(commands) + "\n",
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(f"trace engine failed: {process.stderr.strip()}")
    selected = [line for line in process.stdout.splitlines()
                if line.startswith("info") or line.startswith("done")]
    return hashlib.sha256(("\n".join(selected) + "\n").encode()).hexdigest()


def verify_traces(manifest, engine, tables, openings):
    for start in manifest["starts"]:
        corrected = start["options"]
        old = corrected | start["old_overrides"]
        expected = start["depth8_trace_sha256"]
        require(trace(engine, tables, openings, old), expected, f"old start {start['source_index']}")
        require(
            trace(engine, tables, openings, corrected), expected,
            f"corrected start {start['source_index']}")


def materialize(manifest, space, output):
    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for start in manifest["starts"]:
        spec = centered(space, start["options"])
        path = output / f"start-{start['source_index']:02d}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--engine")
    parser.add_argument("--tables")
    parser.add_argument("--training-book")
    parser.add_argument("--fastchess")
    parser.add_argument("--heldout-book")
    parser.add_argument("--output-spaces")
    args = parser.parse_args()
    manifest, space = audit(args.manifest, args.root)
    supplied = [args.engine, args.tables, args.training_book]
    if any(supplied) and not all(supplied):
        parser.error("--engine, --tables, and --training-book must be supplied together")
    external = {
        "engine": args.engine, "tables": args.tables, "training_book": args.training_book,
        "fastchess": args.fastchess, "heldout_book": args.heldout_book,
    }
    for name, path in external.items():
        if path:
            require(digest(path), manifest["artifacts"][name], name)
    if all(supplied):
        verify_traces(manifest, *supplied)
    if args.output_spaces:
        materialize(manifest, space, args.output_spaces)
    print("recovery benchmark verified")


if __name__ == "__main__":
    main()
