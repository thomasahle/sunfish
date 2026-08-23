#!/usr/bin/env python3
"""Freeze optimizer checkpoint artifacts into one audited recommendation set."""

import argparse
import hashlib
import json
import math
import pathlib


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def save(path, payload):
    target = pathlib.Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def identifier(options):
    payload = json.dumps(options, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def normalize(parameters, reference, options):
    unknown = set(options) - set(reference)
    if unknown:
        raise ValueError(f"unknown recommendation options: {', '.join(sorted(unknown))}")
    values = reference | options
    output = {}
    for parameter in parameters:
        name, kind, value = parameter["name"], parameter["type"], values[parameter["name"]]
        if kind == "integer":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value % 1:
                raise ValueError(f"{name} is not an integer")
            value = int(value)
        elif kind == "real":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} is not real")
            value = float(value)
        else:
            choices = parameter.get("ordered_values") or parameter.get("values")
            if value not in choices:
                raise ValueError(f"{name} is outside its choices")
        if kind in {"integer", "real"} and (
                not math.isfinite(value) or not parameter["min"] <= value <= parameter["max"]):
            raise ValueError(f"{name} is outside its bounds")
        output[name] = value
    return output


def root_for(manifest):
    path = pathlib.Path(manifest).resolve()
    return path.parents[2] if path.parent.name == "tune" else path.parent


def freeze(benchmark_path, analysis_path, sources):
    benchmark_path = pathlib.Path(benchmark_path).resolve()
    analysis_path = pathlib.Path(analysis_path).resolve()
    benchmark = json.loads(benchmark_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    benchmark_sha = digest(benchmark_path)
    if analysis["training_benchmark"]["sha256"] != benchmark_sha:
        raise ValueError("analysis manifest names a different training benchmark")
    space = root_for(benchmark_path) / benchmark["space"]["path"]
    if digest(space) != benchmark["space"]["sha256"]:
        raise ValueError("parameter-space hash disagrees with the benchmark")
    parameters = json.loads(space.read_text())["parameters"]
    reference = normalize(parameters, benchmark["reference"]["options"], {})
    checkpoints = benchmark["budget"]["checkpoints"]
    starts = {item["source_index"]: normalize(parameters, reference, item["options"])
              for item in benchmark["starts"]}
    methods = set(analysis["method_labels"].values())
    fingerprint = {
        "analysis_sha256": digest(analysis_path),
        "training_benchmark_sha256": benchmark_sha,
        "method_source_commit": benchmark["method_source_commit"],
    }
    aliases, artifacts = [], []
    seen = set()
    for source in sources:
        source = pathlib.Path(source)
        records = json.loads(source.read_text())
        labels = {(record.get("method"), record.get("start")) for record in records}
        if len(labels) != 1:
            raise ValueError(f"{source} does not contain one method/start")
        method, start = labels.pop()
        if method not in methods or start not in starts or (method, start) in seen:
            raise ValueError(f"unexpected or duplicate recommendation artifact: {method}/{start}")
        seen.add((method, start))
        by_checkpoint = {record["checkpoint"]: record for record in records}
        if len(by_checkpoint) != len(records) or sorted(by_checkpoint) != checkpoints:
            raise ValueError(f"{method}/{start} does not contain every checkpoint exactly once")
        source_sha = digest(source)
        artifacts.append({"method": method, "start": start, "sha256": source_sha})
        for checkpoint in checkpoints:
            record = by_checkpoint[checkpoint]
            trained = record["trained_games"]
            vintage = record["recommendation_games"]
            if not (0 <= vintage <= trained <= checkpoint):
                raise ValueError(f"invalid game vintage for {method}/{start}/{checkpoint}")
            if checkpoint == benchmark["budget"]["games"] and trained != checkpoint:
                raise ValueError(f"{method}/{start} did not finish its training budget")
            options = normalize(parameters, reference, record["options"])
            if checkpoint == 0 and options != starts[start]:
                raise ValueError(f"{method}/{start} has the wrong checkpoint-zero start")
            aliases.append({
                "method": method, "start": start, "checkpoint": checkpoint,
                "trained_games": trained, "recommendation_games": vintage,
                "configuration": identifier(options), "options": options,
                "recommendation_artifact_sha256": source_sha,
                "benchmark_protocol": fingerprint,
            })
    expected = {(method, start) for method in methods for start in starts}
    if seen != expected:
        missing = ", ".join(f"{method}/{start}" for method, start in sorted(expected - seen))
        raise ValueError(f"missing recommendation artifacts: {missing}")
    aliases.sort(key=lambda record: (record["method"], record["start"], record["checkpoint"]))
    artifacts.sort(key=lambda record: (record["method"], record["start"]))
    configurations = {}
    for record in aliases:
        key = record["configuration"]
        if key in configurations and configurations[key] != record["options"]:
            raise ValueError(f"configuration identifier collision: {key}")
        configurations[key] = record["options"]
    return aliases, {
        "version": 1, "benchmark_protocol": fingerprint,
        "aliases": len(aliases), "unique_configurations": len(configurations),
        "recommendation_artifacts": artifacts, "configurations": configurations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--recommendation", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    aliases, audit = freeze(args.benchmark, args.analysis, args.recommendation)
    output = pathlib.Path(args.output)
    save(output, aliases)
    audit["recommendations_sha256"] = digest(output)
    save(args.audit, audit)


if __name__ == "__main__":
    main()
