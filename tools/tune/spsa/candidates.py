#!/usr/bin/env python3
"""Extract final and tail-averaged candidates from SPSA trajectories."""

import argparse
import json
import pathlib
import sys


sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "logistic_gp"))
import logistic_gp  # noqa: E402


def decode(items, theta):
    result = {}
    for item in items:
        value = theta[item["name"]]
        if item["type"] in {"discrete", "categorical", "boolean"}:
            choices = item.get("ordered_values") or item["values"]
            value = choices[min(max(round(value), 0), len(choices) - 1)]
        elif item["type"] == "integer":
            step = item.get("step", 1)
            value = item["min"] + round((value - item["min"]) / step) * step
            value = min(max(value, item["min"]), item["max"])
        result[item["name"]] = value
    return result


def extract(paths, space_path, tails, results_per_state=None):
    spec = json.loads(pathlib.Path(space_path).read_text())
    items = spec["parameters"]
    space = logistic_gp.MixedSpace(spec)
    output = []
    for path in map(pathlib.Path, paths):
        state = json.loads(path.read_text())
        results = state["results"][:results_per_state]
        if not results:
            continue
        variants = [("final", results[-1]["theta"])]
        for length in tails:
            rows = results[-min(length, len(results)):]
            theta = {
                item["name"]: sum(row["theta"][item["name"]] for row in rows) / len(rows)
                for item in items
            }
            variants.append((f"tail-{length}", theta))
        for variant, theta in variants:
            options = space.knobs(space.canonical(decode(items, theta)))
            output.append({
                "method": "spsa", "trajectory": path.parent.name,
                "variant": variant, "trained_games": 2 * len(results),
                "options": options,
            })
    unique = {}
    for record in output:
        key = json.dumps(record["options"], sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, record)
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("states", nargs="+")
    parser.add_argument("--space", required=True)
    parser.add_argument("--tails", default="50,100")
    parser.add_argument("--results-per-state", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tails = [int(value) for value in args.tails.split(",") if value]
    if any(value <= 0 for value in tails):
        parser.error("tail lengths must be positive")
    if args.results_per_state is not None and args.results_per_state < 1:
        parser.error("results per state must be positive")
    records = extract(args.states, args.space, tails, args.results_per_state)
    pathlib.Path(args.output).write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
