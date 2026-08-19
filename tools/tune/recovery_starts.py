#!/usr/bin/env python3
"""Generate bad-start candidates and symmetric recovery spaces around them."""

import argparse
import json
import math
import pathlib
import random


def sample(parameter, rng):
    low, default, high = parameter["min"], parameter["default"], parameter["max"]
    # This inner half of the domain is exactly the set of centers whose largest
    # symmetric in-domain interval still contains the original default.
    value = rng.uniform((low + default) / 2, (high + default) / 2)
    return round(value) if parameter["type"] == "integer" else value


def distance(parameters, point):
    return math.sqrt(sum(
        ((point[p["name"]] - p["default"]) / (p["max"] - p["min"])) ** 2
        for p in parameters
    ))


def centered(spec, point):
    result = json.loads(json.dumps(spec))
    for parameter in result["parameters"]:
        name = parameter["name"]
        start = point[name]
        radius = min(start - parameter["min"], parameter["max"] - start)
        if parameter["type"] == "integer":
            radius = math.floor(radius)
        if radius <= 0 or abs(parameter["default"] - start) > radius:
            raise ValueError(f"cannot center a recovery interval for {name} at {start}")
        parameter["min"], parameter["default"], parameter["max"] = (
            start - radius, start, start + radius)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-distance", type=float, default=0.4)
    args = parser.parse_args()
    spec = json.loads(pathlib.Path(args.space).read_text())
    parameters = spec["parameters"]
    if any(p["type"] not in {"integer", "real"} for p in parameters):
        parser.error("recovery starts currently require numeric parameters")
    rng = random.Random(args.seed)
    points = []
    while len(points) < args.samples:
        point = {p["name"]: sample(p, rng) for p in parameters}
        if distance(parameters, point) >= args.minimum_distance and point not in points:
            points.append(point)
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for number, point in enumerate(points):
        path = output / f"start-{number:02d}.json"
        path.write_text(json.dumps(centered(spec, point), indent=2) + "\n")
    manifest = {
        "source": str(pathlib.Path(args.space).resolve()),
        "seed": args.seed,
        "reference": {p["name"]: p["default"] for p in parameters},
        "starts": points,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
