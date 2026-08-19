#!/usr/bin/env python3
"""Translate a numeric tuner space into an official CLOP experiment file."""

import argparse
import json
import pathlib
import shlex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="clop")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--runner-arg", action="append", default=[])
    parser.add_argument("--processors", type=int, default=10)
    parser.add_argument("--draw-elo", type=float, default=65)
    parser.add_argument("--correlations", choices=("all", "none"), default="all")
    args = parser.parse_args()
    parameters = json.loads(pathlib.Path(args.space).read_text())["parameters"]
    kinds = {
        ("integer", False): "IntegerParameter",
        ("integer", True): "IntegerGammaParameter",
        ("real", False): "LinearParameter",
        ("real", True): "GammaParameter",
    }
    lines = [
        f"Name {args.name}",
        "Script " + " ".join(map(shlex.quote, [args.runner, *args.runner_arg])),
    ]
    for parameter in parameters:
        key = parameter["type"], parameter.get("transform") == "log"
        if key not in kinds:
            raise ValueError(f"CLOP requires integer or real parameters, not {parameter['type']}")
        lines.append(
            f"{kinds[key]} {parameter['name']} {parameter['min']} {parameter['max']}")
    lines += ["Processor local"] * args.processors
    lines += [
        "Replications 2",
        f"DrawElo {args.draw_elo:g}",
        "H 3",
        f"Correlations {args.correlations}",
    ]
    pathlib.Path(args.output).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
