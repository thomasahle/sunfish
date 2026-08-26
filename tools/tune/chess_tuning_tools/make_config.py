#!/usr/bin/env python3
"""Translate a tuner space into a chess-tuning-tools local configuration."""

import argparse
import json
import pathlib


def option(values):
    return dict(item.split("=", 1) for item in values)


def dimension(parameter):
    kind = parameter["type"]
    if kind == "integer":
        return f"Integer({parameter['min']}, {parameter['max']})"
    if kind == "real":
        prior = ", prior=log-uniform" if parameter.get("transform") == "log" else ""
        return f"Real({parameter['min']}, {parameter['max']}{prior})"
    if kind in {"discrete", "categorical", "boolean"}:
        return repr(tuple(parameter["values"]))
    raise ValueError(f"unsupported parameter type: {kind}")


def required_points(spec):
    parameters = spec["parameters"]
    defaults = {parameter["name"]: parameter["default"] for parameter in parameters}
    result = []
    for override in [{}, *spec.get("required", [])]:
        point = defaults | override
        changed = True
        while changed:
            changed = False
            for clause in spec.get("conditions", []):
                if all(point[name] in values for name, values in clause["when"].items()):
                    for name in clause["reset"]:
                        changed |= point[name] != defaults[name]
                        point[name] = defaults[name]
        values = tuple(point[parameter["name"]] for parameter in parameters)
        if values not in result:
            result.append(values)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-output",
                        help="CSV point that makes the declared defaults CTT's first evaluation")
    parser.add_argument("--candidate", required=True,
                        help="Executable wrapper for the tuned engine")
    parser.add_argument("--baseline", required=True,
                        help="Executable wrapper for the fixed opponent")
    parser.add_argument("--candidate-option", action="append", default=[])
    parser.add_argument("--baseline-option", action="append", default=[])
    parser.add_argument("--openings", required=True)
    parser.add_argument("--tc", default="3+0.1")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--initial-points", type=int, default=24)
    parser.add_argument("--points", type=int, default=1000)
    parser.add_argument("--result-every", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=0,
                        help="stop cleanly after this many evaluated parameter points")
    parser.add_argument("--acquisition", default="mes",
                        choices=("mes", "pvrs", "vr", "ts", "ei", "ttei", "lcb", "mean"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    spec = json.loads(pathlib.Path(args.space).read_text())
    parameters = spec["parameters"]
    config = {
        "engines": [
            {"command": str(pathlib.Path(args.candidate).resolve()),
             "fixed_parameters": option(args.candidate_option)},
            {"command": str(pathlib.Path(args.baseline).resolve()),
             "fixed_parameters": option(args.baseline_option)},
        ],
        "parameter_ranges": {p["name"]: dimension(p) for p in parameters},
        "acq_function": args.acquisition,
        "n_initial_points": args.initial_points,
        "n_points": args.points,
        "random_seed": args.seed,
        "rounds": args.rounds,
        "engine1_tc": args.tc,
        "engine2_tc": args.tc,
        "opening_file": str(pathlib.Path(args.openings).resolve()),
        "adjudicate_draws": True,
        "draw_movenumber": 40,
        "draw_movecount": 8,
        "draw_score": 10,
        "adjudicate_resign": True,
        "resign_movecount": 4,
        "resign_score": 500,
        "concurrency": args.concurrency,
        "plot_every": 0,
        "result_every": args.result_every,
        "logfile": "ctt.log",
    }
    if args.iterations:
        config["max_iterations"] = args.iterations
    if args.start_output:
        config["evaluate_points"] = args.start_output
    pathlib.Path(args.output).write_text(json.dumps(config, indent=2) + "\n")
    if args.start_output:
        rows = [",".join(map(str, [*point, args.rounds])) for point in required_points(spec)]
        pathlib.Path(args.start_output).write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
