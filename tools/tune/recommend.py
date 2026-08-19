#!/usr/bin/env python3
"""Extract comparable optimizer recommendations at fixed game budgets."""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def parameters(path):
    return json.loads(pathlib.Path(path).read_text())["parameters"]


def defaults(items):
    return {item["name"]: item["default"] for item in items}


def metadata(values):
    result = {}
    for item in values:
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        result[key] = value
    return result


def decode(items, point):
    result = {}
    for item, value in zip(items, point):
        if item["type"] == "integer":
            step = item.get("step", 1)
            value = item["min"] + round((float(value) - item["min"]) / step) * step
            value = min(max(value, item["min"]), item["max"])
        elif item["type"] == "real":
            value = float(value)
        elif isinstance(value, np.generic):
            value = value.item()
        result[item["name"]] = value
    return result


def at_checkpoints(method, history, checkpoints, initial, progress=None):
    history = sorted(history, key=lambda item: item[0])
    progress = sorted(progress if progress is not None else [games for games, _ in history])
    output = []
    for checkpoint in checkpoints:
        eligible = [item for item in history if item[0] <= checkpoint]
        recommendation_games, options = eligible[-1] if eligible else (0, initial)
        trained = [games for games in progress if games <= checkpoint]
        output.append({
            "method": method, "checkpoint": checkpoint,
            "trained_games": trained[-1] if trained else 0,
            "recommendation_games": recommendation_games, "options": options,
        })
    return output


def same_engine(study):
    candidate = dict(study["candidate"])
    baseline = dict(study["baseline"])
    candidate.pop("options", None)
    baseline.pop("options", None)
    return candidate == baseline


def gp_recommend(state, space, batches, pair_weight, inducing):
    import adaptive_gp

    if not batches:
        return space.knobs(space.default)
    partial = dict(state)
    partial["batches"] = batches
    allocation = state.get("study", {}).get("allocation", {})
    mean = space.prior_mean
    if allocation.get("learn_mean"):
        mean = adaptive_gp.empirical_mean(
            mean, batches, allocation["initial_design"], space)
    model = adaptive_gp.posterior(
        partial, mean, pair_weight, space, inducing)
    observed = []
    for batch in batches:
        observed.append(space.canonical(batch["knobs"]))
        if batch.get("opponent_knobs") is not None:
            observed.append(space.canonical(batch["opponent_knobs"]))
    rejected = {
        space.canonical(record["knobs"])
        for record in state.get("gates", {}).values() if not record["accepted"]
    }
    if allocation.get("local_acquisition"):
        points = sorted(set(observed) - rejected)
        means, _ = model.predict(points)
        return space.knobs(points[int(np.argmax(means))])
    seeds = sorted(set(space.candidates + observed) - rejected)
    best = adaptive_gp.coordinate_maximum(
        space, seeds, lambda points: model.predict(points)[0], rejected, None, restarts=16)
    return space.knobs(best)


def gp(args, checkpoints):
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "logistic_gp"))
    import adaptive_gp
    import logistic_gp

    state = adaptive_gp.load_state(args.state, 1)
    space = logistic_gp.MixedSpace.load(args.space)
    study = state.get("study", {})
    baseline = study.get("baseline", {}).get("options")
    if ({"candidate", "baseline"} <= study.keys()
            and same_engine(study) and baseline == space.knobs(space.default)):
        space.condition(space.default)
    history = []
    progress = []
    games = 0
    next_checkpoint = 0
    for index, batch in enumerate(state["batches"], 1):
        games += batch["wins"] + batch["draws"] + batch["losses"]
        progress.append(games)
        while next_checkpoint < len(checkpoints) and checkpoints[next_checkpoint] <= games:
            checkpoint = checkpoints[next_checkpoint]
            prefix = state["batches"][:index]
            history.append((games, gp_recommend(
                state, space, prefix, args.pair_weight, args.inducing)))
            next_checkpoint += 1
    return at_checkpoints(
        args.method, history, checkpoints, space.knobs(space.default), progress)


def ctt(args, checkpoints):
    items = parameters(args.space)
    config = json.loads(pathlib.Path(args.config).read_text())
    rounds = config["rounds"]
    with np.load(args.data) as data:
        evaluations = len(data["arr_0"])
        optima = data["arr_3"].tolist() if "arr_3" in data else []
        performance = data["arr_4"].tolist() if "arr_4" in data else []
    if len(optima) != len(performance):
        raise RuntimeError("CTT optimum and performance histories have different lengths")
    if evaluations >= config["n_initial_points"] and not optima:
        raise RuntimeError("CTT fitted a model but recorded no current optimum")
    history = [
        (int(row[0]) * 2 * rounds, decode(items, point))
        for point, row in zip(optima, performance)
    ]
    progress = range(2 * rounds, 2 * rounds * (evaluations + 1), 2 * rounds)
    return at_checkpoints(args.method, history, checkpoints, defaults(items), progress)


def clop(args, checkpoints):
    items = parameters(args.space)
    config = pathlib.Path(args.config).read_text()
    name = next(line.split(maxsplit=1)[1] for line in config.splitlines()
                if line.startswith("Name "))
    source = pathlib.Path(args.config).parent / f"{name}.dat"
    with tempfile.TemporaryDirectory() as directory:
        shutil.copy2(source, pathlib.Path(directory, source.name))
        process = subprocess.run(
            [args.console, "r"], input=config, text=True, cwd=directory,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(f"CLOP replay failed:\n{process.stderr}")
    history = []
    games = 0
    for line in process.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            [float(field) for field in fields[:4]]
        except ValueError:
            continue
        games += 1
        if len(fields) >= 4 + len(items):
            history.append((games, decode(items, fields[-len(items):])))
    return at_checkpoints(
        args.method, history, checkpoints, defaults(items), range(1, games + 1))


def spsa(args, checkpoints):
    items = parameters(args.space)
    state = json.loads(pathlib.Path(args.state).read_text())
    history = []
    games = 0
    for result in state["results"]:
        games += 2 * result["pairs"]
        point = []
        for item in items:
            value = result["theta"][item["name"]]
            if item["type"] in {"discrete", "categorical", "boolean"}:
                values = item.get("ordered_values") or item["values"]
                value = values[min(max(round(value), 0), len(values) - 1)]
            point.append(value)
        history.append((games, decode(items, point)))
    return at_checkpoints(
        args.method, history, checkpoints, defaults(items), [games for games, _ in history])


def rbfopt(args, checkpoints):
    items = parameters(args.space)
    state = json.loads(pathlib.Path(args.state).read_text())
    history = [(row["games"], row["options"]) for row in state["checkpoints"]]
    progress = []
    games = 0
    for evaluation in state["evaluations"]:
        games += 2 * evaluation["pairs"]
        progress.append(games)
    return at_checkpoints(args.method, history, checkpoints, defaults(items), progress)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", default="0,100,200,400,700,1000")
    parser.add_argument("--output")
    parser.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE")
    subparsers = parser.add_subparsers(dest="kind", required=True)

    gp_parser = subparsers.add_parser("gp")
    gp_parser.add_argument("--state", required=True)
    gp_parser.add_argument("--space", required=True)
    gp_parser.add_argument("--method", default="logistic-gp")
    gp_parser.add_argument("--pair-weight", type=float, default=.5)
    gp_parser.add_argument("--inducing", type=int, default=128)

    ctt_parser = subparsers.add_parser("ctt")
    ctt_parser.add_argument("--data", required=True)
    ctt_parser.add_argument("--config", required=True)
    ctt_parser.add_argument("--space", required=True)
    ctt_parser.add_argument("--method", default="ctt-mes")

    clop_parser = subparsers.add_parser("clop")
    clop_parser.add_argument("--console", required=True)
    clop_parser.add_argument("--config", required=True)
    clop_parser.add_argument("--space", required=True)
    clop_parser.add_argument("--method", default="clop")

    spsa_parser = subparsers.add_parser("spsa")
    spsa_parser.add_argument("--state", required=True)
    spsa_parser.add_argument("--space", required=True)
    spsa_parser.add_argument("--method", default="spsa")

    rbf_parser = subparsers.add_parser("rbfopt")
    rbf_parser.add_argument("--state", required=True)
    rbf_parser.add_argument("--space", required=True)
    rbf_parser.add_argument("--method", default="rbfopt")

    args = parser.parse_args()
    checkpoints = sorted(set(map(int, args.checkpoints.split(","))))
    if not checkpoints or checkpoints[0] < 0:
        parser.error("checkpoints must be nonnegative")
    records = globals()[args.kind](args, checkpoints)
    for record in records:
        record.update(metadata(args.meta))
    output = json.dumps(records, indent=2, sort_keys=True) + "\n"
    if args.output:
        pathlib.Path(args.output).write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
