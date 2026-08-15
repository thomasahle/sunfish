#!/usr/bin/env python3
"""Report the current posterior optimum and one-axis profiles."""

import argparse
import json
import math
import pathlib

import adaptive_gp
import logistic_gp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--space", required=True)
    parser.add_argument("--inducing", type=int, default=128)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    args = parser.parse_args()

    state = json.loads(pathlib.Path(args.state).read_text())
    space = logistic_gp.MixedSpace.load(args.space)
    space.condition(space.default)
    model = adaptive_gp.posterior(
        state, space.prior_mean, args.pair_weight, space, args.inducing)
    observed = [space.canonical(batch["knobs"]) for batch in state["batches"]]

    points = sorted(set([*space.candidates, *observed]))
    means = model.predict(points)[0]
    best = points[int(means.argmax())]
    predicted, variance = model.predict([best])
    elo = predicted[0] * logistic_gp.ELO_PER_LOGIT
    error = 1.96 * math.sqrt(variance[0]) * logistic_gp.ELO_PER_LOGIT
    changed = {
        name: value for name, value in space.knobs(best).items()
        if value != space.defaults[name]
    }
    print(f"winner {elo:+.1f} +/- {error:.1f} Elo {json.dumps(changed, sort_keys=True)}")

    axes = state.get("new_axes") or [
        name for name in space.names if name.startswith(("VALUE_", "PST_"))
    ]
    for name in axes:
        parameter = next(parameter for parameter in space.parameters if parameter["name"] == name)
        points = [space.canonical({name: value}) for value in parameter["values"]]
        predictions, variances = model.predict(points)
        index = int(predictions.argmax())
        axis_elo = predictions[index] * logistic_gp.ELO_PER_LOGIT
        axis_error = 1.96 * math.sqrt(variances[index]) * logistic_gp.ELO_PER_LOGIT
        print(f"{name} {parameter['values'][index]} {axis_elo:+.1f} +/- {axis_error:.1f} Elo")


if __name__ == "__main__":
    main()
