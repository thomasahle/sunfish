#!/usr/bin/env python3
"""Report the current posterior optimum and one-axis profiles."""

import argparse
from collections import Counter
import json
import math

import adaptive_gp
import logistic_gp


def describe(label, point, model, space, counts):
    predicted, variance = model.predict([point])
    elo = predicted[0] * logistic_gp.ELO_PER_LOGIT
    error = 1.96 * math.sqrt(variance[0]) * logistic_gp.ELO_PER_LOGIT
    changed = {
        name: value for name, value in space.knobs(point).items()
        if value != space.defaults[name]
    }
    evidence = "anchor" if point == space.default else counts[point]
    print(f"{label} {elo:+.1f} +/- {error:.1f} Elo pairs={evidence} "
          f"{json.dumps(changed, sort_keys=True)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--space", required=True)
    parser.add_argument("--inducing", type=int, default=128)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--all-axes", action="store_true")
    args = parser.parse_args()

    state = adaptive_gp.load_state(args.state, 1)
    space = logistic_gp.MixedSpace.load(args.space)
    space.condition(space.default)
    model = adaptive_gp.posterior(
        state, space.prior_mean, args.pair_weight, space, args.inducing)
    observed = []
    for batch in state["batches"]:
        observed.append(space.canonical(batch["knobs"]))
        if batch.get("opponent_knobs") is not None:
            observed.append(space.canonical(batch["opponent_knobs"]))
    counts = Counter(observed)

    points = sorted(set([*space.candidates, *(point for point in observed if space.contains(point))]))
    best = adaptive_gp.coordinate_maximum(
        space, points, lambda candidates: model.predict(candidates)[0], set(), None, restarts=16)
    challengers = sorted(set(observed))
    challenger_mean = model.predict(challengers)[0]
    challenger_best = challengers[int(challenger_mean.argmax())]
    tested = [space.default, *challengers]
    tested_best = tested[int(model.predict(tested)[0].argmax())]
    describe("model maximum", best, model, space, counts)
    describe("best tested challenger", challenger_best, model, space, counts)
    describe("tested recommendation", tested_best, model, space, counts)

    axes = space.names if args.all_axes else state.get("new_axes") or [
        name for name in space.names if name.startswith(("VALUE_", "PST_"))
    ]
    base = space.knobs(best)
    for name in axes:
        parameter = next(parameter for parameter in space.parameters if parameter["name"] == name)
        points = list(dict.fromkeys(
            space.canonical(base | {name: value}) for value in parameter["values"]))
        predictions, variances = model.predict(points)
        index = int(predictions.argmax())
        value = space.knobs(points[index])[name]
        axis_elo = predictions[index] * logistic_gp.ELO_PER_LOGIT
        axis_error = 1.96 * math.sqrt(variances[index]) * logistic_gp.ELO_PER_LOGIT
        print(f"{name} {value} {axis_elo:+.1f} +/- {axis_error:.1f} Elo")


if __name__ == "__main__":
    main()
