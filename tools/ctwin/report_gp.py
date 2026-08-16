#!/usr/bin/env python3
"""Report exploratory, tested, and statistically supported parameter points."""

import argparse
from collections import Counter
import json
import math

import adaptive_gp
import logistic_gp


def report_domain(space, observed, gate_all, rejected=()):
    """Exclude invalid policies and distinguish the tested recommendation set."""
    candidates = set(space.candidates)
    points = candidates if gate_all else candidates | {
        point for point in observed if space.contains(point)}
    points -= set(rejected)
    tested = {
        point for point in observed if point != space.default and space.contains(point)
        and point in points and (not gate_all or point in candidates)
    }
    return sorted(points), sorted(tested)


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
    gate_all = state["study"]["allocation"].get("gate_all")
    accepted = set()
    rejected = set()
    for record in state.get("gates", {}).values():
        point = space.canonical(record["knobs"])
        (accepted if record["accepted"] else rejected).add(point)
    if gate_all:
        space.candidates = [point for point in space.candidates if point in accepted]
    model = adaptive_gp.posterior(
        state, space.prior_mean, args.pair_weight, space, args.inducing)
    observed = []
    for batch in state["batches"]:
        observed.append(space.canonical(batch["knobs"]))
        if batch.get("opponent_knobs") is not None:
            observed.append(space.canonical(batch["opponent_knobs"]))
    counts = Counter(observed)

    points, challengers = report_domain(space, observed, gate_all, rejected)
    if not challengers:
        raise RuntimeError("the study has no tested challenger to report")
    if gate_all:
        mean = model.predict(points)[0]
        best = points[int(mean.argmax())]
    else:
        best = adaptive_gp.coordinate_maximum(
            space, points, lambda candidates: model.predict(candidates)[0],
            rejected, None, restarts=16)
    challenger_mean = model.predict(challengers)[0]
    challenger_best = challengers[int(challenger_mean.argmax())]
    tested_mean, tested_variance = model.predict(challengers)
    # A noisy multiple-comparison maximum is a lead, not a recommendation.
    supported = tested_mean - 1.96 * tested_variance ** 0.5
    tested_best = challengers[int(supported.argmax())]
    safe = accepted | set(challengers)
    label = "model maximum" if best in safe else "model lead (gate pending)"
    describe(label, best, model, space, counts)
    describe("best tested policy", challenger_best, model, space, counts)
    if supported.max() > 0:
        describe("supported recommendation", tested_best, model, space, counts)
    else:
        print("supported recommendation: none (no tested lower bound exceeds master)")

    axes = space.names if args.all_axes else state.get("new_axes") or [
        name for name in space.names if name.startswith(("VALUE_", "PST_"))
    ]
    base = space.knobs(challenger_best)
    for name in axes:
        parameter = next(parameter for parameter in space.parameters if parameter["name"] == name)
        points = list(dict.fromkeys(
            space.canonical(base | {name: value}) for value in parameter["values"]))
        points = [
            point for point in points
            if point not in rejected and (not gate_all or point in space.candidates)
        ]
        if len(points) < 2:
            continue
        predictions, variances = model.predict(points)
        index = int(predictions.argmax())
        value = space.knobs(points[index])[name]
        axis_elo = predictions[index] * logistic_gp.ELO_PER_LOGIT
        axis_error = 1.96 * math.sqrt(variances[index]) * logistic_gp.ELO_PER_LOGIT
        print(f"{name} {value} {axis_elo:+.1f} +/- {axis_error:.1f} Elo")


if __name__ == "__main__":
    main()
