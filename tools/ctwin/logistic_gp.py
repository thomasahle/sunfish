#!/usr/bin/env python3
"""Suggest intrinsic-LMR experiments with a logistic Gaussian process.

The response is game score against one fixed baseline. Draws contribute half
a success. With the default 0.5 game weight, one color-swapped pair is one
Bernoulli-equivalent trial: its normalized score lies in [0, 1], so its
variance is at most p(1-p) regardless of within-pair correlation. Numeric
coordinates use RBF similarity and categorical coordinates use Hamming
similarity; the space chooses an additive/product blend. A Laplace
approximation gives the posterior over latent log-odds, from which an
upper-confidence acquisition chooses the next arms.

This is a lab scheduler, not engine code. It deliberately searches a finite
space of short, gamma-independent edge-cost rules.
"""

import argparse
import itertools
import json
import math
import pathlib
import re
import sys

import numpy as np
from scipy.special import expit

NUMERIC = ("LMR_MIN_DEPTH", "LMR_LIMIT", "LMR_SAFE_MARGIN")
BOOLEAN = (
    "LMR_SHARED",
    "LMR_CHECK_FULL",
    "LMR_EVASIONS_FULL",
    "LMR_QUIET_ONLY",
    "LMR_NONPAWN_ONLY",
    "LMR_HOT_ONLY",
)
DEFAULT = {
    "LMR_ON": 1,
    "LMR_MIN_DEPTH": 6,
    "LMR_LIMIT": 0,
    "LMR_SAFE_MARGIN": 0,
    **dict.fromkeys(BOOLEAN, 0),
}
ELO_PER_LOGIT = 400 / math.log(10)
BOOLEAN_LOGIT_PRIOR = -0.20
KERNEL_VARIANCE = 0.16


def canonical(knobs):
    """Fill absent coordinates and return one hashable policy vector."""
    full = DEFAULT | knobs
    return tuple(int(full[k]) for k in NUMERIC + BOOLEAN)


def knobs(vector):
    result = {"LMR_ON": 1}
    for key, value in zip(NUMERIC + BOOLEAN, vector):
        if key in NUMERIC or value:
            result[key] = int(value)
    return result


def prior_mean(x):
    """Prefer one-clause policies until game evidence earns complexity."""
    x = np.asarray(x)
    complexity = (x[..., 2] != 0) + np.sum(x[..., len(NUMERIC):] != 0, axis=-1)
    return BOOLEAN_LOGIT_PRIOR * complexity


class MixedSpace:
    """Finite numeric/categorical UCI option space loaded from JSON."""

    def __init__(self, spec):
        if "parameters" in spec:
            numeric_types = {"integer", "real", "discrete"}
            self.numeric = [
                parameter for parameter in spec["parameters"]
                if parameter.get("type", "real") in numeric_types
            ]
            self.categorical = [
                parameter for parameter in spec["parameters"]
                if parameter.get("type") in {"categorical", "boolean"}
            ]
            if len(self.numeric) + len(self.categorical) != len(spec["parameters"]):
                raise ValueError("parameter type must be integer, real, discrete, categorical, or boolean")
        else:
            self.numeric = spec.get("numeric", [])
            self.categorical = spec.get("categorical", [])
        for parameter in self.numeric + self.categorical:
            parameter["values"] = self.parameter_values(parameter)
        self.parameters = self.numeric + self.categorical
        self.names = tuple(parameter["name"] for parameter in self.parameters)
        self.defaults = {
            parameter["name"]: parameter["default"] for parameter in self.parameters
        }
        self.seed_copies = spec.get("seed_copies", {})
        unknown_copies = (set(self.seed_copies) | set(self.seed_copies.values())) - set(self.names)
        if unknown_copies:
            raise ValueError(f"unknown seed-copy parameters: {sorted(unknown_copies)}")
        self.conditions = spec.get("conditions", [])
        unknown = {
            name
            for condition in self.conditions
            for name in [*condition.get("when", {}), *condition.get("reset", [])]
            if name not in self.names
        }
        if unknown:
            raise ValueError(f"unknown conditional parameters: {sorted(unknown)}")
        self.variance = spec.get("kernel_variance", KERNEL_VARIANCE)
        self.hamming = spec.get("hamming", 0.7)
        self.interaction_fraction = spec.get("interaction_fraction", 1.0)
        if not 0 <= self.interaction_fraction <= 1:
            raise ValueError("interaction_fraction must be between zero and one")
        self.interaction_order = spec.get("interaction_order", "all")
        if self.interaction_order not in (2, "all"):
            raise ValueError("interaction_order must be 2 or 'all'")
        self.anchor = None
        # A non-default UCI value is not generically more complex. Studies
        # whose categorical arms add code can opt into a simplicity prior.
        self.clause_prior = spec.get("clause_logit_prior", 0.0)
        self.structural_fraction = spec.get("structural_fraction")
        self._inducing = {}
        choices = [parameter["values"] for parameter in self.parameters]
        self.default = self.canonical(self.defaults)
        maximum = spec.get("max_candidates")
        dictionaries = spec.get("candidates")
        if dictionaries is None:
            size = math.prod(map(len, choices))
            if not maximum or size <= spec.get("max_grid", 100000):
                dictionaries = (dict(zip(self.names, values)) for values in itertools.product(*choices))
            else:
                dictionaries = self.halton_design(
                    self.names, choices, maximum * spec.get("design_oversample", 8))
        candidates = sorted(set(self.canonical(values) for values in dictionaries))
        if maximum and len(candidates) > maximum:
            required = [self.default]
            defaults = self.knobs(self.default)
            axis = [
                self.canonical(defaults | {parameter["name"]: value})
                for parameter in self.parameters for value in parameter["values"]
            ]
            explicit = [self.canonical(values) for values in spec.get("required", [])]
            invalid = [point for point in explicit if not self.contains(point)]
            if invalid:
                raise ValueError(f"required candidates are outside the parameter domain: {invalid}")
            required += axis if len(set(required + axis + explicit)) <= maximum else [
                self.canonical(defaults | {parameter["name"]: value})
                for parameter in self.parameters
                for value in [parameter["values"][0], parameter["values"][-1],
                              *parameter.get("off_values", [])]
            ]
            required += explicit
            if local_count := spec.get("local_interactions"):
                mutations = [
                    (parameter["name"], value)
                    for parameter in self.parameters
                    for value in parameter["values"]
                    if value != parameter["default"]
                ]
                local = [self.canonical(defaults | dict(pair))
                         for pair in itertools.combinations(mutations, 2)
                         if pair[0][0] != pair[1][0]]
                required = self.maximin(sorted(set(local + required)), required, local_count)
            candidates = sorted(set(candidates + required))
            fraction = self.structural_fraction
            if fraction is None:
                candidates = self.maximin(candidates, required, maximum)
            else:
                ordinary = [point for point in candidates if not self.is_structural(point)]
                extremes = [point for point in candidates if self.is_structural(point)]
                ordinary_required = [point for point in required if not self.is_structural(point)]
                extreme_required = [point for point in required if self.is_structural(point)]
                extreme_count = max(len(extreme_required), round(maximum * fraction))
                extreme_count = min(extreme_count, len(extremes))
                ordinary_count = min(maximum - extreme_count, len(ordinary))
                extreme_count = min(maximum - ordinary_count, len(extremes))
                candidates = self.maximin(ordinary, ordinary_required, ordinary_count)
                candidates += self.maximin(extremes, extreme_required, extreme_count)
                candidates.sort()
        self.candidates = candidates

    def inducing_points(self, count):
        """A fixed basis makes online posterior updates order-stable and cheap."""
        count = min(count, len(self.candidates))
        if count not in self._inducing:
            self._inducing[count] = self.maximin(
                self.candidates, [self.default], count)
        return self._inducing[count]

    @staticmethod
    def halton_design(names, choices, count):
        """Generate a deterministic product-space design without materializing it."""
        primes = []
        for candidate in itertools.count(2):
            if len(primes) == len(choices):
                break
            if all(candidate % prime for prime in primes if prime * prime <= candidate):
                primes.append(candidate)

        def coordinate(index, base):
            value, fraction = 0, 1 / base
            while index:
                index, digit = divmod(index, base)
                value += digit * fraction
                fraction /= base
            return value

        for index in range(1, count + 1):
            values = [
                choices[axis][min(int(coordinate(index, prime) * len(choices[axis])),
                                  len(choices[axis]) - 1)]
                for axis, prime in enumerate(primes)
            ]
            yield dict(zip(names, values))

    @staticmethod
    def parameter_values(parameter):
        if "values" in parameter:
            return parameter["values"]
        kind = parameter.get("type", "real")
        if kind == "boolean":
            return [False, True]
        if kind == "integer":
            step = parameter.get("step", 1)
            values = range(parameter["min"], parameter["max"] + 1, step)
            return list(dict.fromkeys([*values, *parameter.get("extra_values", [])]))
        if kind == "real":
            count = parameter["count"]
            if parameter.get("transform", "linear") == "log":
                return np.geomspace(parameter["min"], parameter["max"], count).tolist()
            return np.linspace(parameter["min"], parameter["max"], count).tolist()
        raise ValueError(f"{parameter['name']} needs an explicit values list")

    @classmethod
    def load(cls, path):
        return cls(json.loads(pathlib.Path(path).read_text()))

    def canonical(self, values):
        values = self.defaults | values
        for _ in self.conditions:
            old = values.copy()
            for condition in self.conditions:
                if all(values[name] in choices for name, choices in condition["when"].items()):
                    values.update((name, self.defaults[name]) for name in condition["reset"])
                    values.update(condition.get("set", {}))
            if values == old:
                break
        result = []
        for parameter in self.numeric:
            result.append(values.get(parameter["name"], parameter["default"]))
        for parameter in self.categorical:
            value = values.get(parameter["name"], parameter["default"])
            result.append(parameter["values"].index(value))
        return tuple(result)

    def normalize(self, vector):
        """Collapse vectors that differ only in conditionally inactive parameters."""
        return self.canonical(self.knobs(vector))

    def knobs(self, vector):
        result = {}
        numeric_count = len(self.numeric)
        for parameter, value in zip(self.numeric, vector):
            if parameter.get("type") == "integer":
                value = int(value)
            result[parameter["name"]] = value
        for parameter, index in zip(self.categorical, vector[numeric_count:]):
            result[parameter["name"]] = parameter["values"][int(index)]
        return result

    @property
    def coordinate_values(self):
        """Canonical values available along each numeric or categorical axis."""
        numeric = [tuple(parameter["values"]) for parameter in self.numeric]
        categorical = [tuple(range(len(parameter["values"]))) for parameter in self.categorical]
        return tuple(numeric + categorical)

    def is_structural(self, vector):
        values = self.knobs(vector)
        return any(
            values[parameter["name"]] in parameter.get("off_values", [])
            for parameter in self.parameters
        )

    def contains(self, vector):
        return all(value in choices for value, choices in zip(vector, self.coordinate_values))

    def raw_kernel(self, a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        numeric_count = len(self.numeric)
        similarities = []
        if numeric_count:
            left = self.numeric_coordinates(a[:, :numeric_count])
            right = self.numeric_coordinates(b[:, :numeric_count])
            scales = np.array([self.numeric_scale(parameter) for parameter in self.numeric])
            delta = (left[:, None, :] - right[None, :, :]) / scales
            similarities.extend(np.exp(-0.5 * delta[..., axis] ** 2)
                                for axis in range(numeric_count))
        similarities.extend(
            np.exp(-self.hamming * (a[:, None, axis] != b[None, :, axis]))
            for axis in range(numeric_count, a.shape[1]))
        similarities = np.asarray(similarities)
        main = np.mean(similarities, axis=0)
        if self.interaction_order == 2 and len(similarities) > 1:
            total = np.sum(similarities, axis=0)
            interaction = (total * total - np.sum(similarities * similarities, axis=0))
            interaction /= len(similarities) * (len(similarities) - 1)
        else:
            interaction = np.prod(similarities, axis=0)
        weight = self.interaction_fraction
        return self.variance * ((1 - weight) * main + weight * interaction)

    def kernel(self, a, b):
        covariance = self.raw_kernel(a, b)
        if self.anchor is None:
            return covariance
        anchor = [self.anchor]
        return covariance - self.raw_kernel(a, anchor) @ self.raw_kernel(anchor, b) / self.variance

    def kernel_diagonal(self, points):
        if self.anchor is None:
            return np.full(len(points), self.variance)
        cross = self.raw_kernel(points, [self.anchor])[:, 0]
        return self.variance - cross * cross / self.variance

    def condition(self, point):
        """Condition the prior on one parameter point whose value is exactly zero."""
        self.anchor = self.normalize(point)

    def numeric_coordinates(self, values):
        result = np.asarray(values, dtype=float).copy()
        for index, parameter in enumerate(self.numeric):
            transform = parameter.get("transform", "linear")
            if transform == "log":
                if np.any(result[..., index] <= 0):
                    raise ValueError(f"{parameter['name']} log values must be positive")
                result[..., index] = np.log(result[..., index])
            elif transform != "linear":
                raise ValueError(f"unknown transform {transform!r}")
        return result

    @staticmethod
    def numeric_scale(parameter):
        if "scale" in parameter:
            return parameter["scale"]
        values = np.asarray(parameter["values"], dtype=float)
        if parameter.get("transform", "linear") == "log":
            values = np.log(values)
        span = np.ptp(values)
        return span / 3 if span else 1

    def maximin(self, candidates, required, maximum):
        """Keep required points, then fill with a deterministic maximin design."""
        required = list(dict.fromkeys(required))
        missing = [point for point in required if point not in candidates]
        if missing:
            raise ValueError(f"required candidates are outside the parameter grid: {missing}")
        if len(required) > maximum:
            raise ValueError("max_candidates is smaller than the required axis design")
        selected = required[:]
        selected_set = set(selected)
        similarity = np.zeros(len(candidates))
        for point in selected:
            similarity = np.maximum(similarity, self.kernel([point], candidates)[0])
        while len(selected) < maximum:
            available = np.array([point not in selected_set for point in candidates])
            index = int(np.argmin(np.where(available, similarity, np.inf)))
            point = candidates[index]
            selected.append(point)
            selected_set.add(point)
            similarity = np.maximum(similarity, self.kernel([point], candidates)[0])
        return sorted(selected)

    def prior_mean(self, x):
        x = np.asarray(x)
        numeric_count = len(self.numeric)
        defaults = np.array([
            parameter["values"].index(parameter["default"])
            for parameter in self.categorical
        ])
        complexity = np.sum(x[..., numeric_count:] != defaults, axis=-1)
        return self.clause_prior * complexity


def policies():
    """Finite, floor-aware family of compact intrinsic-LMR policies."""
    result = []
    # The 20k-node calibration reaches depth 7 in only 5% of positions, so a
    # depth-7 policy is almost always the baseline and teaches this screen
    # nothing. It belongs in the later 3+0.1 confirmation, not this space.
    for depth in (4, 5, 6):
        for limit in (-100, -50, 0, 30, 40, 50, 60, 75, 100):
            for safe in (0, 100, 200, 300):
                for bits in range(1 << len(BOOLEAN)):
                    flags = dict(zip(BOOLEAN, ((bits >> i) & 1 for i in range(len(BOOLEAN)))))
                    if flags["LMR_NONPAWN_ONLY"] and not flags["LMR_QUIET_ONLY"]:
                        continue
                # The deterministic screen found hot-only LMR changes at most
                # 0.11% of nodes through depth 7. Do not spend games learning
                # what an exact activity measurement already settled.
                    if flags["LMR_HOT_ONLY"]:
                        continue
                    if safe and (depth < 6 or any(flags.values())):
                        continue
                    if depth == 4 and not (
                            flags["LMR_CHECK_FULL"] and flags["LMR_EVASIONS_FULL"]):
                        continue
                    result.append(canonical({
                        "LMR_MIN_DEPTH": depth,
                        "LMR_LIMIT": limit,
                        "LMR_SAFE_MARGIN": safe,
                        **flags,
                    }))
    return sorted(set(result))


def kernel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    numeric_scale = np.array((1.25, 65.0, 150.0))
    delta = (a[:, None, :len(NUMERIC)] - b[None, :, :len(NUMERIC)]) / numeric_scale
    distance = np.sum(delta * delta, axis=2)
    hamming = np.sum(a[:, None, len(NUMERIC):] != b[None, :, len(NUMERIC):], axis=2)
    return KERNEL_VARIANCE * np.exp(-0.5 * distance - 0.7 * hamming)


class LegacySpace:
    canonical = staticmethod(canonical)
    knobs = staticmethod(knobs)
    kernel = staticmethod(kernel)
    prior_mean = staticmethod(prior_mean)

    @staticmethod
    def kernel_diagonal(points):
        return np.full(len(points), KERNEL_VARIANCE)

    @property
    def candidates(self):
        return policies()

    @property
    def default(self):
        return canonical(DEFAULT)

    structural_fraction = None

    @staticmethod
    def is_structural(_vector):
        return False


class LogisticGP:
    def __init__(self, mean_function=prior_mean, kernel_function=kernel,
                 kernel_diagonal=None, inducing=None):
        self.mean_function = mean_function
        self.kernel = kernel_function
        self.kernel_diagonal = kernel_diagonal
        self.inducing = inducing

    def fit(self, x, success, trials):
        x = np.asarray(x, dtype=float)
        return self.fit_comparisons(x, np.eye(len(x)), success, trials)

    @staticmethod
    def comparison_features(features, means, design):
        means = np.asarray(means, dtype=float)
        if not isinstance(design, tuple):
            design = np.asarray(design, dtype=float)
            return design @ features, design @ means
        left, right = design
        reduced = features[left].copy()
        offset = means[left].copy()
        mask = right >= 0
        reduced[mask] -= features[right[mask]]
        offset[mask] -= means[right[mask]]
        return reduced, offset

    @staticmethod
    def laplace(precision, design, offset, success, trials):
        def objective(centered):
            logits = offset + design @ centered
            probability = expit(logits)
            loss = 0.5 * centered @ precision @ centered
            loss += np.sum(trials * np.logaddexp(0, logits) - success * logits)
            gradient = precision @ centered
            gradient += design.T @ (trials * probability - success)
            return loss, gradient

        centered = np.zeros(len(precision))
        for _ in range(30):
            loss, gradient = objective(centered)
            probability = expit(offset + design @ centered)
            weight = trials * probability * (1 - probability)
            hessian = precision + design.T @ (weight[:, None] * design)
            step = np.linalg.solve(hessian, gradient)
            decrement = gradient @ step
            if decrement < 1e-8:
                break
            scale = 1
            while objective(centered - scale * step)[0] > loss - 0.01 * scale * decrement:
                scale /= 2
                if scale < 1e-8:
                    raise RuntimeError(
                        f"logistic-GP Newton line search failed at decrement {decrement:g}")
            centered -= scale * step
        else:
            raise RuntimeError("logistic-GP Newton fit did not converge")
        probability = expit(offset + design @ centered)
        weight = trials * probability * (1 - probability)
        posterior = np.linalg.inv(precision + design.T @ (weight[:, None] * design))
        return centered, posterior

    def finish(self):
        self.latent = self.mean + self.centered
        self.alpha = self.precision @ self.centered
        self.variance_precision = (
            self.precision - self.precision @ self.posterior @ self.precision)
        return self

    def fit_comparisons(self, x, design, success, trials):
        """Fit binomial observations whose logits are rows of design @ f(x)."""
        observations = np.asarray(x, dtype=float)
        self.x = observations if self.inducing is None else np.asarray(self.inducing, dtype=float)
        success = np.asarray(success, dtype=float)
        trials = np.asarray(trials, dtype=float)
        covariance = self.kernel(self.x, self.x) + np.eye(len(self.x)) * 1e-6
        self.precision = np.linalg.inv(covariance)
        self.mean = self.mean_function(self.x)
        features = (
            np.eye(len(observations)) if self.inducing is None
            else self.kernel(observations, self.x) @ self.precision)
        reduced, offset = self.comparison_features(
            features, self.mean_function(observations), design)
        self.centered, self.posterior = self.laplace(
            self.precision, reduced, offset, success, trials)
        return self.finish()

    def update_comparisons(self, x, design, success, trials):
        """Apply an online Laplace update to a fixed inducing-point posterior."""
        if self.inducing is None:
            raise ValueError("online updates require a fixed inducing basis")
        observations = np.asarray(x, dtype=float)
        success = np.asarray(success, dtype=float)
        trials = np.asarray(trials, dtype=float)
        features = self.kernel(observations, self.x) @ self.precision
        reduced, offset = self.comparison_features(
            features, self.mean_function(observations), design)
        offset += reduced @ self.centered
        delta, self.posterior = self.laplace(
            np.linalg.inv(self.posterior), reduced, offset, success, trials)
        self.centered += delta
        return self.finish()

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        cross = self.kernel(self.x, x)
        mean = self.mean_function(x) + cross.T @ self.alpha
        if self.kernel_diagonal:
            prior_variance = self.kernel_diagonal(x)
        else:
            prior_variance = np.fromiter(
                (self.kernel([point], [point])[0, 0] for point in x), float, len(x))
        variance = prior_variance - np.sum(
            cross * (self.variance_precision @ cross), axis=0)
        return mean, np.maximum(variance, 1e-9)

    def predict_covariance(self, x):
        """Return latent means and their joint posterior covariance."""
        x = np.asarray(x, dtype=float)
        cross = self.kernel(self.x, x)
        mean = self.mean_function(x) + cross.T @ self.alpha
        covariance = self.kernel(x, x) - cross.T @ self.variance_precision @ cross
        return mean, (covariance + covariance.T) / 2

    def predict_cross_covariance(self, left, right):
        """Return posterior covariance between two collections of points."""
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        left_cross = self.kernel(self.x, left)
        right_cross = self.kernel(self.x, right)
        return (
            self.kernel(left, right)
            - left_cross.T @ self.variance_precision @ right_cross)


CHECKPOINT = re.compile(r"A=(\S+) vs B=\S+\s+\+(\d+)\s+=(\d+)\s+-(\d+)")


def read_observations(directory, battery, effective_weight, space=None):
    cells = json.loads(pathlib.Path(battery).read_text())["cells"]
    combined = {}
    names = {}
    modeled = {"LMR_ON", *NUMERIC, *BOOLEAN}
    for path in pathlib.Path(directory).glob("*.log"):
        matches = CHECKPOINT.findall(path.read_text(errors="replace"))
        if not matches:
            continue
        name, wins, draws, losses = matches[-1]
        if name not in cells or name == "baseline":
            continue
        if space is None and any(key.startswith("LMR") and key not in modeled for key in cells[name]):
            continue
        vector = (space.canonical(cells[name]) if space else canonical(cells[name]))
        wins, draws, losses = map(int, (wins, draws, losses))
        success = effective_weight * (wins + draws / 2)
        trials = effective_weight * (wins + draws + losses)
        old_success, old_trials = combined.get(vector, (0, 0))
        combined[vector] = old_success + success, old_trials + trials
        names.setdefault(vector, []).append(name)
    vectors = list(combined)
    success = [combined[x][0] for x in vectors]
    trials = [combined[x][1] for x in vectors]
    return vectors, success, trials, names


def main():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import difftest
    import nodescreen

    parser = argparse.ArgumentParser()
    parser.add_argument("logs")
    parser.add_argument("--battery", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--exploration", type=float, default=1.5)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--screen-depth", type=int, default=6)
    parser.add_argument("--min-node-saving", type=float, default=0.5)
    parser.add_argument("--safe-only", action="store_true")
    parser.add_argument("--binary", default=str(pathlib.Path(__file__).with_name("sunfish_c")))
    parser.add_argument("--tables", default=str(pathlib.Path(__file__).with_name("tables_classic.txt")))
    args = parser.parse_args()

    observed, success, trials, names = read_observations(
        args.logs, args.battery, args.pair_weight)
    if len(observed) < 2:
        raise SystemExit("need results for at least two distinct policies")
    model = LogisticGP().fit(observed, success, trials)
    candidates = policies()
    if args.safe_only:
        candidates = [x for x in candidates if x[2] and not any(x[len(NUMERIC):])]
    mean, variance = model.predict(candidates)
    acquisition = mean + args.exploration * np.sqrt(variance)

    print(f"fit {len(observed)} policies from {sum(trials):.1f} effective games")
    seen_mean, seen_variance = model.predict(observed)
    print("best observed policies:")
    for index in np.argsort(seen_mean)[::-1][:min(5, len(observed))]:
        vector = observed[index]
        print(" ", ",".join(names[vector]),
              f"mean={seen_mean[index] * ELO_PER_LOGIT:+.1f} Elo",
              f"sd={math.sqrt(seen_variance[index]) * ELO_PER_LOGIT:.1f}")
    print("suggested next batch:")
    # Greedy local penalization gives a useful parallel batch instead of ten
    # nearly identical boolean variants around the same unexplored point.
    positions = difftest.load_positions(0)
    base_nodes, _ = nodescreen.run_cell(
        {}, positions, args.screen_depth, args.binary, args.tables)
    remaining = acquisition.copy()
    selected = []
    savings = {}
    while len(selected) < args.count and np.isfinite(remaining).any():
        index = int(np.argmax(remaining))
        node_count, _ = nodescreen.run_cell(
            knobs(candidates[index]), positions, args.screen_depth, args.binary, args.tables)
        saving = 100 * (base_nodes - node_count) / base_nodes
        remaining[index] = -np.inf
        if saving < args.min_node_saving:
            continue
        selected.append(index)
        savings[index] = saving
        similarity = kernel([candidates[index]], candidates)[0]
        remaining -= 0.65 * similarity
    for index in selected:
        vector = candidates[index]
        sd = math.sqrt(variance[index])
        seen = ",".join(names.get(vector, ())) or "new"
        print(json.dumps(knobs(vector), sort_keys=True),
              f"mean={mean[index] * ELO_PER_LOGIT:+.1f} Elo",
              f"sd={sd * ELO_PER_LOGIT:.1f}",
              f"ucb={acquisition[index] * ELO_PER_LOGIT:+.1f}",
              f"nodes={-savings[index]:+.2f}%",
              f"seen={seen}")


if __name__ == "__main__":
    main()
