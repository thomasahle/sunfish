#!/usr/bin/env python3
"""Allocate short fastchess batches with a logistic Gaussian process.

Each lane runs one candidate against the fixed baseline for one color-swapped
opening pair by default.  When a lane finishes, its result is added to the
timed-game posterior and the free lane receives the current best acquisition.
The fixed-node screen is used only as a shrunk prior mean, never as timed data.

Allocation starts with a maximin design.  It then reserves a slowly decaying,
nonzero fraction of batches for maximum-posterior-variance exploration and uses
GP-UCB for the rest.  This prevents a noisy early loss from permanently
discarding a region while still concentrating games around promising policies.
"""

import os

# Small GP matrices are faster single-threaded, and tuning CPU belongs to games.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import asyncio
import copy
import hashlib
import json
import math
import pathlib
import random
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, deque

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import gating  # noqa: E402
import locking  # noqa: E402
import pentanomial  # noqa: E402
import opponent_panel  # noqa: E402
import logistic_gp


SCORE = re.compile(
    r"Score of candidate vs (?:baseline|opponent):\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)")
PAIR_HEADER = re.compile(r"^adaptive-gp-identity ([0-9a-f]{64})$", re.MULTILINE)
PAIR_RESULT = re.compile(
    r"^adaptive-gp-result ([0-9a-f]{64}) (\d+) (\d+) (\d+)$", re.MULTILINE)
UCI_OPTION = re.compile(r"^option name (.+?) type ")
SAVED_STATES = {}
STATE_LOCK = threading.RLock()


def duration(value):
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([smhd]?)", value)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid duration: {value}")
    number, unit = match.groups()
    return float(number) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def validate_options(command, arguments, required):
    """Refuse silent fastchess option loss before spending any games."""
    if not required:
        return
    process = subprocess.run(
        [command, *shlex.split(arguments)], input="uci\nquit\n", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    advertised = {
        match.group(1)
        for line in process.stdout.splitlines()
        if (match := UCI_OPTION.match(line))
    }
    missing = sorted(set(required) - advertised)
    if process.returncode or "uciok" not in process.stdout or missing:
        detail = f"; missing UCI options: {', '.join(missing)}" if missing else ""
        raise RuntimeError(f"cannot validate {command}{detail}\n{process.stderr}")


def load_state(path, start):
    path = pathlib.Path(path)
    if not path.exists():
        return {"next_opening": start, "batches": []}
    state = json.loads(path.read_text())
    offset = state.pop("_journal_offset", 0)
    journal = path.with_suffix(".jsonl")
    if journal.exists():
        with journal.open("rb+") as events:
            events.seek(offset)
            while line := events.readline():
                start = events.tell() - len(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith(b"\n"):
                        events.truncate(start)
                        break
                    raise
                state["batches"].extend(event.get("batches", ()))
                if event.get("gates"):
                    state.setdefault("gates", {}).update(event["gates"])
                state.update(event.get("meta", {}))
    SAVED_STATES[str(path.resolve())] = state_snapshot(state)
    return state


def state_snapshot(state):
    return {
        "batches": len(state.get("batches", ())),
        "gates": set(state.get("gates", ())),
        "meta": copy.deepcopy({
            key: value for key, value in state.items() if key not in {"batches", "gates"}
        }),
    }


def save_state(path, state):
    with STATE_LOCK:
        path = pathlib.Path(path)
        key = str(path.resolve())
        if not path.exists():
            path.with_suffix(".jsonl").unlink(missing_ok=True)
            checkpoint_state(path, state)
            SAVED_STATES[key] = state_snapshot(state)
            return
        old = SAVED_STATES.setdefault(key, state_snapshot(state))
        new = state_snapshot(state)
        event = {
            "batches": state.get("batches", ())[old["batches"]:],
            "gates": {
                name: record for name, record in state.get("gates", {}).items()
                if name not in old["gates"]
            },
            "meta": {
                name: value for name, value in new["meta"].items()
                if old["meta"].get(name) != value
            },
        }
        if any(event.values()):
            journal = path.with_suffix(".jsonl")
            with journal.open("a") as events:
                events.write(json.dumps(event, separators=(",", ":")) + "\n")
        SAVED_STATES[key] = new


def checkpoint_state(path, state):
    with STATE_LOCK:
        path = pathlib.Path(path)
        journal = path.with_suffix(".jsonl")
        payload = state | {"_journal_offset": journal.stat().st_size if journal.exists() else 0}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


def file_identity(path):
    path = pathlib.Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def state_file_identity(path):
    """Identify both halves of an append-journaled state."""
    path = pathlib.Path(path)
    files = [file_identity(path)]
    journal = path.with_suffix(".jsonl")
    if journal.exists():
        files.append(file_identity(journal))
    return files


def engine_identity(command, arguments, options):
    executable = shutil.which(command) or command
    files = []
    for token in [executable, *shlex.split(arguments)]:
        path = pathlib.Path(token)
        if path.is_file():
            files.append(file_identity(path))
    return {
        "command": str(pathlib.Path(executable).resolve()),
        "arguments": arguments,
        "options": options,
        "files": files,
    }


def command_identity(command):
    if not command:
        return None
    argv = shlex.split(command)
    executable = shutil.which(argv[0]) or argv[0]
    files = [file_identity(token) for token in [executable, *argv[1:]]
             if pathlib.Path(token).is_file()]
    return {"argv": argv, "files": files}


def study_identity(args):
    """Describe everything that changes the distribution of game observations."""
    panel = getattr(args, "baseline_panel", [])
    return {
        "version": 3,
        "scheduler": file_identity(__file__),
        "model": file_identity(pathlib.Path(__file__).with_name("logistic_gp.py")),
        "fastchess": file_identity(shutil.which(args.fastchess) or args.fastchess),
        "candidate": engine_identity(args.engine, args.engine_args, {}),
        "baseline": (None if panel else engine_identity(
            args.baseline_engine, args.baseline_args, args.baseline_options)),
        "baseline_panel": [opponent_panel.identity(member, engine_identity, file_identity)
                           for member in panel],
        "panel_selection": ({
            "helper": file_identity(opponent_panel.__file__),
            "seed": getattr(args, "panel_seed", 2026),
        } if panel else None),
        "openings": file_identity(args.openings),
        "opening_schedule": {
            "cycle": args.cycle_openings,
            "order": args.opening_order,
            "seed": args.opening_seed,
        },
        "gate": command_identity(args.gate),
        "gate_timeout": args.gate_timeout,
        "space": file_identity(args.space) if args.space else "legacy",
        "seed_state": state_file_identity(args.seed_state) if args.seed_state else None,
        "tc": args.tc,
        "allocation": {
            name: getattr(args, name)
            for name in ("pairs", "pair_weight", "exploration", "initial_design", "explore_start",
                         "explore_floor", "explore_half_life", "explore_optimism",
                         "explore_confidence",
                         "duel_fraction", "inducing", "seed_selections",
                         "acquisition_restarts", "update_batches", "gate_workers",
                         "gate_all", "gate_design", "axis_design", "learn_mean",
                         "full_axis_design",
                         "local_acquisition", "local_support", "acquisition")
        },
        "total_batches": args.total_batches,
    }


def bind_study(state, identity):
    previous = state.get("study")
    if previous is None:
        if state["batches"] or state.get("pending"):
            raise RuntimeError("state has observations but no study identity; start a new state")
        state["study"] = identity
    elif previous != identity:
        changed = sorted(key for key in identity if previous.get(key) != identity[key])
        raise RuntimeError(f"state belongs to a different study: {', '.join(changed)} changed")


def fixed_baseline_point(args, space):
    """Return the parameter point already represented exactly by the baseline."""
    if not isinstance(space, logistic_gp.MixedSpace):
        return None
    if getattr(args, "baseline_panel", None):
        return None
    candidate = engine_identity(args.engine, args.engine_args, {})
    baseline = engine_identity(args.baseline_engine, args.baseline_args, {})
    if candidate == baseline and args.baseline_options == space.knobs(space.default):
        return space.default
    return None


def aggregate(batches, pair_weight, space, sparse=False):
    totals = {}
    points = set()
    for batch in batches:
        left = space.canonical(batch["knobs"])
        right = batch.get("opponent_knobs")
        right = space.canonical(right) if right is not None else None
        if not space.contains(left) or right is not None and not space.contains(right):
            continue
        points.add(left)
        if right is not None:
            points.add(right)
        key = left, right
        wins, trials = totals.get(key, (0.0, 0.0))
        wins += pair_weight * (batch["wins"] + batch["draws"] / 2)
        trials += pair_weight * (batch["wins"] + batch["draws"] + batch["losses"])
        totals[key] = wins, trials
    points = sorted(points)
    indices = {point: index for index, point in enumerate(points)}
    left = []
    right = []
    success = []
    trials = []
    for (challenger, opponent), (wins, games) in totals.items():
        left.append(indices[challenger])
        right.append(indices[opponent] if opponent is not None else -1)
        success.append(wins)
        trials.append(games)
    endpoints = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    if sparse:
        return points, endpoints, success, trials
    design = np.zeros((len(totals), len(points)))
    design[np.arange(len(totals)), endpoints[0]] = 1
    mask = endpoints[1] >= 0
    design[np.arange(len(totals))[mask], endpoints[1][mask]] = -1
    return points, design, success, trials


def compatible_seed_batches(seed, space):
    """Keep observations that match the new engine on every removed knob."""
    baseline = seed.get("study", {}).get("baseline", {}).get("options", {})
    removed = {name: value for name, value in baseline.items() if name not in space.names}

    def compatible(knobs):
        return all(knobs.get(name, value) == value for name, value in removed.items())

    return [
        batch for batch in seed["batches"]
        if compatible(batch["knobs"])
        and (batch.get("opponent_knobs") is None or compatible(batch["opponent_knobs"]))
    ]


def import_seed_batches(seed, space):
    """Filter old observations and give newly split axes their old semantics."""
    baseline = seed.get("study", {}).get("baseline", {}).get(
        "options", space.defaults)

    def migrate(knobs):
        knobs = dict(knobs)
        for target, source in space.seed_copies.items():
            knobs.setdefault(target, knobs.get(source, baseline[source]))
        return knobs

    imported = [
        batch | {
            "knobs": migrate(batch["knobs"]),
            "opponent_knobs": migrate(batch.get("opponent_knobs") or baseline),
        }
        for batch in compatible_seed_batches(seed, space)
    ]
    return [batch for batch in imported
            if space.contains(space.canonical(batch["knobs"]))
            and space.contains(space.canonical(batch["opponent_knobs"]))]


def source_prior(logs, battery, transfer, space):
    if not logs:
        return space.prior_mean
    reader_space = None if isinstance(space, logistic_gp.LegacySpace) else space
    vectors, success, trials, _ = logistic_gp.read_observations(
        logs, battery, 0.5, reader_space)
    model = logistic_gp.LogisticGP(
        space.prior_mean, space.kernel, space.kernel_diagonal).fit(
        vectors, success, trials)

    def mean(x):
        simple = space.prior_mean(x)
        screened, _ = model.predict(x)
        return simple + transfer * (screened - simple)

    return mean


def empirical_mean(prior, batches, minimum, space):
    """Learn one intercept from the completed, fixed-opponent design."""
    anchored = [
        batch for batch in batches
        if batch.get("opponent_knobs") is None
        and batch.get("allocation", "design") == "design"
    ]
    if len(anchored) < minimum:
        return prior
    anchored = anchored[:minimum]
    score = sum(batch["wins"] + batch["draws"] / 2 for batch in anchored)
    games = sum(batch["wins"] + batch["draws"] + batch["losses"] for batch in anchored)
    probability = min(max(score / games, 1e-3), 1 - 1e-3)
    points = [space.canonical(batch["knobs"]) for batch in anchored]
    correction = math.log(probability / (1 - probability)) - np.mean(prior(points))

    def mean(x):
        return prior(x) + correction

    return mean


def inducing_basis(points, design, trials, space, count):
    """Keep the most-tested comparison endpoints, then cover the remainder."""
    if isinstance(design, tuple):
        information = np.zeros(len(points))
        np.add.at(information, design[0], trials)
        mask = design[1] >= 0
        np.add.at(information, design[1][mask], np.asarray(trials)[mask])
    else:
        information = np.abs(design).T @ trials
    ranked = sorted(range(len(points)), key=lambda i: (-information[i], points[i]))
    required = [space.default]
    required += [points[i] for i in ranked[:max(1, count // 2)]]
    required = list(dict.fromkeys(required))[:count]
    if isinstance(space, logistic_gp.MixedSpace):
        pool = sorted(set([*space.candidates, *required]))
        count = min(count, len(pool))
        return space.maximin(pool, required, count)
    pool = sorted(set([*points, space.default]))
    count = min(count, len(pool))
    remaining = [point for point in pool if point not in required]
    indices = np.linspace(0, len(remaining) - 1, count - len(required), dtype=int)
    return required + [remaining[index] for index in indices]


def posterior(state, mean_function, pair_weight, space, inducing=0):
    points, design, success, trials = aggregate(
        state["batches"], pair_weight, space, sparse=True)
    if not points:
        return None
    basis = None
    if inducing:
        basis = (space.inducing_points(inducing)
                 if isinstance(space, logistic_gp.MixedSpace)
                 else inducing_basis(points, design, trials, space, inducing))
    return logistic_gp.LogisticGP(
        mean_function, space.kernel, space.kernel_diagonal, basis).fit_comparisons(
        points, design, success, trials)


def update_posterior(model, batches, pair_weight, space):
    points, design, success, trials = aggregate(batches, pair_weight, space, sparse=True)
    return model.update_comparisons(points, design, success, trials) if points else model


def design_variance(sites, candidates, space):
    """Residual prior variance after conditioning noiselessly on design sites."""
    variance = space.kernel_diagonal(candidates)
    if not sites:
        return variance
    cross = space.kernel(sites, candidates)
    site_covariance = space.kernel(sites, sites) + np.eye(len(sites)) * 1e-6
    projection = np.linalg.solve(site_covariance, cross)
    return np.maximum(variance - np.sum(cross * projection, axis=0), 0)


def exploration_probability(selections, start, floor, half_life):
    return floor + (start - floor) / math.sqrt(1 + selections / half_life)


def exploitation(mean, variance, args):
    """Score a latent value relative to the fixed zero-Elo baseline."""
    deviation = np.sqrt(variance)
    if getattr(args, "acquisition", "ucb") == "ucb":
        return mean + args.exploration * deviation
    z = mean / deviation
    if args.acquisition == "pi":
        return ndtr(z)
    return mean * ndtr(z) + deviation * np.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def pending_configurations(slots, pairs):
    """Smallest number of parameter choices that can keep every lane busy."""
    return math.ceil(slots / pairs)


class OpeningSchedule:
    """Map an unbounded sequence onto independently shuffled book epochs."""

    def __init__(self, path, seed=0, cycle=False, order="random"):
        self.count = sum(bool(line.strip()) for line in pathlib.Path(path).read_text().splitlines())
        if not self.count:
            raise ValueError("opening book is empty")
        self.seed = seed
        self.cycle = cycle
        self.random = order == "random"
        self.epochs = {}

    def opening(self, sequence):
        if sequence < 1 or not self.cycle and sequence > self.count:
            raise ValueError(f"opening sequence {sequence} exceeds the {self.count}-position book")
        epoch, offset = divmod(sequence - 1, self.count)
        if epoch not in self.epochs:
            order = list(range(1, self.count + 1))
            if self.random:
                random.Random(self.seed + epoch).shuffle(order)
            self.epochs[epoch] = order
        return self.epochs[epoch][offset]


def pair_identity(state, record, pair):
    """Bind a game result to one state generation and frozen comparison."""
    payload = {
        "state": state["state_id"], "study": state["study"],
        "experiment": record["number"], "pair": pair,
        "knobs": record["knobs"], "opponent_knobs": record["opponent_knobs"],
        "opening_sequence": record["opening_sequence"] + pair,
        "opening": record["openings"][pair],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def restore_pending(state, space):
    """Rebuild unfinished experiments and pair tasks from durable state."""
    experiments = {}
    queue = deque()
    for record in state.setdefault("pending", []):
        number = record["number"]
        if number in experiments:
            raise ValueError(f"duplicate pending experiment {number}")
        if len(record["openings"]) != len(record["results"]):
            raise ValueError(f"pending experiment {number} has mismatched pairs")
        baseline_ids = record.setdefault("baseline_ids", [None] * len(record["results"]))
        if len(baseline_ids) != len(record["results"]):
            raise ValueError(f"pending experiment {number} has mismatched panel members")
        if all(result is not None for result in record["results"]):
            raise ValueError(f"pending experiment {number} is already complete")
        expected = [pair_identity(state, record, pair) for pair in range(len(record["results"]))]
        if record.setdefault("identities", expected) != expected:
            raise ValueError(f"pending experiment {number} has a mismatched identity")
        vector = space.canonical(record["knobs"])
        opponent = (space.canonical(record["opponent_knobs"])
                    if record["opponent_knobs"] is not None else None)
        experiments[number] = {"record": record, "vector": vector, "opponent": opponent}
        for pair, result in enumerate(record["results"]):
            if result is None:
                queue.append((number, pair))
            elif len(result) != 3 or sum(result) != 2:
                raise ValueError(f"pending experiment {number} has an invalid pair result")
    return experiments, queue


def resume_tranche(state, batches, pending):
    """Start a batch tranche, or resume the exact budget interrupted earlier."""
    tranche = state.setdefault("tranche", {"batches": batches, "completed": 0})
    if tranche["batches"] != batches:
        raise ValueError(
            f"unfinished tranche used --batches {tranche['batches']}; resume with that value")
    if not 0 <= tranche["completed"] <= batches:
        raise ValueError("invalid completed count in batch tranche")
    remaining = batches - tranche["completed"]
    if not 0 <= pending <= remaining:
        raise ValueError("pending experiments exceed the unfinished batch tranche")
    return tranche, remaining


def pending_batch(record):
    """Turn one completely observed pending experiment into a GP batch."""
    wins, draws, losses = map(sum, zip(*record["results"]))
    return {
        "knobs": record["knobs"], "wins": wins, "draws": draws, "losses": losses,
        "opening": record["openings"][0],
        "opening_sequence": record["opening_sequence"],
        "openings": record["openings"], "allocation": record["allocation"],
        "baseline_ids": sorted(name for name in record.get("baseline_ids", ()) if name is not None),
        "opponent_knobs": record["opponent_knobs"],
    }


def gate_policy(args, state, space, vector):
    """Run and cache a deterministic policy gate; exit 1 means infeasible."""
    knobs = space.knobs(vector)
    payload = {
        "engine": args.engine,
        "engine_args": args.engine_args,
        "options": knobs,
    }
    return gating.policy(
        args.gate, args.gate_timeout, payload,
        state.setdefault("gates", {}), STATE_LOCK)


def selection_state(state):
    """Fork only the small mutable acquisition clock, not the observations."""
    trial = dict(state)
    trial["allocations"] = dict(state.get("allocations", {}))
    return trial


def commit_selection(state, trial):
    for key, value in trial.items():
        if key in {"selections", "allocations", "exploration_credit"} or key.endswith(
                "_structural_credit"):
            state[key] = value


def exploration_stratum(state, mode, space):
    """Reserve the configured share of design/exploration for structural arms."""
    if space.structural_fraction is None:
        return None
    key = f"{mode}_structural_credit"
    credit = state.get(key, 0.0) + space.structural_fraction
    structural = credit >= 1
    state[key] = credit - structural
    return structural


def coordinate_maximum(
        space, seeds, score, active, structural, restarts=4, steps=None, local=False):
    """Deterministically climb a mixed discrete acquisition from diverse seeds."""
    cache = {}
    normalize = getattr(space, "normalize", lambda point: point)
    active = {normalize(point) for point in active}

    def evaluate(points):
        missing = list(dict.fromkeys(point for point in points if point not in cache))
        if missing:
            cache.update(zip(missing, score(missing)))
        return np.asarray([cache[point] for point in points])

    def allowed(point):
        return point not in active and (
            structural is None or space.is_structural(point) == structural)

    seeds = sorted(set(
        normalize(point) for point in seeds if space.contains(normalize(point))
    ))
    if not seeds:
        raise RuntimeError("no available acquisition point")
    seed_scores = evaluate(seeds)
    order = sorted(range(len(seeds)), key=lambda index: (-seed_scores[index], seeds[index]))
    optima = []
    for index in order[:restarts]:
        point = seeds[index]
        value = seed_scores[index] if allowed(point) else -np.inf
        climbed = 0
        while steps is None or climbed < steps:
            neighbors = []
            choices_by_axis = (space.local_coordinate_values(point)
                if local else space.coordinate_values)
            for axis, choices in enumerate(choices_by_axis):
                for choice in choices:
                    neighbor = normalize(point[:axis] + (choice,) + point[axis + 1:])
                    if allowed(neighbor):
                        neighbors.append(neighbor)
            neighbors = list(dict.fromkeys(neighbors))
            values = evaluate(neighbors)
            best = int(np.argmax(values))
            if values[best] <= value + 1e-12:
                break
            point, value = neighbors[best], values[best]
            climbed += 1
        if allowed(point):
            optima.append((value, point))
    if not optima:
        raise RuntimeError("no available acquisition point")
    return min(optima, key=lambda item: (-item[0], item[1]))[1]


def fantasy_variance(model, space, pending, points, variance, effective_trials):
    """Condition variance on noisy pending candidate-opponent comparisons."""
    if not pending:
        return variance
    sites = sorted({point for comparison in pending for point in comparison if point is not None})
    indices = {point: index for index, point in enumerate(sites)}
    if model:
        _, covariance = model.predict_covariance(sites)
        cross = model.predict_cross_covariance(sites, points)
    else:
        covariance = space.kernel(sites, sites)
        cross = space.kernel(sites, points)
    design = np.zeros((len(pending), len(sites)))
    for row, (left, right) in enumerate(pending):
        design[row, indices[left]] = 1
        if right is not None:
            design[row, indices[right]] = -1
    comparison_covariance = design @ covariance @ design.T
    comparison_cross = design @ cross
    # A binomial logit's Fisher information is at most trials / 4.
    noise = np.eye(len(pending)) * 4 / effective_trials
    projection = np.linalg.solve(comparison_covariance + noise, comparison_cross)
    return np.maximum(variance - np.sum(comparison_cross * projection, axis=0), 1e-9)


def predict(model, points, cache=None):
    """Reuse marginal predictions while retaining fresh cross-covariances."""
    if cache is None:
        return model.predict(points)
    missing = list(dict.fromkeys(point for point in points if point not in cache))
    if missing:
        means, variances = model.predict(missing)
        cache.update(zip(missing, zip(means, variances)))
    return tuple(np.asarray(values) for values in zip(*(cache[point] for point in points)))


def choose(state, mean_function, candidates, pending, args, space, model=None,
           forbidden=(), validated=(), observation_counts=None, validated_only=False,
           prediction_cache=None):
    if model is None:
        model = posterior(
            state, mean_function, args.pair_weight, space, getattr(args, "inducing", 0))

    def statistics(points):
        if model:
            return predict(model, points, prediction_cache)
        mean = mean_function(np.asarray(points))
        variance = np.diag(space.kernel(points, points))
        return mean, variance

    def improvement_statistics(points):
        """Posterior difference from the parameter-space starting policy."""
        mean, variance = statistics(points)
        target_mean, target_variance = statistics([space.default])
        if model:
            cross = model.predict_cross_covariance([space.default], points)[0]
        else:
            cross = space.kernel([space.default], points)[0]
        difference_variance = variance + target_variance[0] - 2 * cross
        return mean - target_mean[0], np.maximum(difference_variance, 1e-9)

    mean, variance = statistics(candidates)

    if observation_counts is None:
        observation_counts = Counter(
            space.canonical(batch["knobs"]) for batch in state["batches"])
    observed = {point for point in observation_counts if space.contains(point)}
    opponents = {
        space.canonical(batch["opponent_knobs"])
        for batch in state.get("batches") or () if batch.get("opponent_knobs") is not None
    }
    active = [left for left, _ in pending]
    sites = observed | opponents | set(active)
    selections = state.get("selections")
    if selections is None:
        selections = sum(observation_counts.values())
    probability = exploration_probability(
        selections, args.explore_start, args.explore_floor, args.explore_half_life)
    new_axes = set(state.get("new_axes", ()))
    fresh = set()
    for name in new_axes:
        axis = space.names.index(name)
        seen = {candidate[axis] for candidate in sites}
        options = [candidate for candidate in candidates
                   if candidate[axis] != space.default[axis]]
        if options:
            distance = min(sum(a != b for a, b in zip(candidate, space.default))
                           for candidate in options)
            fresh.update(candidate for candidate in options
                         if candidate[axis] not in seen
                         and sum(a != b for a, b in zip(candidate, space.default)) == distance)
    full_axes = (((set(space.full_axis_design()) | set(space.required))
                  & set(candidates)) - sites - set(forbidden)
                 if getattr(args, "full_axis_design", False) else set())
    fresh_design = bool(fresh or full_axes)
    if fresh_design:
        mode = "design"
        acquisition = variance.copy()
        targets = fresh or full_axes
        for index, candidate in enumerate(candidates):
            if candidate not in targets:
                acquisition[index] = -np.inf
    elif len(sites) < min(args.initial_design, len(candidates)):
        mode = "design"
        acquisition = design_variance(list(sites), candidates, space)
        if getattr(args, "axis_design", False):
            local = set(space.axis_design()) - sites
            if local & set(candidates):
                acquisition = np.where(
                    [candidate in local for candidate in candidates], acquisition, -np.inf)
        if not sites and space.default in candidates:
            acquisition[candidates.index(space.default)] = np.inf
    else:
        credit = state.get("exploration_credit", 0.0) + probability
        if credit >= 1:
            mode = "explore"
            acquisition = variance.copy()
            credit -= 1
        else:
            mode = "ucb"
            values = mean, variance
            if getattr(args, "acquisition", "ucb") != "ucb":
                values = improvement_statistics(candidates)
            acquisition = exploitation(*values, args)
        state["exploration_credit"] = credit
    exploration_pool = []
    if mode == "explore":
        exploration_pool = sorted(
            (set(candidates) | set(validated)) - set(forbidden))
        if not exploration_pool:
            mode = "ucb"
            state["exploration_credit"] += 1
    stratum = None if fresh_design or mode == "ucb" else exploration_stratum(state, mode, space)
    state["selections"] = selections + 1
    state.setdefault("allocations", {}).setdefault(mode, 0)
    state["allocations"][mode] += 1
    if mode != "design" and isinstance(space, logistic_gp.MixedSpace):
        def exploration_variance(points, variance):
            trials = 2 * args.pair_weight * args.pairs
            variance = fantasy_variance(
                model, space, pending, points, np.asarray(variance, dtype=float), trials)
            if model is not None and hasattr(model, "residual_variance"):
                residual = np.minimum(model.residual_variance(points), variance)
                counts = np.asarray([observation_counts[point] for point in points])
                # Logistic Fisher information is at most trials / 4.
                learned = 4 * residual / (trials * counts * residual + 4)
                variance += learned - residual
            return variance

        def score(points):
            point_mean, point_variance = statistics(points)
            if mode == "explore":
                point_variance = exploration_variance(points, point_variance)
                return (point_mean + args.explore_optimism * np.sqrt(point_variance)
                        if args.explore_optimism else point_variance)
            if getattr(args, "acquisition", "ucb") != "ucb":
                return exploitation(*improvement_statistics(points), args)
            point_variance = fantasy_variance(
                model, space, pending, points, point_variance,
                2 * args.pair_weight * args.pairs)
            return exploitation(point_mean, point_variance, args)

        # Pending and completed games both reduce useful exploration uncertainty.
        if mode == "explore":
            pool = exploration_pool
            pool_mean, pool_variance = statistics(pool)
            pool_variance = exploration_variance(pool, pool_variance)
            confidence = getattr(args, "explore_confidence", 1.96)
            supported = max(0, max(pool_mean - confidence * np.sqrt(pool_variance)))
            plausible = {
                point for point, mean, variance in zip(pool, pool_mean, pool_variance)
                if mean + confidence * math.sqrt(variance) >= supported
            }
            if not plausible:
                plausible.add(pool[int(np.argmax(
                    pool_mean + confidence * np.sqrt(pool_variance)))])
            design = [point for point in candidates
                if point not in sites and point not in forbidden and (
                stratum is None or space.is_structural(point) == stratum)]
            matching = [point for point in pool if point in plausible and (
                stratum is None or space.is_structural(point) == stratum)]
            pool = design or matching or [point for point in pool if point in plausible]
            values = score(pool)
            vector = min(zip(values, pool), key=lambda item: (-item[0], item[1]))[1]
        elif args.gate_all or validated_only:
            pool = candidates if args.gate_all else sorted(set(candidates) | set(validated))
            pool = [point for point in pool if point not in forbidden and (
                stratum is None or space.is_structural(point) == stratum)]
            if not pool:
                raise RuntimeError("no available acquisition point")
            values = score(pool)
            vector = min(zip(values, pool), key=lambda item: (-item[0], item[1]))[1]
        else:
            gated = getattr(args, "gate_design", False)
            local = getattr(args, "local_acquisition", False)
            supported = [
                point for point in observed
                if observation_counts[point] >= getattr(args, "local_support", 1)
            ]
            seeds = sorted(set([space.default, *supported])) if local else [
                *candidates, *(validated if gated else observed)]
            optimizer_score = score
            if local and active:
                active_set = set(active)

                def optimizer_score(points):
                    return np.where(
                        [point in active_set for point in points], -np.inf, score(points))
            vector = coordinate_maximum(
                space, seeds, optimizer_score, set(forbidden), stratum,
                args.acquisition_restarts, 1 if gated or local else None, local=local)
    else:
        for index, candidate in enumerate(candidates):
            if candidate in forbidden or candidate in active or (
                    stratum is not None and space.is_structural(candidate) != stratum):
                acquisition[index] = -np.inf
        if not np.isfinite(acquisition).any():
            raise RuntimeError("no available acquisition point")
        vector = candidates[int(np.argmax(acquisition))]
    selected_mean, selected_variance = statistics([vector])
    exact_batches = observation_counts[vector]
    diagnostics = {
        "mode": mode,
        "explore_probability": probability,
        "mean": selected_mean[0],
        "sd": math.sqrt(selected_variance[0]),
        "exact_batches": exact_batches,
        "coverage": len(observed & set(candidates)),
        "unique": len(observed),
        "stratum": "structural" if stratum else "ordinary" if stratum is not None else "free",
    }
    return vector, diagnostics


def choose_opponent(state, mean_function, challenger, args, space, model=None,
                    anchored=None, prediction_cache=None):
    """Choose an anchored, informative rival for a parameter duel."""
    if anchored is None:
        anchored = {
            space.canonical(batch["knobs"])
            for batch in state["batches"]
            if (batch.get("opponent_knobs") is None
                or space.canonical(batch["opponent_knobs"]) == space.default)
        }
    anchored = {point for point in anchored if space.contains(point)}
    anchored.discard(challenger)
    if not anchored:
        return None
    credit = state.get("duel_credit", 0.0) + args.duel_fraction
    if credit < 1 - 1e-12:
        state["duel_credit"] = credit
        return None
    state["duel_credit"] = max(0, credit - 1)
    if model is None:
        model = posterior(
            state, mean_function, args.pair_weight, space, getattr(args, "inducing", 0))
    rivals = sorted(anchored)
    points = [challenger, *rivals]
    mean, variance = predict(model, points, prediction_cache)
    cross = model.predict_cross_covariance([challenger], rivals)[0]
    difference = mean[0] - mean[1:]
    difference_variance = variance[0] + variance[1:] - 2 * cross
    probability = 1 / (1 + np.exp(-difference))
    information = probability * (1 - probability) * np.maximum(difference_variance, 0)
    return rivals[int(np.argmax(information))]

def engine_config(command, name, arguments, options):
    path = pathlib.Path(command)
    command = str(path.resolve()) if path.exists() or "/" in command else command
    result = ["-engine", f"cmd={command}", f"name={name}"]
    if arguments:
        result.append(f"args={arguments}")
    def render(value):
        return str(value).lower() if isinstance(value, bool) else str(value)

    result += [f"option.{key}={render(value)}" for key, value in sorted(options.items())]
    return result


def validate_opening_budget(path, start, batches, pairs):
    openings = sum(1 for line in pathlib.Path(path).read_text().splitlines() if line.strip())
    required = start + batches * pairs - 1
    if required > openings:
        raise ValueError(f"study needs opening {required}, but {path} has {openings}")


def recover_pair(path, identity):
    """Reuse a complete validated pair whose scheduler journal was interrupted."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    output = path.read_text(errors="replace")
    if PAIR_HEADER.findall(output) != [identity]:
        return None
    pentanomial.reject_failures(output)
    if markers := PAIR_RESULT.findall(output):
        marker, *counts = markers[-1]
        if marker != identity:
            return None
        result = tuple(map(int, counts))
        if sum(result) != 2:
            raise ValueError(f"invalid saved pair result in {path}")
        return result
    try:
        results, (wins, losses, draws) = pentanomial.game_results(
            output, subject="candidate")
    except (IndexError, ValueError):
        return None
    return (wins, draws, losses) if len(results) == 2 else None


async def run_pair(args, slot, experiment, pair, identity, vector, opponent,
                   opening, sequence, space):
    if opponent is None:
        member = opponent_panel.select(
            args.baseline_panel, sequence, args.panel_seed) if args.baseline_panel else {
            "name": "baseline", "engine": args.baseline_engine,
            "args": args.baseline_args, "options": args.baseline_options,
        }
        rival = engine_config(member["engine"], "baseline", member["args"], member["options"])
    else:
        member = None
        rival = engine_config(args.engine, "opponent", args.engine_args, space.knobs(opponent))
    command = [
        args.fastchess,
        *engine_config(args.engine, "candidate", args.engine_args, space.knobs(vector)),
        *rival,
        "-each", "proto=uci", f"tc={args.tc}",
        "-openings", f"file={pathlib.Path(args.openings).resolve()}", "format=epd",
        "order=sequential", f"start={opening}",
        "-rounds", "1", "-games", "2", "-repeat", "-concurrency", "1", "-recover",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=4", "score=500",
        "-output", "format=cutechess", "-scoreinterval", "1", "-ratinginterval", "0",
    ]
    name = (f"experiment{experiment:04d}-pair{pair:03d}-opening{opening:06d}-"
            f"{identity}.log")
    path = pathlib.Path(args.logs, name)
    if result := recover_pair(path, identity):
        print(f"[slot {slot}] recovered experiment {experiment} pair {pair}", flush=True)
        return experiment, pair, *result, member["name"] if member else None
    with path.open("wb") as log:
        log.write(f"adaptive-gp-identity {identity}\n".encode())
        log.flush()
        os.fsync(log.fileno())
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    last_score = None
    failure = None
    try:
        with path.open("ab") as log:
            async for line in process.stdout:
                log.write(line)
                text = line.decode(errors="replace").rstrip()
                pentanomial.reject_failures(text)
                failure = failure or opponent_panel.failure(text)
                match = SCORE.search(text)
                if match:
                    last_score = tuple(map(int, match.groups()))
                    print(f"[slot {slot}] {text}", flush=True)
        status = await process.wait()
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
    if status or last_score is None or failure:
        raise RuntimeError(f"slot {slot} failed with status {status}: {failure or 'no score'}")
    wins, losses, draws = last_score
    if wins + draws + losses != 2:
        raise RuntimeError(f"slot {slot} completed {wins + draws + losses} games")
    with path.open("a") as log:
        log.write(f"adaptive-gp-result {identity} {wins} {draws} {losses}\n")
        log.flush()
        os.fsync(log.fileno())
    return experiment, pair, wins, draws, losses, member["name"] if member else None


async def optimize(args):
    with locking.exclusive(args.state):
        await optimize_locked(args)


async def optimize_locked(args):
    pathlib.Path(args.logs).mkdir(parents=True, exist_ok=True)
    state = load_state(args.state, args.start)
    state.setdefault("state_id", secrets.token_hex(16))
    seed = (load_state(args.seed_state, args.start)
            if args.seed_state and not state["batches"] and not state.get("pending") else None)
    openings = OpeningSchedule(
        args.openings, args.opening_seed, args.cycle_openings, args.opening_order)
    space = logistic_gp.MixedSpace.load(args.space) if args.space else logistic_gp.LegacySpace()
    if args.axis_design:
        if not isinstance(space, logistic_gp.MixedSpace):
            raise ValueError("--axis-design requires a JSON parameter space")
        space.candidates = sorted(set(space.candidates + space.axis_design()))
    if args.full_axis_design:
        if not isinstance(space, logistic_gp.MixedSpace):
            raise ValueError("--full-axis-design requires a JSON parameter space")
        space.candidates = sorted(set(space.candidates + space.full_axis_design()))
    if args.baseline_options == "default":
        args.baseline_options = space.knobs(space.default)
    for member in args.baseline_panel:
        if member["options"] == "default":
            member["options"] = space.knobs(space.default)
    bind_study(state, study_identity(args))
    if seed is not None:
        old_baseline = seed.get("study", {}).get("baseline", {}).get("options", {})
        old_names = set(old_baseline)
        for batch in seed["batches"]:
            old_names.update(batch["knobs"])
            old_names.update(batch.get("opponent_knobs") or ())
        imported = import_seed_batches(seed, space)
        state["batches"] = imported
        print(f"[seed] imported {len(imported)}/{len(seed['batches'])} compatible batches",
              flush=True)
        state["next_experiment"] = len(state["batches"])
        state["selections"] = (seed.get("selections", len(seed["batches"]))
            if args.seed_selections is None else args.seed_selections)
        state["allocations"] = {}
        state["new_axes"] = [name for name in space.names if name not in old_names]
        compatible = ("candidate", "gate", "gate_timeout", "space")
        if all(seed["study"].get(name) == state["study"].get(name) for name in compatible):
            state["gates"] = seed.get("gates", {})
    experiments, queue = restore_pending(state, space)
    if args.total_batches is not None:
        if len(state["batches"]) > args.total_batches:
            raise ValueError("state exceeds --total-batches")
        batches = (state["tranche"]["batches"] if "tranche" in state
                   else args.total_batches - len(state["batches"]))
    else:
        batches = args.batches
    tranche, target_batches = resume_tranche(state, batches, len(experiments))
    if not args.cycle_openings:
        future = target_batches - len(experiments)
        validate_opening_budget(
            args.openings, state["next_opening"], future, args.pairs)
    save_state(args.state, state)
    if target_batches == 0:
        state.pop("tranche")
        checkpoint_state(args.state, state)
        return
    options = set().union(*(space.knobs(candidate) for candidate in space.candidates))
    validate_options(args.engine, args.engine_args, options)
    if args.baseline_panel:
        for member in args.baseline_panel:
            validate_options(member["engine"], member["args"], member["options"])
    else:
        validate_options(args.baseline_engine, args.baseline_args, args.baseline_options)
    fixed = fixed_baseline_point(args, space)
    if fixed is not None:
        space.condition(fixed)
    base_mean = source_prior(args.source_logs, args.battery, args.transfer, space)
    mean_function = empirical_mean(
        base_mean, state["batches"], args.initial_design, space) if args.learn_mean else base_mean
    mean_learned = mean_function is not base_mean
    candidates = [candidate for candidate in space.candidates if candidate != fixed]
    if not candidates:
        raise ValueError("parameter space contains no challenger to the fixed baseline")
    if args.safe_only:
        if not isinstance(space, logistic_gp.LegacySpace):
            raise ValueError("--safe-only applies only to the built-in Sunfish LMR space")
        candidates = [x for x in candidates if x[2] and not any(x[len(logistic_gp.NUMERIC):])]
    if args.gate_all or args.gate_design:
        feasible = []
        for offset in range(0, len(candidates), args.gate_workers):
            group = candidates[offset:offset + args.gate_workers]
            accepted = await asyncio.gather(*(
                asyncio.to_thread(gate_policy, args, state, space, candidate)
                for candidate in group
            ))
            feasible.extend(candidate for candidate, passed in zip(group, accepted) if passed)
            save_state(args.state, state)
        print(f"[gate] feasible candidate space: {len(feasible)}/{len(candidates)}", flush=True)
        candidates = feasible
        if args.gate_all:
            space.candidates = sorted(set(candidates + ([fixed] if fixed is not None else [])))
    if not candidates:
        raise ValueError("the policy gate rejected every challenger")
    deadline = None
    if args.wall_time:
        wall_deadline = state.setdefault("wall_deadline", time.time() + args.wall_time)
        deadline = time.monotonic() + max(0, wall_deadline - time.time())
        save_state(args.state, state)
    activity = asyncio.Event()
    running = {}
    completed = 0
    allocation_model = None
    prediction_cache = {}
    modeled_batches = 0
    observation_counts = Counter(
        space.canonical(batch["knobs"]) for batch in state["batches"])
    anchored = {
        space.canonical(batch["knobs"])
        for batch in state["batches"]
        if (batch.get("opponent_knobs") is None
            or space.canonical(batch["opponent_knobs"]) == space.default)
    }
    if experiments:
        finished = sum(result is not None
            for experiment in experiments.values() for result in experiment["record"]["results"])
        print(f"[resume] {len(experiments)} experiments; "
              f"{finished} pairs kept, {len(queue)} pairs remaining", flush=True)

    def refresh_model():
        nonlocal allocation_model, mean_function, mean_learned, modeled_batches
        if args.learn_mean and not mean_learned:
            learned = empirical_mean(
                base_mean, state["batches"], args.initial_design, space)
            if learned is not base_mean:
                mean_function, mean_learned = learned, True
                allocation_model = None
        if allocation_model is None:
            allocation_model = posterior(
                state, mean_function, args.pair_weight, space, args.inducing)
            modeled_batches = len(state["batches"])
            prediction_cache.clear()
        elif len(state["batches"]) - modeled_batches >= args.update_batches:
            allocation_model = (update_posterior(
                allocation_model, state["batches"][modeled_batches:],
                args.pair_weight, space) if args.inducing else posterior(
                state, mean_function, args.pair_weight, space))
            modeled_batches = len(state["batches"])
            prediction_cache.clear()

    def pending_comparisons():
        pending = [
            (experiment["vector"], experiment["opponent"])
            for experiment in experiments.values()
        ]
        pending += [
            (space.canonical(batch["knobs"]),
             space.canonical(batch["opponent_knobs"])
             if batch.get("opponent_knobs") is not None else None)
            for batch in state["batches"][modeled_batches:]
        ]
        return pending

    def gated_configurations(accepted):
        with STATE_LOCK:
            return {
                space.canonical(record["knobs"])
                for record in state.get("gates", {}).values()
                if record["accepted"] == accepted
            }

    def rejected_configurations():
        return gated_configurations(False)

    def validated_configurations():
        return gated_configurations(True)

    def schedule_experiment(vector, diagnostics):
        opponent = choose_opponent(
            state, mean_function, vector, args, space, allocation_model, anchored,
            prediction_cache)
        number = state.get("next_experiment", 0)
        state["next_experiment"] = number + 1
        sequence = state["next_opening"]
        state["next_opening"] += args.pairs
        scheduled = [openings.opening(sequence + offset) for offset in range(args.pairs)]
        record = {
            "number": number, "knobs": space.knobs(vector),
            "opponent_knobs": space.knobs(opponent) if opponent is not None else None,
            "opening_sequence": sequence, "openings": scheduled,
            "results": [None] * args.pairs, "allocation": diagnostics["mode"],
            "baseline_ids": [None] * args.pairs,
        }
        record["identities"] = [pair_identity(state, record, pair)
                                for pair in range(args.pairs)]
        state["pending"].append(record)
        experiments[number] = {"record": record, "vector": vector, "opponent": opponent}
        for pair in range(args.pairs):
            queue.append((number, pair))
        elo = diagnostics["mean"] * logistic_gp.ELO_PER_LOGIT
        error = 1.96 * diagnostics["sd"] * logistic_gp.ELO_PER_LOGIT
        print(
            f"[experiment {number}] {diagnostics['mode']} "
            f"stratum={diagnostics['stratum']} "
            f"opponent={'baseline' if opponent is None else json.dumps(space.knobs(opponent))} "
            f"p_explore={diagnostics['explore_probability']:.2f} "
            f"posterior={elo:+.1f} ± {error:.1f} Elo "
            f"exact_batches={diagnostics['exact_batches']} "
            f"coverage={diagnostics['coverage']}/{len(candidates)} "
            f"observed={diagnostics['unique']} "
            f"{json.dumps(space.knobs(vector), sort_keys=True)}",
            flush=True,
        )

    async def add_experiments(count):
        refresh_model()
        while count:
            size = min(count, args.gate_workers)
            count -= size
            pending = pending_comparisons()
            forbidden = rejected_configurations() | ({fixed} if fixed is not None else set())
            counts = observation_counts.copy()
            proposal_state = selection_state(state)
            proposals = []
            for _ in range(size):
                reservation = selection_state(proposal_state)
                trial = selection_state(reservation)
                reservation_pending = list(pending)
                vector, diagnostics = await asyncio.to_thread(
                    choose, trial, mean_function, candidates, pending, args, space,
                    allocation_model, forbidden | {item[0] for item in proposals},
                    validated_configurations(), counts, False, prediction_cache)
                commit_selection(proposal_state, trial)
                proposals.append([vector, diagnostics, reservation, 0, reservation_pending])
                pending.append((vector, None))

            tasks = {
                asyncio.create_task(asyncio.to_thread(
                    gate_policy, args, state, space, item[0])): item
                for item in proposals
            }
            while tasks:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    item = tasks.pop(task)
                    passed = task.result()
                    if passed:
                        item[3] = -1
                        continue
                    forbidden.add(item[0])
                    item[3] += 1
                    if item[3] >= args.gate_attempts:
                        raise RuntimeError(
                            f"policy gate rejected {args.gate_attempts} consecutive proposals")
                    others = {other[0] for other in proposals if other is not item}
                    trial = selection_state(item[2])
                    item[0], replacement = await asyncio.to_thread(
                        choose, trial, mean_function, candidates, item[4], args, space,
                        allocation_model, forbidden | others, validated_configurations(),
                        counts, args.gate_design, prediction_cache)
                    if replacement["mode"] != item[1]["mode"]:
                        raise AssertionError("gate replacement changed allocation mode")
                    item[1] = replacement
                    tasks[asyncio.create_task(asyncio.to_thread(
                        gate_policy, args, state, space, item[0]))] = item
            commit_selection(state, proposal_state)
            for vector, diagnostics, *_ in proposals:
                schedule_experiment(vector, diagnostics)
            save_state(args.state, state)
            start_queued()

    def start_queued():
        while queue and len(running) < args.slots:
            used = set(running.values())
            slot = next(index for index in range(args.slots) if index not in used)
            number, pair = queue.popleft()
            experiment = experiments[number]
            record = experiment["record"]
            opening = record["openings"][pair]
            task = asyncio.create_task(
                run_pair(args, slot, number, pair, record["identities"][pair],
                         experiment["vector"],
                         experiment["opponent"], opening,
                         record["opening_sequence"] + pair, space))
            running[task] = slot
            activity.set()

    refill = None
    while completed < target_batches:
        expired = deadline is not None and time.monotonic() >= deadline
        if (not expired and refill is None
                and completed + len(experiments) < target_batches
                and len(experiments) <= args.queue_batches - args.refill_batches):
            count = min(
                args.queue_batches - len(experiments),
                target_batches - completed - len(experiments))
            refill = asyncio.create_task(add_experiments(count))
        start_queued()
        activity.clear()
        waiting = set(running)
        wake = None
        if refill is not None:
            waiting.add(refill)
            wake = asyncio.create_task(activity.wait())
            waiting.add(wake)
        if not waiting:
            break
        done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        if wake is not None:
            if wake in done:
                done.remove(wake)
            else:
                wake.cancel()
        if refill in done:
            refill.result()
            done.remove(refill)
            refill = None
        for task in done:
            running.pop(task)
            number, pair, wins, draws, losses, baseline_id = task.result()
            experiment = experiments[number]
            record = experiment["record"]
            if record["results"][pair] is not None:
                raise RuntimeError(f"experiment {number} pair {pair} completed twice")
            record["results"][pair] = [wins, draws, losses]
            record["baseline_ids"][pair] = baseline_id
            if any(result is None for result in record["results"]):
                save_state(args.state, state)
                continue
            vector = experiment["vector"]
            batch = pending_batch(record)
            state["batches"].append(batch)
            observation_counts[vector] += 1
            if experiment["opponent"] is None or experiment["opponent"] == space.default:
                anchored.add(vector)
            # One journal event atomically replaces the reservation with its observation.
            state["pending"].remove(record)
            del experiments[number]
            completed += 1
            tranche["completed"] += 1
            save_state(args.state, state)
            if completed % args.checkpoint_batches == 0:
                checkpoint_state(args.state, state)
            print(f"[experiment {number}] result "
                  f"{batch['wins']}-{batch['losses']}-{batch['draws']}",
                  flush=True)
    if state["pending"]:
        raise RuntimeError("scheduler stopped with pending experiments")
    unfinished = tranche["completed"] < tranche["batches"]
    if unfinished and (deadline is None or time.monotonic() < deadline):
        raise RuntimeError("scheduler stopped before completing its batch tranche")
    state.pop("tranche")
    state.pop("wall_deadline", None)
    checkpoint_state(args.state, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastchess", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--baseline-engine")
    parser.add_argument("--engine-args", default="")
    parser.add_argument("--baseline-args")
    parser.add_argument("--baseline-options", type=lambda value: (
        value if value == "default" else json.loads(value)), default={})
    parser.add_argument("--baseline-panel",
        help="JSON list of deterministic integer-weighted fixed opponents")
    parser.add_argument("--panel-seed", type=int, default=2026,
        help="seed for shuffled weighted opponent blocks")
    parser.add_argument("--space", help="JSON numeric/categorical UCI option space")
    parser.add_argument("--seed-state", help="import completed batches into a new study")
    parser.add_argument("--openings", required=True)
    parser.add_argument("--cycle-openings", action="store_true",
        help="reuse the book in independently shuffled epochs")
    parser.add_argument("--opening-seed", type=int, default=2026)
    parser.add_argument("--opening-order", choices=("random", "sequential"), default="random")
    parser.add_argument("--state", default="adaptive-gp.json")
    parser.add_argument("--logs", default="adaptive-gp-logs")
    parser.add_argument("--battery")
    parser.add_argument("--source-logs")
    parser.add_argument("--transfer", type=float, default=0.5)
    parser.add_argument("--tc", default="3+0.1")
    parser.add_argument("--pairs", type=int, default=1,
        help="paired openings per posterior update (default: 1)")
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--queue-batches", type=int,
        help="pending configurations (default: enough to fill all slots)")
    parser.add_argument("--refill-batches", type=int, default=1,
        help="refill the pending queue after this many completions")
    parser.add_argument("--batches", type=int, default=100,
        help="posterior updates in this crash-resumable invocation")
    parser.add_argument("--total-batches", type=int,
        help="absolute completed-update target; a clean replay performs no more games")
    parser.add_argument("--wall-time", type=duration, default=0,
        help="stop allocating after this duration, e.g. 12h or 3d")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--exploration", type=float, default=1.0)
    parser.add_argument("--acquisition", choices=("ucb", "ei", "pi"), default="ucb")
    parser.add_argument("--initial-design", type=int, default=12)
    parser.add_argument("--axis-design", action="store_true",
        help="start with one-kernel-length probes along each parameter axis")
    parser.add_argument("--full-axis-design", action="store_true",
        help="probe every value of every parameter before model allocation")
    parser.add_argument("--learn-mean", action="store_true",
        help="learn the fixed-opponent GP intercept after the initial design")
    parser.add_argument("--local-acquisition", action="store_true",
        help="validate one-coordinate steps from observed configurations")
    parser.add_argument("--local-support", type=int, default=1,
        help="pairs required before a played point can seed another local step")
    parser.add_argument("--explore-start", type=float, default=0.50)
    parser.add_argument("--explore-floor", type=float, default=0.20)
    parser.add_argument("--explore-half-life", type=float, default=40)
    parser.add_argument("--explore-optimism", type=float, default=0,
        help="explore with mean + K*sd instead of pure variance")
    parser.add_argument("--explore-confidence", type=float, default=1.96,
        help="discard exploration points whose upper bound is supportedly dominated")
    parser.add_argument("--duel-fraction", type=float, default=0.30)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--inducing", type=int, default=0,
        help="sparse GP size; exact inference is the default")
    parser.add_argument("--acquisition-restarts", type=int, default=8)
    parser.add_argument("--update-batches", type=int, default=1,
        help="completed pairs per online posterior update")
    parser.add_argument("--checkpoint-batches", type=int, default=1000,
        help="pairs per compact JSON checkpoint; intervening results use a journal")
    parser.add_argument("--gate", help="command reading a policy JSON object on stdin")
    parser.add_argument("--gate-timeout", type=float, default=60)
    parser.add_argument("--gate-attempts", type=int, default=1000)
    parser.add_argument("--gate-workers", type=int, default=4)
    parser.add_argument("--gate-all", action="store_true",
        help="validate the finite candidate space before allocating games")
    parser.add_argument("--gate-design", action="store_true",
        help="prevalidate the finite design but allow coordinate refinements")
    parser.add_argument("--seed-selections", type=int,
        help="override the imported allocation clock (use 0 to restart it)")
    parser.add_argument("--safe-only", action="store_true")
    args = parser.parse_args()
    args.baseline_panel = opponent_panel.load(args.baseline_panel) if args.baseline_panel else []
    if args.baseline_panel and (args.baseline_engine or args.baseline_args is not None
            or args.baseline_options):
        parser.error("--baseline-panel cannot be combined with single-baseline options")
    args.baseline_engine = args.baseline_engine or args.engine
    args.baseline_args = args.engine_args if args.baseline_args is None else args.baseline_args
    if args.source_logs and not args.battery:
        parser.error("--source-logs requires --battery")
    if (args.gate_all or args.gate_design) and not args.gate:
        parser.error("--gate-all and --gate-design require --gate")
    if not 0 <= args.explore_floor <= args.explore_start <= 1:
        parser.error("require 0 <= --explore-floor <= --explore-start <= 1")
    if not 0 <= args.duel_fraction <= 0.40:
        parser.error("require 0 <= --duel-fraction <= 0.40 to preserve the baseline anchor")
    if args.pair_weight <= 0:
        parser.error("--pair-weight must be positive")
    if min(args.pairs, args.slots, args.batches, args.gate_timeout,
           args.gate_attempts, args.gate_workers) <= 0:
        parser.error("pair, slot, batch, and gate limits must be positive")
    if args.total_batches is not None and args.total_batches < 1:
        parser.error("--total-batches must be positive")
    if args.queue_batches is None:
        args.queue_batches = pending_configurations(args.slots, args.pairs)
    elif args.queue_batches < 1:
        parser.error("--queue-batches must be positive")
    if not 1 <= args.refill_batches <= args.queue_batches:
        parser.error("require 1 <= --refill-batches <= --queue-batches")
    if args.initial_design < 1 or args.explore_half_life <= 0:
        parser.error("--initial-design and --explore-half-life must be positive")
    if args.inducing < 0 or min(
            args.acquisition_restarts, args.update_batches, args.checkpoint_batches,
            args.local_support) < 1:
        parser.error("inducing must be nonnegative; acquisition and update counts must be positive")
    if args.explore_confidence <= 0:
        parser.error("--explore-confidence must be positive")
    if args.explore_optimism < 0 or (
            args.seed_selections is not None and args.seed_selections < 0):
        parser.error("--explore-optimism and --seed-selections cannot be negative")
    asyncio.run(optimize(args))


if __name__ == "__main__":
    main()
