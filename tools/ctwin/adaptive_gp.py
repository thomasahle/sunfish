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
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections import Counter, deque

import numpy as np

import logistic_gp


SCORE = re.compile(
    r"Score of candidate vs (?:baseline|opponent):\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)")
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
    return {
        "version": 3,
        "scheduler": file_identity(__file__),
        "model": file_identity(pathlib.Path(__file__).with_name("logistic_gp.py")),
        "fastchess": file_identity(shutil.which(args.fastchess) or args.fastchess),
        "candidate": engine_identity(args.engine, args.engine_args, {}),
        "baseline": engine_identity(
            args.baseline_engine, args.baseline_args, args.baseline_options),
        "openings": file_identity(args.openings),
        "opening_schedule": {
            "cycle": args.cycle_openings,
            "seed": args.opening_seed,
        },
        "gate": command_identity(args.gate),
        "gate_timeout": args.gate_timeout,
        "space": file_identity(args.space) if args.space else "legacy",
        "seed_state": state_file_identity(args.seed_state) if args.seed_state else None,
        "tc": args.tc,
        "allocation": {
            name: getattr(args, name)
            for name in ("pair_weight", "exploration", "initial_design", "explore_start",
                         "explore_floor", "explore_half_life", "explore_optimism",
                         "explore_confidence",
                         "duel_fraction", "inducing", "seed_selections",
                         "acquisition_restarts", "update_batches", "gate_workers",
                         "gate_all")
        },
    }


def bind_study(state, identity):
    previous = state.get("study")
    if previous is None:
        if state["batches"]:
            raise RuntimeError("state has observations but no study identity; start a new state")
        state["study"] = identity
    elif previous != identity:
        changed = sorted(key for key in identity if previous.get(key) != identity[key])
        raise RuntimeError(f"state belongs to a different study: {', '.join(changed)} changed")


def fixed_baseline_point(args, space):
    """Return the parameter point already represented exactly by the baseline."""
    if not isinstance(space, logistic_gp.MixedSpace):
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
    baseline = seed["study"]["baseline"]["options"]
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
    baseline = seed["study"]["baseline"]["options"]

    def migrate(knobs):
        knobs = dict(knobs)
        for target, source in space.seed_copies.items():
            knobs.setdefault(target, knobs.get(source, baseline[source]))
        return knobs

    return [
        batch | {
            "knobs": migrate(batch["knobs"]),
            "opponent_knobs": migrate(batch.get("opponent_knobs") or baseline),
        }
        for batch in compatible_seed_batches(seed, space)
    ]


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


def pending_configurations(slots, pairs):
    """Smallest number of parameter choices that can keep every lane busy."""
    return math.ceil(slots / pairs)


class OpeningSchedule:
    """Map an unbounded sequence onto independently shuffled book epochs."""

    def __init__(self, path, seed=0, cycle=False):
        self.count = sum(bool(line.strip()) for line in pathlib.Path(path).read_text().splitlines())
        if not self.count:
            raise ValueError("opening book is empty")
        self.seed = seed
        self.cycle = cycle
        self.epochs = {}

    def opening(self, sequence):
        if sequence < 1 or not self.cycle and sequence > self.count:
            raise ValueError(f"opening sequence {sequence} exceeds the {self.count}-position book")
        epoch, offset = divmod(sequence - 1, self.count)
        if epoch not in self.epochs:
            order = list(range(1, self.count + 1))
            random.Random(self.seed + epoch).shuffle(order)
            self.epochs[epoch] = order
        return self.epochs[epoch][offset]


def gate_policy(args, state, space, vector):
    """Run and cache a deterministic policy gate; exit 1 means infeasible."""
    if not args.gate:
        return True
    knobs = space.knobs(vector)
    key = json.dumps(knobs, sort_keys=True, separators=(",", ":"))
    with STATE_LOCK:
        cache = state.setdefault("gates", {})
        if key in cache:
            return cache[key]["accepted"]
    payload = {
        "engine": args.engine,
        "engine_args": args.engine_args,
        "options": knobs,
    }
    started = time.perf_counter()
    process = subprocess.Popen(
        shlex.split(args.gate), text=True, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=os.name != "nt")
    try:
        output = process.communicate(json.dumps(payload), timeout=args.gate_timeout)[0]
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        output = process.communicate()[0] + f"\ntimeout after {args.gate_timeout:g}s"
        process.returncode = 1
    if process.returncode not in (0, 1):
        raise RuntimeError(
            f"policy gate failed with status {process.returncode}:\n{output}")
    accepted = process.returncode == 0
    with STATE_LOCK:
        cache[key] = {
            "accepted": accepted,
            "knobs": knobs,
            "output": output[-2000:],
            "seconds": time.perf_counter() - started,
        }
    print(f"[gate] {'accept' if accepted else 'reject'} "
          f"{cache[key]['seconds']:.2f}s {key}", flush=True)
    return accepted


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


def coordinate_maximum(space, seeds, score, active, structural, restarts=4):
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
        normalize(point) for point in seeds
        if space.contains(normalize(point)) and allowed(normalize(point))
    ))
    if not seeds:
        raise RuntimeError("no available acquisition point")
    seed_scores = evaluate(seeds)
    order = sorted(range(len(seeds)), key=lambda index: (-seed_scores[index], seeds[index]))
    optima = []
    for index in order[:restarts]:
        point = seeds[index]
        value = seed_scores[index]
        while True:
            neighbors = []
            for axis, choices in enumerate(space.coordinate_values):
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
        optima.append((value, point))
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


def choose(state, mean_function, candidates, pending, args, space, model=None,
           forbidden=(), validated=(), observation_counts=None):
    if model is None:
        model = posterior(
            state, mean_function, args.pair_weight, space, getattr(args, "inducing", 0))

    def statistics(points):
        if model:
            return model.predict(points)
        mean = mean_function(np.asarray(points))
        variance = np.diag(space.kernel(points, points))
        return mean, variance

    mean, variance = statistics(candidates)

    if observation_counts is None:
        observation_counts = Counter(
            space.canonical(batch["knobs"]) for batch in state["batches"])
    observed = {point for point in observation_counts if space.contains(point)}
    active = [left for left, _ in pending]
    sites = observed | set(active)
    selections = state.get("selections")
    if selections is None:
        selections = sum(observation_counts.values())
    probability = exploration_probability(
        selections, args.explore_start, args.explore_floor, args.explore_half_life)
    new_axes = set(state.get("new_axes", ()))
    fresh = set()
    if new_axes:
        for candidate in candidates:
            changed = [
                name for name, value, default in zip(space.names, candidate, space.default)
                if value != default
            ]
            if len(changed) == 1 and changed[0] in new_axes and candidate not in sites:
                fresh.add(candidate)
    fresh_design = bool(fresh)
    if fresh_design:
        mode = "design"
        acquisition = variance.copy()
        for index, candidate in enumerate(candidates):
            if candidate not in fresh:
                acquisition[index] = -np.inf
    elif len(sites) < min(args.initial_design, len(candidates)):
        mode = "design"
        acquisition = design_variance(list(sites), candidates, space)
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
            acquisition = mean + args.exploration * np.sqrt(variance)
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
        def score(points):
            point_mean, point_variance = statistics(points)
            point_variance = fantasy_variance(
                model, space, pending, points, point_variance,
                2 * args.pair_weight * args.pairs)
            if mode == "explore":
                return (point_mean + args.explore_optimism * np.sqrt(point_variance)
                        if args.explore_optimism else point_variance)
            return point_mean + args.exploration * np.sqrt(point_variance)

        # Fantasized variance decides whether another pending copy is useful;
        # do not impose a fixed one-copy-per-configuration rule on top of it.
        if mode == "explore":
            pool = exploration_pool
            pool_mean, pool_variance = statistics(pool)
            confidence = getattr(args, "explore_confidence", 1.96)
            supported = max(0, max(pool_mean - confidence * np.sqrt(pool_variance)))
            plausible = {
                point for point, mean, variance in zip(pool, pool_mean, pool_variance)
                if mean + confidence * math.sqrt(variance) >= supported
            }
            if not plausible:
                plausible.add(pool[int(np.argmax(
                    pool_mean + confidence * np.sqrt(pool_variance)))])
            matching = [point for point in pool if point in plausible and (
                stratum is None or space.is_structural(point) == stratum)]
            pool = matching or [point for point in pool if point in plausible]
            values = score(pool)
            vector = min(zip(values, pool), key=lambda item: (-item[0], item[1]))[1]
        elif args.gate_all:
            pool = [point for point in candidates if point not in forbidden and (
                stratum is None or space.is_structural(point) == stratum)]
            if not pool:
                raise RuntimeError("no available acquisition point")
            values = score(pool)
            vector = min(zip(values, pool), key=lambda item: (-item[0], item[1]))[1]
        else:
            vector = coordinate_maximum(
                space, [*candidates, *observed], score, set(forbidden), stratum,
                args.acquisition_restarts)
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
        "unique": len(observed),
        "stratum": "structural" if stratum else "ordinary" if stratum is not None else "free",
    }
    return vector, diagnostics


def choose_opponent(state, mean_function, challenger, args, space, model=None,
                    anchored=None):
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
    mean, covariance = model.predict_covariance(points)
    difference = mean[0] - mean[1:]
    variance = covariance[0, 0] + np.diag(covariance)[1:] - 2 * covariance[0, 1:]
    probability = 1 / (1 + np.exp(-difference))
    information = probability * (1 - probability) * np.maximum(variance, 0)
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


async def run_pair(args, slot, experiment, vector, opponent, opening, space):
    if opponent is None:
        rival = engine_config(
            args.baseline_engine, "baseline", args.baseline_args, args.baseline_options)
    else:
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
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    last_score = None
    name = f"experiment{experiment:04d}-{opening:06d}.log"
    try:
        with pathlib.Path(args.logs, name).open("wb") as log:
            async for line in process.stdout:
                log.write(line)
                text = line.decode(errors="replace").rstrip()
                match = SCORE.search(text)
                if match:
                    last_score = tuple(map(int, match.groups()))
                    print(f"[slot {slot}] {text}", flush=True)
        status = await process.wait()
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()
    if status or last_score is None:
        raise RuntimeError(f"slot {slot} failed with status {status}")
    wins, losses, draws = last_score
    if wins + draws + losses != 2:
        raise RuntimeError(f"slot {slot} completed {wins + draws + losses} games")
    return experiment, wins, draws, losses


async def optimize(args):
    pathlib.Path(args.logs).mkdir(parents=True, exist_ok=True)
    state = load_state(args.state, args.start)
    openings = OpeningSchedule(args.openings, args.opening_seed, args.cycle_openings)
    if not args.cycle_openings:
        validate_opening_budget(args.openings, state["next_opening"], args.batches, args.pairs)
    space = logistic_gp.MixedSpace.load(args.space) if args.space else logistic_gp.LegacySpace()
    if args.baseline_options == "default":
        args.baseline_options = space.knobs(space.default)
    bind_study(state, study_identity(args))
    if args.seed_state and not state["batches"]:
        seed = load_state(args.seed_state, args.start)
        old_baseline = seed["study"]["baseline"]["options"]
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
    save_state(args.state, state)
    options = set().union(*(space.knobs(candidate) for candidate in space.candidates))
    validate_options(args.engine, args.engine_args, options)
    validate_options(args.baseline_engine, args.baseline_args, args.baseline_options)
    fixed = fixed_baseline_point(args, space)
    if fixed is not None:
        if not gate_policy(args, state, space, fixed):
            raise RuntimeError("the fixed baseline fails the policy gate")
        space.condition(fixed)
    mean_function = source_prior(args.source_logs, args.battery, args.transfer, space)
    candidates = [candidate for candidate in space.candidates if candidate != fixed]
    if not candidates:
        raise ValueError("parameter space contains no challenger to the fixed baseline")
    if args.safe_only:
        if not isinstance(space, logistic_gp.LegacySpace):
            raise ValueError("--safe-only applies only to the built-in Sunfish LMR space")
        candidates = [x for x in candidates if x[2] and not any(x[len(logistic_gp.NUMERIC):])]
    if args.gate_all:
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
        space.candidates = sorted(set(candidates + ([fixed] if fixed is not None else [])))
    if not candidates:
        raise ValueError("the policy gate rejected every challenger")
    deadline = None
    if args.wall_time:
        wall_deadline = state.setdefault("wall_deadline", time.time() + args.wall_time)
        deadline = time.monotonic() + max(0, wall_deadline - time.time())
        save_state(args.state, state)
    queue = deque()
    activity = asyncio.Event()
    experiments = {}
    running = {}
    completed = 0
    allocation_model = None
    modeled_batches = 0
    observation_counts = Counter(
        space.canonical(batch["knobs"]) for batch in state["batches"])
    anchored = {
        space.canonical(batch["knobs"])
        for batch in state["batches"]
        if (batch.get("opponent_knobs") is None
            or space.canonical(batch["opponent_knobs"]) == space.default)
    }

    def refresh_model():
        nonlocal allocation_model, modeled_batches
        if allocation_model is None:
            allocation_model = posterior(
                state, mean_function, args.pair_weight, space, args.inducing)
            modeled_batches = len(state["batches"])
        elif len(state["batches"]) - modeled_batches >= args.update_batches:
            allocation_model = (update_posterior(
                allocation_model, state["batches"][modeled_batches:],
                args.pair_weight, space) if args.inducing else posterior(
                    state, mean_function, args.pair_weight, space))
            modeled_batches = len(state["batches"])

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

    def rejected_configurations():
        return {
            space.canonical(record["knobs"])
            for record in state.get("gates", {}).values()
            if not record["accepted"]
        }

    def validated_configurations():
        return {
            space.canonical(record["knobs"])
            for record in state.get("gates", {}).values()
            if record["accepted"]
        }

    def schedule_experiment(vector, diagnostics):
        opponent = choose_opponent(
            state, mean_function, vector, args, space, allocation_model, anchored)
        number = state.get("next_experiment", 0)
        state["next_experiment"] = number + 1
        sequence = state["next_opening"]
        state["next_opening"] += args.pairs
        scheduled = [openings.opening(sequence + offset) for offset in range(args.pairs)]
        experiments[number] = {
            "vector": vector, "opening": scheduled[0], "opening_sequence": sequence,
            "openings": scheduled, "wins": 0, "draws": 0, "losses": 0,
            "allocation": diagnostics["mode"], "opponent": opponent,
        }
        for opening in scheduled:
            queue.append((number, vector, opponent, opening))
        save_state(args.state, state)
        elo = diagnostics["mean"] * logistic_gp.ELO_PER_LOGIT
        error = 1.96 * diagnostics["sd"] * logistic_gp.ELO_PER_LOGIT
        print(
            f"[experiment {number}] {diagnostics['mode']} "
            f"stratum={diagnostics['stratum']} "
            f"opponent={'baseline' if opponent is None else json.dumps(space.knobs(opponent))} "
            f"p_explore={diagnostics['explore_probability']:.2f} "
            f"posterior={elo:+.1f} ± {error:.1f} Elo "
            f"exact_batches={diagnostics['exact_batches']} "
            f"coverage={diagnostics['unique']}/{len(candidates)} "
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
                    validated_configurations(), counts)
                commit_selection(proposal_state, trial)
                proposals.append([vector, diagnostics, reservation, 0, reservation_pending])
                pending.append((vector, None))

            commit_selection(state, proposal_state)
            tasks = {
                asyncio.create_task(asyncio.to_thread(
                    gate_policy, args, state, space, item[0])): item
                for item in proposals
            }
            while tasks:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                released = False
                for task in done:
                    item = tasks.pop(task)
                    passed = task.result()
                    if passed:
                        item[3] = -1
                        schedule_experiment(item[0], item[1])
                        released = True
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
                        counts)
                    if replacement["mode"] != item[1]["mode"]:
                        raise AssertionError("gate replacement changed allocation mode")
                    item[1] = replacement
                    tasks[asyncio.create_task(asyncio.to_thread(
                        gate_policy, args, state, space, item[0]))] = item
                if released:
                    save_state(args.state, state)
                    start_queued()

    def start_queued():
        while queue and len(running) < args.slots:
            used = set(running.values())
            slot = next(index for index in range(args.slots) if index not in used)
            experiment, vector, opponent, opening = queue.popleft()
            task = asyncio.create_task(
                run_pair(args, slot, experiment, vector, opponent, opening, space))
            running[task] = slot
            activity.set()

    refill = None
    while completed < args.batches:
        expired = deadline is not None and time.monotonic() >= deadline
        if (not expired and refill is None
                and len(experiments) <= args.queue_batches - args.refill_batches):
            count = min(
                args.queue_batches - len(experiments),
                args.batches - completed - len(experiments))
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
            number, wins, draws, losses = task.result()
            experiment = experiments[number]
            experiment["wins"] += wins
            experiment["draws"] += draws
            experiment["losses"] += losses
            games = experiment["wins"] + experiment["draws"] + experiment["losses"]
            if games != 2 * args.pairs:
                continue
            vector = experiment["vector"]
            state["batches"].append({
                "knobs": space.knobs(vector),
                "wins": experiment["wins"],
                "draws": experiment["draws"],
                "losses": experiment["losses"],
                "opening": experiment["opening"],
                "opening_sequence": experiment["opening_sequence"],
                "openings": experiment["openings"],
                "allocation": experiment["allocation"],
                "opponent_knobs": (
                    space.knobs(experiment["opponent"])
                    if experiment["opponent"] is not None else None),
            })
            observation_counts[vector] += 1
            if experiment["opponent"] is None or experiment["opponent"] == space.default:
                anchored.add(vector)
            del experiments[number]
            completed += 1
            save_state(args.state, state)
            if completed % args.checkpoint_batches == 0:
                checkpoint_state(args.state, state)
            print(f"[experiment {number}] result "
                  f"{experiment['wins']}-{experiment['losses']}-{experiment['draws']}",
                  flush=True)
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
    parser.add_argument("--space", help="JSON numeric/categorical UCI option space")
    parser.add_argument("--seed-state", help="import completed batches into a new study")
    parser.add_argument("--openings", required=True)
    parser.add_argument("--cycle-openings", action="store_true",
        help="reuse the book in independently shuffled epochs")
    parser.add_argument("--opening-seed", type=int, default=2026)
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
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--wall-time", type=duration, default=0,
        help="stop allocating after this duration, e.g. 12h or 3d")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--exploration", type=float, default=1.0)
    parser.add_argument("--initial-design", type=int, default=12)
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
    parser.add_argument("--seed-selections", type=int,
        help="override the imported allocation clock (use 0 to restart it)")
    parser.add_argument("--safe-only", action="store_true")
    args = parser.parse_args()
    args.baseline_engine = args.baseline_engine or args.engine
    args.baseline_args = args.engine_args if args.baseline_args is None else args.baseline_args
    if args.source_logs and not args.battery:
        parser.error("--source-logs requires --battery")
    if args.gate_all and not args.gate:
        parser.error("--gate-all requires --gate")
    if not 0 <= args.explore_floor <= args.explore_start <= 1:
        parser.error("require 0 <= --explore-floor <= --explore-start <= 1")
    if not 0 <= args.duel_fraction <= 0.40:
        parser.error("require 0 <= --duel-fraction <= 0.40 to preserve the baseline anchor")
    if args.pair_weight <= 0:
        parser.error("--pair-weight must be positive")
    if min(args.pairs, args.slots, args.batches, args.gate_timeout,
           args.gate_attempts, args.gate_workers) <= 0:
        parser.error("pair, slot, batch, and gate limits must be positive")
    if args.queue_batches is None:
        args.queue_batches = pending_configurations(args.slots, args.pairs)
    elif args.queue_batches < 1:
        parser.error("--queue-batches must be positive")
    if not 1 <= args.refill_batches <= args.queue_batches:
        parser.error("require 1 <= --refill-batches <= --queue-batches")
    if args.initial_design < 1 or args.explore_half_life <= 0:
        parser.error("--initial-design and --explore-half-life must be positive")
    if args.inducing < 0 or min(
            args.acquisition_restarts, args.update_batches, args.checkpoint_batches) < 1:
        parser.error("inducing must be nonnegative; acquisition and update counts must be positive")
    if args.explore_confidence <= 0:
        parser.error("--explore-confidence must be positive")
    if args.explore_optimism < 0 or (
            args.seed_selections is not None and args.seed_selections < 0):
        parser.error("--explore-optimism and --seed-selections cannot be negative")
    asyncio.run(optimize(args))


if __name__ == "__main__":
    main()
