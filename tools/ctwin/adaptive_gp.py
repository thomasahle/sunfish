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

import argparse
import asyncio
import hashlib
import json
import math
import pathlib
import re
import shlex
import shutil
import subprocess
from collections import deque

import numpy as np

import logistic_gp


SCORE = re.compile(
    r"Score of candidate vs (?:baseline|opponent):\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)")
UCI_OPTION = re.compile(r"^option name (.+?) type ")


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
    if path.exists():
        return json.loads(path.read_text())
    return {"next_opening": start, "batches": []}


def save_state(path, state):
    path = pathlib.Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_identity(path):
    path = pathlib.Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


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


def study_identity(args):
    """Describe everything that changes the distribution of game observations."""
    return {
        "version": 2,
        "scheduler": file_identity(__file__),
        "model": file_identity(pathlib.Path(__file__).with_name("logistic_gp.py")),
        "fastchess": file_identity(shutil.which(args.fastchess) or args.fastchess),
        "candidate": engine_identity(args.engine, args.engine_args, {}),
        "baseline": engine_identity(
            args.baseline_engine, args.baseline_args, args.baseline_options),
        "openings": file_identity(args.openings),
        "space": file_identity(args.space) if args.space else "legacy",
        "seed_state": file_identity(args.seed_state) if args.seed_state else None,
        "tc": args.tc,
        "allocation": {
            name: getattr(args, name)
            for name in ("pair_weight", "exploration", "initial_design", "explore_start",
                         "explore_floor", "explore_half_life", "explore_optimism",
                         "duel_fraction", "inducing", "seed_selections")
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


def aggregate(batches, pair_weight, space):
    totals = {}
    points = set()
    for batch in batches:
        left = space.canonical(batch["knobs"])
        right = batch.get("opponent_knobs")
        right = space.canonical(right) if right is not None else None
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
    design = np.zeros((len(totals), len(points)))
    success = []
    trials = []
    for row, ((left, right), (wins, games)) in enumerate(totals.items()):
        design[row, indices[left]] = 1
        if right is not None:
            design[row, indices[right]] = -1
        success.append(wins)
        trials.append(games)
    return points, design, success, trials


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
    pool = sorted(set([*points, space.default]))
    count = min(count, len(pool))
    information = np.abs(design).T @ trials
    ranked = sorted(range(len(points)), key=lambda i: (-information[i], points[i]))
    required = [space.default]
    required += [points[i] for i in ranked[:max(1, count // 2)]]
    required = list(dict.fromkeys(required))[:count]
    if isinstance(space, logistic_gp.MixedSpace):
        return space.maximin(pool, required, count)
    remaining = [point for point in pool if point not in required]
    indices = np.linspace(0, len(remaining) - 1, count - len(required), dtype=int)
    return required + [remaining[index] for index in indices]


def posterior(state, mean_function, pair_weight, space, inducing=0):
    points, design, success, trials = aggregate(state["batches"], pair_weight, space)
    if not points:
        return None
    basis = None
    if inducing and len(points) > inducing:
        basis = inducing_basis(points, design, trials, space, inducing)
    return logistic_gp.LogisticGP(
        mean_function, space.kernel, space.kernel_diagonal, basis).fit_comparisons(
        points, design, success, trials)


def design_variance(sites, candidates, space):
    """Residual prior variance after conditioning noiselessly on design sites."""
    covariance = space.kernel(candidates, candidates)
    if not sites:
        return np.diag(covariance).copy()
    cross = space.kernel(sites, candidates)
    site_covariance = space.kernel(sites, sites) + np.eye(len(sites)) * 1e-6
    projection = np.linalg.solve(site_covariance, cross)
    return np.maximum(np.diag(covariance) - np.sum(cross * projection, axis=0), 0)


def exploration_probability(selections, start, floor, half_life):
    return floor + (start - floor) / math.sqrt(1 + selections / half_life)


def pending_configurations(slots, pairs):
    """Smallest number of parameter choices that can keep every lane busy."""
    return math.ceil(slots / pairs)


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
           forbidden=()):
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

    observed = {space.canonical(batch["knobs"]) for batch in state["batches"]}
    active = [left for left, _ in pending]
    sites = observed | set(active)
    selections = state.get("selections", len(state["batches"]))
    probability = exploration_probability(
        selections, args.explore_start, args.explore_floor, args.explore_half_life)
    new_axes = set(state.get("new_axes", ()))
    fresh_design = new_axes and selections < args.initial_design
    if fresh_design:
        mode = "design"
        acquisition = variance.copy()
        for index, candidate in enumerate(candidates):
            changed = [
                name for name, value, default in zip(space.names, candidate, space.default)
                if value != default
            ]
            if len(changed) != 1 or changed[0] not in new_axes or candidate in sites:
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
        vector = coordinate_maximum(
            space, [*candidates, *observed], score, set(forbidden), stratum)
    else:
        for index, candidate in enumerate(candidates):
            if candidate in active or (
                    stratum is not None and space.is_structural(candidate) != stratum):
                acquisition[index] = -np.inf
        vector = candidates[int(np.argmax(acquisition))]
    selected_mean, selected_variance = statistics([vector])
    exact_batches = sum(
        space.canonical(batch["knobs"]) == vector for batch in state["batches"])
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


def choose_opponent(state, mean_function, challenger, args, space, model=None):
    """Choose an anchored, informative rival for a parameter duel."""
    anchored = {
        space.canonical(batch["knobs"])
        for batch in state["batches"]
        if batch.get("opponent_knobs") is None
    }
    anchored.discard(challenger)
    if not anchored:
        return None
    credit = state.get("duel_credit", 0.0) + args.duel_fraction
    if credit < 1:
        state["duel_credit"] = credit
        return None
    state["duel_credit"] = credit - 1
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
    validate_opening_budget(args.openings, state["next_opening"], args.batches, args.pairs)
    space = logistic_gp.MixedSpace.load(args.space) if args.space else logistic_gp.LegacySpace()
    bind_study(state, study_identity(args))
    if args.seed_state and not state["batches"]:
        seed = json.loads(pathlib.Path(args.seed_state).read_text())
        old_baseline = seed["study"]["baseline"]["options"]
        old_names = set(old_baseline)
        for batch in seed["batches"]:
            old_names.update(batch["knobs"])
            old_names.update(batch.get("opponent_knobs") or ())
        state["batches"] = [
            batch | {"opponent_knobs": old_baseline}
            if batch.get("opponent_knobs") is None else batch
            for batch in seed["batches"]
        ]
        state["next_experiment"] = len(state["batches"])
        state["selections"] = args.seed_selections
        state["allocations"] = {}
        state["new_axes"] = [name for name in space.names if name not in old_names]
    save_state(args.state, state)
    options = set().union(*(space.knobs(candidate) for candidate in space.candidates))
    validate_options(args.engine, args.engine_args, options)
    validate_options(args.baseline_engine, args.baseline_args, args.baseline_options)
    fixed = fixed_baseline_point(args, space)
    if fixed is not None:
        space.condition(fixed)
    mean_function = source_prior(args.source_logs, args.battery, args.transfer, space)
    candidates = [candidate for candidate in space.candidates if candidate != fixed]
    if not candidates:
        raise ValueError("parameter space contains no challenger to the fixed baseline")
    if args.safe_only:
        if not isinstance(space, logistic_gp.LegacySpace):
            raise ValueError("--safe-only applies only to the built-in Sunfish LMR space")
        candidates = [x for x in candidates if x[2] and not any(x[len(logistic_gp.NUMERIC):])]
    queue = deque()
    experiments = {}
    running = {}
    completed = 0
    allocation_model = None

    def add_experiment():
        nonlocal allocation_model
        pending = [
            (experiment["vector"], experiment["opponent"])
            for experiment in experiments.values()
        ]
        if allocation_model is None:
            allocation_model = posterior(
                state, mean_function, args.pair_weight, space, args.inducing)
        vector, diagnostics = choose(
            state, mean_function, candidates, pending, args, space, allocation_model,
            [fixed] if fixed is not None else [])
        opponent = None
        if diagnostics["mode"] != "design":
            opponent = choose_opponent(
                state, mean_function, vector, args, space, allocation_model)
        number = state.get("next_experiment", 0)
        state["next_experiment"] = number + 1
        opening = state["next_opening"]
        state["next_opening"] += args.pairs
        experiments[number] = {
            "vector": vector, "opening": opening, "wins": 0, "draws": 0, "losses": 0,
            "allocation": diagnostics["mode"], "opponent": opponent,
        }
        for offset in range(args.pairs):
            queue.append((number, vector, opponent, opening + offset))
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

    while completed < args.batches:
        if len(experiments) <= args.queue_batches - args.refill_batches:
            while (len(experiments) < args.queue_batches
                    and completed + len(experiments) < args.batches):
                add_experiment()
        while queue and len(running) < args.slots:
            used = {value for value in running.values()}
            slot = next(x for x in range(args.slots) if x not in used)
            experiment, vector, opponent, opening = queue.popleft()
            task = asyncio.create_task(
                run_pair(args, slot, experiment, vector, opponent, opening, space))
            running[task] = slot
        done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
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
                "allocation": experiment["allocation"],
                "opponent_knobs": (
                    space.knobs(experiment["opponent"])
                    if experiment["opponent"] is not None else None),
            })
            del experiments[number]
            completed += 1
            allocation_model = None
            save_state(args.state, state)
            print(f"[experiment {number}] result "
                  f"{experiment['wins']}-{experiment['losses']}-{experiment['draws']}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastchess", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--baseline-engine")
    parser.add_argument("--engine-args", default="")
    parser.add_argument("--baseline-args")
    parser.add_argument("--baseline-options", type=json.loads, default={})
    parser.add_argument("--space", help="JSON numeric/categorical UCI option space")
    parser.add_argument("--seed-state", help="import completed batches into a new study")
    parser.add_argument("--openings", required=True)
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
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--exploration", type=float, default=1.0)
    parser.add_argument("--initial-design", type=int, default=12)
    parser.add_argument("--explore-start", type=float, default=0.50)
    parser.add_argument("--explore-floor", type=float, default=0.20)
    parser.add_argument("--explore-half-life", type=float, default=40)
    parser.add_argument("--explore-optimism", type=float, default=0,
        help="explore with mean + K*sd instead of pure variance")
    parser.add_argument("--duel-fraction", type=float, default=0.30)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--inducing", type=int, default=0,
        help="sparse GP size; exact inference is the default")
    parser.add_argument("--seed-selections", type=int, default=0,
        help="continue the allocation clock when importing a state")
    parser.add_argument("--safe-only", action="store_true")
    args = parser.parse_args()
    args.baseline_engine = args.baseline_engine or args.engine
    args.baseline_args = args.engine_args if args.baseline_args is None else args.baseline_args
    if args.source_logs and not args.battery:
        parser.error("--source-logs requires --battery")
    if not 0 <= args.explore_floor <= args.explore_start <= 1:
        parser.error("require 0 <= --explore-floor <= --explore-start <= 1")
    if not 0 <= args.duel_fraction <= 0.40:
        parser.error("require 0 <= --duel-fraction <= 0.40 to preserve the baseline anchor")
    if args.pair_weight <= 0:
        parser.error("--pair-weight must be positive")
    if args.pairs < 1 or args.slots < 1 or args.batches < 1:
        parser.error("--pairs, --slots and --batches must be positive")
    if args.queue_batches is None:
        args.queue_batches = pending_configurations(args.slots, args.pairs)
    elif args.queue_batches < 1:
        parser.error("--queue-batches must be positive")
    if not 1 <= args.refill_batches <= args.queue_batches:
        parser.error("require 1 <= --refill-batches <= --queue-batches")
    if args.initial_design < 1 or args.explore_half_life <= 0:
        parser.error("--initial-design and --explore-half-life must be positive")
    if args.inducing < 0:
        parser.error("--inducing cannot be negative")
    if args.explore_optimism < 0 or args.seed_selections < 0:
        parser.error("--explore-optimism and --seed-selections cannot be negative")
    asyncio.run(optimize(args))


if __name__ == "__main__":
    main()
