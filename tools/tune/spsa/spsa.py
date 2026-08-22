#!/usr/bin/env python3
"""Tune ordered UCI options with the classic Fishtest SPSA schedule."""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import pathlib
import random
import re
import shlex
import shutil
import subprocess
import sys


sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import gating  # noqa: E402
import locking  # noqa: E402
import pentanomial  # noqa: E402
import opponent_panel  # noqa: E402


SCORE = re.compile(r"Score of plus vs minus:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)")
PANEL_SCORE = re.compile(r"Score of candidate vs baseline:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)")
UCI_OPTION = re.compile(r"^option name (.+?) type ")


def engine(command, name, arguments, options):
    command = shutil.which(command) or command
    result = ["-engine", f"cmd={pathlib.Path(command).resolve()}", f"name={name}"]
    if arguments:
        result.append(f"args={arguments}")
    result += [f"option.{key}={value}" for key, value in sorted(options.items())]
    return result


def digest(path):
    path = pathlib.Path(path).resolve()
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return {"path": str(path), "sha256": result.hexdigest()}


def command_identity(command, arguments):
    executable = shutil.which(command) or command
    return {
        "command": str(pathlib.Path(executable).resolve()),
        "arguments": arguments,
        "files": [digest(path) for path in [executable, *shlex.split(arguments)]
                  if pathlib.Path(path).is_file()],
    }


def gate_identity(command):
    if not command:
        return None
    fields = shlex.split(command)
    return command_identity(fields[0], shlex.join(fields[1:]))


def step_identity(study, number, opening, pairs, plus, minus):
    payload = {
        "study": study, "number": number, "opening": opening, "pairs": pairs,
        "plus": plus, "minus": minus,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def recover_step(path, identity, pairs):
    if not path.exists():
        return None
    output = path.read_text(errors="replace")
    if not output.startswith(f"adaptive-spsa-identity {identity}\n"):
        return None
    pentanomial.reject_failures(output)
    matches = SCORE.findall(output)
    if not matches:
        return None
    result = tuple(map(int, matches[-1]))
    return result if sum(result) == 2 * pairs else None


def study_identity(args):
    panel = getattr(args, "baseline_panel", [])
    def configured_engine(command, arguments, options):
        return command_identity(command, arguments) | {"options": options}

    return {
        "version": 3, "runner": digest(__file__),
        "fastchess": digest(shutil.which(args.fastchess) or args.fastchess),
        "engine": command_identity(args.engine, args.engine_args),
        "space": digest(args.space), "openings": digest(args.openings),
        "fixed": sorted(args.fixed_option),
        "initial": sorted(getattr(args, "initial_option", [])),
        "tc": args.tc, "start": args.start,
        "iterations": args.iterations, "pairs_per_step": args.pairs_per_step,
        "slots": args.slots, "seed": args.seed, "a_ratio": args.a_ratio,
        "alpha": args.alpha, "gamma": args.gamma, "c_ratio": args.c_ratio,
        "r_end": args.r_end, "draw_ratio": args.draw_ratio, "precision": args.precision,
        "gate": gate_identity(args.gate), "gate_timeout": args.gate_timeout,
        "gate_attempts": args.gate_attempts,
        "baseline_panel": [opponent_panel.identity(member, configured_engine, digest)
                           for member in panel],
    }


def validate_options(command, arguments, required):
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
        raise RuntimeError(f"cannot validate UCI options: {', '.join(missing)}")


def chi2_95(dimensions):
    """Wilson-Hilferty approximation used by Fishtest's SPSA setup UI."""
    z95 = 1.6448536269514722
    t = 2 / (9 * dimensions)
    return dimensions * (z95 * math.sqrt(t) + 1 - t) ** 3


def option_values(items):
    result = {}
    for item in items:
        name, separator, raw = item.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"bad or duplicate initial option: {item}")
        try:
            result[name] = json.loads(raw)
        except json.JSONDecodeError:
            result[name] = raw
    return result


def load_parameters(path, iterations, args):
    raw = json.loads(pathlib.Path(path).read_text())["parameters"]
    initial = option_values(getattr(args, "initial_option", []))
    unknown = initial.keys() - {item["name"] for item in raw}
    if unknown:
        raise ValueError(f"unknown initial options: {', '.join(sorted(unknown))}")
    parameters = []
    auto_r = args.precision / (
        347.43558552260146 * chi2_95(len(raw)) * (1 - args.draw_ratio) / 8)
    for item in raw:
        kind = item["type"]
        if kind in {"discrete", "categorical", "boolean"}:
            values = item.get("ordered_values") or item.get("values")
            if values is None:
                raise ValueError(f"{item['name']} has no choices")
            if len(values) < 2:
                raise ValueError(f"{item['name']} needs at least two choices")
            value = initial.get(item["name"], item["default"])
            if value not in values:
                raise ValueError(f"{item['name']} initial value is outside its choices")
            low, high, theta = 0, len(values) - 1, values.index(value)
        elif kind in {"integer", "real"}:
            low, high = item["min"], item["max"]
            theta = initial.get(item["name"], item["default"])
            if not isinstance(theta, (int, float)) or not low <= theta <= high:
                raise ValueError(f"{item['name']} initial value is outside its range")
            values = None
        else:
            raise ValueError(f"SPSA cannot tune parameters of type {kind}")
        c_end = item.get("c_end", args.c_ratio * (high - low))
        r_end = item.get("r_end", args.r_end or auto_r)
        parameters.append({
            "name": item["name"], "type": kind, "step": item.get("step", 1),
            "min": low, "max": high, "theta": theta, "values": values,
            "c": c_end * iterations ** args.gamma,
            "a": r_end * c_end ** 2 * (args.a_ratio * iterations + iterations) ** args.alpha,
        })
    return parameters


def perturb(parameters, iteration, args, rng):
    flips = [rng.choice((-1, 1)) for _ in parameters]
    plus, minus, steps = {}, {}, []
    for parameter, flip in zip(parameters, flips):
        k = iteration + 1
        c = parameter["c"] / k ** args.gamma
        r = parameter["a"] / (args.a_ratio * args.iterations + k) ** args.alpha / c ** 2
        low, high, theta = parameter["min"], parameter["max"], parameter["theta"]
        plus[parameter["name"]] = min(max(theta + c * flip, low), high)
        minus[parameter["name"]] = min(max(theta - c * flip, low), high)
        steps.append((c, r, flip))
    return plus, minus, steps


def render(parameters, values, seed):
    """Apply unbiased rounding, then decode ordered choices to UCI values."""
    rng = random.Random(seed)
    result = {}
    for parameter in parameters:
        value = values[parameter["name"]]
        if parameter["type"] == "integer":
            step = parameter["step"]
            scaled = (value - parameter["min"]) / step
            value = parameter["min"] + math.floor(scaled + rng.random()) * step
            value = min(max(value, parameter["min"]), parameter["max"])
        elif parameter.get("values") is not None:
            index = math.floor(value + rng.random())
            index = min(max(index, parameter["min"]), parameter["max"])
            value = parameter["values"][index]
        result[parameter["name"]] = value
    return result


def play(args, study, number, opening, pairs, plus, minus):
    command = [
        args.fastchess,
        *engine(args.engine, "plus", args.engine_args, plus),
        *engine(args.engine, "minus", args.engine_args, minus),
        "-each", "proto=uci", f"tc={args.tc}",
        "-openings", f"file={pathlib.Path(args.openings).resolve()}", "format=epd",
        "order=sequential", f"start={opening}", "-rounds", str(pairs),
        "-games", "2", "-repeat", "-concurrency", str(min(args.slots, pairs)),
        "-recover",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=4", "score=500",
        "-output", "format=cutechess", "-scoreinterval", "1", "-ratinginterval", "0",
    ]
    identity = step_identity(study, number, opening, pairs, plus, minus)
    path = pathlib.Path(args.logs, f"step-{number:06d}-{identity}.log")
    if result := recover_step(path, identity, pairs):
        return result
    with path.open("wb") as output:
        output.write(f"adaptive-spsa-identity {identity}\n".encode())
        output.flush()
        os.fsync(output.fileno())
    with path.open("ab", buffering=0) as output:
        process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        os.fsync(output.fileno())
    text = path.read_text(errors="replace")
    pentanomial.reject_failures(text)
    matches = SCORE.findall(text)
    if process.returncode or not matches:
        raise RuntimeError(f"SPSA step {number} failed with status {process.returncode}")
    wins, losses, draws = map(int, matches[-1])
    if wins + losses + draws != 2 * pairs:
        raise RuntimeError(f"SPSA step {number} completed {wins + losses + draws} games")
    return wins, losses, draws


def play_fixed(args, number, opening, options, member, label):
    command = [
        args.fastchess,
        *engine(args.engine, "candidate", args.engine_args, options),
        *engine(member["engine"], "baseline", member["args"], member["options"]),
        "-each", "proto=uci", f"tc={args.tc}",
        "-openings", f"file={pathlib.Path(args.openings).resolve()}", "format=epd",
        "order=sequential", f"start={opening}", "-rounds", "1",
        "-games", "2", "-repeat", "-concurrency", "1", "-recover",
        "-draw", "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=4", "score=500",
        "-output", "format=cutechess", "-scoreinterval", "1", "-ratinginterval", "0",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pathlib.Path(args.logs, f"step-{number:06d}-{label}.log").write_bytes(process.stdout)
    output = process.stdout.decode(errors="replace")
    matches = PANEL_SCORE.findall(output)
    failure = opponent_panel.failure(output)
    if process.returncode or not matches or failure:
        raise RuntimeError(f"SPSA panel step {number}/{label} failed with status {process.returncode}")
    wins, losses, draws = map(int, matches[-1])
    if wins + losses + draws != 2:
        raise RuntimeError(f"SPSA panel step {number}/{label} completed {wins + losses + draws} games")
    return wins, losses, draws


def play_panel(args, number, opening, sequence, plus, minus):
    """Compare both perturbations with one opponent on the same color pair."""
    member = opponent_panel.select(args.baseline_panel, sequence)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(play_fixed, args, number, opening, options, member, label)
            for options, label in ((plus, "plus"), (minus, "minus"))
        ]
        results = [future.result() for future in futures]
    return member["name"], results


def save(path, state, parameters):
    payload = state | {"parameters": parameters}
    target = pathlib.Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(target)


def optimize(args):
    with locking.exclusive(args.state):
        optimize_locked(args)


def optimize_locked(args):
    pathlib.Path(args.logs).mkdir(parents=True, exist_ok=True)
    args.baseline_panel = getattr(args, "baseline_panel", [])
    fixed = dict(item.split("=", 1) for item in args.fixed_option)
    defaults = {
        item["name"]: item["default"]
        for item in json.loads(pathlib.Path(args.space).read_text())["parameters"]
    }
    for member in args.baseline_panel:
        if member["options"] == "default":
            member["options"] = defaults
    state_path = pathlib.Path(args.state)
    identity = study_identity(args)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("study") != identity:
            raise RuntimeError("state belongs to a different SPSA study")
        parameters = state.pop("parameters")
    else:
        parameters = load_parameters(args.space, args.iterations, args)
        state = {"study": identity, "results": []}
    validate_options(args.engine, args.engine_args, [p["name"] for p in parameters])
    for member in args.baseline_panel:
        validate_options(member["engine"], member["args"], member["options"])
    overlap = fixed.keys() & {p["name"] for p in parameters}
    if overlap:
        raise ValueError(f"fixed and tuned options overlap: {', '.join(sorted(overlap))}")
    opening_count = sum(
        bool(line.strip()) for line in pathlib.Path(args.openings).read_text().splitlines())
    completed = len(state["results"])
    if [result["number"] for result in state["results"]] != list(range(completed)) or any(
            "iteration" not in result or "pairs" not in result for result in state["results"]):
        raise RuntimeError("SPSA state is not a contiguous sequence of updates")
    iteration = sum(result["pairs"] for result in state["results"])
    while iteration < args.iterations:
        number = len(state["results"])
        pairs = min(args.pairs_per_step, args.iterations - iteration)
        rng = random.Random(args.seed + number)
        for attempt in range(args.gate_attempts):
            plus, minus, steps = perturb(parameters, iteration, args, rng)
            seed = 2 * (number if not args.gate else number * args.gate_attempts + attempt)
            plus = render(parameters, plus, seed)
            minus = render(parameters, minus, seed + 1)
            plus.update(fixed)
            minus.update(fixed)
            payloads = [
                {"engine": args.engine, "engine_args": args.engine_args, "options": knobs}
                for knobs in (plus, minus)
            ]
            if all(gating.policy(
                    args.gate, args.gate_timeout, payload,
                    state.setdefault("gates", {})) for payload in payloads):
                break
        else:
            raise RuntimeError(
                f"policy gate rejected {args.gate_attempts} SPSA perturbations")
        opening = (args.start - 1 + iteration) % opening_count + 1
        if args.baseline_panel:
            sequence = args.start + iteration
            opponent, panel_results = play_panel(
                args, number, opening, sequence, plus, minus)
            plus_result, minus_result = panel_results
            result = ((plus_result[0] + plus_result[2] / 2)
                      - (minus_result[0] + minus_result[2] / 2))
            wins = losses = draws = 0
        else:
            opponent = None
            plus_result = minus_result = None
            wins, losses, draws = play(
                args, identity, number, opening, pairs, plus, minus)
            result = wins - losses
        for parameter, (c, r, flip) in zip(parameters, steps):
            parameter["theta"] = min(max(
                parameter["theta"] + r * c * result * flip,
                parameter["min"]), parameter["max"])
        state["results"].append({
            "number": number, "iteration": iteration, "pairs": pairs,
            "opening_pairs": 2 * pairs if opponent is not None else pairs,
            "games": 4 * pairs if opponent is not None else 2 * pairs,
            "opening": opening, "plus": plus, "minus": minus,
            "wins": wins, "draws": draws, "losses": losses,
            "opponent": opponent, "plus_result": plus_result,
            "minus_result": minus_result, "gradient": result,
            "theta": {p["name"]: p["theta"] for p in parameters},
        })
        iteration += pairs
        save(args.state, state, parameters)
        values = " ".join(f"{p['name']}={p['theta']:.3g}" for p in parameters)
        outcome = (f"{wins}-{losses}-{draws}" if opponent is None else
            f"{opponent} plus={plus_result} minus={minus_result}")
        print(f"[{iteration}/{args.iterations} steps] {outcome} {values}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastchess", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--engine-args", default="")
    parser.add_argument("--baseline-panel",
        help="evaluate both perturbations against one weighted-panel member")
    parser.add_argument("--space", required=True)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--state", default="spsa.json")
    parser.add_argument("--logs", default="spsa-logs")
    parser.add_argument("--tc", default="3+0.1")
    parser.add_argument("--iterations", type=int, required=True,
                        help="total SPSA update steps")
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--pairs-per-step", type=int, default=5)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fixed-option", action="append", default=[])
    parser.add_argument("--initial-option", action="append", default=[])
    parser.add_argument("--a-ratio", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.602)
    parser.add_argument("--gamma", type=float, default=0.101)
    parser.add_argument("--c-ratio", type=float, default=1 / 6)
    parser.add_argument("--r-end", type=float)
    parser.add_argument("--draw-ratio", type=float, default=0.2)
    parser.add_argument("--precision", type=float, default=0.5)
    parser.add_argument("--gate", help="command reading a policy JSON object on stdin")
    parser.add_argument("--gate-timeout", type=float, default=60)
    parser.add_argument("--gate-attempts", type=int, default=100)
    args = parser.parse_args()
    args.baseline_panel = opponent_panel.load(args.baseline_panel) if args.baseline_panel else []
    if args.baseline_panel and (args.pairs_per_step != 1 or args.slots != 1):
        parser.error("panel-anchored SPSA requires --pairs-per-step 1 --slots 1")
    if min(args.iterations, args.slots, args.pairs_per_step,
           args.gate_timeout, args.gate_attempts) < 1:
        parser.error("iteration, slot, pair, and gate limits must be positive")
    optimize(args)


if __name__ == "__main__":
    main()
