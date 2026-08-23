#!/usr/bin/env python3
"""Tune UCI options from paired games with RBFOpt's noisy MSRSM."""

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
from rbfopt import RbfoptAlgorithm, RbfoptBlackBox, RbfoptSettings

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import gating  # noqa: E402
import pentanomial  # noqa: E402


UCI_OPTION = re.compile(r"^option name (.+?) type ")


def options(values):
    return dict(item.split("=", 1) for item in values)


def engine(command, name, arguments, values):
    executable = shutil.which(command) or command
    result = ["-engine", f"cmd={pathlib.Path(executable).resolve()}", f"name={name}"]
    if arguments:
        result.append(f"args={arguments}")
    result += [f"option.{key}={value}" for key, value in sorted(values.items())]
    return result


def digest(path):
    path = pathlib.Path(path).resolve()
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return str(path), result.hexdigest()


def command_identity(command, arguments):
    executable = shutil.which(command) or command
    files = [digest(path) for path in [executable, *shlex.split(arguments)]
             if pathlib.Path(path).is_file()]
    return str(pathlib.Path(executable).resolve()), arguments, files


def gate_identity(command):
    if not command:
        return None
    fields = shlex.split(command)
    return command_identity(fields[0], shlex.join(fields[1:]))


def evaluation_identity(study, number, opening, pairs, mode, knobs):
    payload = {
        "study": study, "number": number, "opening": opening,
        "pairs": pairs, "mode": mode, "options": knobs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def recover_evaluation(path, identity, pairs):
    if not path.exists():
        return None
    output = path.read_text(errors="replace")
    if not output.startswith(f"rbfopt-match-identity {identity}\n"):
        return None
    pentanomial.reject_failures(output)
    try:
        counts, wdl = pentanomial.parse(output)
    except (IndexError, ValueError):
        return None
    return (counts, wdl) if sum(counts) == pairs else None


def save_model(optimizer, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    optimizer.save_to_file(temporary)
    with temporary.open("rb") as saved:
        os.fsync(saved.fileno())
    temporary.replace(path)


def study_identity(args):
    return {
        "version": 2,
        "runner": digest(__file__),
        "fastchess": digest(shutil.which(args.fastchess) or args.fastchess),
        "candidate": command_identity(args.engine, args.engine_args),
        "baseline": command_identity(
            args.baseline_engine or args.engine,
            args.baseline_args if args.baseline_args is not None else args.engine_args),
        "space": digest(args.space), "openings": digest(args.openings),
        "tc": args.tc, "fixed": options(args.fixed_option),
        "baseline_options": options(args.baseline_option),
        "noisy_pairs": args.noisy_pairs, "accurate_pairs": args.accurate_pairs,
        "seed": args.seed, "gate": gate_identity(args.gate),
        "gate_timeout": args.gate_timeout,
    }


def validate(command, arguments, required):
    process = subprocess.run(
        [command, *shlex.split(arguments)], input="uci\nquit\n", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    advertised = {match.group(1) for line in process.stdout.splitlines()
                  if (match := UCI_OPTION.match(line))}
    missing = sorted(set(required) - advertised)
    if process.returncode or "uciok" not in process.stdout or missing:
        raise RuntimeError(f"cannot validate UCI options: {', '.join(missing)}")


class Space:
    def __init__(self, path):
        self.parameters = json.loads(pathlib.Path(path).read_text())["parameters"]
        self.low, self.high, self.kind, self.start = [], [], [], []
        for parameter in self.parameters:
            kind = parameter["type"]
            if kind == "integer":
                step = parameter.get("step", 1)
                count = round((parameter["max"] - parameter["min"]) / step)
                self.low.append(0)
                self.high.append(count)
                self.kind.append("I")
                self.start.append((parameter["default"] - parameter["min"]) / step)
            elif kind == "real":
                transform = math.log if parameter.get("transform") == "log" else lambda x: x
                self.low.append(transform(parameter["min"]))
                self.high.append(transform(parameter["max"]))
                self.kind.append("R")
                self.start.append(transform(parameter["default"]))
            elif kind in {"boolean", "categorical", "discrete"}:
                values = parameter.get("ordered_values") or parameter["values"]
                self.low.append(0)
                self.high.append(len(values) - 1)
                self.kind.append("C")
                self.start.append(values.index(parameter["default"]))
            else:
                raise ValueError(f"unsupported parameter type: {kind}")

    def decode(self, point):
        result = {}
        for parameter, coordinate in zip(self.parameters, point):
            kind = parameter["type"]
            if kind == "integer":
                value = parameter["min"] + round(coordinate) * parameter.get("step", 1)
            elif kind == "real":
                value = math.exp(coordinate) if parameter.get("transform") == "log" else coordinate
            else:
                values = parameter.get("ordered_values") or parameter["values"]
                value = values[round(coordinate)]
            result[parameter["name"]] = value
        return result


class ChessBox(RbfoptBlackBox):
    def __init__(self, args, space):
        self.args = args
        self.space = space
        self.state = self.load_state()
        self.opening_count = sum(
            bool(line.strip()) for line in pathlib.Path(args.openings).read_text().splitlines())

    def load_state(self):
        path = pathlib.Path(self.args.state)
        state = json.loads(path.read_text()) if path.exists() else {
            "next_opening": self.args.start, "games": 0, "evaluations": [], "checkpoints": []}
        identity = study_identity(self.args)
        if "study" in state and state["study"] != identity:
            raise RuntimeError("state belongs to a different RBFOpt study")
        state["study"] = identity
        return state

    def save(self):
        path = pathlib.Path(self.args.state)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w") as output:
            output.write(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)

    def get_dimension(self):
        return len(self.space.parameters)

    def get_var_lower(self):
        return np.asarray(self.space.low, dtype=float)

    def get_var_upper(self):
        return np.asarray(self.space.high, dtype=float)

    def get_var_type(self):
        return np.asarray(self.space.kind)

    def has_evaluate_noisy(self):
        return True

    def aggregate(self, knobs):
        counts = [0] * 5
        for evaluation in self.state["evaluations"]:
            if evaluation["options"] == knobs:
                counts = [a + b for a, b in zip(counts, evaluation["counts"])]
        return counts

    def play(self, point, requested_pairs, mode):
        remaining = self.args.games - self.state["games"]
        pairs = min(requested_pairs, remaining // 2)
        if pairs < 1:
            raise RuntimeError("game budget exhausted inside an RBFOpt evaluation")
        knobs = self.space.decode(point) | options(self.args.fixed_option)
        payload = {
            "engine": self.args.engine,
            "engine_args": self.args.engine_args,
            "options": knobs,
        }
        if not gating.policy(
                self.args.gate, self.args.gate_timeout, payload,
                self.state.setdefault("gates", {})):
            return 0.5, 0, 0
        opening = self.state["next_opening"]
        command = [
            self.args.fastchess,
            *engine(self.args.engine, "candidate", self.args.engine_args, knobs),
            *engine(self.args.baseline_engine or self.args.engine, "baseline",
                    self.args.baseline_args if self.args.baseline_args is not None
                    else self.args.engine_args, options(self.args.baseline_option)),
            "-each", "proto=uci", f"tc={self.args.tc}",
            "-openings", f"file={pathlib.Path(self.args.openings).resolve()}", "format=epd",
            "order=sequential", f"start={opening}", "-rounds", str(pairs), "-games", "2",
            "-repeat", "-concurrency", str(min(pairs, self.args.slots)), "-recover",
            "-draw", "movenumber=40", "movecount=8", "score=10",
            "-resign", "movecount=4", "score=500", "-output", "format=cutechess",
            "-scoreinterval", "1", "-ratinginterval", "0",
        ]
        number = len(self.state["evaluations"])
        identity = evaluation_identity(
            self.state["study"], number, opening, pairs, mode, knobs)
        path = pathlib.Path(self.args.logs, f"evaluation-{number:06d}-{identity}.log")
        recovered = recover_evaluation(path, identity, pairs)
        if recovered is None:
            with path.open("wb") as output:
                output.write(f"rbfopt-match-identity {identity}\n".encode())
                output.flush()
                os.fsync(output.fileno())
            with path.open("ab", buffering=0) as output:
                process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
                os.fsync(output.fileno())
            recovered = recover_evaluation(path, identity, pairs)
            if process.returncode or recovered is None:
                raise RuntimeError(
                    f"fastchess evaluation {number} failed with {process.returncode}")
        counts, wdl = recovered
        self.state["evaluations"].append({
            "number": number, "mode": mode, "opening": opening, "pairs": pairs,
            "options": knobs, "counts": counts, "wins": wdl[0], "losses": wdl[1],
            "draws": wdl[2],
        })
        self.state["games"] += 2 * pairs
        self.state["next_opening"] = (opening - 1 + pairs) % self.opening_count + 1
        combined = self.aggregate(knobs)
        mean, deviation, interval = pentanomial.posterior(combined)
        value = mean - 0.5
        print(f"[{self.state['games']}/{self.args.games}] {mode} {wdl[0]}-{wdl[1]}-{wdl[2]} "
              f"loss={value:+.4f} +- {1.96 * deviation:.4f} {knobs}", flush=True)
        return value, interval[0] - 0.5 - value, interval[1] - 0.5 - value

    def evaluate(self, point):
        return self.play(point, self.args.accurate_pairs, "accurate")[0]

    def evaluate_noisy(self, point):
        return np.asarray(self.play(point, self.args.noisy_pairs, "noisy"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fastchess", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--baseline-engine")
    parser.add_argument("--engine-args", default="")
    parser.add_argument("--baseline-args")
    parser.add_argument("--fixed-option", action="append", default=[])
    parser.add_argument("--baseline-option", action="append", default=[])
    parser.add_argument("--space", required=True)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--state", default="rbfopt.json")
    parser.add_argument("--model", default="rbfopt.pkl")
    parser.add_argument("--logs", default="rbfopt-logs")
    parser.add_argument("--tc", default="3+0.1")
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--noisy-pairs", type=int, default=1)
    parser.add_argument("--accurate-pairs", type=int, default=10)
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gate", help="command reading a policy JSON object on stdin")
    parser.add_argument("--gate-timeout", type=float, default=60)
    args = parser.parse_args()
    if min(args.games // 2, args.noisy_pairs, args.accurate_pairs, args.slots,
           args.start, args.gate_timeout) < 1:
        parser.error("games, pair counts, slots, and start must be positive")
    pathlib.Path(args.logs).mkdir(parents=True, exist_ok=True)
    space = Space(args.space)
    validate(args.engine, args.engine_args,
             [p["name"] for p in space.parameters] + list(options(args.fixed_option)))
    validate(args.baseline_engine or args.engine,
             args.baseline_args if args.baseline_args is not None else args.engine_args,
             options(args.baseline_option))
    model = pathlib.Path(args.model)
    if model.exists():
        optimizer = RbfoptAlgorithm.load_from_file(model)
        optimizer.bb.args = args
        expected = study_identity(args)
        if optimizer.bb.state.get("study") != expected:
            raise RuntimeError("model belongs to a different RBFOpt study")
        optimizer.bb.space = space
        optimizer.settings.do_local_search = False
        optimizer.l_settings.do_local_search = False
        optimizer.bb.save()
    elif pathlib.Path(args.state).exists():
        raise RuntimeError("RBFOpt state exists without its authoritative model")
    else:
        black_box = ChessBox(args, space)
        settings = RbfoptSettings(
            algorithm="MSRSM", global_search_method="genetic", rand_seed=args.seed,
            do_local_search=False,
            max_iterations=1000000, max_evaluations=1000000, max_noisy_evaluations=1000000,
            max_noisy_iterations=1000000, max_noisy_restarts=1000000,
            save_state_interval=1000000)
        optimizer = RbfoptAlgorithm(settings, black_box, init_node_pos=[space.start])
    with pathlib.Path(args.logs, "rbfopt.log").open("a") as output:
        optimizer.set_output_stream(output)
        idle = 0
        while optimizer.bb.state["games"] + 2 <= args.games:
            before = optimizer.bb.state["games"]
            _, point, *_ = optimizer.optimize(pause_after_iters=1)
            if optimizer.bb.state["games"] > before:
                optimizer.bb.state["checkpoints"].append({
                    "games": optimizer.bb.state["games"],
                    "options": optimizer.bb.space.decode(point),
                })
                idle = 0
            else:
                idle += 1
            save_model(optimizer, model)
            optimizer.bb.save()
            if idle == 100:
                raise RuntimeError("RBFOpt made no evaluation in 100 iterations")


if __name__ == "__main__":
    main()
