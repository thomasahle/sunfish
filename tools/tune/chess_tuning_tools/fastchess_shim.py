#!/usr/bin/env python3
"""Translate CTT's cutechess engine configs into native fastchess options."""

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
import pentanomial  # noqa: E402


SETOPTION = re.compile(r"setoption name (.*?) value (.*)")


def engines():
    return {engine["name"]: engine for engine in json.loads(
        pathlib.Path("engines.json").read_text())}


def translate(argv, configs):
    result = [os.environ.get("FASTCHESS", "fastchess"),
              "-config", "discard=true", "outname=fastchess-state.json",
              "-output", "format=cutechess", "-autosaveinterval", "0"]
    index = 0
    while index < len(argv):
        if argv[index] == "-pgnout" and index + 1 < len(argv):
            result += ["-pgnout", f"file={argv[index + 1]}"]
            index += 2
            continue
        if argv[index] != "-engine":
            if argv[index] != "-debug":
                result.append(argv[index])
            index += 1
            continue
        index += 1
        fields = []
        while index < len(argv) and not argv[index].startswith("-"):
            fields.append(argv[index])
            index += 1
        values = dict(field.split("=", 1) for field in fields if "=" in field)
        engine = configs[values.pop("conf")]
        result += ["-engine", f"cmd={engine['command']}", f"name={engine['name']}",
                   f"proto={engine.get('protocol', 'uci')}"]
        for command in engine.get("initStrings", []):
            if match := SETOPTION.fullmatch(command):
                result.append(f"option.{match.group(1)}={match.group(2)}")
        result += [field for field in fields
                   if not field.startswith("conf=") and field != "restart=auto"]
    return result


def save(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        output.write(json.dumps(state, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def commit_openings(iteration):
    path = pathlib.Path(os.environ["CTT_OPENING_STATE"])
    progress = json.loads(path.read_text())
    pending = progress.get("pending")
    if pending and pending["iteration"] < iteration:
        progress["next_opening"] = pending["next_opening"]
        progress["games"] = pending["games"]
        progress.pop("pending")
        save(path, progress)


def complete_log(path, identity, rounds):
    if not path.exists():
        return None
    output = path.read_text(errors="replace")
    header = f"ctt-match-identity {identity}\n"
    if not output.startswith(header):
        return None
    pentanomial.reject_failures(output)
    try:
        results, _ = pentanomial.game_results(output)
    except (IndexError, ValueError):
        return None
    return output[len(header):] if len(results) == 2 * rounds else None


def sequence_openings(command):
    state = os.environ.get("CTT_OPENING_STATE")
    if not state:
        return command, None
    study = os.environ.get("CTT_STUDY_ID")
    iteration = os.environ.get("CTT_ITERATION")
    if not study or iteration is None:
        raise RuntimeError("CTT_OPENING_STATE requires CTT_STUDY_ID and CTT_ITERATION")
    iteration = int(iteration)
    opening = command.index("-openings")
    rounds = int(command[command.index("-rounds") + 1])
    book = pathlib.Path(command[opening + 1].split("=", 1)[1])
    count = sum(bool(line.strip()) for line in book.read_text().splitlines())
    path = pathlib.Path(state)
    progress = {
        "version": 2, "study": study,
        "next_opening": int(os.environ.get("CTT_OPENING_START", 1)), "games": 0,
    }
    if path.exists():
        progress = json.loads(path.read_text())
        if progress.get("version") != 2 or progress.get("study") != study:
            raise RuntimeError("opening state belongs to a different CTT study")
    pending = progress.pop("pending", None)
    if pending and pending["iteration"] < iteration:
        progress["next_opening"] = pending["next_opening"]
        progress["games"] = pending["games"]
        pending = None
    if pending and pending["iteration"] != iteration:
        raise RuntimeError("CTT iteration and pending opening do not agree")
    start = pending["opening"] if pending else progress["next_opening"]
    command[command.index("order=random", opening)] = "order=sequential"
    command.insert(opening + 4, f"start={start}")
    identity = hashlib.sha256(json.dumps(
        {"study": study, "iteration": iteration, "command": command},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if pending and pending["identity"] != identity:
        raise RuntimeError("restarted CTT iteration changed its match identity")
    if not pending:
        pending = {
            "iteration": iteration, "identity": identity, "opening": start,
            "next_opening": (start - 1 + rounds) % count + 1,
            "games": progress["games"] + 2 * rounds,
        }
    progress["pending"] = pending
    save(path, progress)
    logs = pathlib.Path(os.environ.get("CTT_MATCH_DIR", path.parent / "ctt-match-cache"))
    logs.mkdir(parents=True, exist_ok=True)
    return command, (logs / f"iteration-{iteration:06d}-{identity}.log", identity, rounds)


def main():
    if sys.argv[1:2] == ["--commit-iteration"]:
        commit_openings(int(sys.argv[2]))
        return 0
    command, transaction = sequence_openings(translate(sys.argv[1:], engines()))
    if transaction:
        path, identity, rounds = transaction
        if output := complete_log(path, identity, rounds):
            print(output, end="", flush=True)
            return 0
        with path.open("w") as log:
            log.write(f"ctt-match-identity {identity}\n")
            log.flush()
            os.fsync(log.fileno())
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace")
    failure = None
    output = path.open("a") if transaction else None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if output:
                output.write(line)
            try:
                pentanomial.reject_failures(line)
            except pentanomial.EngineFailure as error:
                failure = failure or error
    finally:
        if output:
            output.flush()
            os.fsync(output.fileno())
            output.close()
    status = process.wait()
    if failure:
        print(failure, file=sys.stderr)
        status = status or 1
    if status == 0 and transaction and complete_log(path, identity, rounds) is None:
        status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
