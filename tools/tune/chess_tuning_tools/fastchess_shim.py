#!/usr/bin/env python3
"""Translate CTT's cutechess engine configs into native fastchess options."""

import json
import os
import pathlib
import re
import subprocess
import sys


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


def sequence_openings(command):
    state = os.environ.get("CTT_OPENING_STATE")
    if not state:
        return command, None
    opening = command.index("-openings")
    rounds = int(command[command.index("-rounds") + 1])
    book = pathlib.Path(command[opening + 1].split("=", 1)[1])
    count = sum(bool(line.strip()) for line in book.read_text().splitlines())
    path = pathlib.Path(state)
    progress = {"next_opening": int(os.environ.get("CTT_OPENING_START", 1)), "games": 0}
    if path.exists():
        text = path.read_text()
        try:
            progress.update(json.loads(text))
        except json.JSONDecodeError:
            progress["next_opening"] = int(text)
    start = progress["next_opening"]
    following_games = progress["games"] + 2 * rounds
    command[command.index("order=random", opening)] = "order=sequential"
    command.insert(opening + 4, f"start={start}")
    progress.update(
        next_opening=(start - 1 + rounds) % count + 1, games=following_games)
    return command, (path, progress)


def advance_openings(checkpoint):
    if checkpoint is None:
        return
    path, following = checkpoint
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(following, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    command, checkpoint = sequence_openings(translate(sys.argv[1:], engines()))
    status = subprocess.call(command)
    if not status:
        advance_openings(checkpoint)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
