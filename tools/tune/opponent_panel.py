"""Deterministic integer-weighted opponent panels for game-result tuners."""

import json
import pathlib
import random
import re


ENGINE_FAILURE = re.compile(
    r"not responsive|illegal move|"
    r"\b(?:disconnect(?:ed|s)?|stall(?:ed|s)?|crash(?:ed|es)?|forfeit(?:ed|s)?)\b"
    r"(?!\s*[:=]\s*0\b)", re.IGNORECASE)
IDENTITY_FIELDS = ("source", "revision", "license")


def failure(text):
    """Return the first engine-failure marker hidden by fastchess recovery."""
    match = ENGINE_FAILURE.search(text)
    return match.group(0) if match else None


def load(path):
    members = json.loads(pathlib.Path(path).read_text())
    if not isinstance(members, list) or not members:
        raise ValueError("opponent panel must be a nonempty JSON list")
    names = set()
    for member in members:
        required = {"name", "engine", "weight"}
        if not isinstance(member, dict) or not required <= member.keys():
            raise ValueError("each panel member needs name, engine, and weight")
        if member["name"] in names or not isinstance(member["name"], str):
            raise ValueError("panel names must be unique strings")
        if not isinstance(member["weight"], int) or member["weight"] < 1:
            raise ValueError("panel weights must be positive integers")
        if not isinstance(member.get("options", {}), (dict, str)):
            raise ValueError("panel options must be an object or 'default'")
        if isinstance(member.get("options"), str) and member["options"] != "default":
            raise ValueError("the only panel options string is 'default'")
        if any(not isinstance(member.get(field), str) or not member[field]
               for field in IDENTITY_FIELDS if field in member):
            raise ValueError("panel identity fields must be nonempty strings")
        files = member.setdefault("identity_files", [])
        if not isinstance(files, list) or any(not isinstance(path, str) or not path for path in files):
            raise ValueError("panel identity_files must be a list of paths")
        member.setdefault("args", "")
        member.setdefault("options", {})
        names.add(member["name"])
    return members


def identity(member, engine, file_digest):
    """Pin one member's executable, configuration, provenance, and extra files."""
    result = {
        "name": member["name"], "weight": member["weight"],
        "engine": engine(member["engine"], member["args"], member["options"]),
        "identity_files": [file_digest(path) for path in member.get("identity_files", [])],
    }
    result.update({field: member[field] for field in IDENTITY_FIELDS if field in member})
    return result


def select(members, sequence, seed=2026):
    """Choose from a reproducibly shuffled block with the exact panel weights."""
    if sequence < 1:
        raise ValueError("panel sequence must be one-based")
    block = [member for member in members for _ in range(member["weight"])]
    epoch, offset = divmod(sequence - 1, len(block))
    random.Random(f"{seed}:{epoch}").shuffle(block)
    return block[offset]
