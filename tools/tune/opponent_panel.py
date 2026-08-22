"""Deterministic integer-weighted opponent panels for game-result tuners."""

import json
import pathlib
import re


ENGINE_FAILURE = re.compile(
    r"disconnects|not responsive|illegal move|stalls|crash|forfeit", re.IGNORECASE)


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
        member.setdefault("args", "")
        member.setdefault("options", {})
        names.add(member["name"])
    return members


def select(members, sequence):
    """Map a paired-opening sequence reproducibly onto the weighted panel."""
    offset = sequence % sum(member["weight"] for member in members)
    for member in members:
        if offset < member["weight"]:
            return member
        offset -= member["weight"]
    raise AssertionError("unreachable panel offset")
