#!/usr/bin/env python3
"""Freeze and bind tuner invocations to one cross-method training protocol."""

import argparse
import hashlib
import json
import pathlib


class ProtocolError(ValueError):
    """The manifest or effective runner configuration is inconsistent."""


def canonical_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(protocol):
    """Return a copy with all derived canonical identities refreshed."""
    value = json.loads(json.dumps(protocol))
    for start in value.get("starts", {}).values():
        start["options_canonical_sha256"] = canonical_digest(start["options"])
    for method in value.get("methods", {}).values():
        payload = {key: item for key, item in method.items() if key != "canonical_sha256"}
        method["canonical_sha256"] = canonical_digest(payload)
    payload = {key: item for key, item in value.items() if key != "canonical_sha256"}
    value["canonical_sha256"] = canonical_digest(payload)
    return value


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def _equal(actual, expected):
    return json.dumps(actual, sort_keys=True, separators=(",", ":")) == json.dumps(
        expected, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def _sealed(record, label):
    expected = record.get("canonical_sha256")
    payload = {key: value for key, value in record.items() if key != "canonical_sha256"}
    _require(expected == canonical_digest(payload), f"{label} canonical identity mismatch")


def _file_identity(path, expected, label):
    path = pathlib.Path(path)
    _require(path.is_file(), f"{label} does not exist: {path}")
    identity = {"sha256": file_digest(path)}
    _require(identity["sha256"] == expected.get("sha256"), f"{label} byte identity mismatch")
    if "canonical_sha256" in expected:
        try:
            value = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError(f"{label} is not canonical JSON") from error
        identity["canonical_sha256"] = canonical_digest(value)
        _require(
            identity["canonical_sha256"] == expected["canonical_sha256"],
            f"{label} canonical identity mismatch",
        )
    return identity


def _validate_training(training):
    paired = training.get("paired_observations")
    games = training.get("games")
    checkpoints = training.get("checkpoints")
    openings = training.get("openings", {})
    panel = training.get("panel", {})
    _require(isinstance(paired, int) and paired > 0, "paired-observation budget must be positive")
    _require(games == 2 * paired, "game budget must equal two games per paired observation")
    _require(
        isinstance(checkpoints, list) and checkpoints == sorted(set(checkpoints)),
        "accepted-game checkpoints must be sorted and unique",
    )
    _require(checkpoints and checkpoints[0] == 0 and checkpoints[-1] == games,
             "accepted-game checkpoints must include zero and the full budget")
    _require(all(isinstance(value, int) and value % 2 == 0 for value in checkpoints),
             "accepted-game checkpoints must end on complete pairs")
    _require(training.get("time_control"), "time control is required")
    _require(openings.get("order") == "sequential", "training openings must be sequential")
    _require(openings.get("cycle") is False, "training openings must not cycle")
    _require(isinstance(openings.get("first_line"), int) and openings["first_line"] > 0,
             "first opening line must be positive")
    _require(openings.get("artifact"), "training openings must name their artifact")
    _require(isinstance(panel.get("seed"), int), "opponent-panel seed must be an integer")
    _require(panel.get("artifact"), "opponent panel must name its artifact")
    _require(panel.get("schedule"), "opponent-panel schedule identity is required")


def _validate_method(method_id, method, protocol):
    _sealed(method, f"method {method_id}")
    per_update = method.get("paired_observations_per_update")
    paired = protocol["training"]["paired_observations"]
    _require(isinstance(per_update, int) and per_update > 0,
             f"method {method_id} needs a positive observation cost")
    _require(paired % per_update == 0,
             f"method {method_id} does not divide the observation budget")
    game_cost = 2 * per_update
    _require(all(checkpoint % game_cost == 0 for checkpoint in protocol["training"]["checkpoints"]),
             f"accepted-game checkpoints split a {method_id} optimizer update")
    _require(isinstance(method.get("settings"), dict),
             f"method {method_id} settings are required")
    _require(isinstance(method.get("implementation"), dict) and method["implementation"],
             f"method {method_id} implementation identity is required")
    crn = method.get("common_random_numbers")
    expected_crn = {"same_opening": True, "same_panel_member": True}
    if method_id == "spsa":
        _require(per_update == 2, "SPSA must spend two paired observations per update")
        _require(crn == expected_crn, "SPSA common-random-numbers policy must be explicit")
    else:
        _require(crn is None, f"method {method_id} must not share observations within an update")
    if method_id == "logistic_gp":
        settings = method["settings"]
        required = {
            "pairs": 1,
            "total_batches": paired,
            "duel_fraction": 0.0,
            "opening_order": "sequential",
            "cycle_openings": False,
            "panel_seed": protocol["training"]["panel"]["seed"],
        }
        _require(all(settings.get(key) == value for key, value in required.items()),
                 "logistic GP settings disagree with the common training protocol")


def load(path):
    """Load and internally verify a protocol manifest."""
    path = pathlib.Path(path)
    try:
        protocol = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read training protocol: {path}") from error
    _require(isinstance(protocol, dict), "training protocol must be a JSON object")
    _require(protocol.get("version") == 1, "unsupported training-protocol version")
    _require(isinstance(protocol.get("id"), str) and protocol["id"], "protocol id is required")
    _sealed(protocol, "protocol")
    _validate_training(protocol.get("training", {}))
    artifacts = protocol.get("artifacts")
    _require(isinstance(artifacts, dict) and artifacts, "protocol artifacts are required")
    for name, record in artifacts.items():
        _require(isinstance(record, dict) and set(record) <= {"sha256", "canonical_sha256"},
                 f"invalid artifact identity: {name}")
        _require(_digest(record.get("sha256")), f"artifact {name} needs a byte identity")
        _require("canonical_sha256" not in record or _digest(record["canonical_sha256"]),
                 f"artifact {name} has an invalid canonical identity")
    starts = protocol.get("starts")
    _require(isinstance(starts, dict) and starts, "protocol starts are required")
    for start_id, start in starts.items():
        _require(isinstance(start_id, str) and start_id, "start ids must be nonempty strings")
        _require(isinstance(start, dict), f"invalid start record: {start_id}")
        options = start.get("options")
        _require(isinstance(options, dict) and options, f"start {start_id} options are required")
        _require(start.get("options_canonical_sha256") == canonical_digest(options),
                 f"start {start_id} option identity mismatch")
        space = start.get("space", {})
        _require(set(space) == {"sha256", "canonical_sha256"},
                 f"start {start_id} needs exact space identities")
        _require(all(_digest(value) for value in space.values()),
                 f"start {start_id} has invalid space identities")
    methods = protocol.get("methods")
    _require(isinstance(methods, dict) and methods, "registered tuner methods are required")
    for method_id, method in methods.items():
        _require(isinstance(method_id, str) and method_id, "method ids must be nonempty strings")
        _require(isinstance(method, dict), f"invalid method record: {method_id}")
        _validate_method(method_id, method, protocol)
    return protocol


def bind(path, method_id, start_id, artifacts, space, training, settings):
    """Verify one effective runner invocation and return its portable identity."""
    path = pathlib.Path(path)
    protocol = load(path)
    _require(method_id in protocol["methods"], f"unregistered tuner method: {method_id}")
    _require(start_id in protocol["starts"], f"unregistered recovery start: {start_id}")
    _require(_equal(training, protocol["training"]), "runner training settings disagree with protocol")
    method = protocol["methods"][method_id]
    _require(_equal(settings, method["settings"]),
             f"runner settings disagree with method {method_id}")
    expected_artifacts = protocol["artifacts"]
    _require(set(artifacts) == set(expected_artifacts), "runner artifact set disagrees with protocol")
    identities = {
        name: _file_identity(artifacts[name], expected, f"artifact {name}")
        for name, expected in expected_artifacts.items()
    }
    start = protocol["starts"][start_id]
    space_identity = _file_identity(space, start["space"], f"start {start_id} space")
    space_value = json.loads(pathlib.Path(space).read_text())
    defaults = {
        parameter["name"]: parameter["default"]
        for parameter in space_value.get("parameters", ())
    }
    _require(_equal(defaults, start["options"]),
             f"start {start_id} options do not match the space defaults")
    openings = protocol["training"]["openings"]
    book = pathlib.Path(artifacts[openings["artifact"]])
    lines = book.read_text().splitlines()
    _require(all(line.strip() for line in lines), "training book must not contain blank lines")
    last = openings["first_line"] + protocol["training"]["paired_observations"] - 1
    _require(last <= len(lines),
             f"training needs unique opening line {last}, but the book has {len(lines)}")
    selected = lines[openings["first_line"] - 1:last]
    keys = [" ".join(line.split()[:4]) if len(line.split()) >= 4 else line.strip()
            for line in selected]
    _require(len(set(keys)) == len(keys), "training opening slice contains duplicate positions")
    panel_artifact = protocol["training"]["panel"]["artifact"]
    _require(panel_artifact in identities, "opponent-panel artifact is not pinned")
    protocol_payload = {
        key: value for key, value in protocol.items() if key != "canonical_sha256"
    }
    method_payload = {key: value for key, value in method.items() if key != "canonical_sha256"}
    return {
        "protocol": {
            "id": protocol["id"],
            "sha256": file_digest(path),
            "canonical_sha256": canonical_digest(protocol_payload),
        },
        "method": {"id": method_id, "canonical_sha256": canonical_digest(method_payload)},
        "start": {
            "id": start_id,
            "options_canonical_sha256": start["options_canonical_sha256"],
            "space": space_identity,
        },
        "artifacts": identities,
        "training": protocol["training"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refresh nested training-protocol identities")
    parser.add_argument("draft", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.draft.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(freeze(value), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
