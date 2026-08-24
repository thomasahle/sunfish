#!/usr/bin/env python3
"""Audit a sealed Sunfish tuning consensus without consulting validation games."""

import argparse
import hashlib
import json
import pathlib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(root):
    manifest = root / "manifest-sha256.txt"
    seal = root / "SEALED"
    if not manifest.is_file() or not seal.is_file():
        raise RuntimeError(f"unsealed artifact: {root}")
    fields = dict(line.split(maxsplit=1) for line in seal.read_text().splitlines()[1:])
    if fields.get("manifest-sha256.txt") != digest(manifest):
        raise RuntimeError("seal does not bind manifest")
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"manifest mismatch: {relative}")


def domain(record):
    values = record.get("ordered_values", record.get("values"))
    if values is None and record["type"] == "integer":
        values = range(record["min"], record["max"] + 1, record.get("step", 1))
    if values is None:
        raise RuntimeError(f"unsupported parameter domain: {record['name']}")
    return list(values)


def canonical(options, space):
    defaults = {record["name"]: record["default"] for record in space["parameters"]}
    result = dict(options)
    for _ in range(len(space.get("conditions", [])) + 1):
        old = dict(result)
        for clause in space.get("conditions", []):
            if all(result[name] in values for name, values in clause["when"].items()):
                for name in clause.get("reset", []):
                    result[name] = defaults[name]
                result.update(clause.get("set", {}))
        if result == old:
            return result
    raise RuntimeError("space canonicalization did not converge")


def verify_eventual_space(parameters):
    domains = {name: domain(record) for name, record in parameters.items()}
    positive_reductions = ("NULL_CUT_RED", "NULL_RED", "IID_RED")
    if any(value < 1 for name in positive_reductions for value in domains[name]):
        raise RuntimeError("recursive probe reductions must remain positive")
    if any(value < 0 for value in domains["NULL_SPAN"]):
        raise RuntimeError("the scoring-null interval must remain finite")
    if any(value not in (0, 1, 2) for value in domains["FUEL_NULL"]):
        raise RuntimeError("fuel debt lies outside the proved bounded domain")
    if any(value not in (1, 2) for value in domains["LMR_RED"]):
        raise RuntimeError("LMR debt lies outside the proved bounded domain")


def verify_selection(candidate, parameters, space):
    names = list(parameters)
    domains = {name: domain(record) for name, record in parameters.items()}
    lanes = [canonical(options, space) for options in candidate["lane_optima"]]
    if len(lanes) != 20:
        raise RuntimeError("combined consensus must contain twenty lane optima")

    def coordinate(options):
        return tuple(domains[name].index(options[name]) / max(1, len(domains[name]) - 1)
                     for name in names)

    unique = {}
    for options in lanes:
        payload = json.dumps(options, sort_keys=True, separators=(",", ":"))
        unique.setdefault(payload, {"options": options, "support": 0})["support"] += 1
    points = [coordinate(options) for options in lanes]
    choices = []
    for payload, record in unique.items():
        point = coordinate(record["options"])
        distance = sum(sum(abs(a - b) for a, b in zip(point, other)) for other in points)
        choices.append((-record["support"], distance,
                        hashlib.sha256(payload.encode()).hexdigest(), record))
    _, distance, checksum, winner = min(choices)
    evidence = {"support": winner["support"], "total_normalized_l1": distance,
                "canonical_sha256": checksum}
    if candidate["selected"] != winner["options"]:
        raise RuntimeError("selected candidate is not the preregistered consensus")
    if candidate["selection_evidence"] != evidence:
        raise RuntimeError("selection evidence does not reproduce")


def mechanism_status(selected):
    limit = selected["NULL_LIMIT"]
    scoring_off = not limit or selected["NULL_SPAN"] == 0 or selected["NULL_MIN_DEPTH"] >= 30
    fuel_off = not limit or selected["FUEL_NULL"] == 0 or selected["FUEL_MIN_DEPTH"] >= 30
    return {
        "qsearch_off": selected["QS"] >= 3000,
        "scoring_null_off": scoring_off,
        "fuel_null_off": fuel_off,
        "all_null_off": scoring_off and fuel_off,
        "lmr_off": selected["LMR_LIMIT"] == 0 or selected["LMR_MIN_DEPTH"] >= 30,
        "caps_off": selected["FUT_CAP_DEPTH"] < 0,
        "iid_on": bool(selected["IID"]),
    }


def audit(root, space_path):
    root, space_path = pathlib.Path(root).resolve(), pathlib.Path(space_path)
    verify_seal(root)
    candidate = json.loads((root / "candidate.json").read_text())
    space = json.loads(space_path.read_text())
    parameters = {record["name"]: record for record in space["parameters"]}
    verify_eventual_space(parameters)
    verify_selection(candidate, parameters, space)
    selected = candidate["selected"]
    if set(selected) != set(parameters):
        raise RuntimeError("candidate and parameter-space axes differ")
    for name, value in selected.items():
        if value not in domain(parameters[name]):
            raise RuntimeError(f"out-of-domain value: {name}={value}")
    selected = canonical(selected, space)
    limit = selected["NULL_LIMIT"]
    compact = dict(selected)
    if not compact["IID"]:
        compact["IID_MIN_DEPTH"] = parameters["IID_MIN_DEPTH"]["default"]
        compact["IID_RED"] = parameters["IID_RED"]["default"]
    fuel = compact["FUEL_NULL"] if limit and compact["FUEL_MIN_DEPTH"] < 99 else 0
    lmr = (compact["LMR_RED"] if compact["LMR_LIMIT"]
           and compact["LMR_MIN_DEPTH"] < 99 else 0)
    defaults = {name: record["default"] for name, record in parameters.items()}
    return {
        "schema": "sunfish-consensus-prevalidation-audit-v1",
        "consensus": str(root),
        "consensus_seal_sha256": digest(root / "SEALED"),
        "literal": selected,
        "compact": compact,
        "literal_changes": {name: [defaults[name], value]
                            for name, value in selected.items() if value != defaults[name]},
        "compact_changes": {name: [defaults[name], value]
                            for name, value in compact.items() if value != defaults[name]},
        "mechanisms": mechanism_status(selected),
        "proof": {
            "maximum_real_edge_cost": 1 + fuel + lmr,
            "shallow_null_first_depth": selected["NULL_MIN_DEPTH"] + 1,
            "shallow_null_last_depth": selected["NULL_MIN_DEPTH"] + selected["NULL_SPAN"],
            "positive_null_cap_upper": selected["NULL_LIMIT"] - 1
                                       + selected["NULL_CAP_MARGIN"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("consensus", type=pathlib.Path)
    parser.add_argument("space", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.consensus, args.space), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
