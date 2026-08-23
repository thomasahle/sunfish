#!/usr/bin/env python3
"""Select recovery finalists and make a conservative confirmation decision."""

import argparse
import json
import pathlib
import sys

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).parent))
import freeze_recommendations as freezer  # noqa: E402
import plot_recovery  # noqa: E402


PAIR_SCORES = {0, .25, .5, .75, 1}
digest, save = freezer.digest, freezer.save


def study(benchmark_path, analysis_path):
    benchmark_path = pathlib.Path(benchmark_path).resolve()
    analysis_path = pathlib.Path(analysis_path).resolve()
    benchmark = json.loads(benchmark_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    benchmark_sha = digest(benchmark_path)
    if analysis["training_benchmark"]["sha256"] != benchmark_sha:
        raise ValueError("analysis manifest names a different training benchmark")
    space_path = freezer.root_for(benchmark_path) / benchmark["space"]["path"]
    if digest(space_path) != benchmark["space"]["sha256"]:
        raise ValueError("parameter-space hash disagrees with the benchmark")
    parameters = json.loads(space_path.read_text())["parameters"]
    reference = freezer.normalize(parameters, benchmark["reference"]["options"], {})
    starts = {
        item["source_index"]: freezer.normalize(parameters, reference, item["options"])
        for item in benchmark["starts"]
    }
    fingerprint = {
        "analysis_sha256": digest(analysis_path),
        "training_benchmark_sha256": benchmark_sha,
        "method_source_commit": benchmark["method_source_commit"],
    }
    return benchmark, analysis, parameters, reference, starts, fingerprint


def identity_hashes(identity):
    return sorted(item["sha256"] for item in identity.get("files", []))


def verify_match(match, pairs):
    scores = match.get("pair_scores")
    if not isinstance(scores, list) or len(scores) != pairs or any(score not in PAIR_SCORES for score in scores):
        raise ValueError("validation match has invalid or incomplete pair scores")
    counts = match.get("pentanomial")
    if not isinstance(counts, list) or len(counts) != 5 or sum(counts) != pairs:
        raise ValueError("validation match has invalid pentanomial counts")
    if counts != [scores.count(value) for value in (1, .75, .5, .25, 0)]:
        raise ValueError("validation pair scores disagree with its pentanomial counts")
    wins, losses, draws = (match.get(name) for name in ("wins", "losses", "draws"))
    if not all(isinstance(value, int) and value >= 0 for value in (wins, losses, draws)):
        raise ValueError("validation match has invalid game counts")
    if wins + losses + draws != 2 * pairs or abs(2 * sum(scores) - wins - draws / 2) > 1e-9:
        raise ValueError("validation match scores disagree with its game counts")


def verify_protocol(protocol, benchmark, phase, recommendation_sha):
    expected = benchmark["artifacts"]
    if protocol.get("tc") != benchmark["budget"]["time_control"]:
        raise ValueError("validation time control disagrees with the benchmark")
    for name, key in (("pairs", "pairs"), ("start", "start")):
        if protocol.get(name) != phase[key]:
            raise ValueError(f"validation {name} disagrees with the analysis manifest")
    if protocol.get("opening_slice_sha256") != phase["slice_sha256"]:
        raise ValueError("validation opening slice disagrees with the analysis manifest")
    if protocol.get("opening_format") != "epd":
        raise ValueError("validation opening format is not EPD")
    if protocol.get("openings", {}).get("sha256") != expected["heldout_book"]:
        raise ValueError("validation used the wrong held-out opening book")
    if protocol.get("fastchess", {}).get("sha256") != expected["fastchess"]:
        raise ValueError("validation used the wrong fastchess artifact")
    if protocol.get("recommendations", {}).get("sha256") != recommendation_sha:
        raise ValueError("validation used a different recommendation artifact")
    candidate, baseline = protocol.get("candidate", {}), protocol.get("baseline", {})
    engine_files = sorted((expected["engine"], expected["tables"]))
    if identity_hashes(candidate) != engine_files or identity_hashes(baseline) != engine_files:
        raise ValueError("validation used the wrong engine or evaluation tables")
    if candidate.get("arguments") != baseline.get("arguments"):
        raise ValueError("candidate and baseline used different engine arguments")
    baseline_options = {name: str(value) for name, value in benchmark["reference"]["options"].items()}
    if protocol.get("baseline_options") != baseline_options:
        raise ValueError("validation baseline options disagree with the benchmark")


def verify_validation(path, benchmark, phase, parameters, reference, starts, fingerprint,
                      methods, checkpoints, recommendation_sha):
    payload = json.loads(pathlib.Path(path).read_text())
    verify_protocol(payload.get("protocol", {}), benchmark, phase, recommendation_sha)
    records = payload.get("records", [])
    expected = {(method, start, checkpoint) for method in methods for start in starts
                for checkpoint in checkpoints}
    indexed = {}
    for record in records:
        key = record.get("method"), record.get("start"), record.get("checkpoint")
        if key in indexed or key not in expected:
            raise ValueError(f"unexpected or duplicate validation record {key}")
        options = freezer.normalize(parameters, reference, record.get("options", {}))
        config = freezer.identifier(options)
        if record.get("configuration") != config or record.get("validation") != config:
            raise ValueError(f"validation record {key} has the wrong configuration identity")
        if record.get("benchmark_protocol") != fingerprint:
            raise ValueError(f"validation record {key} has the wrong benchmark fingerprint")
        if key[2] == 0 and options != starts[key[1]]:
            raise ValueError(f"validation record {key} has the wrong degraded start")
        if config not in payload.get("matches", {}):
            raise ValueError(f"validation record {key} has no completed match")
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError("validation is missing required method/start/checkpoint records")
    configurations = {record["validation"] for record in records}
    if set(payload.get("matches", {})) != configurations:
        raise ValueError("validation contains missing or unrelated matches")
    for match in payload["matches"].values():
        verify_match(match, phase["pairs"])
    return payload, indexed


def verify_recommendations(payload, path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError("recommendation artifact hash disagrees with its audit")
    records = json.loads(pathlib.Path(path).read_text())
    for record in records:
        record["validation"] = freezer.identifier(record["options"])
    if payload["records"] != records:
        raise ValueError("validation records disagree with the recommendation artifact")


def contrast(comparisons, candidate, reference):
    for row in comparisons:
        if (row["method_a"], row["method_b"]) == (candidate, reference):
            return row["ci_low"], row["ci_high"]
        if (row["method_a"], row["method_b"]) == (reference, candidate):
            return -row["ci_high"], -row["ci_low"]
    raise ValueError(f"missing primary contrast for {candidate} and {reference}")


def select_finalists(recoveries, comparisons):
    ranking = sorted(recoveries, key=lambda method: (-recoveries[method], method))
    if len(ranking) < 2:
        raise ValueError("primary selection needs at least two methods")
    leader, runner_up = ranking[:2]
    finalists = {leader, runner_up}
    for method in ranking[2:]:
        low, high = contrast(comparisons, method, runner_up)
        if low <= 0 <= high:
            finalists.add(method)
    return ranking, leader, runner_up, sorted(finalists)


def primary(args):
    benchmark, analysis, parameters, reference, starts, fingerprint = study(
        args.benchmark, args.analysis)
    audit = json.loads(pathlib.Path(args.audit).read_text())
    if audit.get("benchmark_protocol") != fingerprint:
        raise ValueError("recommendation audit has the wrong benchmark fingerprint")
    recommendation_sha = audit.get("recommendations_sha256")
    methods = sorted(analysis["method_labels"].values())
    phase, checkpoint = analysis["primary"], analysis["primary"]["checkpoint"]
    payload, indexed = verify_validation(
        args.validation, benchmark, phase, parameters, reference, starts, fingerprint,
        methods, benchmark["budget"]["checkpoints"], recommendation_sha)
    verify_recommendations(payload, args.recommendations, recommendation_sha)
    if audit.get("aliases") != len(payload["records"]):
        raise ValueError("recommendation audit has the wrong alias count")
    artifact_rows = audit.get("recommendation_artifacts", [])
    artifacts = {(item["method"], item["start"]): item["sha256"] for item in artifact_rows}
    expected_artifacts = {(method, start) for method in methods for start in starts}
    if len(artifact_rows) != len(artifacts) or set(artifacts) != expected_artifacts:
        raise ValueError("recommendation audit has the wrong source-artifact grid")
    if any(record.get("recommendation_artifact_sha256")
           != artifacts[record["method"], record["start"]] for record in payload["records"]):
        raise ValueError("recommendation alias has the wrong source-artifact hash")
    configurations = {record["configuration"]: record["options"] for record in payload["records"]}
    if audit.get("configurations") != configurations:
        raise ValueError("recommendation audit has different normalized configurations")
    raw = plot_recovery.add_gains(plot_recovery.rows([args.validation]))
    comparisons = plot_recovery.paired_comparisons(
        raw, checkpoint, phase["bootstrap_replicates"], phase["bootstrap_seed"], starts)
    recoveries = {}
    for row in comparisons:
        recoveries[row["method_a"]] = row["recovery_a"]
        recoveries[row["method_b"]] = row["recovery_b"]
    ranking, leader, runner_up, finalists = select_finalists(recoveries, comparisons)
    selected = []
    for method in finalists:
        for start in sorted(starts):
            for point in (0, checkpoint):
                record = dict(indexed[method, start, point])
                record.pop("validation", None)
                selected.append(record)
    selected.sort(key=lambda record: (record["method"], record["start"], record["checkpoint"]))
    save(args.confirmation_recommendations, selected)
    result = {
        "version": 1, "status": "confirmation-required", "benchmark_protocol": fingerprint,
        "selection_rule": phase["finalists"], "leader": leader, "runner_up": runner_up,
        "finalists": finalists,
        "ranking": [{"method": method, "recovery_elo": recoveries[method]} for method in ranking],
        "comparisons": comparisons,
        "sources": {
            "recommendation_audit_sha256": digest(args.audit),
            "primary_validation_sha256": digest(args.validation),
            "confirmation_recommendations_sha256": digest(args.confirmation_recommendations),
        },
    }
    save(args.output, result)


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    scaled = np.rint(4 * values).astype(int)
    if np.any(np.abs(4 * values - scaled) > 1e-9):
        raise ValueError("sign-flip differences are not quarter-integers")
    scaled = scaled[scaled != 0]
    observed = int(scaled.sum())
    distribution = np.ones(1, dtype=np.longdouble)
    offset = 0
    for magnitude in np.abs(scaled):
        following = np.zeros(len(distribution) + 2 * magnitude, dtype=np.longdouble)
        following[:len(distribution)] += distribution / 2
        following[2 * magnitude:] += distribution / 2
        distribution = following
        offset += int(magnitude)
    tail = distribution[max(0, observed + offset):].sum()
    return np.longdouble(tail / distribution.sum())


def holm(hypotheses):
    count = len(hypotheses)
    running = np.longdouble(0)
    for rank, hypothesis in enumerate(sorted(hypotheses, key=lambda item: (item["_p"], item["name"]))):
        running = max(running, min(np.longdouble(1), (count - rank) * hypothesis["_p"]))
        hypothesis["adjusted_p_value"] = float(running)
    for hypothesis in hypotheses:
        hypothesis["p_value"] = float(hypothesis.pop("_p"))
    return hypotheses


def confirmation_tests(scores):
    matrices = {method: np.atleast_2d(values) for method, values in scores.items()}
    shapes = {values.shape for values in matrices.values()}
    if len(shapes) != 1:
        raise ValueError("confirmation scores do not share a start/opening grid")
    start_count, _ = shapes.pop()
    hypotheses = []
    for method in sorted(matrices):
        values = matrices[method]
        hypotheses.append({
            "name": f"{method}>zero", "kind": "versus-zero", "method": method,
            "score_difference": float(np.mean(values) - .5),
            "elo_difference": float(plot_recovery.logistic_elo(np.mean(values))),
            "_p": exact_sign_flip(np.sum(values, axis=0) - start_count / 2),
        })
    for candidate in sorted(matrices):
        for rival in sorted(matrices):
            if candidate == rival:
                continue
            difference = matrices[candidate] - matrices[rival]
            hypotheses.append({
                "name": f"{candidate}>{rival}", "kind": "pairwise",
                "method": candidate, "rival": rival,
                "score_difference": float(np.mean(difference)),
                "elo_difference": float(
                    plot_recovery.logistic_elo(np.mean(matrices[candidate]))
                    - plot_recovery.logistic_elo(np.mean(matrices[rival]))),
                "_p": exact_sign_flip(np.sum(difference, axis=0)),
            })
    return holm(hypotheses)


def confirmation(args):
    benchmark, analysis, parameters, reference, starts, fingerprint = study(
        args.benchmark, args.analysis)
    selection = json.loads(pathlib.Path(args.selection).read_text())
    if selection.get("benchmark_protocol") != fingerprint:
        raise ValueError("primary selection has the wrong benchmark fingerprint")
    finalists = selection["finalists"]
    recommendation_sha = selection["sources"]["confirmation_recommendations_sha256"]
    phase, checkpoint = analysis["confirmation"], analysis["primary"]["checkpoint"]
    payload, indexed = verify_validation(
        args.validation, benchmark, phase, parameters, reference, starts, fingerprint,
        finalists, (0, checkpoint), recommendation_sha)
    verify_recommendations(payload, args.recommendations, recommendation_sha)
    scores, estimates = {}, {}
    for method in finalists:
        values, master_elo, recovered_elo = [], [], []
        for start in sorted(starts):
            record = indexed[method, start, checkpoint]
            initial = indexed[method, start, 0]
            final_scores = payload["matches"][record["validation"]]["pair_scores"]
            initial_scores = payload["matches"][initial["validation"]]["pair_scores"]
            final_elo = float(plot_recovery.logistic_elo(np.mean(final_scores)))
            master_elo.append(final_elo)
            recovered_elo.append(
                final_elo - float(plot_recovery.logistic_elo(np.mean(initial_scores))))
            values += final_scores
        scores[method] = np.asarray(values, dtype=float).reshape(len(starts), phase["pairs"])
        estimates[method] = {
            "score": float(np.mean(values)), "elo_vs_master": float(np.mean(master_elo)),
            "recovery_elo": float(np.mean(recovered_elo)),
        }
    hypotheses = confirmation_tests(scores)
    alpha = phase["alpha"]
    winners = []
    for method in finalists:
        required = [item for item in hypotheses if item["method"] == method]
        if all(item["score_difference"] > 0 and item["adjusted_p_value"] <= alpha
               for item in required):
            winners.append(method)
    result = {
        "version": 1, "status": "winner" if len(winners) == 1 else "inconclusive",
        "winner": winners[0] if len(winners) == 1 else None,
        "benchmark_protocol": fingerprint, "finalists": finalists,
        "estimates": estimates,
        "method": phase["test"], "alternative": phase["alternative"],
        "correction": phase["correction"], "family": phase["family"],
        "alpha": alpha, "hypotheses": hypotheses,
        "sources": {
            "primary_selection_sha256": digest(args.selection),
            "confirmation_validation_sha256": digest(args.validation),
        },
    }
    save(args.output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--analysis", required=True)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    first = subparsers.add_parser("primary")
    first.add_argument("--validation", required=True)
    first.add_argument("--recommendations", required=True)
    first.add_argument("--audit", required=True)
    first.add_argument("--output", required=True)
    first.add_argument("--confirmation-recommendations", required=True)
    first.set_defaults(run=primary)
    final = subparsers.add_parser("confirmation")
    final.add_argument("--validation", required=True)
    final.add_argument("--recommendations", required=True)
    final.add_argument("--selection", required=True)
    final.add_argument("--output", required=True)
    final.set_defaults(run=confirmation)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
