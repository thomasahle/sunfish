#!/usr/bin/env python3
"""Turn held-out optimizer validations into learning-curve CSV and plots."""

import argparse
import csv
import itertools
import json
import math
import pathlib
import statistics


def rows(paths):
    output = []
    for path in paths:
        payload = json.loads(pathlib.Path(path).read_text())
        protocol = payload.get("protocol", {})
        opening = protocol.get("openings", {})
        protocol_identity = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
        for record in payload["records"]:
            match = payload["matches"][record["validation"]]
            output.append(record | {
                "elo": match["elo"], "error": match["error"],
                "wins": match["wins"], "draws": match["draws"],
                "losses": match["losses"], "pair_scores": match.get("pair_scores"),
                "validation_start": protocol.get("start"),
                "validation_pairs": protocol.get("pairs"),
                "validation_openings": opening.get("sha256"),
                "validation_protocol": protocol_identity if protocol else None,
            })
    return output


def logistic_elo(score):
    """Convert an expected score against the reference to logistic Elo."""
    import numpy as np

    score = np.clip(score, 1e-9, 1 - 1e-9)
    return 400 / math.log(10) * np.log(score / (1 - score))


def paired_comparisons(records, checkpoint=1000, replicates=100000, seed=20260822,
                       expected_starts=(5, 15, 23)):
    """Directly compare methods using shared, start-stratified bootstrap draws."""
    import numpy as np

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    indexed = {}
    for record in records:
        key = record["method"], record.get("start"), record["checkpoint"]
        if key in indexed:
            raise ValueError(f"duplicate recovery record {key}")
        indexed[key] = record
    methods = sorted({method for method, _, _ in indexed})
    if len(methods) < 2:
        raise ValueError(f"checkpoint {checkpoint} needs at least two methods")
    method_starts = {
        method: {start for candidate, start, point in indexed
                 if candidate == method and point == checkpoint}
        for method in methods
    }
    expected = (set(expected_starts) if expected_starts is not None
                else next((value for value in method_starts.values() if value), set()))
    if None in expected or not expected:
        raise ValueError(f"checkpoint {checkpoint} has missing start labels")
    for method in methods:
        if method_starts[method] != expected:
            raise ValueError(f"checkpoint {checkpoint} has misaligned starts for {method}")
    starts = sorted(expected, key=str)

    scores, initial, slices, counts = {}, {}, set(), set()
    for start in starts:
        baselines, baseline_ids = [], []
        for method in methods:
            record = indexed[method, start, checkpoint]
            baseline = indexed.get((method, start, 0))
            if baseline is None:
                raise ValueError(f"missing checkpoint 0 for {method}, start {start}")
            for item in (record, baseline):
                vector = item.get("pair_scores")
                if not vector:
                    raise ValueError(f"missing pair scores for {method}, start {start}")
                if any(value not in (0, .25, .5, .75, 1) for value in vector):
                    raise ValueError(f"invalid pair score for {method}, start {start}")
                declared = item.get("validation_pairs")
                if declared is not None and declared != len(vector):
                    raise ValueError(f"declared pair count disagrees for {method}, start {start}")
                identity = tuple(item.get(name) for name in (
                    "validation_start", "validation_pairs", "validation_openings",
                    "validation_protocol"))
                slices.add(identity)
                if any(value is None for value in identity):
                    raise ValueError(f"incomplete validation identity for {method}, start {start}")
            scores[method, start] = np.asarray(record["pair_scores"], dtype=float)
            baselines.append(tuple(baseline["pair_scores"]))
            baseline_ids.append(baseline.get("validation"))
            counts.add(len(record["pair_scores"]))
            counts.add(len(baseline["pair_scores"]))
        if len(set(baselines)) != 1:
            raise ValueError(f"checkpoint 0 scores disagree at start {start}")
        if None in baseline_ids or len(set(baseline_ids)) != 1:
            raise ValueError(f"checkpoint 0 aliases disagree at start {start}")
        initial[start] = np.asarray(baselines[0], dtype=float)
    if len(counts) != 1:
        raise ValueError("misaligned opening-pair counts")
    if len(slices) > 1:
        raise ValueError("validation records use different opening slices")

    observed = np.empty((len(starts), len(methods)))
    raw = np.empty_like(observed)
    for s, start in enumerate(starts):
        base = logistic_elo(initial[start].mean())
        for m, method in enumerate(methods):
            observed[s, m] = logistic_elo(scores[method, start].mean()) - base
            raw[s, m] = np.mean(scores[method, start] - initial[start])

    rng = np.random.default_rng(seed)
    recovery = np.zeros((replicates, len(methods)))
    pair_count = counts.pop()
    for offset in range(0, replicates, 4096):
        stop = min(offset + 4096, replicates)
        for start in starts:
            sample = rng.integers(pair_count, size=(stop - offset, pair_count))
            base = logistic_elo(initial[start][sample].mean(axis=1))
            for m, method in enumerate(methods):
                recovery[offset:stop, m] += (
                    logistic_elo(scores[method, start][sample].mean(axis=1)) - base)
    recovery /= len(starts)

    output = []
    for a, b in itertools.combinations(range(len(methods)), 2):
        difference = recovery[:, a] - recovery[:, b]
        low, high = np.percentile(difference, [2.5, 97.5])
        output.append({
            "checkpoint": checkpoint, "method_a": methods[a], "method_b": methods[b],
            "starts": len(starts), "pairs_per_start": pair_count,
            "replicates": replicates, "seed": seed,
            "recovery_a": observed[:, a].mean(),
            "recovery_b": observed[:, b].mean(),
            "elo_difference": observed[:, a].mean() - observed[:, b].mean(),
            "ci_low": low, "ci_high": high,
            "score_difference": raw[:, a].mean() - raw[:, b].mean(),
        })
    return output


def paired_error(candidate, baseline):
    if not candidate or len(candidate) != len(baseline) or len(candidate) < 2:
        return None
    candidate_mean, baseline_mean = statistics.mean(candidate), statistics.mean(baseline)
    scale = 400 / math.log(10)
    candidate_gradient = scale / max(1e-9, candidate_mean * (1 - candidate_mean))
    baseline_gradient = scale / max(1e-9, baseline_mean * (1 - baseline_mean))
    influence = [
        candidate_gradient * (value - candidate_mean)
        - baseline_gradient * (start - baseline_mean)
        for value, start in zip(candidate, baseline)
    ]
    return 1.96 * statistics.stdev(influence) / math.sqrt(len(influence))


def add_gains(records):
    starts = {
        (record["method"], record.get("start")): record
        for record in records if record["checkpoint"] == 0
    }
    for record in records:
        start = starts[(record["method"], record.get("start"))]
        record["gain"] = record["elo"] - start["elo"]
        if record["validation"] == start["validation"]:
            record["gain_error"] = 0
            continue
        error = paired_error(record.get("pair_scores"), start.get("pair_scores"))
        record["gain_error"] = error if error is not None else math.hypot(
            record["error"], start["error"])
    return records


def aggregate(group, value, error):
    values = [record[value] for record in group]
    within = sum((record[error] / 1.96) ** 2 for record in group) / len(group) ** 2
    between = statistics.variance(values) / len(group) if len(group) > 1 else 0
    return statistics.mean(values), 1.96 * math.sqrt(within + between)


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault((record["method"], record["checkpoint"]), []).append(record)
    output = []
    for (method, checkpoint), group in sorted(groups.items()):
        elo, error = aggregate(group, "elo", "error")
        row = {
            "method": method, "checkpoint": checkpoint,
            "trained_games": statistics.mean(record["trained_games"] for record in group),
            "starts": len(group), "elo": elo, "error": error,
        }
        if all("gain" in record for record in group):
            row["gain"], row["gain_error"] = aggregate(group, "gain", "gain_error")
        output.append(row)
    return output


def write_csv(path, records, fields):
    with pathlib.Path(path).open("w", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def plot(path, raw, summary, title, value, error, ylabel):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8, 5))
    methods = sorted({record["method"] for record in summary})
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {method: palette[index % len(palette)] for index, method in enumerate(methods)}
    for method in methods:
        starts = {record.get("start") for record in raw if record["method"] == method}
        for start in sorted(starts, key=str):
            line = sorted(
                (record for record in raw
                 if record["method"] == method and record.get("start") == start),
                key=lambda record: record["trained_games"],
            )
            axis.plot(
                [record["trained_games"] for record in line],
                [record[value] for record in line],
                color=colors[method], linewidth=1, alpha=.2,
            )
        line = sorted(
            (record for record in summary if record["method"] == method),
            key=lambda record: record["trained_games"],
        )
        axis.errorbar(
            [record["trained_games"] for record in line],
            [record[value] for record in line],
            yerr=[record[error] for record in line],
            color=colors[method], marker="o", capsize=3, label=method,
        )
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="Training games used", ylabel=ylabel, title=title)
    axis.grid(alpha=.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", action="append", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--title", default="Optimizer recovery from degraded parameters")
    parser.add_argument("--primary-checkpoint", type=int, default=1000)
    parser.add_argument("--bootstrap-replicates", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    parser.add_argument("--recovery-start", type=int, action="append")
    args = parser.parse_args()
    raw = add_gains(rows(args.validation))
    summary = summarize(raw)
    comparisons = paired_comparisons(
        raw, args.primary_checkpoint, args.bootstrap_replicates, args.bootstrap_seed,
        args.recovery_start or (5, 15, 23))
    prefix = pathlib.Path(args.output_prefix)
    write_csv(prefix.with_suffix(".raw.csv"), raw, [
        "method", "start", "checkpoint", "trained_games", "recommendation_games",
        "elo", "error", "gain", "gain_error", "wins", "draws", "losses", "validation",
    ])
    write_csv(prefix.with_suffix(".csv"), summary, [
        "method", "checkpoint", "trained_games", "starts", "elo", "error", "gain", "gain_error",
    ])
    write_csv(prefix.with_suffix(".paired.csv"), comparisons, [
        "checkpoint", "method_a", "method_b", "starts", "pairs_per_start", "replicates", "seed",
        "recovery_a", "recovery_b", "elo_difference", "ci_low", "ci_high", "score_difference",
    ])
    plot(prefix.with_suffix(".svg"), raw, summary, args.title, "gain", "gain_error",
         "Held-out Elo recovered from start")
    plot(prefix.with_suffix(".absolute.svg"), raw, summary, args.title, "elo", "error",
         "Held-out Elo vs baseline")


if __name__ == "__main__":
    main()
