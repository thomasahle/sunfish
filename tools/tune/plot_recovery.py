#!/usr/bin/env python3
"""Turn held-out optimizer validations into learning-curve CSV and plots."""

import argparse
import csv
import json
import math
import pathlib
import statistics


def rows(paths):
    output = []
    for path in paths:
        payload = json.loads(pathlib.Path(path).read_text())
        for record in payload["records"]:
            match = payload["matches"][record["validation"]]
            output.append(record | {
                "elo": match["elo"], "error": match["error"],
                "wins": match["wins"], "draws": match["draws"],
                "losses": match["losses"], "pair_scores": match.get("pair_scores"),
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
    args = parser.parse_args()
    raw = add_gains(rows(args.validation))
    summary = summarize(raw)
    prefix = pathlib.Path(args.output_prefix)
    write_csv(prefix.with_suffix(".raw.csv"), raw, [
        "method", "start", "checkpoint", "trained_games", "recommendation_games",
        "elo", "error", "gain", "gain_error", "wins", "draws", "losses", "validation",
    ])
    write_csv(prefix.with_suffix(".csv"), summary, [
        "method", "checkpoint", "trained_games", "starts", "elo", "error", "gain", "gain_error",
    ])
    plot(prefix.with_suffix(".svg"), raw, summary, args.title, "gain", "gain_error",
         "Held-out Elo recovered from start")
    plot(prefix.with_suffix(".absolute.svg"), raw, summary, args.title, "elo", "error",
         "Held-out Elo vs baseline")


if __name__ == "__main__":
    main()
