#!/usr/bin/env python3
"""Pool SPSA perturbation matches for offline preference modelling."""

import argparse
import json
import pathlib


def pool(paths, results_per_state=None):
    batches = []
    gates = {}
    sources = []
    for path in map(pathlib.Path, paths):
        state = json.loads(path.read_text())
        results = state["results"][:results_per_state]
        sources.append({
            "path": str(path.resolve()),
            "available_results": len(state["results"]),
            "used_results": len(results),
        })
        gates.update(state.get("gates", {}))
        for result in results:
            if result.get("opponent"):
                for side in ("plus", "minus"):
                    wins, losses, draws = result[f"{side}_result"]
                    batches.append({
                        "knobs": result[side], "opponent_knobs": None,
                        "wins": wins, "draws": draws, "losses": losses,
                        "opening": result["opening"], "baseline_ids": [result["opponent"]],
                        "allocation": "spsa-panel-perturbation",
                    })
            else:
                batches.append({
                    "knobs": result["plus"], "opponent_knobs": result["minus"],
                    "wins": result["wins"], "draws": result["draws"],
                    "losses": result["losses"], "opening": result["opening"],
                    "allocation": "spsa-perturbation",
                })
    return {
        "next_opening": 1,
        "study": {"allocation": {"gate_all": False}, "sources": sources},
        "batches": batches,
        "gates": gates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("states", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--results-per-state", type=int)
    args = parser.parse_args()
    if args.results_per_state is not None and args.results_per_state < 1:
        parser.error("results per state must be positive")
    output = pathlib.Path(args.output)
    output.write_text(json.dumps(
        pool(args.states, args.results_per_state), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
