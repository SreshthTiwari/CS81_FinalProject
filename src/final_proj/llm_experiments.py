#!/usr/bin/env python3
"""Run 40 LLM full-pipeline experiments using the same simulator as test_llm_full_pipeline.py."""

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure the current script directory is on sys.path so top-level test modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_llm_full_pipeline
from test_llm_full_pipeline import FullPipelineSimulator


def main():
    parser = argparse.ArgumentParser(description="Run repeated LLM full-pipeline experiments")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for experiment reproducibility")
    parser.add_argument("--noise-rate", type=float, default=0.05, help="Base noise rate for the simulator")
    parser.add_argument("--timesteps", type=int, default=3, help="Number of temporal timesteps for each scenario")
    parser.add_argument("--trials-per-mode", type=int, default=10, help="Number of trials to run per mode")
    parser.add_argument("--no-window", action="store_true", help="Disable visualization windows")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON file for experiment results")
    parser.add_argument("--csv-output", type=str, default=None, help="Optional output CSV file for trial results")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"

    simulator = FullPipelineSimulator(
        map_path=str(map_path),
        noise_rate=args.noise_rate,
        num_timesteps=args.timesteps,
        seed=args.seed,
    )

    modes = simulator.MODES
    experiments = []

    for mode in modes:
        for trial_index in range(args.trials_per_mode):
            trial_id = len(experiments) + 1
            print(f"\n=== EXPERIMENT {trial_id}/{len(modes) * args.trials_per_mode}: mode={mode} trial={trial_index + 1} ===")
            result = simulator.run(mode=mode, show_window=not args.no_window)
            experiments.append({
                "trial_id": trial_id,
                "mode": mode,
                "seed": args.seed + trial_index,
                "success": result.get("success", False),
                "final_path_exists": result.get("final_path_exists", False),
                "history_length": len(result.get("history", [])),
            })

    summary = {
        "total_trials": len(experiments),
        "per_mode": {},
        "total_successes": sum(1 for e in experiments if e["success"]),
    }

    for mode in modes:
        mode_results = [e for e in experiments if e["mode"] == mode]
        success_count = sum(1 for e in mode_results if e["success"])
        summary["per_mode"][mode] = {
            "trials": len(mode_results),
            "success_count": success_count,
            "success_rate": success_count / len(mode_results) if mode_results else 0.0,
        }

    print(json.dumps(summary, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "experiments": experiments}, f, indent=2)

    if args.csv_output:
        with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["trial_id", "mode", "seed", "success", "final_path_exists", "history_length"],
            )
            writer.writeheader()
            writer.writerows(experiments)


if __name__ == "__main__":
    main()
