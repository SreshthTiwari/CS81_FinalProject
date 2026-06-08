#!/usr/bin/env python3
"""Run a simple occupancy Kalman baseline on uncertain sensor noise and report success rate."""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from final_proj.environment.map_loader import MapLoader
from final_proj.environment.corruption import Corruptor
from final_proj.planning.astar import astar


class KalmanOccupancyFilter:
    def __init__(self, shape, original_grid, process_variance=0.01, measurement_variance=0.25):
        self.shape = shape
        self.original_grid = original_grid
        self.state = np.full(shape, 0.5, dtype=float)
        self.variance = np.full(shape, 1.0, dtype=float)
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def predict(self):
        self.variance += self.process_variance

    def update(self, measurement):
        mask = measurement != -1
        if not np.any(mask):
            return

        z = np.zeros(self.shape, dtype=float)
        z[measurement == 1] = 1.0
        z[measurement == 0] = 0.0

        p = self.variance[mask]
        kalman_gain = p / (p + self.measurement_variance)
        self.state[mask] = self.state[mask] + kalman_gain * (z[mask] - self.state[mask])
        self.variance[mask] = (1.0 - kalman_gain) * p

    def step(self, measurement):
        self.predict()
        self.update(measurement)

    def to_cost_map(self, latest_measurement):
        cost_map = np.full(self.shape, np.inf, dtype=float)
        blocked_mask = (latest_measurement == 1) | (self.original_grid == 1)
        free_mask = ~blocked_mask
        cost_map[free_mask] = 1.0 + 4.0 * (1.0 - self.state[free_mask])
        cost_map = np.clip(cost_map, 1.0, 5.0)
        return cost_map


class KalmanBaselineTestRunner:
    MODES = ["uncertain_sparse", "uncertain_clustered"]

    def __init__(self, map_path, trials=20, noise_rate=0.05, num_timesteps=3, seed=None, csv_path=None):
        self.map_loader = MapLoader(map_path)
        self.grid_original = self.map_loader.get_grid().copy()
        self.noise_rate = noise_rate
        self.num_timesteps = num_timesteps
        self.trials = trials
        self.seed = seed
        self.csv_path = csv_path
        self.corruptor = Corruptor(corruption_rate=noise_rate)

    def _seed(self, value):
        if value is not None:
            random.seed(value)
            np.random.seed(value)

    def _find_random_valid_cells(self):
        free_cells = np.where(self.grid_original == 0)
        options = list(zip(free_cells[1], free_cells[0]))
        if not options:
            raise RuntimeError("No free cells available on the map")
        start = tuple(int(v) for v in random.choice(options))
        goal = tuple(int(v) for v in random.choice(options))
        while goal == start:
            goal = tuple(int(v) for v in random.choice(options))
        return start, goal

    def _is_valid_path(self, path):
        if path is None:
            return False
        return all(self.grid_original[y, x] == 0 for x, y in path)

    def _generate_uncertainty_sequence(self, mode):
        if mode == "uncertain_sparse":
            return self.corruptor.inject_random_corruption(
                self.grid_original,
                corruption_rate=self.noise_rate,
                change_over_time=True,
                num_timesteps=self.num_timesteps
            )

        if mode == "uncertain_clustered":
            return self.corruptor.generate_clustered_uncertainty_sequence(
                self.grid_original,
                num_timesteps=self.num_timesteps,
                num_clusters=max(3, int(self.noise_rate * 20)),
                cluster_radius=2,
                jitter=1
            )

        raise ValueError(f"Unsupported mode: {mode}")

    def run_trial(self, trial_index, mode):
        self._seed(self.seed + trial_index if self.seed is not None else None)
        start, goal = self._find_random_valid_cells()

        sequence = self._generate_uncertainty_sequence(mode)

        kalman = KalmanOccupancyFilter(self.grid_original.shape, self.grid_original)
        for measurement in sequence:
            kalman.step(measurement)

        cost_map = kalman.to_cost_map(sequence[-1])
        estimated_path = astar(cost_map, start, goal)
        path_is_valid = self._is_valid_path(estimated_path)

        return {
            "trial": trial_index,
            "mode": mode,
            "seed": self.seed + trial_index if self.seed is not None else None,
            "start_x": start[0],
            "start_y": start[1],
            "goal_x": goal[0],
            "goal_y": goal[1],
            "path_found": estimated_path is not None,
            "path_valid": path_is_valid,
            "path_length": len(estimated_path) if estimated_path is not None else None,
        }

    def run(self):
        results = []
        half = self.trials // 2
        mode_sequence = ["uncertain_sparse"] * half + ["uncertain_clustered"] * (self.trials - half)

        for trial_index, mode in enumerate(mode_sequence):
            results.append(self.run_trial(trial_index, mode))

        success_count = sum(1 for r in results if r["path_found"] and r["path_valid"])

        if self.csv_path:
            self._write_csv(results)

        return success_count, results

    def _write_csv(self, results):
        if self.csv_path is None:
            return

        fieldnames = [
            "trial",
            "mode",
            "seed",
            "start_x",
            "start_y",
            "goal_x",
            "goal_y",
            "path_found",
            "path_valid",
            "path_length",
        ]
        with open(self.csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Run a Kalman baseline and save detailed results to CSV")
    parser.add_argument("--trials", type=int, default=20, help="Number of Kalman test trials")
    parser.add_argument("--noise-rate", type=float, default=0.05, help="Corruption noise rate for uncertain sparse observations")
    parser.add_argument("--timesteps", type=int, default=3, help="Number of noisy timesteps to integrate")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for reproducible trials")
    parser.add_argument("--csv", type=str, default="kalman_baseline_results.csv", help="CSV file to save detailed trial results")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"
    runner = KalmanBaselineTestRunner(
        map_path=str(map_path),
        trials=args.trials,
        noise_rate=args.noise_rate,
        num_timesteps=args.timesteps,
        seed=args.seed,
        csv_path=args.csv,
    )
    success_count, _ = runner.run()
    print(success_count)


if __name__ == "__main__":
    main()
