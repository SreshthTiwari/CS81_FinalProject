#!/usr/bin/env python3
"""LLM targeted test runner.

Runs a fixed set of sparse-vs-clustered corruption tests and saves results to CSV.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from final_proj.environment.map_loader import MapLoader
from final_proj.environment.corruption import Corruptor
from final_proj.environment.context_extractor import ContextExtractor
from final_proj.planning.astar import astar
from final_proj.llm.prompt_builder import PromptBuilder
from final_proj.llm.response_parser import ResponseParser
from final_proj.llm.client import LLMClient
from final_proj.planning.replanner import Replanner


def build_llm_replanner():
    try:
        client = LLMClient()
        builder = PromptBuilder()
        parser = ResponseParser()
        extractor = ContextExtractor()
        replanner = Replanner(builder, parser, client, extractor)
        return replanner
    except Exception as e:
        print(f"[WARN] Failed to initialize LLM replanner: {e}")
        return None


def to_builtin(obj):
    if isinstance(obj, dict):
        return {to_builtin(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if obj is None:
        return None
    if hasattr(obj, "tolist"):
        return to_builtin(obj.tolist())
    return str(obj)


class LLMTargetedTestRunner:
    def __init__(self, map_path: str, sparse_tests: int = 5, clustered_tests: int = 5, noise_rate: float = 0.05):
        self.map_loader = MapLoader(map_path)
        self.grid_original = self.map_loader.get_grid().copy()
        self.height, self.width = self.grid_original.shape
        self.noise_rate = noise_rate
        self.replanner = build_llm_replanner()
        self.sparse_tests = sparse_tests
        self.clustered_tests = clustered_tests

    def find_random_valid_cells(self, num_pairs=1):
        free_cells = np.where(self.grid_original == 0)
        free_cells = list(zip(free_cells[1], free_cells[0]))

        pairs = []
        for _ in range(num_pairs):
            start = random.choice(free_cells)
            goal = random.choice(free_cells)
            attempts = 0
            while start == goal and attempts < 100:
                goal = random.choice(free_cells)
                attempts += 1
            if start != goal:
                pairs.append((start, goal))

        return pairs

    def make_cost_map(self, grid):
        cost_map = np.full(grid.shape, np.inf, dtype=float)
        cost_map[grid == 0] = 1.0
        cost_map[grid == -1] = 5.0
        return cost_map

    def generate_sparse_corruption(self, grid):
        corruptor = Corruptor(corruption_rate=self.noise_rate)
        return corruptor.inject_random_corruption(grid)

    def generate_clustered_corruption(self, grid, start, goal, radius=3):
        corrupted = grid.copy()
        path = astar(self.make_cost_map(grid), start, goal)
        if not path:
            return corrupted

        center = path[len(path) // 2]
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x = center[0] + dx
                y = center[1] + dy
                if 0 <= x < self.width and 0 <= y < self.height and corrupted[y, x] != 1:
                    corrupted[y, x] = -1
        return corrupted

    def classify_answer(self, decision):
        if not decision:
            return "no_decision"
        action = decision.get("recommended_action")
        if action == "plan_through":
            return "keep_moving"
        if action in ("avoid", "increase_cost"):
            return "replan"
        return "uncertain"

    def run_test(self, test_id, test_type, start, goal):
        grid = self.grid_original.copy()
        if test_type == "sparse":
            corrupted_grid = self.generate_sparse_corruption(grid)
            expected_answer = "keep_moving"
        else:
            corrupted_grid = self.generate_clustered_corruption(grid, start, goal)
            expected_answer = "replan"

        corrupted_map_cost = self.make_cost_map(corrupted_grid)
        corrupted_path = astar(corrupted_map_cost, start, goal)
        corrupted_exists = corrupted_path is not None
        corrupted_length = len(corrupted_path) if corrupted_exists else 0

        decision = None
        modified_path_exists = False
        modified_length = 0
        if self.replanner is not None:
            robot_pose = {"x": start[0], "y": start[1]}
            try:
                modified_grid, decision = self.replanner.replan(
                    corrupted_grid.copy(), start, goal, corrupted_path or [], robot_pose
                )
                if modified_grid is not None:
                    modified_cost = self.make_cost_map(modified_grid)
                    modified_path = astar(modified_cost, start, goal)
                    modified_path_exists = modified_path is not None
                    modified_length = len(modified_path) if modified_path_exists else 0
            except Exception as e:
                print(f"[WARN] Test {test_id} replanning failed: {e}")

        actual_answer = self.classify_answer(decision)
        success = actual_answer == expected_answer
        corrupted_cells = np.where(corrupted_grid != grid)
        corrupted_count = len(corrupted_cells[0])

        return {
            "test_id": test_id,
            "test_type": test_type,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "success": success,
            "start_x": start[0],
            "start_y": start[1],
            "goal_x": goal[0],
            "goal_y": goal[1],
            "corrupted_cells": corrupted_count,
            "corrupted_path_exists": corrupted_exists,
            "corrupted_path_length": corrupted_length,
            "modified_path_exists": modified_path_exists,
            "modified_path_length": modified_length,
            "decision_label": decision.get("label") if decision else "",
            "decision_confidence": decision.get("confidence") if decision else "",
            "decision_reason": decision.get("reason") if decision else "",
            "decision_action": decision.get("recommended_action") if decision else "",
        }

    def run_all(self, output_csv: str):
        results = []
        pairs = self.find_random_valid_cells(num_pairs=self.sparse_tests + self.clustered_tests)

        for i in range(self.sparse_tests):
            start, goal = pairs[i]
            results.append(self.run_test(i + 1, "sparse", start, goal))

        for i in range(self.clustered_tests):
            start, goal = pairs[self.sparse_tests + i]
            results.append(self.run_test(i + 1 + self.sparse_tests, "clustered", start, goal))

        success_count = sum(1 for row in results if row["success"])
        success_rate = success_count / len(results) if results else 0.0

        fieldnames = [
            "test_id",
            "test_type",
            "expected_answer",
            "actual_answer",
            "success",
            "start_x",
            "start_y",
            "goal_x",
            "goal_y",
            "corrupted_cells",
            "corrupted_path_exists",
            "corrupted_path_length",
            "modified_path_exists",
            "modified_path_length",
            "decision_label",
            "decision_confidence",
            "decision_reason",
            "decision_action"
        ]

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Wrote {len(results)} test rows to {output_csv}")
        print(f"Overall success rate: {success_rate:.2%} ({success_count}/{len(results)})")
        return results


def main():
    parser = argparse.ArgumentParser(description="Run targeted LLM corruption tests and save results to CSV")
    parser.add_argument("--output", default="llm_targeted_test_results.csv", help="CSV output file")
    parser.add_argument("--sparse", type=int, default=5, help="Number of sparse corruption tests")
    parser.add_argument("--clustered", type=int, default=5, help="Number of clustered corruption tests")
    parser.add_argument("--noise", type=float, default=0.05, help="Sparse corruption rate")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"

    runner = LLMTargetedTestRunner(
        map_path=str(map_path),
        sparse_tests=args.sparse,
        clustered_tests=args.clustered,
        noise_rate=args.noise,
    )
    runner.run_all(args.output)


if __name__ == "__main__":
    main()
