#!/usr/bin/env python3
"""Dynamic obstacle and uncertainty test runner for LLM-assisted navigation."""

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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


class LLMDynamicObstacleTestRunner:
    def __init__(self, map_path: str, uncertain_sparse: int = 5, uncertain_clustered: int = 5,
                 blocked_moving: int = 5, blocked_permanent: int = 5, noise_rate: float = 0.05,
                 sparse_rate: float = None, show_window: bool = False, num_timesteps: int = 3):
        self.map_loader = MapLoader(map_path)
        self.grid_original = self.map_loader.get_grid().copy()
        self.height, self.width = self.grid_original.shape
        self.noise_rate = noise_rate
        self.sparse_rate = sparse_rate if sparse_rate is not None else noise_rate
        self.corruptor = Corruptor(corruption_rate=self.noise_rate)
        self.replanner = build_llm_replanner()
        self.uncertain_sparse = uncertain_sparse
        self.uncertain_clustered = uncertain_clustered
        self.blocked_moving = blocked_moving
        self.blocked_permanent = blocked_permanent
        self.show_window = show_window
        self.num_timesteps = num_timesteps

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

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

    def generate_sparse_uncertainty(self, grid):
        return self.corruptor.inject_random_corruption(grid, corruption_rate=self.sparse_rate)

    def generate_clustered_uncertainty(self, grid, start, goal, radius=3):
        corrupted = grid.copy()
        path = astar(self.make_cost_map(grid), start, goal)
        if not path:
            return corrupted

        center = path[len(path) // 2]
        x, y = center
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(self.width, x + radius + 1)
        y2 = min(self.height, y + radius + 1)
        # Ensure the clustered block lies on the planned path center
        return self.corruptor.inject_block(corrupted, (x1, y1), (x2, y2))

    def generate_moving_blockage(self, grid, start, goal):
        corrupted = grid.copy()
        path = astar(self.make_cost_map(grid), start, goal)
        if not path:
            return corrupted

        center = path[len(path) // 2]
        x, y = center
        if corrupted[y, x] == 0:
            corrupted[y, x] = 1

        if len(path) > 3:
            neighbor = path[min(len(path) // 2 + 1, len(path) - 1)]
            nx, ny = neighbor
            if corrupted[ny, nx] == 0:
                corrupted[ny, nx] = 1

        return corrupted

    def generate_permanent_blockage(self, grid, start, goal, radius=2):
        corrupted = grid.copy()
        path = astar(self.make_cost_map(grid), start, goal)
        if not path:
            return corrupted

        center = path[len(path) // 2]
        x, y = center
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(self.width, x + radius + 1)
        y2 = min(self.height, y + radius + 1)
        return self.corruptor.inject_block(corrupted, (x1, y1), (x2, y2), blocked_value=1)

    def classify_answer(self, decision):
        if not decision:
            return "no_decision"

        action = decision.get("recommended_action")
        if action in ("plan_through", "keep_moving"):
            return "keep_moving"
        if action in ("replan_immediately", "avoid", "increase_cost", "replan"):
            return "replan"
        if action in ("wait", "inspect", "wait_and_reinspect"):
            return "wait"
        return "uncertain"

    def display_test_maps(self, test_id, test_type, corrupted_grid, corrupted_path, timestep_grids, start, goal, expected_answer, actual_answer, decision):
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib not installed; cannot display map window.")
            return

        cmap = ListedColormap(["#ff9999", "#ffffff", "#000000"])
        norm = BoundaryNorm([-1, 0, 1, 2], cmap.N)

        def plot_grid(ax, grid, title, path=None):
            ax.imshow(grid, cmap=cmap, norm=norm, origin="lower")
            if path:
                xs, ys = zip(*path)
                ax.plot(xs, ys, color="#ffff00", linewidth=2, marker="o", markersize=4)
            ax.scatter([start[0]], [start[1]], c="#00ff00", s=80, marker="*", edgecolors="#000000", linewidths=0.5, label="start")
            ax.scatter([goal[0]], [goal[1]], c="#0000ff", s=80, marker="X", edgecolors="#000000", linewidths=0.5, label="goal")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc="upper right", fontsize="small")

        if timestep_grids:
            n = len(timestep_grids)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
            if n == 1:
                axes = [axes]
            for idx, grid in enumerate(timestep_grids):
                plot_grid(axes[idx], grid, f"t={idx}", path=(corrupted_path if idx == 0 else None))
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_grid(ax, corrupted_grid, f"{test_type}", path=corrupted_path)

        fig.suptitle(
            f"Test {test_id}: {test_type} | expected={expected_answer} actual={actual_answer} | "
            f"decision={decision.get('recommended_action') if decision else 'None'}"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=True)
        plt.close(fig)

    def run_test(self, test_id, test_type, start, goal):
        grid = self.grid_original.copy()
        expected_answer = ""
        use_temporal = False
        timestep_grids = []

        if test_type == "uncertain_sparse":
            # Create a temporal sequence of sparse noise so uncertainty changes across timesteps
            expected_answer = "keep_moving"
            timestep_grids = self.corruptor.inject_random_corruption(
                grid,
                corruption_rate=self.sparse_rate,
                change_over_time=True,
                num_timesteps=self.num_timesteps
            )
            use_temporal = True
            corrupted_grid = timestep_grids[0]
        elif test_type == "uncertain_clustered":
            # Clustered uncertainty: place a fixed cluster on the planned path; no temporal change
            corrupted_grid = self.generate_clustered_uncertainty(grid, start, goal)
            expected_answer = "replan"
            timestep_grids = [corrupted_grid.copy() for _ in range(self.num_timesteps)]
            use_temporal = True
        elif test_type == "blocked_moving":
            # Generate timestep sequence for moving obstacle
            use_temporal = True
            expected_answer = "wait"
            path = astar(self.make_cost_map(grid), start, goal)
            if path:
                path_segment = path[max(0, len(path)//2 - 2):min(len(path), len(path)//2 + 5)]
                timestep_grids = self.corruptor.generate_moving_obstacle_sequence(grid, path_segment, num_timesteps=self.num_timesteps)
            else:
                use_temporal = False
                corrupted_grid = grid.copy()
        elif test_type == "blocked_permanent":
            # Generate timestep sequence for permanent obstacle
            use_temporal = True
            expected_answer = "replan"
            path = astar(self.make_cost_map(grid), start, goal)
            if path:
                center_idx = len(path) // 2
                center = path[center_idx]
                x, y = center
                x1 = max(0, x - 2)
                y1 = max(0, y - 2)
                x2 = min(self.width, x + 3)
                y2 = min(self.height, y + 3)
                timestep_grids = self.corruptor.generate_permanent_obstacle_sequence(grid, (x1, y1, x2, y2), num_timesteps=self.num_timesteps)
            else:
                use_temporal = False
                corrupted_grid = grid.copy()
        else:
            corrupted_grid = grid.copy()

        original_cost = self.make_cost_map(self.grid_original)
        original_path = astar(original_cost, start, goal)
        
        if use_temporal and timestep_grids:
            corrupted_grid = timestep_grids[0]
        
        corrupted_cost = self.make_cost_map(corrupted_grid)
        corrupted_path = astar(corrupted_cost, start, goal)
        corrupted_exists = corrupted_path is not None
        corrupted_length = len(corrupted_path) if corrupted_exists else 0

        decision = None
        modified_path_exists = False
        modified_length = 0
        llm_prompt = ""
        llm_response = ""

        if self.replanner is not None:
            try:
                # determine situation_type for prompt semantics
                situation_type = None
                if test_type == "uncertain_sparse":
                    situation_type = "uncertain"
                elif test_type in ("uncertain_clustered", "blocked_moving", "blocked_permanent"):
                    situation_type = "new_blockage"

                target_cell = None
                if use_temporal and timestep_grids:
                    if test_type == "blocked_moving":
                        target_cell = path_segment[len(path_segment) // 2] if 'path_segment' in locals() else None
                    elif test_type == "blocked_permanent":
                        target_cell = center if 'center' in locals() else None

                if use_temporal and timestep_grids:
                    # Use temporal replanner for blocked tests
                    modified_grid, decision = self.replanner.replan_temporal(
                        grids=timestep_grids,
                        start=start,
                        goal=goal,
                        original_path=original_path,
                        robot_pose={"x": start[0], "y": start[1]},
                        situation_type=situation_type,
                        target_cell=target_cell
                    )
                else:
                    # Use regular replanner for uncertainty tests
                    modified_grid, decision = self.replanner.replan(
                        corrupted_grid.copy(),
                        start,
                        goal,
                        corrupted_path or [],
                        {"x": start[0], "y": start[1]},
                        original_path=original_path,
                        situation_type=situation_type
                    )

                llm_prompt = self.replanner.last_prompt or ""
                llm_response = self.replanner.last_response or ""

                if modified_grid is not None:
                    modified_cost = self.make_cost_map(modified_grid)
                    modified_path = astar(modified_cost, start, goal)
                    modified_path_exists = modified_path is not None
                    modified_length = len(modified_path) if modified_path_exists else 0
            except Exception as e:
                self.logger.warning(f"[WARN] Test {test_id} replanning failed: {e}")

        actual_answer = self.classify_answer(decision)
        success = actual_answer == expected_answer
        corrupted_cells = np.where(corrupted_grid != grid)
        corrupted_count = len(corrupted_cells[0])

        self.logger.info("""
============================================================
TEST %s: %s
expected_answer=%s actual_answer=%s success=%s
start=(%s,%s) goal=(%s,%s)
corrupted_cells=%s corrupted_path_exists=%s corrupted_path_length=%s
modified_path_exists=%s modified_path_length=%s
decision_label=%s decision_confidence=%s decision_reason=%s decision_action=%s movement_pattern=%s
""".strip(),
            test_id,
            test_type,
            expected_answer,
            actual_answer,
            success,
            start[0], start[1],
            goal[0], goal[1],
            corrupted_count,
            corrupted_exists,
            corrupted_length,
            modified_path_exists,
            modified_length,
            decision.get("label") if decision else "",
            decision.get("confidence") if decision else "",
            decision.get("reason") if decision else "",
            decision.get("recommended_action") if decision else "",
            decision.get("movement_pattern") if decision else ""
        )
        if llm_prompt:
            self.logger.info("LLM prompt:\n%s", llm_prompt)
        if llm_response:
            self.logger.info("LLM response:\n%s", llm_response)

        if self.show_window:
            self.display_test_maps(
                test_id=test_id,
                test_type=test_type,
                corrupted_grid=corrupted_grid,
                corrupted_path=corrupted_path,
                timestep_grids=timestep_grids,
                start=start,
                goal=goal,
                expected_answer=expected_answer,
                actual_answer=actual_answer,
                decision=decision
            )

        movement_pattern = decision.get("movement_pattern", "") if decision else ""
        timestep_sequence = []
        if timestep_grids:
            timestep_sequence = [grid.tolist() for grid in timestep_grids]

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
            "movement_pattern": movement_pattern,
            "llm_prompt": llm_prompt,
            "llm_response": llm_response,
            "num_timesteps": len(timestep_sequence),
            "timestep_sequence": json.dumps(timestep_sequence),
            "temporal_test": bool(timestep_sequence),
        }

    def run_all(self, output_csv: str):
        total_tests = (
            self.uncertain_sparse +
            self.uncertain_clustered +
            self.blocked_moving +
            self.blocked_permanent
        )
        results = []
        pairs = self.find_random_valid_cells(num_pairs=total_tests)

        index = 0
        for i in range(self.uncertain_sparse):
            start, goal = pairs[index]
            results.append(self.run_test(i + 1, "uncertain_sparse", start, goal))
            index += 1

        for i in range(self.uncertain_clustered):
            start, goal = pairs[index]
            results.append(self.run_test(self.uncertain_sparse + i + 1, "uncertain_clustered", start, goal))
            index += 1

        for i in range(self.blocked_moving):
            start, goal = pairs[index]
            results.append(self.run_test(self.uncertain_sparse + self.uncertain_clustered + i + 1, "blocked_moving", start, goal))
            index += 1

        for i in range(self.blocked_permanent):
            start, goal = pairs[index]
            results.append(self.run_test(
                self.uncertain_sparse + self.uncertain_clustered + self.blocked_moving + i + 1,
                "blocked_permanent", start, goal
            ))
            index += 1

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
            "decision_action",
            "movement_pattern",
            "llm_prompt",
            "llm_response",
            "num_timesteps",
            "temporal_test",
            "timestep_sequence"
        ]

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Wrote {len(results)} test rows to {output_csv}")
        print(f"Overall success rate: {success_rate:.2%} ({success_count}/{len(results)})")
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run uncertain and blocked dynamic obstacle LLM tests and save results to CSV"
    )
    parser.add_argument("--output", default="llm_dynamic_obstacle_test_results.csv", help="CSV output file")
    parser.add_argument("--uncertain-sparse", type=int, default=5, help="Number of sparse uncertainty tests")
    parser.add_argument("--uncertain-clustered", type=int, default=5, help="Number of clustered uncertainty tests")
    parser.add_argument("--blocked-moving", type=int, default=5, help="Number of blocked moving obstacle tests")
    parser.add_argument("--blocked-permanent", type=int, default=5, help="Number of blocked permanent obstacle tests")
    parser.add_argument("--noise", type=float, default=0.05, help="Baseline uncertainty rate")
    parser.add_argument("--sparse-rate", type=float, default=None, help="Noise rate for sparse uncertainty tests")
    parser.add_argument("--show-window", action="store_true", help="Display corrupted map windows for each test")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"

    runner = LLMDynamicObstacleTestRunner(
        map_path=str(map_path),
        uncertain_sparse=args.uncertain_sparse,
        uncertain_clustered=args.uncertain_clustered,
        blocked_moving=args.blocked_moving,
        blocked_permanent=args.blocked_permanent,
        noise_rate=args.noise,
        sparse_rate=args.sparse_rate,
        show_window=args.show_window,
    )
    runner.run_all(args.output)


if __name__ == "__main__":
    main()
