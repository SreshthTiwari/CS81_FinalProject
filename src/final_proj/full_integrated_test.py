#!/usr/bin/env python3
"""Integrated test combining skill storage with the full LLM navigation pipeline."""

import argparse
import json
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
from final_proj.planning.replanner import Replanner
from final_proj.llm.prompt_builder import PromptBuilder
from final_proj.llm.response_parser import ResponseParser
from final_proj.llm.client import LLMClient


def split_path_into_segments(path, num_segments=3):
    if not path or num_segments <= 0:
        return []

    L = len(path)
    segments = []
    prev_end = 0
    for i in range(num_segments):
        start_idx = prev_end
        end_idx = ((i + 1) * L) // num_segments - 1 if i < num_segments - 1 else L - 1
        segment = path[start_idx:end_idx + 1]
        if segment:
            segments.append(segment)
            prev_end = end_idx
    return segments


def combine_paths(paths):
    combined = []
    for p in paths:
        if not p:
            continue
        if not combined:
            combined.extend(p)
        else:
            combined.extend(p[1:])
    return combined


class FullIntegratedTest:
    MODES = ["uncertain_sparse", "uncertain_clustered", "blocked_moving", "blocked_permanent"]

    def __init__(self, map_path, corruption_rate=0.05, num_timesteps=5, seed=None, mode="blocked_moving"):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.mode = mode

        self.map_loader = MapLoader(map_path)
        self.grid_original = self.map_loader.get_grid().copy()
        self.corruptor = Corruptor(corruption_rate=corruption_rate)
        self.replanner = Replanner(PromptBuilder(), ResponseParser(), LLMClient(), ContextExtractor())
        self.num_timesteps = num_timesteps

    def make_cost_map(self, grid):
        cost_map = np.full(grid.shape, np.inf, dtype=float)
        cost_map[grid == 0] = 1.0
        cost_map[grid == -1] = 5.0
        return cost_map

    def pick_free_cells(self):
        free_cells = np.where(self.grid_original == 0)
        options = list(zip(free_cells[1], free_cells[0]))
        self.rng.shuffle(options)
        return options

    def _trim_path_at_corruption(self, path, grid):
        if not path or grid is None:
            return path

        trimmed = []
        for x, y in path:
            trimmed.append((x, y))
            if grid[y, x] != 0:
                break
        return trimmed

    def extract_obstacle_cells(self, grid):
        if grid is None:
            return []
        mask = (grid == 1) & (self.grid_original == 0)
        ys, xs = np.where(mask)
        return list(zip(xs, ys))

    def extract_uncertain_cells(self, grid):
        if grid is None:
            return []
        mask = (grid == -1) & (self.grid_original == 0)
        ys, xs = np.where(mask)
        return list(zip(xs, ys))

    def build_history_entry(self, **kwargs):
        entry = {
            "grid": kwargs.get("grid").copy() if kwargs.get("grid") is not None else None,
            "path": kwargs.get("path"),
            "planned_path": kwargs.get("planned_path"),
            "actual_path": kwargs.get("actual_path"),
            "display_path": kwargs.get("display_path"),
            "step_type": kwargs.get("step_type"),
            "title": kwargs.get("title"),
            "advice": kwargs.get("advice"),
            "start": kwargs.get("start"),
            "end": kwargs.get("end"),
            "final_goal": kwargs.get("final_goal"),
            "checkpoints": kwargs.get("checkpoints"),
            "obstacle_cells": kwargs.get("obstacle_cells"),
            "uncertain_cells": kwargs.get("uncertain_cells"),
        }
        return entry

    def generate_scenario(self, mode, path_cp2_to_goal):
        if mode == "uncertain_sparse":
            return self.corruptor.inject_random_corruption(
                self.grid_original,
                corruption_rate=self.corruptor.corruption_rate,
                change_over_time=True,
                num_timesteps=self.num_timesteps
            )

        if mode == "uncertain_clustered":
            corrupted_grid = self.generate_clustered_uncertainty(self.grid_original, path_cp2_to_goal)
            return [corrupted_grid.copy() for _ in range(self.num_timesteps)]

        if mode == "blocked_moving":
            path_segment = path_cp2_to_goal[max(0, len(path_cp2_to_goal) // 2 - 2):min(len(path_cp2_to_goal), len(path_cp2_to_goal) // 2 + 5)]
            num_steps = max(self.num_timesteps, 5)
            return self.corruptor.generate_moving_obstacle_sequence(self.grid_original, path_segment, num_timesteps=num_steps)

        if mode == "blocked_permanent":
            if not path_cp2_to_goal:
                return [self.grid_original.copy() for _ in range(self.num_timesteps)]
            center = path_cp2_to_goal[len(path_cp2_to_goal) // 2]
            x, y = center
            x1 = max(0, x - 2)
            y1 = max(0, y - 2)
            x2 = min(self.grid_original.shape[1], x + 3)
            y2 = min(self.grid_original.shape[0], y + 3)
            return self.corruptor.generate_permanent_obstacle_sequence(self.grid_original, (x1, y1, x2, y2), num_timesteps=self.num_timesteps)

        return [self.grid_original.copy() for _ in range(self.num_timesteps)]

    def generate_clustered_uncertainty(self, grid, path, radius=3):
        corrupted = grid.copy()
        if not path:
            return corrupted

        center = path[len(path) // 2]
        x, y = center
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(self.grid_original.shape[1], x + radius + 1)
        y2 = min(self.grid_original.shape[0], y + radius + 1)
        return self.corruptor.inject_block(corrupted, (x1, y1), (x2, y2), blocked_value=-1)

    def run(self, show_window=True):
        options = self.pick_free_cells()
        start1 = tuple(int(v) for v in self.rng.choice(options))
        goal = tuple(int(v) for v in self.rng.choice(options))
        while goal == start1:
            goal = tuple(int(v) for v in self.rng.choice(options))

        new_goal = tuple(int(v) for v in self.rng.choice(options))
        while new_goal == start1 or new_goal == goal:
            new_goal = tuple(int(v) for v in self.rng.choice(options))

        cost = self.make_cost_map(self.grid_original)
        full_path = astar(cost, start1, goal)
        if not full_path:
            raise RuntimeError("Unable to compute initial clean path")

        segments = split_path_into_segments(full_path, num_segments=3)
        checkpoints = [segments[i][-1] for i in range(len(segments) - 1)]

        history = []
        history.append(self.build_history_entry(
            grid=self.grid_original,
            path=full_path,
            planned_path=full_path,
            actual_path=full_path,
            display_path=full_path,
            step_type="original",
            title="Stage 1: Clean A* path and stored skill",
            advice="Original path is computed on a clean map and stored as a skill with checkpoints.",
            start=start1,
            end=goal,
            checkpoints=checkpoints,
            obstacle_cells=[],
            uncertain_cells=[]
        ))

        start2 = tuple(int(v) for v in self.rng.choice(options))
        while start2 == start1 or start2 == new_goal:
            start2 = tuple(int(v) for v in self.rng.choice(options))

        checkpoint1 = checkpoints[0]
        checkpoint2 = checkpoints[1]

        seg_a = astar(cost, start2, checkpoint1)
        if not seg_a:
            raise RuntimeError("Unable to compute new path to checkpoint")

        path_cp2_to_goal = astar(cost, checkpoint2, new_goal)
        if not path_cp2_to_goal:
            raise RuntimeError("Unable to compute path from checkpoint 2 to new goal")

        history.append(self.build_history_entry(
            grid=self.grid_original,
            path=seg_a,
            planned_path=combine_paths([seg_a, segments[1], path_cp2_to_goal]),
            actual_path=combine_paths([seg_a, segments[1], path_cp2_to_goal]),
            display_path=combine_paths([seg_a, segments[1], path_cp2_to_goal]),
            step_type="new_route",
            title="Stage 2: New path to stored checkpoint",
            advice="Robot plans from a new start to a stored checkpoint using A*.",
            start=start2,
            end=checkpoint1,
            final_goal=new_goal,
            checkpoints=[checkpoint1, checkpoint2],
            obstacle_cells=[],
            uncertain_cells=[]
        ))

        seg_b = segments[1]
        reuse_seg_b = False
        if seg_b and seg_b[0] == checkpoint1 and seg_b[-1] == checkpoint2:
            reuse_seg_b = True

        if reuse_seg_b:
            history.append(self.build_history_entry(
                grid=self.grid_original,
                path=seg_b,
                planned_path=combine_paths([seg_b, path_cp2_to_goal]),
                actual_path=combine_paths([seg_b, path_cp2_to_goal]),
                display_path=combine_paths([seg_b, path_cp2_to_goal]),
                step_type="reused_skill",
                title="Stage 3: Reuse stored skill segment",
                advice="Robot reuses the stored skill segment from checkpoint 1 to checkpoint 2.",
                start=checkpoint1,
                end=checkpoint2,
                final_goal=new_goal,
                checkpoints=[checkpoint1, checkpoint2],
                obstacle_cells=[],
                uncertain_cells=[]
            ))
        else:
            seg_b_plan = astar(cost, checkpoint1, checkpoint2)
            if not seg_b_plan:
                raise RuntimeError("Unable to compute path from checkpoint1 to checkpoint2 when reuse not possible")
            history.append(self.build_history_entry(
                grid=self.grid_original,
                path=seg_b_plan,
                planned_path=combine_paths([seg_b_plan, path_cp2_to_goal]),
                actual_path=combine_paths([seg_b_plan, path_cp2_to_goal]),
                display_path=combine_paths([seg_b_plan, path_cp2_to_goal]),
                step_type="planned_segment",
                title="Stage 3: Plan segment between checkpoints",
                advice="No stored skill available for this segment; planning via A*.",
                start=checkpoint1,
                end=checkpoint2,
                final_goal=new_goal,
                checkpoints=[checkpoint1, checkpoint2],
                obstacle_cells=[],
                uncertain_cells=[]
            ))

        timestep_grids = self.generate_scenario(self.mode, path_cp2_to_goal)
        current_time = 0
        current_grid = timestep_grids[current_time]
        current_plan_path = path_cp2_to_goal
        current_path = astar(self.make_cost_map(current_grid), checkpoint2, new_goal)

        if self.mode == "blocked_moving":
            advice_stage4 = "A moving blockage is introduced on the remaining route to the goal."
        elif self.mode == "blocked_permanent":
            advice_stage4 = "A permanent blockage is placed on the remaining route; replanning may be required."
        elif self.mode == "uncertain_clustered":
            advice_stage4 = "A cluster of uncertain observations appears near the remaining route; the LLM may choose to replan or wait if the route becomes clearer."
        else:
            advice_stage4 = "Sparse uncertain observations appear near the route; these are likely transient."

        history.append(self.build_history_entry(
            grid=current_grid,
            path=current_plan_path,
            planned_path=current_plan_path,
            actual_path=self._trim_path_at_corruption(current_plan_path, current_grid),
            display_path=self._trim_path_at_corruption(current_plan_path, current_grid),
            step_type="corrupted",
            title=f"Stage 4: Corrupted map at t={current_time}",
            advice=advice_stage4,
            start=checkpoint2,
            end=new_goal,
            final_goal=new_goal,
            checkpoints=[checkpoint2],
            obstacle_cells=self.extract_obstacle_cells(current_grid),
            uncertain_cells=self.extract_uncertain_cells(current_grid)
        ))

        situation_type = "uncertain" if self.mode == "uncertain_sparse" else "new_blockage"
        consecutive_waits = 0

        if situation_type == "new_blockage" and len(timestep_grids) > 2:
            for _ in range(2):
                if current_time + 1 < len(timestep_grids):
                    current_time += 1
                    current_grid = timestep_grids[current_time]
                    current_path = astar(self.make_cost_map(current_grid), checkpoint2, new_goal)
                    history.append(self.build_history_entry(
                        grid=current_grid,
                        path=current_plan_path,
                        planned_path=current_plan_path,
                        actual_path=self._trim_path_at_corruption(current_plan_path, current_grid),
                        display_path=self._trim_path_at_corruption(current_plan_path, current_grid),
                        step_type="observation",
                        title=f"Stage 5: Observing timestep t={current_time}",
                        advice="Gathering temporal context before querying the LLM.",
                        start=checkpoint2,
                        end=new_goal,
                        checkpoints=[checkpoint2],
                        obstacle_cells=self.extract_obstacle_cells(current_grid),
                        uncertain_cells=self.extract_uncertain_cells(current_grid)
                    ))
        elif current_time + 1 < len(timestep_grids):
            current_time += 1
            current_grid = timestep_grids[current_time]
            current_path = astar(self.make_cost_map(current_grid), checkpoint2, new_goal)
            history.append(self.build_history_entry(
                grid=current_grid,
                path=current_plan_path,
                planned_path=current_plan_path,
                actual_path=self._trim_path_at_corruption(current_plan_path, current_grid),
                display_path=self._trim_path_at_corruption(current_plan_path, current_grid),
                step_type="observation",
                title=f"Stage 5: Observing timestep t={current_time}",
                advice="Gathering temporal context before querying the LLM.",
                start=checkpoint2,
                end=new_goal,
                checkpoints=[checkpoint2],
                obstacle_cells=self.extract_obstacle_cells(current_grid),
                uncertain_cells=self.extract_uncertain_cells(current_grid)
            ))

        loop_count = 0
        reached = False
        while loop_count < 10:
            loop_count += 1
            remaining_grids = timestep_grids[current_time:]
            if not remaining_grids:
                break

            if len(remaining_grids) > 1:
                modified_grid, decision = self.replanner.replan_temporal(
                    grids=remaining_grids,
                    start=checkpoint2,
                    goal=new_goal,
                    original_path=current_plan_path,
                    robot_pose={"x": checkpoint2[0], "y": checkpoint2[1]},
                    situation_type=situation_type
                )
            else:
                modified_grid, decision = self.replanner.replan(
                    current_grid.copy(),
                    checkpoint2,
                    new_goal,
                    current_plan_path,
                    {"x": checkpoint2[0], "y": checkpoint2[1]},
                    original_path=current_plan_path,
                    situation_type=situation_type
                )

            action = decision.get("recommended_action", "uncertain") if decision else "uncertain"
            reason = decision.get("reason", "") if decision else "No decision"
            history.append(self.build_history_entry(
                grid=current_grid,
                path=current_plan_path,
                planned_path=current_plan_path,
                actual_path=self._trim_path_at_corruption(current_plan_path, current_grid),
                display_path=self._trim_path_at_corruption(current_plan_path, current_grid),
                step_type="query",
                title=f"Stage 6: LLM query and decision",
                advice=f"LLM recommends {action}. Reason: {reason}",
                start=checkpoint2,
                end=new_goal,
                checkpoints=[checkpoint2],
                obstacle_cells=self.extract_obstacle_cells(current_grid),
                uncertain_cells=self.extract_uncertain_cells(current_grid)
            ))

            if action in ("keep_moving", "plan_through"):
                history.append(self.build_history_entry(
                    grid=current_grid,
                    path=current_path,
                    planned_path=current_path,
                    actual_path=current_path,
                    display_path=current_path,
                    step_type="keep_moving",
                    title="Stage 7: Action execution - keep moving",
                    advice="Following current route to goal.",
                    start=checkpoint2,
                    end=new_goal,
                    checkpoints=[checkpoint2],
                    obstacle_cells=self.extract_obstacle_cells(current_grid),
                    uncertain_cells=self.extract_uncertain_cells(current_grid)
                ))
                reached = current_path is not None
                break

            if action in ("wait", "wait_and_reinspect", "inspect"):
                consecutive_waits += 1
                next_time = current_time + 1
                if next_time >= len(timestep_grids):
                    history.append(self.build_history_entry(
                        grid=current_grid,
                        path=current_path,
                        planned_path=current_path,
                        actual_path=self._trim_path_at_corruption(current_path, current_grid),
                        display_path=self._trim_path_at_corruption(current_path, current_grid),
                        step_type="final",
                        title="Stage 7: No more observations",
                        advice="No further timesteps; acting on current knowledge.",
                        start=checkpoint2,
                        end=new_goal,
                        checkpoints=[checkpoint2],
                        obstacle_cells=self.extract_obstacle_cells(current_grid),
                        uncertain_cells=self.extract_uncertain_cells(current_grid)
                    ))
                    break
                current_time = next_time
                current_grid = timestep_grids[current_time]
                current_path = astar(self.make_cost_map(current_grid), checkpoint2, new_goal)
                history.append(self.build_history_entry(
                    grid=current_grid,
                    path=current_plan_path,
                    planned_path=current_plan_path,
                    actual_path=self._trim_path_at_corruption(current_plan_path, current_grid),
                    display_path=self._trim_path_at_corruption(current_plan_path, current_grid),
                    step_type="observation",
                    title=f"Stage 7: Re-observing timestep t={current_time}",
                    advice="Waiting and re-inspecting before querying the LLM again.",
                    start=checkpoint2,
                    end=new_goal,
                    checkpoints=[checkpoint2],
                    obstacle_cells=self.extract_obstacle_cells(current_grid),
                    uncertain_cells=self.extract_uncertain_cells(current_grid)
                ))
                continue

            if action in ("replan", "replan_immediately", "avoid"):
                current_grid = modified_grid
                current_path = astar(self.make_cost_map(current_grid), checkpoint2, new_goal)
                current_plan_path = current_path
                history.append(self.build_history_entry(
                    grid=current_grid,
                    path=current_plan_path,
                    planned_path=current_plan_path,
                    actual_path=current_plan_path,
                    display_path=current_plan_path,
                    step_type="replan",
                    title="Stage 7: Action execution - replan",
                    advice="LLM decides to replan around the blockage.",
                    start=checkpoint2,
                    end=new_goal,
                    checkpoints=[checkpoint2],
                    obstacle_cells=self.extract_obstacle_cells(current_grid),
                    uncertain_cells=self.extract_uncertain_cells(current_grid)
                ))
                reached = current_path is not None
                break

            break

        history.append(self.build_history_entry(
            grid=current_grid,
            path=current_plan_path,
            planned_path=current_plan_path,
            actual_path=current_plan_path,
            display_path=current_plan_path,
            step_type="final",
            title="Stage 8: Integration finished",
            advice="Skill storage and LLM planning have been exercised together.",
            start=checkpoint2,
            end=new_goal,
            checkpoints=[checkpoint2],
            obstacle_cells=self.extract_obstacle_cells(current_grid),
            uncertain_cells=self.extract_uncertain_cells(current_grid)
        ))

        if show_window and MATPLOTLIB_AVAILABLE:
            self.display_history(history)

        return {
            "mode": self.mode,
            "history": history,
            "success": reached,
            "final_path_exists": current_path is not None
        }

    def display_history(self, history):
        cmap = ListedColormap(["#999999", "#ffffff", "#000000"])
        norm = BoundaryNorm([-1, 0, 1, 2], cmap.N)

        plt.ion()
        for step in history:
            fig, ax = plt.subplots(figsize=(8, 8))
            grid = step["grid"]
            if grid is None:
                continue

            ax.imshow(grid, cmap=ListedColormap(["#999999", "#ffffff", "#000000"]),
                      norm=BoundaryNorm([-1, 0, 1, 2], 3), origin="lower")

            display_path = step.get("display_path")
            if display_path:
                xs, ys = zip(*display_path)
                ax.plot(xs, ys, color="#ffff00", linewidth=3, alpha=0.9, label="Display path", zorder=10)

            start = step.get("start") or (history[0]["path"][0] if history and history[0].get("path") else None)
            if start:
                ax.scatter([start[0]], [start[1]], c="#00ff00", s=200, marker="*", edgecolors="#000000", linewidths=2, label="Start", zorder=12)

            goal = step.get("end")
            if goal:
                ax.scatter([goal[0]], [goal[1]], c="#0000ff", s=200, marker="X", edgecolors="#000000", linewidths=2, label="Goal", zorder=12)

            final_goal = step.get("final_goal")
            if final_goal and final_goal != goal:
                ax.scatter([final_goal[0]], [final_goal[1]], c="#0033cc", s=200, marker="X", edgecolors="#000000", linewidths=2, label="Final goal", zorder=12)

            checkpoints = step.get("checkpoints") or []
            if checkpoints:
                cp_xs, cp_ys = zip(*checkpoints)
                ax.scatter(cp_xs, cp_ys, c="#ff9900", s=180, marker="D", edgecolors="#000000", linewidths=2, label="Checkpoint", zorder=12)
                for i, (cx, cy) in enumerate(checkpoints, start=1):
                    ax.annotate(f"CP{i}", xy=(cx, cy), xytext=(4, 4), textcoords="offset points",
                                fontsize=9, fontweight='bold', color="#ff9900")

            uncertain_cells = step.get("uncertain_cells") or []
            if uncertain_cells:
                ux, uy = zip(*uncertain_cells)
                size = 25 if self.mode == "uncertain_sparse" else 100
                ax.scatter(ux, uy, c="#ff3333", s=size, marker="s", edgecolors='black', linewidths=0.6,
                           alpha=1.0, label="Uncertain", zorder=4)

            obstacle_cells = step.get("obstacle_cells") or []
            if obstacle_cells:
                ox, oy = zip(*obstacle_cells)
                ax.scatter(ox, oy, c="#aa0000", s=120, marker="s", edgecolors='black', linewidths=0.6,
                           alpha=0.9, label="Obstacle", zorder=6)

            ax.set_title(step.get("title", ""))
            ax.set_xlim(-0.5, step["grid"].shape[1] - 0.5)
            ax.set_ylim(-0.5, step["grid"].shape[0] - 0.5)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend(loc='upper left', fontsize=9)
            plt.show()
            input("Press Enter to continue...")
            plt.close(fig)
        plt.ioff()


def main():
    parser = argparse.ArgumentParser(description="Run a full integrated LLM planner + skill storage visualization test")
    parser.add_argument("--mode", choices=FullIntegratedTest.MODES, default="blocked_moving", help="Corruption mode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-window", action="store_true", help="Do not open visualization")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"
    integrator = FullIntegratedTest(map_path=str(map_path), seed=args.seed, mode=args.mode)
    result = integrator.run(show_window=not args.no_window)
    print(json.dumps({
        "mode": result["mode"],
        "success": result["success"],
        "final_path_exists": result["final_path_exists"],
        "steps": len(result["history"]),
        "last_title": result["history"][-1]["title"] if result["history"] else None,
    }, indent=2))


if __name__ == "__main__":
    main()
