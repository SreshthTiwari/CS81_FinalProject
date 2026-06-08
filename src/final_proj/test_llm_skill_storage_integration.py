#!/usr/bin/env python3
"""Integration test: LLM planning + skill storage with staged visualization.

Phases:
1. Compute a clean A* path and store it as a skill with checkpoints.
2. Start from a new location, connect to a stored checkpoint, reuse a stored
   segment when exactly at a checkpoint and endpoints match.
3. After checkpoint 2, inject moving corruption and run the LLM temporal
   replanner; visualize stages.
"""

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


class InMemorySkillStore:
    def __init__(self):
        self.store = {}

    def save_skill(self, key, skill_obj):
        self.store[key] = skill_obj

    def load_skill(self, key):
        return self.store.get(key)


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


class LLMSkillStorageIntegration:
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
        cost = np.full(grid.shape, np.inf, dtype=float)
        cost[grid == 0] = 1.0
        cost[grid == -1] = 5.0
        return cost

    def pick_free_cells(self):
        free_cells = np.where(self.grid_original == 0)
        options = list(zip(free_cells[1], free_cells[0]))
        self.rng.shuffle(options)
        return options

    def build_history_entry(self, grid, path, title, advice, start, end, whole_path=None, checkpoints=None, reused=False, endpoint_label=None, overall_goal=None, obstacle_cells=None):
        entry = {
            "grid": grid.copy(),
            "path": path,
            "start": start,
            "end": end,
            "whole_path": whole_path,
            "checkpoints": checkpoints,
            "is_reused": reused,
            "title": title,
            "advice": advice,
            "endpoint_label": endpoint_label,
            "overall_goal": overall_goal,
            "obstacle_cells": obstacle_cells,
        }

        # infer uncertain cells that were injected (grid == -1) but were free in the original map
        uncertain_cells = None
        try:
            mask = (grid == -1) & (self.grid_original == 0)
            ys, xs = np.where(mask)
            uncertain_cells = list(zip(xs, ys))
        except Exception:
            uncertain_cells = None

        entry["uncertain_cells"] = uncertain_cells
        return entry

    def extract_dynamic_obstacle_cells(self, grid):
        obstacle_mask = (grid == 1) & (self.grid_original == 0)
        ys, xs = np.where(obstacle_mask)
        return list(zip(xs, ys))

    def run(self, show_window=True):
        history = []
        executed_segments = []

        options = self.pick_free_cells()
        start1 = tuple(int(v) for v in self.rng.choice(options))
        goal = tuple(int(v) for v in self.rng.choice(options))
        while goal == start1:
            goal = tuple(int(v) for v in self.rng.choice(options))

        # pick a new random final goal for the second run (displayed throughout)
        new_goal = tuple(int(v) for v in self.rng.choice(options))
        while new_goal == start1 or new_goal == goal:
            new_goal = tuple(int(v) for v in self.rng.choice(options))

        cost = self.make_cost_map(self.grid_original)
        full_path = astar(cost, start1, goal)
        if not full_path:
            raise RuntimeError("Unable to compute initial clean path")

        segments = split_path_into_segments(full_path, num_segments=3)
        checkpoints = [segments[i][-1] for i in range(len(segments) - 1)]
        skill = {
            "segments": segments,
            "checkpoints": checkpoints,
            "origin_start": start1,
            "origin_goal": goal,
        }
        store = InMemorySkillStore()
        store.save_skill("skill_A", skill)

        original_goal = goal
        history.append(self.build_history_entry(
            grid=self.grid_original,
            path=full_path,
            title=f"Stage 1: Clean A* path and stored skill\nStart1 → Goal",
            advice="Original path is computed on a clean map and stored as a skill with checkpoints.",
            start=start1,
            end=original_goal,
            whole_path=full_path,
            checkpoints=checkpoints,
            reused=False,
            endpoint_label="Goal",
            overall_goal=original_goal,
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
            title="Stage 2: New path to stored checkpoint",
            advice="Robot plans from a new start to a stored checkpoint using A*.",
            start=start2,
            end=checkpoint1,
            whole_path=combine_paths([seg_a, segments[1], path_cp2_to_goal]),
            checkpoints=[checkpoint1, checkpoint2],
            reused=False,
            endpoint_label="Checkpoint",
            overall_goal=None,
        ))

        seg_b = segments[1]
        reuse_seg_b = False
        if seg_b and seg_b[0] == checkpoint1 and seg_b[-1] == checkpoint2:
            reuse_seg_b = True

        if reuse_seg_b:
            history.append(self.build_history_entry(
                grid=self.grid_original,
                path=seg_b,
                title="Stage 3: Reuse stored skill segment",
                advice="Robot reuses the stored skill segment from checkpoint 1 to checkpoint 2.",
                start=checkpoint1,
                end=checkpoint2,
                whole_path=combine_paths([seg_b, path_cp2_to_goal]),
                checkpoints=[checkpoint1, checkpoint2],
                reused=True,
                endpoint_label="Checkpoint",
                overall_goal=None,
            ))
            executed_segments.append({"path": seg_b, "reused": True, "title": "Checkpoint 1 → Checkpoint 2 (reused skill)"})
            print(f"  Segment B: reused skill segment (len {len(seg_b)})")
        else:
            seg_b_plan = astar(cost, checkpoint1, checkpoint2)
            if not seg_b_plan:
                raise RuntimeError("Unable to compute path from checkpoint1 to checkpoint2 when reuse not possible")
            history.append(self.build_history_entry(
                grid=self.grid_original,
                path=seg_b_plan,
                title="Stage 3: Plan segment between checkpoints",
                advice="No stored skill available for this segment; planning via A*.",
                start=checkpoint1,
                end=checkpoint2,
                whole_path=combine_paths([seg_b_plan, path_cp2_to_goal]),
                checkpoints=[checkpoint1, checkpoint2],
                reused=False,
                endpoint_label="Checkpoint",
                overall_goal=None,
            ))
            executed_segments.append({"path": seg_b_plan, "reused": False, "title": "Checkpoint 1 → Checkpoint 2"})
            print(f"  Segment B: planned (len {len(seg_b_plan)})")

        path_cp2_to_goal = astar(cost, checkpoint2, new_goal)
        if not path_cp2_to_goal:
            raise RuntimeError("Unable to compute path from checkpoint 2 to new goal")

        # Generate timestep grids according to selected mode (like test_llm_full_pipeline.py)
        if self.mode == "blocked_moving":
            # Extract a short middle segment from the path
            path_segment = path_cp2_to_goal[max(0, len(path_cp2_to_goal) // 2 - 2):min(len(path_cp2_to_goal), len(path_cp2_to_goal) // 2 + 5)]
            timestep_grids = self.corruptor.generate_moving_obstacle_sequence(self.grid_original, path_segment, num_timesteps=self.num_timesteps)
        elif self.mode == "blocked_permanent":
            path = path_cp2_to_goal
            center = path[len(path) // 2]
            x, y = center
            x1 = max(0, x - 2)
            y1 = max(0, y - 2)
            x2 = min(self.grid_original.shape[1], x + 3)
            y2 = min(self.grid_original.shape[0], y + 3)
            timestep_grids = self.corruptor.generate_permanent_obstacle_sequence(self.grid_original, (x1, y1, x2, y2), num_timesteps=self.num_timesteps)
        elif self.mode == "uncertain_clustered":
            # Generate clustered uncertainty with one cluster on the path
            timestep_grids = self.corruptor.generate_clustered_uncertainty_sequence(
                self.grid_original,
                num_timesteps=self.num_timesteps,
                num_clusters=max(3, int(self.corruptor.corruption_rate * 20)),
                cluster_radius=2,
                jitter=1,
                force_centers=[path_cp2_to_goal[len(path_cp2_to_goal) // 2]],
            )
        elif self.mode == "uncertain_sparse":
            timestep_grids = self.corruptor.inject_random_corruption(
                self.grid_original,
                corruption_rate=self.corruptor.corruption_rate,
                change_over_time=True,
                num_timesteps=self.num_timesteps
            )
        else:
            timestep_grids = self.corruptor.generate_moving_obstacle_sequence(self.grid_original, path_cp2_to_goal, num_timesteps=self.num_timesteps)

        current_time = 0
        current_grid = timestep_grids[current_time]
        current_path = path_cp2_to_goal
        consecutive_waits = 0

        # Stage 4: Initial corruption observation
        if self.mode == "blocked_moving":
            advice_stage4 = "A moving blockage is introduced on the remaining route to the goal."
        elif self.mode == "blocked_permanent":
            advice_stage4 = "A permanent blockage is placed on the remaining route; replanning may be required."
        elif self.mode == "uncertain_clustered":
            advice_stage4 = "A cluster of uncertain observations appears near the remaining route; consider waiting or replanning."
        elif self.mode == "uncertain_sparse":
            advice_stage4 = "Sparse uncertain observations appear near the route; these are likely transient."
        else:
            advice_stage4 = "Corruption injected on the remaining route to the goal."

        history.append(self.build_history_entry(
            grid=current_grid,
            path=current_path,
            title="Stage 4: Corruption injected after checkpoint 2",
            advice=advice_stage4,
            start=checkpoint2,
            end=new_goal,
            whole_path=current_path,
            checkpoints=[checkpoint2],
            reused=False,
            endpoint_label="Goal",
            overall_goal=new_goal,
            obstacle_cells=self.extract_dynamic_obstacle_cells(current_grid),
        ))

        # Stage 5: Auto-advance 2 timesteps for new_blockage scenarios (like test_llm_full_pipeline.py)
        # This gathers temporal context before the first LLM query
        situation_type = "uncertain" if self.mode == "uncertain_sparse" else "new_blockage"
        
        if situation_type == "new_blockage" and len(timestep_grids) > 2:
            for _ in range(2):
                if current_time + 1 < len(timestep_grids):
                    current_time += 1
                    current_grid = timestep_grids[current_time]
                    current_path = astar(self.make_cost_map(current_grid), checkpoint2, new_goal)
                    history.append(self.build_history_entry(
                        grid=current_grid,
                        path=current_path,
                        title=f"Stage 5: Observing timestep t={current_time}",
                        advice="The robot observes and gathers temporal context before querying the LLM.",
                        start=checkpoint2,
                        end=new_goal,
                        whole_path=current_path,
                        checkpoints=[checkpoint2],
                        reused=False,
                        endpoint_label="Goal",
                        overall_goal=new_goal,
                        obstacle_cells=self.extract_dynamic_obstacle_cells(current_grid),
                    ))
        else:
            # For uncertain_sparse, show one observation timestep
            if current_time + 1 < len(timestep_grids):
                current_time += 1
                current_grid = timestep_grids[current_time]
                current_path = astar(self.make_cost_map(current_grid), checkpoint2, new_goal)
                history.append(self.build_history_entry(
                    grid=current_grid,
                    path=current_path,
                    title=f"Stage 5: Observing timestep t={current_time}",
                    advice="The robot observes uncertain cells and gathers context.",
                    start=checkpoint2,
                    end=new_goal,
                    whole_path=current_path,
                    checkpoints=[checkpoint2],
                    reused=False,
                    endpoint_label="Goal",
                    overall_goal=new_goal,
                    obstacle_cells=self.extract_dynamic_obstacle_cells(current_grid),
                ))

        # LLM decision loop (like test_llm_full_pipeline.py)
        decision_loop_count = 0
        while decision_loop_count < 10:
            decision_loop_count += 1
            remaining_grids = timestep_grids[current_time:]
            if len(remaining_grids) == 0:
                break

            # Query LLM with remaining timesteps (full temporal data)
            if len(remaining_grids) > 1:
                modified_grid, decision = self.replanner.replan_temporal(
                    grids=remaining_grids,
                    start=checkpoint2,
                    goal=new_goal,
                    original_path=current_path,
                    robot_pose={"x": checkpoint2[0], "y": checkpoint2[1]},
                    situation_type=situation_type
                )
            else:
                modified_grid, decision = self.replanner.replan(
                    current_grid.copy(),
                    checkpoint2,
                    new_goal,
                    current_path,
                    {"x": checkpoint2[0], "y": checkpoint2[1]},
                    original_path=current_path,
                    situation_type=situation_type
                )

            action = decision.get("recommended_action", "uncertain") if decision else "uncertain"
            reason = decision.get("reason", "") if decision else "No decision"
            
            history.append(self.build_history_entry(
                grid=current_grid,
                path=current_path,
                title=f"Stage 6: LLM query and decision",
                advice=f"LLM recommends {action}. Reason: {reason}",
                start=checkpoint2,
                end=new_goal,
                whole_path=current_path,
                checkpoints=[checkpoint2],
                reused=False,
                endpoint_label="Goal",
                overall_goal=new_goal,
                obstacle_cells=self.extract_dynamic_obstacle_cells(current_grid),
            ))

            if action in ("keep_moving", "plan_through"):
                history.append(self.build_history_entry(
                    grid=current_grid,
                    path=current_path,
                    title="Stage 7: Action execution - keep moving",
                    advice="The LLM decides to continue on the current route and reach the goal.",
                    start=checkpoint2,
                    end=new_goal,
                    whole_path=current_path,
                    checkpoints=[checkpoint2],
                    reused=False,
                    endpoint_label="Goal",
                    overall_goal=new_goal,
                    obstacle_cells=self.extract_dynamic_obstacle_cells(current_grid),
                ))
                break

            if action in ("wait", "wait_and_reinspect", "inspect"):
                consecutive_waits += 1
                next_time = current_time + 1
                if next_time >= len(timestep_grids):
                    current_path = astar(self.make_cost_map(current_grid), checkpoint2, new_goal)
                    history.append(self.build_history_entry(
                        grid=current_grid,
                        path=current_path,
                        title="Stage 7: No more observations",
                        advice="The robot has no further timesteps to observe and must act on current knowledge.",
                        start=checkpoint2,
                        end=new_goal,
                        whole_path=current_path,
                        checkpoints=[checkpoint2],
                        reused=False,
                        endpoint_label="Goal",
                        overall_goal=new_goal,
                        obstacle_cells=self.extract_dynamic_obstacle_cells(current_grid),
                    ))
                    break
                current_time = next_time
                current_grid = timestep_grids[current_time]
                current_path = astar(self.make_cost_map(current_grid), checkpoint2, new_goal)
                history.append(self.build_history_entry(
                    grid=current_grid,
                    path=current_path,
                    title=f"Stage 7: Re-observing timestep t={current_time}",
                    advice="The robot waits and observes again before querying the LLM.",
                    reused=False,
                    endpoint_label="Goal",
                    overall_goal=new_goal,
                    obstacle_cells=self.extract_dynamic_obstacle_cells(current_grid),
                ))
                continue

            if action in ("replan", "replan_immediately", "avoid"):
                current_grid = modified_grid
                new_cost = self.make_cost_map(current_grid)
                replanned_path = astar(new_cost, checkpoint2, new_goal)
                current_path = replanned_path
                history.append(self.build_history_entry(
                    grid=current_grid,
                    path=current_path,
                    title="Stage 7: Action execution - replan",
                    advice="The LLM decides the blockage is permanent and replans around it.",
                    start=checkpoint2,
                    end=new_goal,
                    whole_path=current_path,
                    checkpoints=[checkpoint2],
                    reused=False,
                    endpoint_label="Goal",
                    overall_goal=new_goal,
                    obstacle_cells=self.extract_dynamic_obstacle_cells(current_grid),
                ))
                break

            break

        history.append(self.build_history_entry(
            grid=current_grid,
            path=current_path,
            title="Stage 8: Integration finished",
            advice="Skill storage and LLM planning have been exercised in sequence.",
            start=checkpoint2,
            end=new_goal,
            whole_path=current_path,
            checkpoints=[checkpoint2],
            reused=False,
            endpoint_label="Goal",
            overall_goal=new_goal,
            obstacle_cells=self.extract_dynamic_obstacle_cells(current_grid),
        ))

        if show_window and MATPLOTLIB_AVAILABLE:
            display_history(history)
        elif not MATPLOTLIB_AVAILABLE:
            print("matplotlib is not installed; skipping visualization")

        return history


def display_history(history):
    # base colormap: uncertain/background grey, free white, occupied black
    cmap = ListedColormap(["#999999", "#ffffff", "#000000"])
    norm = BoundaryNorm([-1, 0, 1, 2], cmap.N)

    plt.ion()
    for index, step in enumerate(history):
        fig, ax = plt.subplots(figsize=(10, 10))
        try:
            ax.imshow(step["grid"], cmap=cmap, norm=norm, origin="lower")
        except ValueError as e:
            if "BoundaryNorm" in str(e):
                ax.imshow(step["grid"], cmap=cmap, origin="lower")
            else:
                raise

        if step.get("whole_path"):
            whole = step["whole_path"]
            if whole:
                wx, wy = zip(*whole)
                ax.plot(wx, wy, color="#ff0000", linestyle="--", linewidth=2, alpha=0.5, label="Remaining route")

        path = step.get("path")
        if path:
            xs, ys = zip(*path)
            ax.plot(xs, ys, color="#ffff00", linewidth=3, alpha=0.9, label="Current segment")
            ax.plot(xs, ys, color="#ffff00", linewidth=5, alpha=0.3)

        if step.get("checkpoints"):
            cp_xs, cp_ys = zip(*step["checkpoints"])
            ax.scatter(cp_xs, cp_ys, c="#ff9900", s=200, marker="D", edgecolors="#000000", linewidths=2, label="Stored checkpoint", zorder=6)
            for i, (cx, cy) in enumerate(step["checkpoints"], start=1):
                ax.annotate(f"CP{i}", xy=(cx, cy), xytext=(6, 6), textcoords="offset points",
                           fontsize=10, fontweight='bold', color="#ff9900",
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Overlay uncertain cells (injected) in red so noise is visible. Use stored list
        uncertain_cells = step.get("uncertain_cells") or []
        if uncertain_cells:
            ux, uy = zip(*uncertain_cells)
            ax.scatter(ux, uy, c="#ff3333", s=80, marker="s", edgecolors='black', linewidths=0.4,
                       alpha=0.95, label="Uncertain / noisy observation", zorder=7)

        # Overlay newly observed obstacles (grid value == 1 but original grid was free)
        obstacle_cells = step.get("obstacle_cells") or []
        if obstacle_cells:
            ox, oy = zip(*obstacle_cells)
            ax.scatter(ox, oy, c="#aa0000", s=140, marker="s", edgecolors='black', linewidths=0.7,
                       alpha=0.95, label="Observed moving obstacle", zorder=8)

        start = step.get("start")
        if start:
            ax.scatter([start[0]], [start[1]], c="#00ff00", s=250, marker="*", edgecolors="#000000", linewidths=2, label="Start", zorder=6)

        # Prefer displaying the overall final goal (X) if provided so it's visible across stages
        goal_cell = step.get("overall_goal") or step.get("end")
        if goal_cell:
            label = step.get("endpoint_label", "Goal")
            color = "#0000ff" if label == "Goal" else "#ff9900"
            marker = "X" if label == "Goal" else "D"
            ax.scatter([goal_cell[0]], [goal_cell[1]], c=color, s=250, marker=marker, edgecolors="#000000", linewidths=2, label=label, zorder=6)
            if label != "Goal":
                ax.annotate(label, xy=(goal_cell[0], goal_cell[1]), xytext=(6, 6), textcoords="offset points",
                           fontsize=10, fontweight='bold', color=color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        if step.get("is_reused"):
            ax.text(0.5, 0.95, "REUSED SKILL SEGMENT", transform=ax.transAxes,
                    fontsize=12, fontweight='bold', color="#cc0000", ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_title(step.get("title", ""), fontsize=13, fontweight='bold', pad=15)
        if step.get("advice"):
            ax.text(0.02, 0.02, step["advice"], transform=ax.transAxes, fontsize=11,
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        ax.set_xlim(-0.5, step["grid"].shape[1] - 0.5)
        ax.set_ylim(-0.5, step["grid"].shape[0] - 0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper left', fontsize=9)

        if index < len(history) - 1:
            response = input(f"\n→ Stage {index + 1}/{len(history)}: Press Enter to continue, or 'q' to quit: ")
            if response.lower() == 'q':
                plt.close('all')
                print("Visualization closed by user")
                return
        else:
            input(f"\n→ Final stage {index + 1}/{len(history)}: Press Enter to close: ")

        plt.close(fig)

    plt.ioff()
    print("Visualization complete")


def main():
    parser = argparse.ArgumentParser(description="Run an integrated LLM planner + skill storage visualization test")
    parser.add_argument("--no-window", action="store_true", help="Do not open visualization windows")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. If omitted, the test uses a random seed.")
    parser.add_argument("--mode", choices=["uncertain_sparse", "uncertain_clustered", "blocked_moving", "blocked_permanent"], default="blocked_moving", help="Type of corruption/mode for the test")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"
    integrator = LLMSkillStorageIntegration(map_path=str(map_path), seed=args.seed, mode=args.mode)
    history = integrator.run(show_window=not args.no_window)
    print(json.dumps({
        "steps": len(history),
        "last_title": history[-1]["title"] if history else None,
    }, indent=2))


if __name__ == "__main__":
    main()
