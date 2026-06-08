#!/usr/bin/env python3
"""Run an end-to-end LLM-guided navigation pipeline and visualize the decision loop."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib import patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
import logging

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Silence noisy BoundaryNorm invertible exceptions raised inside Matplotlib
if MATPLOTLIB_AVAILABLE:
    try:
        import matplotlib.cbook as _cbook

        _orig_exc_printer = getattr(_cbook, "_exception_printer", None)

        def _filtered_exception_printer(exc):
            try:
                if isinstance(exc, ValueError) and "BoundaryNorm is not invertible" in str(exc):
                    logger.debug("Suppressed Matplotlib BoundaryNorm invertible ValueError in event handler")
                    return
            except Exception:
                pass
            if _orig_exc_printer:
                return _orig_exc_printer(exc)

        if _orig_exc_printer is not None:
            _cbook._exception_printer = _filtered_exception_printer
    except Exception:
        # If we cannot patch matplotlib, don't fail — it's optional for visualization
        logger.debug("Could not install BoundaryNorm exception filter", exc_info=True)
    # Additionally, wrap traceback.print_exc to suppress noisy BoundaryNorm prints
    try:
        import traceback, sys as _sys
        _orig_print_exc = traceback.print_exc

        def _filtered_print_exc(limit=None, file=None, chain=True):
            exc = _sys.exc_info()[1]
            try:
                if isinstance(exc, ValueError) and "BoundaryNorm is not invertible" in str(exc):
                    logger.debug("Suppressed traceback.print_exc for BoundaryNorm not invertible")
                    return
            except Exception:
                pass
            return _orig_print_exc(limit=limit, file=file, chain=chain)

        traceback.print_exc = _filtered_print_exc
    except Exception:
        logger.debug("Could not patch traceback.print_exc", exc_info=True)
    # Also wrap the CallbackRegistry.process to catch BoundaryNorm ValueError
    try:
        import matplotlib.backend_bases as _backend_bases
        _orig_cr_process = getattr(_backend_bases.CallbackRegistry, 'process', None)

        def _filtered_cr_process(self, s, *args, **kwargs):
            try:
                return _orig_cr_process(self, s, *args, **kwargs)
            except Exception as e:
                try:
                    if isinstance(e, ValueError) and 'BoundaryNorm is not invertible' in str(e):
                        logger.debug('Suppressed BoundaryNorm ValueError in CallbackRegistry.process')
                        return
                except Exception:
                    pass
                raise

        if _orig_cr_process is not None:
            _backend_bases.CallbackRegistry.process = _filtered_cr_process
    except Exception:
        logger.debug('Could not patch CallbackRegistry.process', exc_info=True)
    # As a last-resort simple filter, replace sys.excepthook to catch the specific
    # BoundaryNorm ValueError and print a short warning instead of a full traceback.
    try:
        import sys as _sys
        _orig_excepthook = _sys.excepthook

        def _simple_excepthook(exc_type, exc, tb):
            try:
                if exc_type is ValueError and "BoundaryNorm is not invertible" in str(exc):
                    # Compact warning instead of long traceback
                    logger.warning("BoundaryNorm not invertible (suppressed): %s", exc)
                    return
            except Exception:
                pass
            return _orig_excepthook(exc_type, exc, tb)

        _sys.excepthook = _simple_excepthook
    except Exception:
        logger.debug('Could not install simple sys.excepthook filter', exc_info=True)

from final_proj.environment.map_loader import MapLoader
from final_proj.environment.corruption import Corruptor
from final_proj.environment.context_extractor import ContextExtractor
from final_proj.planning.astar import astar
from final_proj.llm.prompt_builder import PromptBuilder
from final_proj.llm.response_parser import ResponseParser
from final_proj.llm.client import LLMClient
from final_proj.planning.replanner import Replanner


class FullPipelineSimulator:
    MODES = ["uncertain_sparse", "uncertain_clustered", "blocked_moving", "blocked_permanent"]

    def __init__(self, map_path, noise_rate=0.05, sparse_rate=None, num_timesteps=3, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.map_loader = MapLoader(map_path)
        self.grid_original = self.map_loader.get_grid().copy()
        self.noise_rate = noise_rate
        self.sparse_rate = sparse_rate if sparse_rate is not None else noise_rate
        self.num_timesteps = num_timesteps
        self.corruptor = Corruptor(corruption_rate=self.noise_rate)
        self.replanner = Replanner(PromptBuilder(), ResponseParser(), LLMClient(), ContextExtractor())

    def find_random_valid_cells(self):
        free_cells = np.where(self.grid_original == 0)
        free_cells = list(zip(free_cells[1], free_cells[0]))
        start = tuple(int(v) for v in random.choice(free_cells))
        goal = tuple(int(v) for v in random.choice(free_cells))
        while goal == start:
            goal = tuple(int(v) for v in random.choice(free_cells))
        return start, goal

    def make_cost_map(self, grid):
        cost_map = np.full(grid.shape, np.inf, dtype=float)
        cost_map[grid == 0] = 1.0
        cost_map[grid == -1] = 5.0
        return cost_map

    def generate_scenario(self, mode, start, goal):
        grid = self.grid_original.copy()

        if mode == "uncertain_sparse":
            return self.corruptor.inject_random_corruption(
                grid,
                corruption_rate=self.sparse_rate,
                change_over_time=True,
                num_timesteps=self.num_timesteps
            )

        if mode == "uncertain_clustered":
            corrupted_grid = self.generate_clustered_uncertainty(grid, start, goal)
            return [corrupted_grid.copy() for _ in range(self.num_timesteps)]

        if mode == "blocked_moving":
            path = astar(self.make_cost_map(grid), start, goal)
            if not path:
                return [grid.copy() for _ in range(self.num_timesteps)]
            path_segment = path[max(0, len(path) // 2 - 2):min(len(path), len(path) // 2 + 5)]
            num_steps = max(self.num_timesteps, 5)
            if num_steps != self.num_timesteps:
                print(f"Note: using {num_steps} timesteps for blocked_moving to show obstacle movement")
            return self.corruptor.generate_moving_obstacle_sequence(grid, path_segment, num_timesteps=num_steps)

        if mode == "blocked_permanent":
            path = astar(self.make_cost_map(grid), start, goal)
            if not path:
                return [grid.copy() for _ in range(self.num_timesteps)]
            center = path[len(path) // 2]
            x, y = center
            x1 = max(0, x - 2)
            y1 = max(0, y - 2)
            x2 = min(self.grid_original.shape[1], x + 3)
            y2 = min(self.grid_original.shape[0], y + 3)
            return self.corruptor.generate_permanent_obstacle_sequence(grid, (x1, y1, x2, y2), num_timesteps=self.num_timesteps)

        return [grid.copy() for _ in range(self.num_timesteps)]

    def generate_clustered_uncertainty(self, grid, start, goal, radius=3):
        corrupted = grid.copy()
        path = astar(self.make_cost_map(grid), start, goal)
        if not path:
            return corrupted

        center = path[len(path) // 2]
        x, y = center
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(self.grid_original.shape[1], x + radius + 1)
        y2 = min(self.grid_original.shape[0], y + radius + 1)
        return self.corruptor.inject_block(corrupted, (x1, y1), (x2, y2), blocked_value=-1)

    def describe_action(self, action):
        if action in ("wait", "wait_and_reinspect", "inspect"):
            return "wait and re-inspect"
        if action in ("replan", "replan_immediately", "avoid"):
            return "replan"
        if action in ("keep_moving", "plan_through"):
            return "keep moving"
        return action

    def _mode_description(self, mode):
        descriptions = {
            "uncertain_sparse": "Sparse random sensor noise appears along the path; the LLM should decide whether to continue, pause, or replan based on uncertainty.",
            "uncertain_clustered": "A localized cluster of uncertain/noisy cells occludes the path; the LLM should decide whether to proceed or wait for more information.",
            "blocked_moving": "A temporary moving blockage passes over the path; the LLM should decide whether to wait for it to move and then proceed.",
            "blocked_permanent": "A permanent obstacle blocks the path; the LLM should decide to replan around the blockage.",
        }
        return descriptions.get(mode, "A temporal navigation scenario with corrupted observations.")

    def _trim_path_at_corruption(self, path, grid):
        if not path or grid is None:
            return path

        trimmed = []
        for x, y in path:
            trimmed.append((x, y))
            if grid[y, x] != 0:
                break
        return trimmed

    def run(self, mode=None, show_window=True):
        if mode is None or mode == "random":
            mode = random.choice(self.MODES)

        start, goal = self.find_random_valid_cells()
        original_cost = self.make_cost_map(self.grid_original)
        original_path = astar(original_cost, start, goal)
        if original_path is None:
            raise RuntimeError("Failed to find original path on clean map")

        timestep_grids = self.generate_scenario(mode, start, goal)
        current_time = 0
        history = []
        
        # Stage 1: Show original and corrupted grids
        print("=" * 60)
        print(f"STAGE 1: CORRUPTION SETUP")
        print("=" * 60)
        print(f"Scenario: mode={mode}, start={start}, goal={goal}")
        
        # Determine corruption type and expected answer
        if mode == "uncertain_sparse":
            corruption_desc = "Sparse random sensor noise (~noise varies per timestep)"
            expected_answer = "keep_moving or wait (depends on noise severity)"
        elif mode == "uncertain_clustered":
            corruption_desc = "Clustered sensor uncertainty (fixed noise region)"
            expected_answer = "keep_moving or wait (depends on cost)"
        elif mode == "blocked_moving":
            corruption_desc = "Moving obstacle (3x3 block moves across and then off the path)"
            expected_answer = "wait_and_reinspect then plan_through if the path clears"
        else:  # blocked_permanent
            corruption_desc = "Permanent blockage (static 3x3 obstacle)"
            expected_answer = "replan immediately (permanent blockage)"
        
        print(f"Corruption type: {corruption_desc}")
        print(f"Expected LLM answer: {expected_answer}")
        print(f"Mode description: {self._mode_description(mode)}")
        print(f"Original path length: {len(original_path)}")
        
        history.append({
            "grid": self.grid_original,
            "path": original_path,
            "planned_path": original_path,
            "actual_path": original_path,
            "display_path": original_path,
            "step_type": "original",
            "title": f"ORIGINAL PATH (clean map)\n{corruption_desc}",
            "advice": "Baseline A* planner"
        })

        current_grid = timestep_grids[current_time]
        current_path = astar(self.make_cost_map(current_grid), start, goal)
        current_plan_path = original_path
        history.append({
            "grid": current_grid,
            "path": current_plan_path,
            "planned_path": current_plan_path,
            "actual_path": self._trim_path_at_corruption(current_plan_path, current_grid),
            "display_path": self._trim_path_at_corruption(current_plan_path, current_grid),
            "step_type": "corrupted",
            "title": f"Corrupted map at t={current_time}\nExpected answer: {expected_answer}",
            "advice": ""
        })

        situation_type = "uncertain" if mode == "uncertain_sparse" else "new_blockage"
        reached = False
        loop_count = 0
        consecutive_waits = 0

        # For blocked scenarios, auto-advance 2 timesteps to gather temporal context before first query
        if situation_type == "new_blockage" and len(timestep_grids) > 2:
            print("\n" + "=" * 60)
            print(f"STAGE 2: TIMESTEP ADVANCEMENT")
            print("=" * 60)
            print("Auto-advancing 2 timesteps to gather temporal context...")
            for _ in range(2):
                if current_time + 1 < len(timestep_grids):
                    current_time += 1
                    current_grid = timestep_grids[current_time]
                    current_path = astar(self.make_cost_map(current_grid), start, goal)
                    history.append({
                        "grid": current_grid,
                        "path": current_plan_path,
                        "planned_path": current_plan_path,
                        "actual_path": self._trim_path_at_corruption(current_plan_path, current_grid),
                        "display_path": self._trim_path_at_corruption(current_plan_path, current_grid),
                        "step_type": "observation",
                        "title": f"Observing timestep t={current_time}\n(gathering temporal context)",
                        "advice": ""
                    })
                    print(f"  → Advanced to t={current_time}")
            print(f"Ready to query LLM with temporal data at t={current_time}")
        else:
            print("\nNo auto-advance needed (uncertain scenario or insufficient timesteps)")

        while loop_count < 10:
            loop_count += 1
            remaining_grids = timestep_grids[current_time:]
            if len(remaining_grids) == 0:
                break

            print("\n" + "=" * 60)
            print(f"STAGE 3: LLM QUERY & DECISION (Loop {loop_count})")
            print("=" * 60)
            print(f"Current timestep: t={current_time}")
            print(f"Remaining timesteps available: {len(remaining_grids)}")

            if len(remaining_grids) > 1:
                modified_grid, decision = self.replanner.replan_temporal(
                    grids=remaining_grids,
                    start=start,
                    goal=goal,
                    original_path=original_path,
                    robot_pose={"x": start[0], "y": start[1]},
                    situation_type=situation_type
                )
            else:
                modified_grid, decision = self.replanner.replan(
                    current_grid.copy(),
                    start,
                    goal,
                    current_path or [],
                    {"x": start[0], "y": start[1]},
                    original_path=original_path,
                    situation_type=situation_type
                )

            action = decision.get("recommended_action", "uncertain")
            reason = decision.get('reason','')
            advice_text = f"LLM: {action}\nReason: {reason}"
            
            print(f"\n→ LLM Decision:")
            print(f"  Action: {action}")
            print(f"  Reason: {reason}")
            
            history.append({
                "grid": current_grid,
                "path": current_plan_path,
                "planned_path": current_plan_path,
                "actual_path": self._trim_path_at_corruption(current_plan_path, current_grid),
                "display_path": self._trim_path_at_corruption(current_plan_path, current_grid),
                "step_type": "query",
                "title": f"At t={current_time}: LLM Query\n{self.describe_action(action).upper()}",
                "advice": advice_text
            })

            if action in ("keep_moving", "plan_through"):
                print("\n" + "=" * 60)
                print(f"STAGE 4: ACTION EXECUTION - KEEP MOVING")
                print("=" * 60)
                reached = current_path is not None
                print(f"✓ Following current path to goal")
                print(f"  Path exists: {reached}")
                history.append({
                    "grid": current_grid,
                    "path": current_path,
                    "planned_path": current_path,
                    "actual_path": current_path,
                    "display_path": current_path,
                    "step_type": "keep_moving",
                    "title": f"Action: KEEP MOVING\n(Path length: {len(current_path) if current_path else 0})",
                    "advice": advice_text
                })
                break

            if action in ("wait", "wait_and_reinspect", "inspect"):
                consecutive_waits += 1
                print(f"\n→ Waiting and re-inspecting...")
                print(f"  Wait count: {consecutive_waits}/2")
                
                if consecutive_waits >= 2:
                    print(f"  ✓ Waited 2 timesteps - will query LLM again after advancing")
                    consecutive_waits = 0
                
                next_time = current_time + 1
                if next_time >= len(timestep_grids):
                    print("  ✗ No further timesteps available - finishing")
                    history.append({
                        "grid": current_grid,
                        "path": current_path,
                        "planned_path": current_path,
                        "actual_path": self._trim_path_at_corruption(current_path, current_grid),
                        "display_path": self._trim_path_at_corruption(current_path, current_grid),
                        "step_type": "final",
                        "title": "No more observations available",
                        "advice": advice_text
                    })
                    break
                
                current_time = next_time
                current_grid = timestep_grids[current_time]
                current_path = astar(self.make_cost_map(current_grid), start, goal)
                next_title = "Advanced to t={time}\nWait & Re-inspect".format(time=current_time)
                if mode == "blocked_moving":
                    next_title = f"Advanced to t={current_time}\nObstacle may have moved; re-evaluate"
                elif mode == "blocked_permanent":
                    next_title = f"Advanced to t={current_time}\nChecking if path still blocked"
                history.append({
                    "grid": current_grid,
                    "path": current_plan_path,
                    "planned_path": current_plan_path,
                    "actual_path": self._trim_path_at_corruption(current_plan_path, current_grid),
                    "display_path": self._trim_path_at_corruption(current_plan_path, current_grid),
                    "step_type": "observation",
                    "title": next_title,
                    "advice": f"Waiting... (count={consecutive_waits})"
                })
                continue

            if action in ("replan", "replan_immediately", "avoid"):
                print("\n" + "=" * 60)
                print(f"STAGE 4: ACTION EXECUTION - REPLAN")
                print("=" * 60)
                print(f"✓ Replanning with modified grid")
                current_grid = modified_grid
                current_path = astar(self.make_cost_map(current_grid), start, goal)
                current_plan_path = current_path
                history.append({
                    "grid": current_grid,
                    "path": current_plan_path,
                    "planned_path": current_plan_path,
                    "actual_path": current_plan_path,
                    "display_path": current_plan_path,
                    "step_type": "replan",
                    "title": f"Replanned path\n(new length: {len(current_plan_path) if current_plan_path else 0})",
                    "advice": advice_text
                })
                reached = current_path is not None
                print(f"  New path exists: {reached}")
                break

            history.append({
                "grid": current_grid,
                "path": current_path,
                "planned_path": current_path,
                "actual_path": self._trim_path_at_corruption(current_path, current_grid),
                "display_path": self._trim_path_at_corruption(current_path, current_grid),
                "step_type": "final",
                "title": "Action inconclusive",
                "advice": advice_text
            })
            break

        print("\n" + "=" * 60)
        print(f"STAGE 5: FINAL RESULT")
        print("=" * 60)
        print(f"Pipeline completed after {loop_count} decision loops")
        print(f"Total history entries: {len(history)}")
        print(f"Final path exists: {current_path is not None}")
        print(f"Success: {reached}")
        print("=" * 60)

        # Add final result to history
        history.append({
            "grid": current_grid,
            "path": current_plan_path,
            "planned_path": current_plan_path,
            "actual_path": current_plan_path,
            "display_path": current_plan_path,
            "step_type": "final",
            "title": f"FINAL: Success={reached}, Path={'exists' if current_plan_path else 'not found'}",
            "advice": "Pipeline complete"
        })

        if show_window:
            self.display_history(history, start, goal, mode)

        return {
            "mode": mode,
            "start": start,
            "goal": goal,
            "history": history,
            "success": reached,
            "final_path_exists": current_path is not None
        }

    def display_history_animated(self, history, start, goal, mode):
        """Display pipeline step-by-step with animation"""
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib is not installed; cannot display visualization.")
            for step in history:
                print(step["title"], step["advice"])
            return

        cmap = ListedColormap(["#999999", "#ffffff", "#000000"])
        norm = BoundaryNorm([-1, 0, 1, 2], cmap.N)

        plt.ion()  # Enable interactive mode
        
        # Group history into logical stages
        stages = self._group_history_into_stages(history, mode)
        
        stage_descriptions = {
            0: "Corruption Setup - Original and corrupted maps with expected answer",
            1: "Corruption View - Detailed corruption details",
            2: "Timestep Advancement - Robot observing over time",
            3: "LLM Query & Decision - What the LLM sees and decides",
            4: "Action Execution - Robot follows LLM advice",
            5: "Final Result - Pipeline completion",
        }
        
        for stage_idx, stage in enumerate(stages):
            # Create figure for this stage
            n_items = len(stage)
            cols = min(n_items, 2)
            rows = (n_items + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = list(axes)
            else:
                axes = [ax for row in axes for ax in row]
            
            # Hide unused axes
            for ax in axes[n_items:]:
                ax.axis('off')
            
            # Plot each item in the stage
            for item_idx, step in enumerate(stage):
                ax = axes[item_idx]
                grid = step["grid"]
                path = step["path"]
                title = step["title"]
                advice = step["advice"]
                
                # Draw grid (use BoundaryNorm when possible; fallback on error)
                try:
                    im = ax.imshow(grid, cmap=cmap, norm=norm, origin="lower")
                except ValueError as e:
                    msg = str(e)
                    if "BoundaryNorm" in msg or "not invertible" in msg or "invertible" in msg:
                        logger.warning("BoundaryNorm not invertible; falling back to default imshow. Error: %s", e)
                        im = ax.imshow(grid, cmap=cmap, origin="lower")
                    else:
                        raise

                # Highlight corruption relative to the original clean map
                corrupted_mask = (grid != self.grid_original) & (grid != 0)
                if np.any(corrupted_mask):
                    ys, xs = np.where(corrupted_mask)
                    ax.scatter(xs, ys, c="#ff0000", s=30, marker="s", alpha=0.4, edgecolors="none", zorder=4)
                
                # Draw planned full path (lighter line)
                planned_path = step.get("planned_path", path)
                if planned_path and len(planned_path) > 0:
                    xs, ys = zip(*planned_path)
                    ax.plot(xs, ys, color="#ffff66", linewidth=2, alpha=0.25, zorder=3)

                # Draw actual robot motion path (stronger dashed/orange to show executed route)
                actual_path = step.get("actual_path", step.get("display_path", path))
                if actual_path and len(actual_path) > 0:
                    xs, ys = zip(*actual_path)
                    if step.get("step_type") == "original":
                        line_style = '-'
                        line_color = "#ffff00"
                    else:
                        line_style = '--'
                        line_color = "#ffb600"
                    ax.plot(xs, ys, color=line_color, linewidth=3, linestyle=line_style,
                            marker="o", markersize=5, markerfacecolor=line_color,
                            markeredgecolor="#000000", alpha=0.9, zorder=5)

                # Draw a red rectangle around the moving blockage for blocked_moving scenarios
                if mode == "blocked_moving" and np.any(corrupted_mask):
                    ys_block, xs_block = np.where(corrupted_mask)
                    x_min, x_max = xs_block.min(), xs_block.max()
                    y_min, y_max = ys_block.min(), ys_block.max()
                    width = x_max - x_min + 1
                    height = y_max - y_min + 1
                    rect = patches.Rectangle((x_min - 0.5, y_min - 0.5), width, height,
                                             linewidth=2, edgecolor="#ff0000", facecolor="none",
                                             linestyle=":" , zorder=6)
                    ax.add_patch(rect)
                    tag_x = min(x_max + 1, grid.shape[1] - 1)
                    tag_y = min(y_max + 1, grid.shape[0] - 1)
                    ax.annotate("Moving obstacle", xy=((x_min + x_max) / 2, (y_min + y_max) / 2),
                                xytext=(tag_x, tag_y), color="#ff0000", fontsize=10,
                                arrowprops=dict(arrowstyle="->", color="#ff0000"), zorder=7)

                # Draw start and goal
                ax.scatter([start[0]], [start[1]], c="#00ff00", s=150, marker="*", edgecolors="#000000", linewidths=1.5, label="Start", zorder=5)
                ax.scatter([goal[0]], [goal[1]], c="#0000ff", s=150, marker="X", edgecolors="#000000", linewidths=1.5, label="Goal", zorder=5)

                # Set title and info text
                ax.set_title(title, fontsize=11, fontweight='bold', wrap=True, pad=10)
                if advice:
                    ax.text(0.02, 0.02, advice, transform=ax.transAxes, fontsize=9, 
                           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                if step.get("step_type") == "original":
                    ax.text(0.02, 0.95, "Start = green star\nGoal = blue X\nBaseline A* planner", transform=ax.transAxes, fontsize=9,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.02, 0.95, "Start = green star\nGoal = blue X", transform=ax.transAxes, fontsize=9,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                legend_text = "Red = current blockage\nYellow solid = planned path\nOrange dashed = actual motion"
                ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
                        fontsize=8, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Wait for user or auto-advance
            if stage_idx < len(stages) - 1:
                response = input(f"\n→ Press Enter to continue to Stage {stage_idx + 2}, or 'q' to quit: ")
                if response.lower() == 'q':
                    plt.close('all')
                    print("Visualization closed by user")
                    return
            else:
                input(f"\n→ Pipeline complete! Press Enter to close visualization: ")
            
            plt.close(fig)
        
        plt.ioff()

    def _group_history_into_stages(self, history, mode):
        """Group history entries into logical stages for animated display"""
        if not history:
            return []
        
        stages = []
        
        # Stage grouping logic - use title keywords to identify stages
        stage_keywords = {
            "ORIGINAL": 0,           # Original path
            "Corrupted": 1,          # Corrupted map
            "Observing": 2,          # Timestep advancement
            "LLM Query": 3,          # LLM query
            "Advanced": 3,           # Timestep during waiting
            "KEEP": 4,               # Keep moving action
            "Replanned": 4,          # Replan action
            "FINAL": 5,              # Final result
        }
        
        current_stage_idx = -1
        current_stage_items = []
        
        for step in history:
            title = step["title"]
            
            # Determine stage from keywords
            stage_idx = -1
            for keyword, idx in stage_keywords.items():
                if keyword in title:
                    stage_idx = idx
                    break
            
            # If stage changed, save previous stage and start new one
            if stage_idx != -1 and stage_idx != current_stage_idx:
                if current_stage_items:
                    stages.append(current_stage_items)
                current_stage_items = [step]
                current_stage_idx = stage_idx
            else:
                current_stage_items.append(step)
        
        if current_stage_items:
            stages.append(current_stage_items)
        
        # Ensure we have stages
        if not stages:
            stages = [[step] for step in history]
        
        return stages

    def display_history(self, history, start, goal, mode):
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib is not installed; cannot display visualization.")
            for step in history:
                print(step["title"], step["advice"])
            return

        self.display_history_animated(history, start, goal, mode)


def main():
    parser = argparse.ArgumentParser(description="Run a full LLM-guided navigation pipeline and visualize the decision loop")
    parser.add_argument("--mode", choices=FullPipelineSimulator.MODES + ["random"], default="random", help="Scenario mode")
    parser.add_argument("--noise", type=float, default=0.05, help="Base noise/corruption rate")
    parser.add_argument("--sparse-rate", type=float, default=None, help="Sparse uncertainty rate")
    parser.add_argument("--num-timesteps", type=int, default=3, help="Number of timesteps for temporal scenarios")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-window", action="store_true", help="Do not display visualizations")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"

    simulator = FullPipelineSimulator(
        map_path=str(map_path),
        noise_rate=args.noise,
        sparse_rate=args.sparse_rate,
        num_timesteps=args.num_timesteps,
        seed=args.seed
    )
    result = simulator.run(mode=args.mode, show_window=not args.no_window)
    print(json.dumps({
        "mode": result["mode"],
        "start": result["start"],
        "goal": result["goal"],
        "success": result["success"],
        "final_path_exists": result["final_path_exists"],
        "steps": len(result["history"])
    }, indent=2))


if __name__ == "__main__":
    main()
