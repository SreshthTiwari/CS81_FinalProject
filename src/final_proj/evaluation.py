#!/usr/bin/env python3
"""
Comprehensive evaluation script comparing:
1. Baseline A* planner
2. A* with map corruption
3. Full system (LLM replanner + skill memory)

Metrics collected:
- Success rate (reached goal or not)
- Path length (cells traveled)
- Number of replans
- Time to goal
- Replanning time
"""

import argparse
import json
import time
import random
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from final_proj.environment.map_loader import MapLoader
from final_proj.environment.corruption import Corruptor
from final_proj.environment.context_extractor import ContextExtractor
from final_proj.planning.astar import astar
from final_proj.memory.skill_store import SkillStore

# Optional LLM imports
try:
    from final_proj.planning.replanner import Replanner
    from final_proj.llm.prompt_builder import PromptBuilder
    from final_proj.llm.response_parser import ResponseParser
    from final_proj.llm.client import LLMClient
    HAS_LLM = True
except (ImportError, ModuleNotFoundError) as e:
    print(f"[WARN] LLM modules not available: {e}")
    Replanner = None
    PromptBuilder = None
    ResponseParser = None
    LLMClient = None
    HAS_LLM = False


@dataclass
class NavigationMetrics:
    """Metrics for a single navigation trial"""
    scenario: str  # baseline, corruption, full_system
    trial_id: int
    start_cell: tuple
    goal_cell: tuple
    success: bool
    path_length: int
    num_replans: int
    total_time: float
    replanning_time: float
    path: list = None
    corrupted_cells: list = None
    
    def to_dict(self):
        d = asdict(self)
        d['path'] = len(self.path) if self.path else 0
        d['corrupted_cells'] = len(self.corrupted_cells) if self.corrupted_cells else 0
        d['start_cell'] = list(d['start_cell']) if d['start_cell'] else None
        d['goal_cell'] = list(d['goal_cell']) if d['goal_cell'] else None
        # Convert numpy types to native Python types
        for key in d:
            if isinstance(d[key], (np.integer, np.floating)):
                d[key] = float(d[key]) if isinstance(d[key], np.floating) else int(d[key])
        return d


class NavigationEvaluator:
    def __init__(self, map_path: str):
        """Initialize evaluator with a map"""
        self.map_loader = MapLoader(map_path)
        self.grid_original = self.map_loader.get_grid().copy()
        self.resolution = self.map_loader.get_resolution()
        self.origin = self.map_loader.get_origin()
        
        # Load skill store
        pkg_root = Path(__file__).resolve().parent  # evaluation.py's directory
        skills_path = pkg_root / 'final_proj' / 'data' / 'skills.json'
        self.skill_store = SkillStore(str(skills_path))
        
        # Initialize LLM components (optional)
        self.replanner = None
        self.llm_debug = False
        if HAS_LLM and LLMClient is not None:
            try:
                client = LLMClient()
                builder = PromptBuilder()
                parser = ResponseParser()
                extractor = ContextExtractor()
                self.replanner = Replanner(builder, parser, client, extractor)
                print("[INFO] LLM Replanner initialized")
            except Exception as e:
                print(f"[WARN] Failed to initialize LLM Replanner: {e}")
        else:
            print("[INFO] Running in baseline mode (no LLM)")

        
        self.metrics = []
    
    def find_random_valid_cells(self, grid, num_pairs=1):
        """Find random start and goal cells in free space"""
        free_cells = np.where(grid == 0)
        free_cells = list(zip(free_cells[1], free_cells[0]))  # (x, y) format
        
        pairs = []
        for _ in range(num_pairs):
            start = random.choice(free_cells)
            goal = random.choice(free_cells)
            # Ensure they're different and reasonably far apart
            attempts = 0
            while start == goal and attempts < 100:
                goal = random.choice(free_cells)
                attempts += 1
            if start != goal:
                pairs.append((start, goal))
        
        return pairs
    
    def make_cost_map(self, grid):
        """Create cost map from occupancy grid"""
        cost_map = np.full(grid.shape, np.inf, dtype=float)
        cost_map[grid == 0] = 1.0
        cost_map[grid == -1] = 5.0  # Unknown cells have higher cost
        return cost_map

    def make_display(self, grid):
        display = np.zeros_like(grid, dtype=float)
        display[grid == 0] = 1.0
        display[grid == 1] = 0.0
        display[grid == -1] = 0.5
        return display

    def plot_path(self, ax, grid, path, start, goal, title, corrupted_cells=None):
        display = self.make_display(grid)
        ax.imshow(display, cmap="gray", origin="lower")

        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="red", linewidth=2, label="path")

        if start is not None:
            ax.scatter([start[0]], [start[1]], c="blue", s=40, label="start")
        if goal is not None:
            ax.scatter([goal[0]], [goal[1]], c="green", s=40, label="goal")

        if corrupted_cells:
            if isinstance(corrupted_cells, tuple):
                ys, xs = corrupted_cells
            else:
                xs = [c[0] for c in corrupted_cells]
                ys = [c[1] for c in corrupted_cells]
            ax.scatter(xs, ys, c="yellow", s=2, alpha=0.2, label="corruption")

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize="small")

    def plot_llm_debug(self, ax, prompt, response, decision):
        ax.axis('off')
        prompt_text = prompt or "(no prompt generated)"
        response_text = response or "(no response)"
        decision_text = decision if decision is not None else "(no decision)"

        text = (
            f"LLM PROMPT:\n{prompt_text}\n\n"
            f"LLM RESPONSE:\n{response_text}\n\n"
            f"PARSED DECISION:\n{decision_text}"
        )
        ax.text(0.01, 0.99, text, transform=ax.transAxes, va='top', ha='left', fontsize=8, family='monospace')
        ax.set_title("LLM communication")

    def setup_animation_panel(self, ax, grid, path, start, goal, title, corrupted_cells=None):
        display = self.make_display(grid)
        ax.imshow(display, cmap="gray", origin="lower")

        if corrupted_cells:
            if isinstance(corrupted_cells, tuple):
                ys, xs = corrupted_cells
            else:
                xs = [c[0] for c in corrupted_cells]
                ys = [c[1] for c in corrupted_cells]
            ax.scatter(xs, ys, c="yellow", s=2, alpha=0.2)

        if path:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, color="red", linewidth=1, alpha=0.3)
            progress_line, = ax.plot([], [], color="red", linewidth=2)
        else:
            progress_line = None

        ax.scatter([start[0]], [start[1]], c="blue", s=40)
        ax.scatter([goal[0]], [goal[1]], c="green", s=40)

        robot_dot, = ax.plot([], [], marker="o", color="cyan", markersize=10)
        step_text = ax.text(
            0.02, 0.95,
            "", transform=ax.transAxes,
            color="white", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round")
        )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        return {
            'path': path,
            'progress_line': progress_line,
            'robot_dot': robot_dot,
            'step_text': step_text,
        }

    def animate_trial(self, start_cell, goal_cell, corruption_pct=0.95, interval=250, force_llm=False):
        grid = self.grid_original.copy()
        cost_map = self.make_cost_map(grid)
        baseline_path = astar(cost_map, start_cell, goal_cell)

        corruptor = Corruptor(corruption_rate=corruption_pct)
        corrupted_grid = corruptor.inject_random_corruption(grid)
        corrupted_cost = self.make_cost_map(corrupted_grid)
        corrupted_path = astar(corrupted_cost, start_cell, goal_cell)

        replanned_path = None
        modified_grid = None
        decision = None
        if self.replanner is not None and (corrupted_path is None or force_llm):
            try:
                robot_pose = {'x': start_cell[0], 'y': start_cell[1]}
                modified_grid, decision = self.replanner.replan(
                    corrupted_grid.copy(), start_cell, goal_cell, baseline_path or [], robot_pose
                )
                if self.llm_debug:
                    debug_info = self.replanner.get_debug_info()
                    print("\n[LLM DEBUG] prompt:")
                    print(debug_info['prompt'])
                    print("\n[LLM DEBUG] response:")
                    print(debug_info['response'])
                    print("\n[LLM DEBUG] parsed decision:")
                    print(decision)
                if modified_grid is not None:
                    modified_cost = self.make_cost_map(modified_grid)
                    replanned_path = astar(modified_cost, start_cell, goal_cell)
            except Exception as e:
                print(f"[WARN] Replanning animation failed: {e}")
        elif self.replanner is None:
            print("[INFO] No LLM replanner available for animation.")
        else:
            print("[INFO] Corrupted map still has a valid path; LLM replanning was not triggered.")

        plots = 3 if self.replanner is not None else 2
        fig, axes = plt.subplots(1, plots, figsize=(5 * plots, 5))
        if plots == 2:
            axes = [axes[0], axes[1]]

        panels = []
        panels.append(self.setup_animation_panel(
            axes[0], grid, baseline_path, start_cell, goal_cell, "Baseline A*"
        ))
        panels.append(self.setup_animation_panel(
            axes[1], corrupted_grid, corrupted_path, start_cell, goal_cell,
            f"Corrupted map ({int(corruption_pct * 100)}%)",
            corrupted_cells=np.where(corrupted_grid != grid)
        ))

        if self.replanner is not None:
            panels.append(self.setup_animation_panel(
                axes[2], modified_grid if modified_grid is not None else corrupted_grid,
                replanned_path, start_cell, goal_cell,
                "Replanned path" if replanned_path is not None else "Replan failed",
                corrupted_cells=np.where(corrupted_grid != grid)
            ))

        max_frames = max(
            len(baseline_path) if baseline_path else 1,
            len(corrupted_path) if corrupted_path else 1,
            len(replanned_path) if replanned_path else 1,
        )

        def update(frame):
            artists = []
            for panel in panels:
                path = panel['path']
                robot_dot = panel['robot_dot']
                progress_line = panel['progress_line']
                step_text = panel['step_text']

                if path:
                    step = min(frame, len(path) - 1)
                    xs = [p[0] for p in path[:step + 1]]
                    ys = [p[1] for p in path[:step + 1]]
                    if progress_line is not None:
                        progress_line.set_data(xs, ys)
                        artists.append(progress_line)
                    robot_dot.set_data(xs[-1], ys[-1])
                    step_text.set_text(f"step {step + 1}/{len(path)}")
                else:
                    robot_dot.set_data([], [])
                    if progress_line is not None:
                        progress_line.set_data([], [])
                    step_text.set_text("FAILED" if panel['progress_line'] is None else "NO PATH")

                artists.append(robot_dot)
                artists.append(step_text)

            return artists

        animation = FuncAnimation(fig, update, frames=range(max_frames), interval=interval, blit=True, repeat=False)
        plt.tight_layout()
        plt.show()

        if self.llm_debug and self.replanner is not None:
            debug_info = self.replanner.get_debug_info()
            fig_dbg, ax_dbg = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_llm_debug(ax_dbg, debug_info.get('prompt'), debug_info.get('response'), decision)
            plt.tight_layout()
            plt.show()

    def visualize_trial(self, start_cell, goal_cell, corruption_pct=0.95, force_llm=False):
        grid = self.grid_original.copy()
        cost_map = self.make_cost_map(grid)
        baseline_path = astar(cost_map, start_cell, goal_cell)

        corruptor = Corruptor(corruption_rate=corruption_pct)
        corrupted_grid = corruptor.inject_random_corruption(grid)
        corrupted_cost = self.make_cost_map(corrupted_grid)
        corrupted_path = astar(corrupted_cost, start_cell, goal_cell)

        replanned_path = None
        modified_grid = None
        decision = None
        if self.replanner is not None and (corrupted_path is None or force_llm):
            try:
                robot_pose = {'x': start_cell[0], 'y': start_cell[1]}
                modified_grid, decision = self.replanner.replan(
                    corrupted_grid.copy(), start_cell, goal_cell, baseline_path or [], robot_pose
                )
                if self.llm_debug:
                    debug_info = self.replanner.get_debug_info()
                    print("\n[LLM DEBUG] prompt:")
                    print(debug_info['prompt'])
                    print("\n[LLM DEBUG] response:")
                    print(debug_info['response'])
                    print("\n[LLM DEBUG] parsed decision:")
                    print(decision)
                if modified_grid is not None:
                    modified_cost = self.make_cost_map(modified_grid)
                    replanned_path = astar(modified_cost, start_cell, goal_cell)
            except Exception as e:
                print(f"[WARN] Replanning visualization failed: {e}")
        elif self.replanner is None:
            print("[INFO] No LLM replanner available for visualization.")
        else:
            print("[INFO] Corrupted map still has a valid path; LLM replanning was not triggered.")

        plots = 3 if self.replanner is not None else 2
        fig, axes = plt.subplots(1, plots, figsize=(5 * plots, 5))

        if plots == 2:
            axes = [axes[0], axes[1]]

        self.plot_path(axes[0], grid, baseline_path, start_cell, goal_cell, "Baseline A*")
        self.plot_path(
            axes[1], corrupted_grid, corrupted_path, start_cell, goal_cell,
            f"Corrupted map ({int(corruption_pct * 100)}%)", corrupted_cells=np.where(corrupted_grid != grid)
        )

        if self.replanner is not None:
            self.plot_path(
                axes[2], modified_grid if modified_grid is not None else corrupted_grid,
                replanned_path, start_cell, goal_cell,
                "Replanned path" if replanned_path is not None else "Replan failed",
                corrupted_cells=np.where(corrupted_grid != grid)
            )

        plt.tight_layout()
        plt.show()

        if self.llm_debug and self.replanner is not None:
            debug_info = self.replanner.get_debug_info()
            fig_dbg, ax_dbg = plt.subplots(1, 1, figsize=(8, 6))
            self.plot_llm_debug(ax_dbg, debug_info.get('prompt'), debug_info.get('response'), decision)
            plt.tight_layout()
            plt.show()

    def baseline_astar(self, start_cell, goal_cell, num_trials=5):
        """Baseline: pure A* on original map"""
        print("\n[BASELINE] Running A* on original map...")
        
        results = []
        for trial in range(num_trials):
            grid = self.grid_original.copy()
            cost_map = self.make_cost_map(grid)
            
            start_time = time.time()
            path = astar(cost_map, start_cell, goal_cell)
            elapsed = time.time() - start_time
            
            success = path is not None
            path_length = len(path) if path else 0
            
            metric = NavigationMetrics(
                scenario='baseline',
                trial_id=trial,
                start_cell=start_cell,
                goal_cell=goal_cell,
                success=success,
                path_length=path_length,
                num_replans=0,
                total_time=elapsed,
                replanning_time=0.0,
                path=path,
                corrupted_cells=[]
            )
            results.append(metric)
            print(f"  Trial {trial+1}: Success={success}, Path length={path_length}, Time={elapsed:.3f}s")
        
        return results
    
    def with_corruption(self, start_cell, goal_cell, corruption_pct=0.1, num_trials=5):
        """With corruption: A* fails on corrupted map, then optional replanning"""
        print(f"\n[CORRUPTION] Corrupting {corruption_pct*100:.0f}% of map and replanning...")
        
        results = []
        for trial in range(num_trials):
            grid = self.grid_original.copy()
            corruptor = Corruptor(corruption_rate=corruption_pct)
            corrupted_grid = corruptor.inject_random_corruption(grid)
            
            # Track which cells were corrupted
            corrupted_cells = np.where(corrupted_grid != grid)
            corrupted_cells = list(zip(corrupted_cells[1], corrupted_cells[0]))
            
            cost_map = self.make_cost_map(corrupted_grid)
            
            # First attempt on corrupted map
            start_time = time.time()
            path = astar(cost_map, start_cell, goal_cell)
            initial_time = time.time() - start_time
            
            replanning_time = 0.0
            num_replans = 0
            
            # If failed, try LLM replanning
            if path is None and self.replanner is not None:
                replan_start = time.time()
                try:
                    robot_pose = {'x': start_cell[0], 'y': start_cell[1]}
                    modified_grid, decision = self.replanner.replan(
                        corrupted_grid.copy(), start_cell, goal_cell, [], robot_pose
                    )
                    if modified_grid is not None:
                        cost_map = self.make_cost_map(modified_grid)
                        path = astar(cost_map, start_cell, goal_cell)
                        num_replans = 1
                except Exception as e:
                    print(f"    Replanning failed: {e}")
                
                replanning_time = time.time() - replan_start
            
            success = path is not None
            path_length = len(path) if path else 0
            total_time = initial_time + replanning_time
            
            metric = NavigationMetrics(
                scenario='corruption',
                trial_id=trial,
                start_cell=start_cell,
                goal_cell=goal_cell,
                success=success,
                path_length=path_length,
                num_replans=num_replans,
                total_time=total_time,
                replanning_time=replanning_time,
                path=path,
                corrupted_cells=corrupted_cells
            )
            results.append(metric)
            print(f"  Trial {trial+1}: Success={success}, Replans={num_replans}, "
                  f"Path={path_length}, Time={total_time:.3f}s")
        
        return results
    
    def full_system(self, start_cell, goal_cell, corruption_pct=0.1, num_trials=5):
        """Full system: corruption + LLM replanning + skill memory"""
        print(f"\n[FULL SYSTEM] With corruption + LLM + skills...")
        
        results = []
        for trial in range(num_trials):
            grid = self.grid_original.copy()
            corruptor = Corruptor(corruption_rate=corruption_pct)
            corrupted_grid = corruptor.inject_random_corruption(grid)
            
            # Track which cells were corrupted
            corrupted_cells = np.where(corrupted_grid != grid)
            corrupted_cells = list(zip(corrupted_cells[1], corrupted_cells[0]))
            
            cost_map = self.make_cost_map(corrupted_grid)
            
            # First attempt
            start_time = time.time()
            path = astar(cost_map, start_cell, goal_cell)
            initial_time = time.time() - start_time
            
            replanning_time = 0.0
            num_replans = 0
            
            # Replanning with LLM
            if path is None and self.replanner is not None:
                replan_start = time.time()
                try:
                    robot_pose = {'x': start_cell[0], 'y': start_cell[1]}
                    modified_grid, decision = self.replanner.replan(
                        corrupted_grid.copy(), start_cell, goal_cell, [], robot_pose
                    )
                    if modified_grid is not None:
                        cost_map = self.make_cost_map(modified_grid)
                        path = astar(cost_map, start_cell, goal_cell)
                        num_replans += 1
                except Exception:
                    pass
                
                replanning_time = time.time() - replan_start
            
            success = path is not None
            path_length = len(path) if path else 0
            total_time = initial_time + replanning_time
            
            metric = NavigationMetrics(
                scenario='full_system',
                trial_id=trial,
                start_cell=start_cell,
                goal_cell=goal_cell,
                success=success,
                path_length=path_length,
                num_replans=num_replans,
                total_time=total_time,
                replanning_time=replanning_time,
                path=path,
                corrupted_cells=corrupted_cells
            )
            results.append(metric)
            print(f"  Trial {trial+1}: Success={success}, Replans={num_replans}, "
                  f"Path={path_length}, Time={total_time:.3f}s")
        
        return results
    
    def run_evaluation(self, num_scenarios=5, trials_per_scenario=5):
        """Run complete evaluation with multiple scenarios"""
        print("=" * 60)
        print("NAVIGATION EVALUATION")
        print("=" * 60)
        
        # Generate test scenarios
        scenarios = self.find_random_valid_cells(self.grid_original, num_pairs=num_scenarios)
        
        all_results = []
        for scenario_idx, (start, goal) in enumerate(scenarios):
            print(f"\n{'='*60}")
            print(f"SCENARIO {scenario_idx + 1}/{num_scenarios}")
            print(f"Start: {start}, Goal: {goal}")
            print(f"{'='*60}")
            
            # Run all three approaches
            baseline_results = self.baseline_astar(start, goal, trials_per_scenario)
            corruption_results = self.with_corruption(start, goal, corruption_pct=0.950, num_trials=trials_per_scenario)
            full_results = self.full_system(start, goal, corruption_pct=0.950, num_trials=trials_per_scenario)
            
            all_results.extend(baseline_results)
            all_results.extend(corruption_results)
            all_results.extend(full_results)
        
        self.metrics = all_results
        return all_results
    
    def generate_report(self, output_file="evaluation_results.json"):
        """Save results to JSON and print summary"""
        results_dict = []
        for m in self.metrics:
            d = m.to_dict()
            # Ensure all values are JSON serializable
            d_clean = {}
            for k, v in d.items():
                if isinstance(v, (np.integer, np.floating)):
                    d_clean[k] = float(v) if isinstance(v, np.floating) else int(v)
                elif isinstance(v, (tuple, list)):
                    d_clean[k] = [int(x) if isinstance(x, (np.integer, int)) else float(x) if isinstance(x, (np.floating, float)) else x for x in v]
                else:
                    d_clean[k] = v
            results_dict.append(d_clean)
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        # Print summary statistics
        self.print_summary()
    
    def print_summary(self):
        """Print summary statistics by scenario"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        scenarios = set(m.scenario for m in self.metrics)
        
        for scenario in sorted(scenarios):
            scenario_metrics = [m for m in self.metrics if m.scenario == scenario]
            
            successes = sum(1 for m in scenario_metrics if m.success)
            success_rate = successes / len(scenario_metrics) if scenario_metrics else 0
            
            successful_paths = [m.path_length for m in scenario_metrics if m.success and m.path_length > 0]
            avg_path_length = np.mean(successful_paths) if successful_paths else 0
            
            avg_replans = np.mean([m.num_replans for m in scenario_metrics])
            avg_time = np.mean([m.total_time for m in scenario_metrics])
            avg_replan_time = np.mean([m.replanning_time for m in scenario_metrics])
            
            print(f"\n{scenario.upper().replace('_', ' ')}:")
            print(f"  Success Rate:       {success_rate*100:.1f}% ({successes}/{len(scenario_metrics)})")
            print(f"  Avg Path Length:    {avg_path_length:.1f} cells")
            print(f"  Avg Replans:        {avg_replans:.2f}")
            print(f"  Total Time:         {avg_time:.3f}s")
            print(f"  Replanning Time:    {avg_replan_time:.3f}s")
        
        # Comparison
        print("\n" + "-"*60)
        print("IMPROVEMENTS (vs. BASELINE):")
        print("-"*60)
        
        baseline = [m for m in self.metrics if m.scenario == 'baseline']
        corruption = [m for m in self.metrics if m.scenario == 'corruption']
        full = [m for m in self.metrics if m.scenario == 'full_system']
        
        if baseline and corruption:
            baseline_success = sum(1 for m in baseline if m.success) / len(baseline)
            corruption_success = sum(1 for m in corruption if m.success) / len(corruption)
            print(f"\nCorruption Recovery Success Rate:    {corruption_success*100:.1f}% (vs. {baseline_success*100:.1f}% baseline)")
        
        if baseline and full:
            baseline_success = sum(1 for m in baseline if m.success) / len(baseline)
            full_success = sum(1 for m in full if m.success) / len(full)
            print(f"Full System Success Rate:            {full_success*100:.1f}% (vs. {baseline_success*100:.1f}% baseline)")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Navigation evaluation and visualization")
    parser.add_argument("--visualize", action="store_true", help="Show a visual trial instead of running the full evaluation")
    parser.add_argument("--animate", action="store_true", help="Show an animated trial instead of running the full evaluation")
    parser.add_argument("--start", type=int, nargs=2, metavar=("X", "Y"), help="Start cell coordinates")
    parser.add_argument("--goal", type=int, nargs=2, metavar=("X", "Y"), help="Goal cell coordinates")
    parser.add_argument("--corruption", type=float, default=0.95, help="Corruption rate for visualization or evaluation")
    parser.add_argument("--force-llm", action="store_true", help="Force LLM replanning even if the corrupted map still has a valid path")
    parser.add_argument("--llm-debug", action="store_true", help="Print the LLM prompt and response during replanning")
    parser.add_argument("--scenarios", type=int, default=5, help="Number of random scenarios for evaluation")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per scenario")
    args = parser.parse_args()

    # Use house map - resolve relative to the final_proj package
    pkg_root = Path(__file__).resolve().parent  # evaluation.py's directory
    map_path = str(pkg_root / 'final_proj' / 'data' / 'map.yaml')

    print(f"Loading map: {map_path}")

    evaluator = NavigationEvaluator(map_path)

    evaluator.llm_debug = args.llm_debug

    if args.animate:
        if args.start is None or args.goal is None:
            start, goal = evaluator.find_random_valid_cells(evaluator.grid_original, num_pairs=1)[0]
            print(f"Animating random start/goal {start} -> {goal}")
        else:
            start = tuple(args.start)
            goal = tuple(args.goal)
        evaluator.animate_trial(start, goal, corruption_pct=args.corruption, force_llm=args.force_llm)
    elif args.visualize:
        if args.start is None or args.goal is None:
            start, goal = evaluator.find_random_valid_cells(evaluator.grid_original, num_pairs=1)[0]
            print(f"Visualizing random start/goal {start} -> {goal}")
        else:
            start = tuple(args.start)
            goal = tuple(args.goal)
        evaluator.visualize_trial(start, goal, corruption_pct=args.corruption, force_llm=args.force_llm)
    else:
        results = evaluator.run_evaluation(num_scenarios=args.scenarios, trials_per_scenario=args.trials)
        evaluator.generate_report("evaluation_results.json")
        print("\n[DONE] Evaluation complete!")


if __name__ == '__main__':
    main()
