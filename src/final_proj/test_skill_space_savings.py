#!/usr/bin/env python3
"""Measure space savings from skill storage and reuse."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_proj.environment.map_loader import MapLoader
from final_proj.planning.astar import astar
from final_proj.memory.skill_store import SkillStore


def make_cost_map(grid):
    cost_map = np.full(grid.shape, np.inf, dtype=float)
    cost_map[grid == 0] = 1.0
    cost_map[grid == -1] = 5.0
    return cost_map


def find_random_cells(grid, count=2):
    """Find random free cells on the grid."""
    free_cells = np.where(grid == 0)
    free_cells = list(zip(free_cells[1], free_cells[0]))
    random.shuffle(free_cells)
    # Convert numpy types to native Python types
    return [(int(x), int(y)) for x, y in free_cells[:count]]


def split_path_into_segments(path, num_segments=3):
    """Split a path into checkpointed segments."""
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


def extract_start_context(grid, cell, radius=3):
    """Extract local context around a start cell."""
    patch = []
    cx, cy = cell
    h, w = grid.shape
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            x = cx + dx
            y = cy + dy
            if 0 <= x < w and 0 <= y < h:
                row.append(int(grid[y, x]))
            else:
                row.append(1)
        patch.append(row)
    return patch


def compute_json_size(obj):
    """Compute the size in bytes of a JSON-serializable object."""
    return len(json.dumps(obj).encode('utf-8'))


def run_space_savings_test(map_path, num_trials=50, seed=None, show_window=False):
    """
    Run space savings analysis.
    
    For each trial:
    1. Generate a random start and goal
    2. Compute A* path
    3. Record full path
    4. Split path into segments and store as skills
    5. Measure file sizes and compute savings
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    map_loader = MapLoader(map_path)
    grid = map_loader.get_grid().copy()
    cost_map = make_cost_map(grid)

    full_paths_storage = []
    skills_storage = SkillStore("/tmp/skill_store_test.json")

    results = {
        "trials": [],
        "total_full_path_bytes": 0,
        "total_skills_bytes": 0,
        "total_skills_created": 0,
        "average_path_length": 0.0,
        "average_segments_per_path": 0.0,
    }

    for trial_idx in range(num_trials):
        try:
            # Generate random start and goal
            cells = find_random_cells(grid, 2)
            if len(cells) < 2:
                continue
            start, goal = cells[0], cells[1]

            # Compute path
            path = astar(cost_map, start, goal)
            if not path or len(path) < 3:
                continue

            # Store full path
            full_path_entry = {
                "trial": trial_idx,
                "start": start,
                "goal": goal,
                "path": [(int(x), int(y)) for x, y in path],
            }
            full_paths_storage.append(full_path_entry)

            # Split into segments and store as skills
            num_segments = max(2, len(path) // 30)  # At least 2 segments, or 1 per ~30 cells
            segments = split_path_into_segments(path, num_segments=num_segments)

            for seg_idx, segment in enumerate(segments):
                if len(segment) < 2:
                    continue

                start_context = extract_start_context(grid, segment[0])
                end_context = extract_start_context(grid, segment[-1])
                
                path_pattern = {
                    "segment_index": seg_idx,
                    "segment_length": len(segment),
                    "segment": [(int(x), int(y)) for x, y in segment],
                }

                skill_id = skills_storage.add_skill(
                    start_context=start_context,
                    end_context=end_context,
                    path_pattern=path_pattern,
                    outcome="success"
                )

                results["total_skills_created"] += 1

            results["trials"].append({
                "trial": trial_idx,
                "start": start,
                "goal": goal,
                "path_length": int(len(path)),
                "num_segments": int(len(segments)),
            })

        except Exception as e:
            print(f"[Trial {trial_idx}] Error: {e}")
            continue

    # Compute sizes
    full_paths_size = compute_json_size(full_paths_storage)
    skills_all = skills_storage.get_all_skills()
    skills_size = compute_json_size(skills_all)

    results["total_full_path_bytes"] = full_paths_size
    results["total_skills_bytes"] = skills_size

    if full_paths_size > 0:
        compression_ratio = skills_size / full_paths_size
        savings = full_paths_size - skills_size
    else:
        compression_ratio = 0.0
        savings = 0

    successful_trials = len(results["trials"])
    if successful_trials > 0:
        total_path_length = sum(t["path_length"] for t in results["trials"])
        total_segments = sum(t["num_segments"] for t in results["trials"])
        results["average_path_length"] = total_path_length / successful_trials
        results["average_segments_per_path"] = total_segments / successful_trials

    results["successful_trials"] = successful_trials
    results["space_savings_bytes"] = savings
    results["compression_ratio"] = compression_ratio
    results["space_savings_percent"] = (1 - compression_ratio) * 100 if compression_ratio > 0 else 0.0

    print("\n" + "="*70)
    print("SPACE SAVINGS ANALYSIS")
    print("="*70)
    print(f"Successful trials: {successful_trials}/{num_trials}")
    print(f"Total skills stored: {results['total_skills_created']}")
    print(f"Average path length: {results['average_path_length']:.1f} cells")
    print(f"Average segments per path: {results['average_segments_per_path']:.1f}")
    print(f"\nStorage comparison:")
    print(f"  Full paths (naive): {full_paths_size:,} bytes")
    print(f"  Skill segments:     {skills_size:,} bytes")
    print(f"  Space saved:        {savings:,} bytes ({results['space_savings_percent']:.1f}%)")
    print(f"  Compression ratio:  {compression_ratio:.3f}")
    print("="*70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Measure space savings from skill storage")
    parser.add_argument("--num-trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for detailed results")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"

    results = run_space_savings_test(
        map_path=str(map_path),
        num_trials=args.num_trials,
        seed=args.seed,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
