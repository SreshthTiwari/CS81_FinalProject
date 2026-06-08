#!/usr/bin/env python3
"""Measure reuse potential of stored skills across multiple missions."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_proj.environment.map_loader import MapLoader
from final_proj.planning.astar import astar


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


def path_similarity(path_a, path_b, max_dist=10):
    """
    Check if two paths share a common segment of at least max_dist cells.
    Returns the maximum shared segment length.
    """
    if not path_a or not path_b:
        return 0

    max_match = 0
    for i in range(len(path_a) - max_dist + 1):
        segment_a = set(path_a[i:i+max_dist])
        for j in range(len(path_b) - max_dist + 1):
            segment_b = set(path_b[j:j+max_dist])
            overlap = len(segment_a & segment_b)
            if overlap > max_match:
                max_match = overlap

    # Also check for longer contiguous matches
    for i in range(len(path_a)):
        for j in range(len(path_b)):
            match_len = 0
            while (i + match_len < len(path_a) and 
                   j + match_len < len(path_b) and 
                   path_a[i + match_len] == path_b[j + match_len]):
                match_len += 1
            if match_len > max_match:
                max_match = match_len

    return max_match


def run_reuse_potential_test(map_path, num_missions=100, missions_per_batch=10, seed=None):
    """
    Run reuse potential analysis.
    
    1. Generate initial mission paths and store segments
    2. Generate new missions and check how many segments could be reused
    3. Calculate potential space and compute savings
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    map_loader = MapLoader(map_path)
    grid = map_loader.get_grid().copy()
    cost_map = make_cost_map(grid)

    results = {
        "missions": [],
        "total_unique_segments": 0,
        "total_new_missions_analyzed": 0,
        "reuses_found": 0,
        "total_cells_reused": 0,
        "average_reuse_per_mission": 0.0,
    }

    # Phase 1: Generate initial segments library
    stored_segments = []
    for mission_idx in range(missions_per_batch):
        try:
            cells = find_random_cells(grid, 2)
            if len(cells) < 2:
                continue
            start, goal = cells[0], cells[1]

            path = astar(cost_map, start, goal)
            if not path or len(path) < 3:
                continue

            segments = split_path_into_segments(path, num_segments=max(2, len(path) // 30))
            for seg in segments:
                if len(seg) >= 5:
                    stored_segments.append({
                        "path": seg,
                        "length": len(seg),
                    })

        except Exception as e:
            print(f"[Initial mission {mission_idx}] Error: {e}")
            continue

    results["total_unique_segments"] = len(stored_segments)

    # Phase 2: Analyze how many new missions can reuse segments
    for mission_idx in range(missions_per_batch, num_missions):
        try:
            cells = find_random_cells(grid, 2)
            if len(cells) < 2:
                continue
            start, goal = cells[0], cells[1]

            path = astar(cost_map, start, goal)
            if not path or len(path) < 3:
                continue

            # Check for reusable segments in this path
            mission_reuses = 0
            mission_cells_reused = 0
            reused_segment_ids = []

            for seg_idx, segment in enumerate(stored_segments):
                stored_path = segment["path"]
                similarity = path_similarity(path, stored_path, max_dist=min(len(stored_path), 10))
                if similarity >= len(stored_path) * 0.7:  # At least 70% match
                    mission_reuses += 1
                    mission_cells_reused += len(stored_path)
                    reused_segment_ids.append(seg_idx)

            if mission_reuses > 0:
                results["reuses_found"] += 1
                results["total_cells_reused"] += mission_cells_reused

            results["missions"].append({
                "mission": mission_idx,
                "path_length": int(len(path)),
                "reusable_segments": mission_reuses,
                "cells_reused": mission_cells_reused,
            })
            results["total_new_missions_analyzed"] += 1

        except Exception as e:
            print(f"[New mission {mission_idx}] Error: {e}")
            continue

    if results["total_new_missions_analyzed"] > 0:
        results["average_reuse_per_mission"] = (
            results["total_cells_reused"] / results["total_new_missions_analyzed"]
        )

    # Estimate savings
    # If we can reuse stored segments, we save the cost of recomputing A* for those segments
    # Assume A* computation cost is roughly proportional to path length squared (due to search space)
    # Rough estimate: computing a segment costs segment_length^2 operations
    total_compute_saved = sum(m["cells_reused"] ** 1.5 for m in results["missions"])

    results["missions_with_reuse"] = results["reuses_found"]
    results["reuse_percentage"] = (
        (results["reuses_found"] / results["total_new_missions_analyzed"] * 100)
        if results["total_new_missions_analyzed"] > 0 else 0.0
    )
    results["estimated_computation_savings"] = int(total_compute_saved)

    return results


def main():
    parser = argparse.ArgumentParser(description="Measure reuse potential of stored skills")
    parser.add_argument("--num-missions", type=int, default=100, help="Total number of missions to analyze")
    parser.add_argument("--initial-batch", type=int, default=10, help="Number of initial missions for skill library")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for detailed results")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"

    results = run_reuse_potential_test(
        map_path=str(map_path),
        num_missions=args.num_missions,
        missions_per_batch=args.initial_batch,
        seed=args.seed,
    )

    print("\n" + "="*70)
    print("SKILL REUSE POTENTIAL ANALYSIS")
    print("="*70)
    print(f"Initial skill library size: {results['total_unique_segments']} segments")
    print(f"Missions analyzed for reuse: {results['total_new_missions_analyzed']}")
    print(f"Missions with reusable segments: {results['missions_with_reuse']}")
    print(f"Reuse rate: {results['reuse_percentage']:.1f}%")
    print(f"Average cells reused per mission: {results['average_reuse_per_mission']:.1f}")
    print(f"Total cells reused across all missions: {results['total_cells_reused']}")
    print(f"Estimated computation savings: ~{results['estimated_computation_savings']:,} units")
    print("="*70 + "\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
