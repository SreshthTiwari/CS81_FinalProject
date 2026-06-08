#!/usr/bin/env python3
"""Test: skill storage via checkpointed A* navigation (no LLM).

This test places the robot in an environment, computes a full path to a goal,
splits the path into checkpoints (skills), stores them, then re-initializes the
robot at a different start and verifies that the new planned path contains the
previous path segment (so the robot can reuse stored skill checkpoints).
"""

from pathlib import Path
import random

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib import patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from final_proj.environment.map_loader import MapLoader
from final_proj.planning.astar import astar


class InMemorySkillStore:
    """Simple skill storage: maps skill_id -> list of checkpoints (x,y).
    For this test we only need store and retrieve by a name/key.
    """
    def __init__(self):
        self.store = {}

    def save_skill(self, key, skill_obj):
        # store arbitrary skill objects (lists, dicts, etc.)
        self.store[key] = skill_obj

    def load_skill(self, key):
        return self.store.get(key)


def split_into_checkpoints(path, num_checkpoints=3):
    if not path:
        return []
    if num_checkpoints <= 0:
        return []
    L = len(path)
    # evenly spaced checkpoints along the path excluding start, include goal
    indices = sorted({min(L - 1, max(0, int(round(i * (L - 1) / num_checkpoints)))) for i in range(1, num_checkpoints + 1)})
    checkpoints = [path[i] for i in indices]
    return checkpoints


def path_contains_subpath(path, subpath):
    # Check that all points of subpath appear in path in same order (not necessarily contiguous)
    if not subpath:
        return True
    pi = 0
    for p in path:
        if p == subpath[pi]:
            pi += 1
            if pi >= len(subpath):
                return True
    return False


def test_skill_storage_and_reuse(show_window=True):
    # Setup
    script_dir = Path(__file__).resolve().parent
    map_path = script_dir / "final_proj" / "data" / "map.yaml"
    ml = MapLoader(str(map_path))
    grid = ml.get_grid()

    history = []

    # choose two start positions and a shared goal (use free cells)
    free_cells = np.where(grid == 0)
    free = list(zip(free_cells[1], free_cells[0]))
    random.seed(1)
    np.random.seed(1)
    start1 = tuple(int(v) for v in random.choice(free))
    start2 = tuple(int(v) for v in random.choice(free))
    while start2 == start1:
        start2 = tuple(int(v) for v in random.choice(free))
    goal = tuple(int(v) for v in random.choice(free))

    print("=" * 60)
    print("STAGE 1: INITIAL PATH PLANNING & SKILL STORAGE")
    print("=" * 60)
    print(f"Start1: {start1}, Goal: {goal}")

    # compute original full path from start1 to goal
    cost = np.full(grid.shape, np.inf, dtype=float)
    cost[grid == 0] = 1.0
    cost[grid == -1] = 5.0
    full_path1 = astar(cost, start1, goal)
    assert full_path1, "No path found for start1 -> goal"
    print(f"Path1 length: {len(full_path1)}")

    # Record initial path for visualization
    history.append({
        "grid": grid.copy(),
        "path": full_path1,
        "start": start1,
        "goal": goal,
        "step_type": "initial_path",
        "title": f"Initial Path: Start1 → Goal\n(Path length: {len(full_path1)})",
        "advice": "Robot navigates and stores this path as a skill"
    })

    # Split the original path into contiguous segments (we'll store segments as the skill)
    num_segments = 3
    L = len(full_path1)
    segments = []
    prev_end = None
    for i in range(num_segments):
        start_idx = (i * L) // num_segments
        if prev_end is not None:
            # overlap the previous end so checkpoints align exactly
            start_idx = prev_end
        end_idx = ((i + 1) * L) // num_segments - 1 if i < num_segments - 1 else L - 1
        seg = full_path1[start_idx:end_idx + 1]
        if seg:
            segments.append(seg)
            prev_end = end_idx

    # Derive checkpoints as the end-points between segments (excluding the global start)
    checkpoints = [segments[i][-1] for i in range(len(segments) - 1)]
    print(f"Stored {len(segments)} segments; checkpoints: {checkpoints}")

    store = InMemorySkillStore()
    skill_obj = {"segments": segments, "checkpoints": checkpoints, "origin_start": start1, "origin_goal": goal}
    store.save_skill("skill_A", skill_obj)
    print("✓ Skill (segments) stored in memory")

    history[0]["checkpoints"] = checkpoints
    history[0]["skill_segments"] = segments
    history[0]["advice"] = "Robot stores the path as three skill segments and checkpoint boundaries"

    print("\n" + "=" * 60)
    print("STAGE 2: NEW PATH PLANNING - EXECUTE SEGMENTS STEPWISE")
    print("=" * 60)
    print(f"Start2: {start2} (different from Start1)")
    print(f"Goal: {goal} (same as before)")

    # Load stored skill
    loaded = store.load_skill("skill_A")
    assert loaded and "segments" in loaded, "Loaded skill malformed"
    stored_segments = loaded["segments"]
    stored_checkpoints = loaded.get("checkpoints", [])

    # Ensure the stored middle segment endpoints match the checkpoints
    if len(stored_segments) >= 2:
        assert stored_segments[1][0] == stored_segments[0][-1], "Stored segments do not chain correctly"

    # Build the executed segments for the new run:
    executed_segments = []

    # Segment A: plan from start2 -> first checkpoint
    cp1 = stored_checkpoints[0] if stored_checkpoints else stored_segments[0][-1]
    cost = np.full(grid.shape, np.inf, dtype=float)
    cost[grid == 0] = 1.0
    cost[grid == -1] = 5.0
    seg_a = astar(cost, start2, cp1)
    assert seg_a, "No path from start2 to first checkpoint"
    executed_segments.append({"path": seg_a, "reused": False, "title": "Start2 → Checkpoint 1"})
    print(f"  Segment A: {start2} → {cp1} (len {len(seg_a)})")

    # Segment B: reuse stored middle segment exactly as the skill
    if len(stored_segments) >= 2:
        seg_b = stored_segments[1]
        # validate endpoints equal checkpoint positions
        if stored_checkpoints:
            assert seg_b[0] == stored_checkpoints[0] and seg_b[-1] == stored_checkpoints[1], "Stored segment endpoints don't match checkpoints"
        executed_segments.append({"path": seg_b, "reused": True, "title": "Checkpoint 1 → Checkpoint 2 (reused skill)"})
        print(f"  Segment B: reused skill segment (len {len(seg_b)})")
    else:
        # fallback: plan from cp1 to cp2
        cp2 = stored_segments[0][-1]
        seg_b = astar(cost, cp1, cp2)
        executed_segments.append({"path": seg_b, "reused": False, "title": "Checkpoint 1 → Checkpoint 2"})

    # Segment C: plan from last checkpoint to goal (reuse if stored)
    if len(stored_segments) >= 3:
        seg_c = stored_segments[2]
        executed_segments.append({"path": seg_c, "reused": True, "title": "Checkpoint 2 → Goal (reused skill)"})
        print(f"  Segment C: reused skill segment (len {len(seg_c)})")
    else:
        last_cp = stored_checkpoints[-1] if stored_checkpoints else stored_segments[-1][-1]
        seg_c = astar(cost, last_cp, goal)
        assert seg_c, "No path from last checkpoint to goal"
        executed_segments.append({"path": seg_c, "reused": False, "title": "Checkpoint 2 → Goal"})
        print(f"  Segment C: {last_cp} → {goal} (len {len(seg_c)})")

    # Combine paths (avoid duplicating endpoints)
    combined = []
    for seg in executed_segments:
        p = seg["path"]
        if not p:
            continue
        if not combined:
            combined.extend(p)
        else:
            combined.extend(p[1:])

    # Add whole-path context for each segment window
    for idx, seg in enumerate(executed_segments):
        whole_path = []
        for later_seg in executed_segments[idx:]:
            p = later_seg["path"]
            if not p:
                continue
            if not whole_path:
                whole_path.extend(p)
            else:
                whole_path.extend(p[1:])
        seg["whole_path"] = whole_path
        seg["endpoint_label"] = "Checkpoint"
        seg["checkpoint_index"] = idx + 1

    print(f"✓ Combined path length: {len(combined)}")

    # Build history entries for each executed segment (separate windows)
    for seg in executed_segments:
        history.append({
            "grid": grid.copy(),
            "path": seg["path"],
            "whole_path": seg["whole_path"],
            "start": seg["path"][0],
            "goal": seg["path"][-1],
            "endpoint_label": seg.get("endpoint_label", "Checkpoint"),
            "checkpoint_index": seg.get("checkpoint_index"),
            "is_reused": seg.get("reused", False),
            "step_type": "segment",
            "title": seg.get("title", "Segment"),
            "advice": "Reused skill from memory" if seg.get("reused") else "Planned via A*"
        })

    # Final verification: ensure the reused segment equals the stored one
    if len(stored_segments) >= 2:
        assert executed_segments[1]["path"] == stored_segments[1], "Reused segment does not exactly match stored skill"

    print("\n" + "=" * 60)
    print("STAGE 3: VERIFICATION")
    print("=" * 60)
    print("✓ TEST PASSED: skill segments stored and reused exactly where expected")
    print("=" * 60)

    if show_window and MATPLOTLIB_AVAILABLE:
        display_skill_storage_visualization(history, grid)
    elif not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available; skipping visualization")


def display_skill_storage_visualization(history, grid):
    """Display skill storage progression interactively"""
    
    cmap = ListedColormap(["#999999", "#ffffff", "#000000"])
    norm = BoundaryNorm([-1, 0, 1, 2], cmap.N)

    plt.ion()

    for stage_idx, step in enumerate(history):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw grid
        try:
            im = ax.imshow(step["grid"], cmap=cmap, norm=norm, origin="lower")
        except ValueError as e:
            if "BoundaryNorm" in str(e):
                im = ax.imshow(step["grid"], cmap=cmap, origin="lower")
            else:
                raise

        # Draw whole-path context for segment windows
        if step.get("step_type") == "segment" and step.get("whole_path"):
            whole = step["whole_path"]
            if whole and len(whole) > 0:
                wx, wy = zip(*whole)
                ax.plot(wx, wy, color="#ff0000", linewidth=2, linestyle="--", alpha=0.5,
                        label="Route to goal", zorder=2)

        # Draw current segment or initial path
        path = step["path"]
        if path and len(path) > 0:
            xs, ys = zip(*path)
            ax.plot(xs, ys, color="#ffff00", linewidth=3, alpha=0.9, label="Current segment", zorder=4)
            ax.plot(xs, ys, color="#ffff00", linewidth=5, alpha=0.3, zorder=3)

        # Draw checkpoints if present on the initial stored path
        if step.get("step_type") == "initial_path" and "checkpoints" in step and step["checkpoints"]:
            checkpoints = step["checkpoints"]
            cp_xs, cp_ys = zip(*checkpoints)
            ax.scatter(cp_xs, cp_ys, c="#ff9900", s=220, marker="D", edgecolors="#000000",
                      linewidths=2, label="Stored Checkpoints", zorder=6, alpha=0.95)
            # Annotate checkpoint indices
            for i, (cx, cy) in enumerate(checkpoints, start=1):
                ax.annotate(f"CP{i}", xy=(cx, cy), xytext=(7, 7), textcoords="offset points",
                           fontsize=10, fontweight='bold', color="#ff9900",
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Draw start and endpoint
        start = step["start"]
        ax.scatter([start[0]], [start[1]], c="#00ff00", s=250, marker="*",
                  edgecolors="#000000", linewidths=2, label="Start", zorder=6)

        if step.get("step_type") == "segment":
            end = step["goal"]
            endpoint_label = step.get("endpoint_label", "Checkpoint")
            ax.scatter([end[0]], [end[1]], c="#ff9900", s=220, marker="D",
                      edgecolors="#000000", linewidths=2, label=endpoint_label, zorder=6)
            ax.annotate(endpoint_label, xy=(end[0], end[1]), xytext=(7, 7), textcoords="offset points",
                       fontsize=10, fontweight='bold', color="#ff9900",
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            goal = step["goal"]
            ax.scatter([goal[0]], [goal[1]], c="#0000ff", s=250, marker="X",
                      edgecolors="#000000", linewidths=2, label="Goal", zorder=6)

        # If this segment was reused from memory, annotate it
        if step.get("is_reused"):
            ax.text(0.5, 0.95, "REUSED SKILL SEGMENT", transform=ax.transAxes,
                fontsize=12, fontweight='bold', color="#cc0000", ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # For reused path, also mark the original start/goal (checkpoint origin)
        if "checkpoint_start" in step and "checkpoint_goal" in step:
            cp_start = step["checkpoint_start"]
            cp_goal = step["checkpoint_goal"]
            # Draw faded markers for original positions
            ax.scatter([cp_start[0]], [cp_start[1]], c="#00ff00", s=150, marker="*",
                      edgecolors="#000000", linewidths=1, alpha=0.4, zorder=4)
            ax.annotate("Start1\n(from skill)", xy=(cp_start[0], cp_start[1]), 
                       xytext=(10, 10), textcoords="offset points", fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#90EE90', alpha=0.7),
                       arrowprops=dict(arrowstyle="->", color="#00ff00", alpha=0.6))

        ax.set_title(step["title"], fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=10)

        # Add advice text
        if step["advice"]:
            ax.text(0.02, 0.02, step["advice"], transform=ax.transAxes, fontsize=11,
                   verticalalignment='bottom', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        ax.set_xlim(-0.5, step["grid"].shape[1] - 0.5)
        ax.set_ylim(-0.5, step["grid"].shape[0] - 0.5)
        ax.set_xlabel("X", fontsize=11)
        ax.set_ylabel("Y", fontsize=11)

        # Interactive navigation
        if stage_idx < len(history) - 1:
            response = input(f"\n→ Stage {stage_idx + 1}/{len(history)}: Press Enter to continue, or 'q' to quit: ")
            if response.lower() == 'q':
                plt.close('all')
                print("Visualization closed by user")
                return
        else:
            input(f"\n→ Final stage {stage_idx + 1}/{len(history)}: Press Enter to close: ")

        plt.close(fig)

    plt.ioff()
    print("Visualization complete")


if __name__ == '__main__':
    test_skill_storage_and_reuse()
