from pathlib import Path
import math
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from final_proj.environment.map_loader import MapLoader
from final_proj.memory.skill_store import SkillStore
from final_proj.planning.astar import astar


def neighbors(cell, grid):
    x, y = cell
    height, width = grid.shape
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 0:
            yield (nx, ny)


def bfs_reachable(grid, start):
    q = deque([start])
    visited = {start}
    while q:
        current = q.popleft()
        for nb in neighbors(current, grid):
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return visited


def farthest_cell(start, reachable):
    sx, sy = start
    best = None
    best_dist = -1
    for cell in reachable:
        x, y = cell
        dist = abs(x - sx) + abs(y - sy)
        if dist > best_dist:
            best_dist = dist
            best = cell
    return best


def nearest_free_cell(grid, center, search_radius=30):
    cx, cy = center
    height, width = grid.shape
    best = None
    best_dist = 10**9
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            x = cx + dx
            y = cy + dy
            if 0 <= x < width and 0 <= y < height and grid[y, x] == 0:
                dist = abs(dx) + abs(dy)
                if dist < best_dist:
                    best_dist = dist
                    best = (x, y)
    return best


def extract_patch(grid, center, size=5):
    half = size // 2
    cx, cy = center
    height, width = grid.shape
    patch = []
    for dy in range(-half, half + 1):
        row = []
        for dx in range(-half, half + 1):
            x = cx + dx
            y = cy + dy
            if 0 <= x < width and 0 <= y < height:
                row.append(int(grid[y, x]))
            else:
                row.append(1)
        patch.append(row)
    return patch


def goal_relative_direction(start, goal):
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    angle = math.atan2(dy, dx)
    if -math.pi / 4 <= angle < math.pi / 4:
        return "east"
    if math.pi / 4 <= angle < 3 * math.pi / 4:
        return "north"
    if -3 * math.pi / 4 <= angle < -math.pi / 4:
        return "south"
    return "west"


def extract_turn_pattern(path):
    if not path or len(path) < 3:
        return []

    turns = []
    for i in range(2, len(path)):
        x1, y1 = path[i - 2]
        x2, y2 = path[i - 1]
        x3, y3 = path[i]

        d1 = (x2 - x1, y2 - y1)
        d2 = (x3 - x2, y3 - y2)

        if d1 == d2:
            turns.append("straight")
        elif d1[0] * d2[0] + d1[1] * d2[1] == 0:
            turns.append("turn")
        else:
            turns.append("other")
    return turns


def simple_context(grid, start, goal, path):
    return {
        "start_patch": extract_patch(grid, start, size=5),
        "goal_patch": extract_patch(grid, goal, size=5),
        "goal_direction": goal_relative_direction(start, goal),
        "turn_pattern": extract_turn_pattern(path),
        "path_length": len(path)
    }


def context_similarity(a, b):
    patch_score = 0.0
    a_patch = a.get("start_patch", [])
    b_patch = b.get("start_patch", [])
    if a_patch and b_patch and len(a_patch) == len(b_patch):
        same = 0
        total = 0
        for ra, rb in zip(a_patch, b_patch):
            for va, vb in zip(ra, rb):
                total += 1
                if va == vb:
                    same += 1
        patch_score = same / total if total > 0 else 0.0

    dir_score = 1.0 if a.get("goal_direction") == b.get("goal_direction") else 0.0

    turns_a = a.get("turn_pattern", [])
    turns_b = b.get("turn_pattern", [])
    if not turns_a or not turns_b:
        turn_score = 0.0
    else:
        overlap = min(len(turns_a), len(turns_b))
        same = sum(1 for i in range(overlap) if turns_a[i] == turns_b[i])
        turn_score = same / overlap if overlap > 0 else 0.0

    len_a = a.get("path_length", 0)
    len_b = b.get("path_length", 0)
    length_score = 1.0 - abs(len_a - len_b) / max(len_a, len_b, 1)

    return 0.4 * patch_score + 0.2 * dir_score + 0.2 * turn_score + 0.2 * length_score


def retrieve_skill(store, query_context, min_score=0.45):
    best_skill = None
    best_score = -1.0

    for skill in store.get_all_skills():
        stored_context = skill.get("start_context", {})
        score = context_similarity(stored_context, query_context)
        if score > best_score:
            best_score = score
            best_skill = skill

    if best_skill is not None and best_score >= min_score:
        return best_skill, best_score

    return None, best_score


def make_display(grid):
    display = np.zeros_like(grid, dtype=float)
    display[grid == 0] = 1.0
    display[grid == 1] = 0.0
    display[grid == -1] = 0.5
    return display


def plot_path(ax, grid, start, goal, path, title, skill_path=None):
    display = make_display(grid)
    ax.imshow(display, cmap="gray", origin="lower")
    if path is not None and len(path) > 0:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="red", linewidth=2, label="current path")
    if skill_path is not None and len(skill_path) > 0:
        xs = [p[0] for p in skill_path]
        ys = [p[1] for p in skill_path]
        ax.plot(xs, ys, color="blue", linewidth=2, linestyle="--", label="reused skill")
    ax.scatter([start[0]], [start[1]], c="green", s=50, label="start")
    ax.scatter([goal[0]], [goal[1]], c="yellow", s=50, label="goal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")


def main():
    map_path = Path("/root/ros2_ws/src/pa3/maze.yml")
    loader = MapLoader(map_path)
    grid = loader.load()

    store_path = Path("/root/ros2_ws/src/final_proj/data/skills.json")
    store = SkillStore(store_path)

    robot_guess = (grid.shape[1] - 10, grid.shape[0] - 10)
    start = nearest_free_cell(grid, robot_guess, search_radius=40)

    if start is None:
        print("No start found")
        return

    reachable = bfs_reachable(grid, start)
    goal = farthest_cell(start, reachable)

    if goal is None or goal == start:
        print("No goal found")
        return

    path = astar(grid, start, goal, allow_unknown=False)
    if path is None:
        print("No path found")
        return

    query_context = simple_context(grid, start, goal, path)
    print("query context:")
    print(query_context)

    reused_skill = None
    reused_skill_path = None

    if len(store.get_all_skills()) == 0:
        skill_id = store.add_skill(
            start_context=query_context,
            end_context=query_context,
            path_pattern=path,
            outcome="success"
        )
        print("saved first skill id:", skill_id)
        print("run again to test retrieval")
    else:
        found_skill, score = retrieve_skill(store, query_context, min_score=0.45)

        if found_skill is None:
            print("no reusable skill found")
            print("best score:", score)
        else:
            print("reusable skill found")
            print("score:", score)
            print("skill id:", found_skill["id"])
            print("skill:", found_skill)
            reused_skill = found_skill
            reused_skill_path = found_skill.get("path_pattern", [])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plot_path(
        ax,
        grid,
        start,
        goal,
        path,
        "skill reuse visualization",
        skill_path=reused_skill_path
    )
    plt.tight_layout()
    plt.show()

    if reused_skill is not None:
        same_skill = store.find_skill_by_id(reused_skill["id"])
        print("retrieved by id:")
        print(same_skill)


if __name__ == "__main__":
    main()