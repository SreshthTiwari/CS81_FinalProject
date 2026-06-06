from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from final_proj.environment.map_loader import MapLoader
from final_proj.memory.skill_store import SkillStore
from final_proj.memory.similarity import best_matching_skill
from final_proj.planning.costmap import build_base_cost_map, apply_skill_bias
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
    if abs(dx) >= abs(dy):
        return "east" if dx >= 0 else "west"
    return "north" if dy >= 0 else "south"

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

def make_context(grid, start, goal, path):
    return {
        "start_patch": extract_patch(grid, start, size=5),
        "goal_patch": extract_patch(grid, goal, size=5),
        "goal_direction": goal_relative_direction(start, goal),
        "turn_pattern": extract_turn_pattern(path),
        "path_length": len(path)
    }

def make_display(grid):
    display = np.zeros_like(grid, dtype=float)
    display[grid == 0] = 1.0
    display[grid == 1] = 0.0
    display[grid == -1] = 0.5
    return display

def plot_paths(grid, base_path, biased_path, start, goal, title):
    display = make_display(grid)
    plt.figure(figsize=(5, 5))
    plt.imshow(display, cmap="gray", origin="lower")

    if base_path is not None:
        xs = [p[0] for p in base_path]
        ys = [p[1] for p in base_path]
        plt.plot(xs, ys, color="red", linewidth=2, label="baseline a*")

    if biased_path is not None:
        xs = [p[0] for p in biased_path]
        ys = [p[1] for p in biased_path]
        plt.plot(xs, ys, color="blue", linewidth=2, linestyle="--", label="skill-biased a*")

    plt.scatter([start[0]], [start[1]], c="green", s=60, label="start")
    plt.scatter([goal[0]], [goal[1]], c="yellow", s=60, label="goal")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

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

    base_cost = build_base_cost_map(grid)
    base_path = astar(base_cost, start, goal)
    if base_path is None:
        print("No baseline path found")
        return

    query_context = make_context(grid, start, goal, base_path)

    if len(store.get_all_skills()) == 0:
        skill_id = store.add_skill(
            start_context=query_context,
            end_context=query_context,
            path_pattern=base_path,
            outcome="success"
        )
        print("saved first skill id:", skill_id)
        plot_paths(grid, base_path, None, start, goal, "baseline path saved as skill")
        return

    best_skill, score = best_matching_skill(query_context, store.get_all_skills())
    if best_skill is None:
        print("no reusable skill found")
        print("best score:", score)
        plot_paths(grid, base_path, None, start, goal, "no skill match")
        return

    print("best matching skill id:", best_skill["id"])
    print("similarity score:", score)

    skill_path = best_skill.get("path_pattern", [])
    biased_cost = apply_skill_bias(base_cost, skill_path, bias_strength=max(0.2, 1.0 - score), neighborhood=1)
    biased_path = astar(biased_cost, start, goal)

    if biased_path is None:
        print("no biased path found, falling back to baseline")
        biased_path = base_path

    print("baseline path length:", len(base_path))
    print("biased path length:", len(biased_path))

    plot_paths(grid, base_path, biased_path, start, goal, "skill-biased planning")

    same_skill = store.find_skill_by_id(best_skill["id"])
    print("retrieved by id:")
    print(same_skill)

if __name__ == "__main__":
    main()