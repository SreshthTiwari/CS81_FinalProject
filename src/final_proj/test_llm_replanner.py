from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from final_proj.environment.map_loader import MapLoader
from final_proj.environment.context_extractor import ContextExtractor
from final_proj.planning.astar import astar
from final_proj.planning.costmap import build_base_cost_map
from final_proj.planning.replanner import Replanner
from final_proj.llm.prompt_builder import PromptBuilder
from final_proj.llm.response_parser import ResponseParser
from final_proj.llm.client import LLMClient


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


def corrupt_path_cells(grid, path, radius=1):
    corrupted = grid.copy()
    if not path:
        return corrupted
    mid_index = len(path) // 2
    x, y = path[mid_index]
    height, width = grid.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < width and 0 <= ny < height:
                corrupted[ny, nx] = -1
    return corrupted


def make_display(grid):
    display = np.zeros_like(grid, dtype=float)
    display[grid == 0] = 1.0
    display[grid == 1] = 0.0
    display[grid == -1] = 0.5
    return display


def plot_path(ax, grid, path, start, goal, title, corrupted_cells=None):
    display = make_display(grid)
    ax.imshow(display, cmap="gray", origin="lower")
    if path is not None and len(path) > 0:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, color="red", linewidth=2)
    ax.scatter([start[0]], [start[1]], c="blue", s=40, label="start")
    ax.scatter([goal[0]], [goal[1]], c="green", s=40, label="goal")
    if corrupted_cells:
        cx = [p[0] for p in corrupted_cells]
        cy = [p[1] for p in corrupted_cells]
        ax.scatter(cx, cy, c="yellow", s=20, label="corruption")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    map_path = Path("/root/ros2_ws/src/pa3/maze.yml")
    loader = MapLoader(map_path)
    grid = loader.load()

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

    print("Start:", start)
    print("Goal:", goal)

    clean_cost = build_base_cost_map(grid)
    clean_path = astar(clean_cost, start, goal)
    if clean_path is None:
        print("No clean path found")
        return

    print("Clean path length:", len(clean_path))
    print("Clean path:", clean_path)

    corrupted_grid = corrupt_path_cells(grid, clean_path, radius=1)
    corruption_cells = []
    mid_index = len(clean_path) // 2
    mid_x, mid_y = clean_path[mid_index]
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx = mid_x + dx
            ny = mid_y + dy
            if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                corruption_cells.append((nx, ny))

    corrupted_cost = build_base_cost_map(corrupted_grid)
    corrupted_path = astar(corrupted_cost, start, goal)
    if corrupted_path is None:
        print("No path found on corrupted map")
        corrupted_path = clean_path
    else:
        print("Corrupted path length:", len(corrupted_path))
        print("Corrupted path:", corrupted_path)

    extractor = ContextExtractor(patch_size=7)
    builder = PromptBuilder()
    parser = ResponseParser()
    client = LLMClient()
    replanner = Replanner(builder, parser, client, extractor)

    uncertain_cells = replanner.get_path_neighborhood(corrupted_grid, corrupted_path)
    print("Uncertain cells near path:", uncertain_cells)

    if not uncertain_cells:
        print("No uncertain cells near path, using corrupted path directly")
        fig, axes = plt.subplots(1, 3, figsize=(5, 5))
        plot_path(axes[0], grid, clean_path, start, goal, "original map + clean path")
        plot_path(axes[1], corrupted_grid, corrupted_path, start, goal, "corrupted map + path", corruption_cells)
        plot_path(axes[2], corrupted_grid, corrupted_path, start, goal, "replanned map + path", corruption_cells)
        plt.tight_layout()
        plt.show()
        return

    robot_pose = [0.0, 0.0, 0.0]
    modified_grid, decision = replanner.replan(
        grid=corrupted_grid,
        start=start,
        goal=goal,
        path=corrupted_path,
        robot_pose=robot_pose,
        skill_context=[]
    )

    print("LLM decision:", decision)

    modified_cost = build_base_cost_map(modified_grid)
    new_path = astar(modified_cost, start, goal)
    if new_path is None:
        print("No replanned path found")
        new_path = corrupted_path
    else:
        print("Replanned path length:", len(new_path))
        print("Replanned path:", new_path)

    fig, axes = plt.subplots(1, 3, figsize=(5, 5))
    plot_path(axes[0], grid, clean_path, start, goal, "original map + clean path")
    plot_path(axes[1], corrupted_grid, corrupted_path, start, goal, "corrupted map + path", corruption_cells)
    plot_path(axes[2], modified_grid, new_path, start, goal, "replanned map + new path", corruption_cells)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()