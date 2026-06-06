import math
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from final_proj.environment.map_loader import MapLoader
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


def find_first_free_cell(grid):
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0:
                return (x, y)
    return None


def main():
    map_path = Path("/root/ros2_ws/src/pa3/maze.yml")
    loader = MapLoader(map_path)
    grid = loader.load()

    # start = find_first_free_cell(grid)
    start = (25, 25)
    if start is None:
        print("No free start cell found")
        return

    reachable = bfs_reachable(grid, start)
    goal = farthest_cell(start, reachable)
    # goal = (2, 1)
    if goal is None or goal == start:
        print("No valid goal found")
        return

    path = astar(grid, start, goal, allow_unknown=False)
    if path is None:
        print("No path found")
        return

    print("Start:", start)
    print("Goal:", goal)
    print("Path length:", len(path))

    display = np.zeros_like(grid, dtype=float)
    display[grid == 0] = 1.0
    display[grid == 1] = 0.0
    display[grid == -1] = 0.5

    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]

    plt.figure(figsize=(3, 3))
    plt.imshow(display, cmap="gray", origin="lower")
    plt.plot(path_x, path_y, color="red", linewidth=2)
    plt.scatter([start[0]], [start[1]], c="blue", s=50, label="start")
    plt.scatter([goal[0]], [goal[1]], c="green", s=50, label="goal")
    plt.legend()
    plt.title("Occupancy Grid with A* Path")
    plt.show()


if __name__ == "__main__":
    main()