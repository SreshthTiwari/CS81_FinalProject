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


def main():
    map_path = Path("/root/ros2_ws/src/pa3/maze.yml")
    loader = MapLoader(map_path)
    grid = loader.load()

    print("Map loaded")
    print("Grid shape:", grid.shape)
    print("Resolution:", loader.get_resolution())
    print("Origin:", loader.get_origin())

    robot_guess = (grid.shape[1] - 10, grid.shape[0] - 10)
    start = nearest_free_cell(grid, robot_guess, search_radius=40)

    if start is None:
        print("No start found")
        return

    reachable = bfs_reachable(grid, start)
    goal = farthest_cell(start, reachable)

    if goal is None or goal == start:
        print("No valid goal found")
        return

    print("Start:", start)
    print("Goal:", goal)
    print("Reachable cells:", len(reachable))

    path = astar(grid, start, goal, allow_unknown=False)
    if path is None:
        print("No path found")
        return

    print("Path length:", len(path))
    print("Path:", path)

    display = np.zeros_like(grid, dtype=float)
    display[grid == 0] = 1.0
    display[grid == 1] = 0.0
    display[grid == -1] = 0.5

    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]

    plt.figure(figsize=(6, 6))
    plt.imshow(display, cmap="gray", origin="lower")
    plt.plot(path_x, path_y, color="red", linewidth=2)
    plt.scatter([start[0]], [start[1]], c="blue", s=40, label="start")
    plt.scatter([goal[0]], [goal[1]], c="green", s=40, label="goal")
    plt.legend()
    plt.title("A* Path Only")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()