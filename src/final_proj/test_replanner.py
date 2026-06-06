# loads the clean map
# finds start and goal
# plans path
# injects a corruption block in the middle
# replans on corrupted map
# uses the replanner pipeline if uncertain cells are involved

from pathlib import Path
from collections import deque

from final_proj.environment.map_loader import MapLoader
from final_proj.environment.corruption import Corruptor
from final_proj.environment.context_extractor import ContextExtractor
from final_proj.planning.astar import astar
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

    path = astar(grid, start, goal, allow_unknown=False)

    if path is None:
        print("No initial path found")
        return

    print("Initial path length:", len(path))
    print("Initial path:", path)

    corruptor = Corruptor(corruption_rate=0.1)

    mid_x = grid.shape[1] // 2
    mid_y = grid.shape[0] // 2
    corrupted_grid = corruptor.inject_block(grid, (mid_x - 10, mid_y - 10), (mid_x + 10, mid_y + 10))

    print("Injected corruption block at:", (mid_x - 10, mid_y - 10), (mid_x + 10, mid_y + 10))

    corrupted_path = astar(corrupted_grid, start, goal, allow_unknown=False)

    if corrupted_path is None:
        print("No path found on corrupted map")
    else:
        print("Corrupted path length:", len(corrupted_path))
        print("Corrupted path:", corrupted_path)

    extractor = ContextExtractor(patch_size=7)
    builder = PromptBuilder()
    parser = ResponseParser()
    client = LLMClient()
    replanner = Replanner(builder, parser, client, extractor)

    uncertain_cells = replanner.find_uncertain_on_path(corrupted_grid, corrupted_path if corrupted_path else path)
    print("Uncertain cells on path:", uncertain_cells)

    if not uncertain_cells:
        print("No uncertain cells on path, skipping replanning")
        return

    robot_pose = [0.0, 0.0, 0.0]
    modified_grid, decision = replanner.replan(
        grid=corrupted_grid,
        start=start,
        goal=goal,
        path=corrupted_path if corrupted_path else path,
        robot_pose=robot_pose,
        skill_context=[]
    )

    print("LLM decision:", decision)

    new_path = astar(modified_grid, start, goal, allow_unknown=False)

    if new_path is None:
        print("No replanned path found")
        return

    print("Replanned path length:", len(new_path))
    print("Replanned path:", new_path)


if __name__ == "__main__":
    main()