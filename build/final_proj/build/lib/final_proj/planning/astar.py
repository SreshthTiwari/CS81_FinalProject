import heapq
import math
import numpy as np

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def astar(cost_map, start, goal):
    height, width = cost_map.shape
    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in closed:
            continue

        if current == goal:
            return reconstruct_path(came_from, current)

        closed.add(current)
        x, y = current

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = x + dx
            ny = y + dy

            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue

            cell_cost = cost_map[ny, nx]
            if not np.isfinite(cell_cost):
                continue

            neighbor = (nx, ny)
            tentative_g = g_score[current] + cell_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None