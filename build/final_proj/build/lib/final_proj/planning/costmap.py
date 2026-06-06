import numpy as np

def build_base_cost_map(grid):
    cost_map = np.full(grid.shape, np.inf, dtype=float)
    cost_map[grid == 0] = 1.0
    cost_map[grid == -1] = 3.0
    return cost_map

def apply_skill_bias(cost_map, skill_path, bias_strength=0.4, neighborhood=1):
    biased = cost_map.copy()
    if skill_path is None or len(skill_path) == 0:
        return biased

    height, width = biased.shape

    for x, y in skill_path:
        for dy in range(-neighborhood, neighborhood + 1):
            for dx in range(-neighborhood, neighborhood + 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height and np.isfinite(biased[ny, nx]):
                    if dx == 0 and dy == 0:
                        biased[ny, nx] = min(biased[ny, nx], max(0.2, biased[ny, nx] * bias_strength))
                    else:
                        biased[ny, nx] = min(biased[ny, nx], max(0.5, biased[ny, nx] * (bias_strength + 0.3)))

    return biased

def apply_region_penalty(cost_map, cells, penalty=5.0):
    adjusted = cost_map.copy()
    if cells is None:
        return adjusted

    height, width = adjusted.shape
    for x, y in cells:
        if 0 <= x < width and 0 <= y < height and np.isfinite(adjusted[y, x]):
            adjusted[y, x] += penalty
    return adjusted