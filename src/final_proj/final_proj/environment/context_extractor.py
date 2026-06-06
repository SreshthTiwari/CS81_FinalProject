import numpy as np

class ContextExtractor:
    def __init__(self, patch_size=7):
        self.patch_size = patch_size

    def extract_patch(self, grid, center):
        half = self.patch_size // 2
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

    def find_uncertain_cells(self, grid):
        ys, xs = np.where(grid == -1)
        return list(zip(xs.tolist(), ys.tolist()))