import numpy as np

class Corruptor:
    def __init__(self, corruption_rate=0.1, uncertain_value=-1):
        self.corruption_rate = corruption_rate
        self.uncertain_value = uncertain_value

    def inject_random_corruption(self, grid):
        corrupted = grid.copy()
        free_or_occupied = np.where((corrupted == 0) | (corrupted == 1))
        num_cells = len(free_or_occupied[0])
        num_corrupt = int(num_cells * self.corruption_rate)
        if num_corrupt == 0:
            return corrupted
        indices = np.random.choice(num_cells, num_corrupt, replace=False)
        rows = free_or_occupied[0][indices]
        cols = free_or_occupied[1][indices]
        corrupted[rows, cols] = self.uncertain_value
        return corrupted

    def inject_block(self, grid, top_left, bottom_right):
        corrupted = grid.copy()
        x1, y1 = top_left
        x2, y2 = bottom_right
        corrupted[y1:y2, x1:x2] = self.uncertain_value
        return corrupted