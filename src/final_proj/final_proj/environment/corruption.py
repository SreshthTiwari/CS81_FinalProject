import numpy as np

class Corruptor:
    def __init__(self, corruption_rate=0.1, uncertain_value=-1):
        self.corruption_rate = corruption_rate
        self.uncertain_value = uncertain_value

    def inject_random_corruption(self, grid, corruption_rate=None, change_over_time=False, num_timesteps=1, jitter=0.02):
        """Inject random uncertain cells into the grid.

        If change_over_time is False (default) returns a single corrupted grid.
        If change_over_time is True returns a list of `num_timesteps` corrupted grids
        with slightly varying corruption patterns (rate jittered by `jitter`).
        """
        if corruption_rate is None:
            corruption_rate = self.corruption_rate

        def single_corrupt(rate):
            corrupted = grid.copy()
            free_or_occupied = np.where((corrupted == 0) | (corrupted == 1))
            num_cells = len(free_or_occupied[0])
            num_corrupt = int(num_cells * rate)
            if num_corrupt == 0:
                return corrupted
            indices = np.random.choice(num_cells, num_corrupt, replace=False)
            rows = free_or_occupied[0][indices]
            cols = free_or_occupied[1][indices]
            corrupted[rows, cols] = self.uncertain_value
            return corrupted

        if not change_over_time or num_timesteps <= 1:
            return single_corrupt(corruption_rate)

        sequence = []
        for t in range(num_timesteps):
            # jitter the corruption rate to create time-varying noise
            rate = max(0.0, min(1.0, corruption_rate + np.random.uniform(-jitter, jitter)))
            sequence.append(single_corrupt(rate))

        return sequence

    def inject_random_blockage(self, grid, blockage_rate=None):
        corrupted = grid.copy()
        if blockage_rate is None:
            blockage_rate = self.corruption_rate
        free_cells = np.where(corrupted == 0)
        num_cells = len(free_cells[0])
        num_block = int(num_cells * blockage_rate)
        if num_block == 0:
            return corrupted
        indices = np.random.choice(num_cells, num_block, replace=False)
        rows = free_cells[0][indices]
        cols = free_cells[1][indices]
        corrupted[rows, cols] = 1
        return corrupted

    def inject_block(self, grid, top_left, bottom_right, blocked_value=1):
        corrupted = grid.copy()
        x1, y1 = top_left
        x2, y2 = bottom_right
        corrupted[y1:y2, x1:x2] = blocked_value
        return corrupted

    def generate_moving_obstacle_sequence(self, grid, obstacle_path, num_timesteps=3):
        """Generate a sequence of grids where an obstacle moves along a path.
        
        Args:
            grid: original grid
            obstacle_path: list of (x, y) positions where obstacle moves
            num_timesteps: number of timesteps to generate
            
        Returns:
            list of grids, one per timestep
        """
        sequence = []
        half = 1

        if not obstacle_path:
            return [grid.copy() for _ in range(num_timesteps)]

        # Choose distinct positions along the path so the obstacle clearly moves.
        if len(obstacle_path) >= num_timesteps and num_timesteps > 1:
            indices = [int(round(i * (len(obstacle_path) - 1) / (num_timesteps - 1))) for i in range(num_timesteps)]
        else:
            indices = list(range(len(obstacle_path)))

        positions = [obstacle_path[i] for i in indices]

        # If we still need extra steps, continue moving from the last position.
        while len(positions) < num_timesteps:
            last_x, last_y = positions[-1]
            if len(positions) == 1:
                # move horizontally if only one anchor position is available
                next_x = min(last_x + 1, grid.shape[1] - 1)
                next_y = last_y
            else:
                prev_x, prev_y = positions[-2]
                dx = last_x - prev_x
                dy = last_y - prev_y
                next_x = min(max(last_x + dx, 0), grid.shape[1] - 1)
                next_y = min(max(last_y + dy, 0), grid.shape[0] - 1)
            if (next_x, next_y) == positions[-1]:
                next_x = min(last_x + 1, grid.shape[1] - 1)
            positions.append((next_x, next_y))

        for x, y in positions[:num_timesteps]:
            corrupted = grid.copy()
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < corrupted.shape[1] and 0 <= ny < corrupted.shape[0]:
                        if corrupted[ny, nx] == 0:
                            corrupted[ny, nx] = 1
            sequence.append(corrupted)

        return sequence

    def generate_permanent_obstacle_sequence(self, grid, blocked_region, num_timesteps=3):
        """Generate a sequence of grids where an obstacle stays in place.
        
        Args:
            grid: original grid
            blocked_region: tuple (x1, y1, x2, y2) defining blocked rectangle
            num_timesteps: number of timesteps to generate
            
        Returns:
            list of grids with same blockage, one per timestep
        """
        corrupted = self.inject_block(grid, blocked_region[:2], blocked_region[2:], blocked_value=1)
        sequence = [corrupted.copy() for _ in range(num_timesteps)]
        return sequence
