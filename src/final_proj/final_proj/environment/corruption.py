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
        """Generate a sequence of grids where an obstacle moves along a path, then off.
        
        Args:
            grid: original grid
            obstacle_path: list of (x, y) positions where obstacle moves
            num_timesteps: number of timesteps to generate
            
        Returns:
            list of grids, one per timestep
            
        Behavior: Obstacle moves ON the path for the first ~2 timesteps, then moves OFF the path
        for later timesteps so the path becomes clear again.
        """
        sequence = []
        half = 1

        if not obstacle_path or len(obstacle_path) == 0:
            return [grid.copy() for _ in range(num_timesteps)]

        positions = []
        on_path_steps = min(2, max(1, num_timesteps - 1))  # obstacle on path for first 1-2 timesteps

        # Phase 1: Obstacle moves ON the path for early timesteps
        for t in range(on_path_steps):
            if num_timesteps > 1 and on_path_steps > 1:
                idx = int(round(t * (len(obstacle_path) - 1) / (on_path_steps - 1)))
            else:
                idx = len(obstacle_path) // 2  # middle of path
            idx = min(idx, len(obstacle_path) - 1)
            positions.append(obstacle_path[idx])

        # Phase 2: Obstacle moves OFF the path for later timesteps
        if len(positions) < num_timesteps:
            # Compute perpendicular direction from the end of the on-path segment
            last_on_path = positions[-1]
            if len(positions) >= 2:
                prev_on_path = positions[-2]
                dx = last_on_path[0] - prev_on_path[0]
                dy = last_on_path[1] - prev_on_path[1]
                # Perpendicular: rotate 90 degrees
                perp_x, perp_y = -dy, dx
            else:
                # Fallback perpendicular direction
                perp_x, perp_y = 1, 0

            # Normalize perpendicular direction
            if perp_x == 0 and perp_y == 0:
                perp_x, perp_y = 1, 0

            # Move obstacle off the path by stepping in perpendicular direction
            current_x, current_y = last_on_path
            step_size = 1
            for t in range(on_path_steps, num_timesteps):
                # Try moving in perpendicular direction with increasing offset
                offset = (t - on_path_steps + 1) * step_size
                target_x = current_x + perp_x * offset
                target_y = current_y + perp_y * offset

                # If out of bounds or hit wall, try opposite direction
                if not (0 <= target_x < grid.shape[1] and 0 <= target_y < grid.shape[0]) or grid[target_y, target_x] != 0:
                    target_x = current_x - perp_x * offset
                    target_y = current_y - perp_y * offset

                # If still invalid, clamp to boundaries
                target_x = max(0, min(target_x, grid.shape[1] - 1))
                target_y = max(0, min(target_y, grid.shape[0] - 1))

                positions.append((target_x, target_y))

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

    def generate_clustered_uncertainty_sequence(self, grid, num_timesteps=3, num_clusters=6, cluster_radius=2, jitter=1, force_centers=None):
        """Generate clustered uncertain cells across the grid.

        Creates small clusters (blobs) of uncertain cells that resemble occluding obstacles.
        Clusters can jitter slightly over time to simulate sensing variation.
        """
        def sample_cluster_cells(cx, cy, radius):
            cells = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # prefer cells within a circle to make roundish blobs
                    if dx * dx + dy * dy <= radius * radius:
                        x = cx + dx
                        y = cy + dy
                        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                            if grid[y, x] == 0:
                                cells.append((x, y))
            return cells

        free_cells = np.column_stack(np.where(grid == 0))
        if len(free_cells) == 0:
            return [grid.copy() for _ in range(num_timesteps)]

        sequence = []
        # pick initial cluster centers. allow forcing some centers (e.g., on a path)
        centers = []
        if force_centers:
            for (cx, cy) in force_centers:
                # clamp and only keep if free in original grid
                ncx = int(min(max(cx, 0), grid.shape[1] - 1))
                ncy = int(min(max(cy, 0), grid.shape[0] - 1))
                if grid[ncy, ncx] == 0:
                    centers.append((ncx, ncy))

        remaining = max(0, min(num_clusters, len(free_cells)) - len(centers))
        if remaining > 0:
            # sample additional random centers from free cells
            available_idx = np.setdiff1d(np.arange(len(free_cells)), np.array([], dtype=int))
            # If we already used some forced centers that coincide with free_cells, try to avoid duplicates by sampling
            # randomly; simplest is to sample without replacement from all free_cells
            chosen_idx = np.random.choice(len(free_cells), size=remaining, replace=False)
            for i in chosen_idx:
                centers.append(tuple(free_cells[i][::-1]))  # convert (row,col)->(x,y)

        for t in range(num_timesteps):
            corrupted = grid.copy()
            # jitter centers slightly over time to simulate sensor variation
            jittered_centers = []
            for (cx, cy) in centers:
                if jitter > 0:
                    dx = int(np.round(np.random.uniform(-jitter, jitter)))
                    dy = int(np.round(np.random.uniform(-jitter, jitter)))
                else:
                    dx = dy = 0
                ncx = min(max(cx + dx, 0), grid.shape[1] - 1)
                ncy = min(max(cy + dy, 0), grid.shape[0] - 1)
                jittered_centers.append((ncx, ncy))

            # for each cluster, mark a small blob as uncertain
            for (cx, cy) in jittered_centers:
                cluster_cells = sample_cluster_cells(cx, cy, cluster_radius)
                for (x, y) in cluster_cells:
                    if corrupted[y, x] == 0:
                        corrupted[y, x] = self.uncertain_value

            sequence.append(corrupted)

        return sequence
