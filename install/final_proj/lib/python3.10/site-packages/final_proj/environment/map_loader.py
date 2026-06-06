import yaml
import numpy as np
from PIL import Image
from pathlib import Path

class MapLoader:
    def __init__(self, yaml_path):
        self.yaml_path = Path(yaml_path)
        self.data = None
        self.image = None
        self.grid = None

    def load(self):
        with open(self.yaml_path, "r") as f:
            self.data = yaml.safe_load(f)

        image_path = self.yaml_path.parent / self.data["image"]
        self.image = Image.open(image_path).convert("L")
        img_array = np.array(self.image).astype(np.float32)

        img_array = 255.0 - img_array

        occupancy_threshold = self.data.get("occupied_thresh", 0.65)
        free_threshold = self.data.get("free_thresh", 0.196)

        normalized = img_array / 255.0
        grid = np.full(normalized.shape, -1, dtype=int)
        grid[normalized >= occupancy_threshold] = 1
        grid[normalized <= free_threshold] = 0

        self.grid = np.flipud(grid)
        return self.grid

    def get_grid(self):
        if self.grid is None:
            return self.load()
        return self.grid

    def get_resolution(self):
        return self.data["resolution"]

    def get_origin(self):
        return self.data["origin"]