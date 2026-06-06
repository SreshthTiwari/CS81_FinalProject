import yaml
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

def load_variant(yaml_path, flipud=False, invert=False):
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    image_path = yaml_path.parent / data["image"]
    image = Image.open(image_path).convert("L")
    img_array = np.array(image).astype(np.float32)

    if invert:
        img_array = 255.0 - img_array

    normalized = img_array / 255.0
    occupied_thresh = data.get("occupied_thresh", 0.65)
    free_thresh = data.get("free_thresh", 0.196)

    grid = np.full(normalized.shape, -1, dtype=int)
    grid[normalized >= occupied_thresh] = 1
    grid[normalized <= free_thresh] = 0

    if flipud:
        grid = np.flipud(grid)

    return grid

def show_grid(ax, grid, title):
    display = np.zeros_like(grid, dtype=float)
    display[grid == 0] = 1.0
    display[grid == 1] = 0.0
    display[grid == -1] = 0.5
    ax.imshow(display, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def main():
    yaml_path = "/root/ros2_ws/src/pa3/maze.yml"

    variants = [
        (False, False, "no flip, no invert"),
        (True, False, "flip, no invert"),
        (False, True, "no flip, invert"),
        (True, True, "flip, invert"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (flipud, invert, title) in zip(axes, variants):
        grid = load_variant(yaml_path, flipud=flipud, invert=invert)
        show_grid(ax, grid, title)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()