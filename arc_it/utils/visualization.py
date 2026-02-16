"""Visualization utilities for ARC grids."""

from typing import List, Optional

import numpy as np


ARC_CMAP = [
    (0, 0, 0),        # 0: black
    (0, 116, 217),     # 1: blue
    (255, 65, 54),     # 2: red
    (46, 204, 64),     # 3: green
    (255, 220, 0),     # 4: yellow
    (170, 170, 170),   # 5: gray
    (240, 18, 190),    # 6: magenta
    (255, 133, 27),    # 7: orange
    (127, 219, 255),   # 8: light blue
    (177, 13, 201),    # 9: maroon
    (30, 30, 30),      # 10: IGNORE (dark bg)
    (255, 255, 255),   # 11: PAD (white border)
]


def grid_to_rgb_array(grid: List[List[int]], cell_size: int = 10) -> np.ndarray:
    """Convert an ARC grid to an RGB numpy array for display.

    Args:
        grid: 2D list of integer color values (0-11).
        cell_size: Pixel size of each grid cell.

    Returns:
        RGB uint8 array of shape (H*cell_size, W*cell_size, 3).
    """
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = ARC_CMAP[min(grid[r][c], len(ARC_CMAP) - 1)]
            img[r * cell_size:(r + 1) * cell_size,
                c * cell_size:(c + 1) * cell_size] = color
    return img


def show_task(
    train_examples: list,
    test_input: Optional[list] = None,
    test_output: Optional[list] = None,
    prediction: Optional[list] = None,
    title: str = "",
) -> None:
    """Display an ARC task using matplotlib (if available).

    Works headless (saves to file) if no display is detected.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    n_train = len(train_examples)
    n_cols = max(n_train, 1) + (1 if test_input else 0)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for i, ex in enumerate(train_examples):
        axes[0, i].imshow(grid_to_rgb_array(ex["input"]))
        axes[0, i].set_title(f"Train {i+1} In")
        axes[0, i].axis("off")
        axes[1, i].imshow(grid_to_rgb_array(ex["output"]))
        axes[1, i].set_title(f"Train {i+1} Out")
        axes[1, i].axis("off")

    if test_input is not None:
        col = n_train
        axes[0, col].imshow(grid_to_rgb_array(test_input))
        axes[0, col].set_title("Test In")
        axes[0, col].axis("off")
        if prediction is not None:
            axes[1, col].imshow(grid_to_rgb_array(prediction))
            axes[1, col].set_title("Prediction")
        elif test_output is not None:
            axes[1, col].imshow(grid_to_rgb_array(test_output))
            axes[1, col].set_title("Test Out (GT)")
        axes[1, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig
