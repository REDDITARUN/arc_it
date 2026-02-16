"""Canvas operations: padding, translation, cropping for ARC grids.

ARC grids are variable-size (up to 30x30) with 10 colors (0-9).
We place them on a fixed-size canvas (default 64x64) with:
  - IGNORE_INDEX (10): padding for empty space
  - PAD_INDEX (11): boundary markers around the output shape

This follows the VARC approach which proved that a fixed canvas
with translation augmentation significantly improves generalization.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

IGNORE_INDEX = 10
PAD_INDEX = 11
NUM_COLORS = 12  # 10 ARC colors + IGNORE + PAD

ARC_COLORS_RGB = np.array([
    [0, 0, 0],        # 0: black
    [0, 116, 217],     # 1: blue
    [255, 65, 54],     # 2: red
    [46, 204, 64],     # 3: green
    [255, 220, 0],     # 4: yellow
    [170, 170, 170],   # 5: gray
    [240, 18, 190],    # 6: magenta
    [255, 133, 27],    # 7: orange
    [127, 219, 255],   # 8: light blue
    [177, 13, 201],    # 9: maroon
    [30, 30, 30],      # 10: IGNORE (dark background)
    [255, 255, 255],   # 11: PAD (white boundary)
], dtype=np.uint8)


def pad_grid_to_canvas(
    grid: List[List[int]],
    canvas_size: int = 64,
    x_offset: int = 1,
    y_offset: int = 1,
    mark_boundary: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Place an ARC grid onto a fixed-size canvas with padding and translation.

    Args:
        grid: 2D list of integers (0-9), variable size up to 30x30.
        canvas_size: Fixed canvas dimension (default 64).
        x_offset: Horizontal offset for placement (>= 1 for border).
        y_offset: Vertical offset for placement (>= 1 for border).
        mark_boundary: If True, add PAD_INDEX markers around the grid
                       (right edge + bottom edge) to indicate output shape.

    Returns:
        canvas: (canvas_size, canvas_size) tensor, values 0-11.
        mask: (canvas_size, canvas_size) binary mask, 1 = valid pixel.
        height: Original grid height.
        width: Original grid width.
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    max_usable = canvas_size - 2  # leave 1px border on each side
    if height > max_usable or width > max_usable:
        raise ValueError(
            f"Grid size ({height}, {width}) exceeds canvas usable area "
            f"({max_usable}, {max_usable}). Canvas size = {canvas_size}."
        )

    canvas = torch.full((canvas_size, canvas_size), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((canvas_size, canvas_size), dtype=torch.long)

    values = torch.tensor(grid, dtype=torch.long)
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = values
    mask[y_offset:y_offset + height, x_offset:x_offset + width] = 1

    if mark_boundary:
        # Right edge marker
        canvas[y_offset:y_offset + height, x_offset + width] = PAD_INDEX
        # Bottom edge marker (including corner)
        canvas[y_offset + height, x_offset:x_offset + width + 1] = PAD_INDEX
        # Extend mask to include boundary markers
        mask[y_offset:y_offset + height + 1, x_offset:x_offset + width + 1] = 1

    return canvas, mask, height, width


def random_offset(
    grid_height: int,
    grid_width: int,
    canvas_size: int,
    rng: np.random.RandomState,
) -> Tuple[int, int]:
    """Generate random (x, y) offset for translation augmentation.

    Ensures the grid + 1px boundary markers fit within the canvas.
    """
    max_usable = canvas_size - 2
    max_x = max_usable - grid_width
    max_y = max_usable - grid_height

    x_offset = rng.randint(1, max(1, max_x) + 1) if max_x > 0 else 1
    y_offset = rng.randint(1, max(1, max_y) + 1) if max_y > 0 else 1

    return x_offset, y_offset


def resolution_scale(
    grid: List[List[int]],
    scale_factor: int,
) -> List[List[int]]:
    """Upsample a grid by repeating each pixel scale_factor times.

    A 5x7 grid with scale_factor=2 becomes 10x14.
    """
    if scale_factor <= 1:
        return grid
    arr = np.array(grid)
    scaled = np.repeat(np.repeat(arr, scale_factor, axis=0), scale_factor, axis=1)
    return scaled.tolist()


def random_scale_factor(
    grid_height: int,
    grid_width: int,
    canvas_size: int,
    rng: np.random.RandomState,
) -> int:
    """Pick a random scale factor that keeps the grid within the canvas."""
    max_usable = canvas_size - 2
    max_dim = max(grid_height, grid_width)
    max_scale = max_usable // max_dim if max_dim > 0 else 1
    if max_scale <= 1:
        return 1
    return rng.randint(1, max_scale + 1)


def crop_prediction_from_canvas(
    prediction: torch.Tensor,
    x_offset: int,
    y_offset: int,
    scale_factor: int = 1,
) -> List[List[int]]:
    """Extract the predicted grid from a canvas, reversing padding and scaling.

    Args:
        prediction: (canvas_size, canvas_size) integer tensor (argmax output).
        x_offset: The x offset used during canvas placement.
        y_offset: The y offset used during canvas placement.
        scale_factor: Resolution scale factor used during augmentation.

    Returns:
        Cropped grid as 2D list of integers (0-9).
    """
    pred_np = prediction.cpu().numpy()
    # Crop from offset
    cropped = pred_np[y_offset:, x_offset:]

    # Find boundaries using PAD_INDEX
    len_x = 0
    len_y = 0
    if cropped.size > 0:
        while len_x < cropped.shape[1] and cropped[0, len_x] != PAD_INDEX:
            len_x += 1
        while len_y < cropped.shape[0] and cropped[len_y, 0] != PAD_INDEX:
            len_y += 1

    if len_x == 0 or len_y == 0:
        return [[0]]

    grid = cropped[:len_y, :len_x]

    # Downsample via majority vote if resolution was scaled
    if scale_factor > 1:
        out_h = len_y // scale_factor
        out_w = len_x // scale_factor
        downsampled = []
        for i in range(out_h):
            row = []
            for j in range(out_w):
                block = grid[
                    i * scale_factor:(i + 1) * scale_factor,
                    j * scale_factor:(j + 1) * scale_factor,
                ].flatten()
                counts = np.bincount(block, minlength=NUM_COLORS)
                row.append(int(np.argmax(counts)))
            downsampled.append(row)
        return downsampled

    return grid.tolist()
