"""Data loading, augmentation, and preprocessing for ARC-AGI datasets."""

from arc_it.data.canvas import (
    IGNORE_INDEX,
    PAD_INDEX,
    NUM_COLORS,
    ARC_COLORS_RGB,
    pad_grid_to_canvas,
    crop_prediction_from_canvas,
)
from arc_it.data.augmentation import (
    rotate_grid,
    flip_grid,
    transpose_grid,
    permute_colors,
    get_geometric_augmentations,
    augment_task,
)
from arc_it.data.dataset import ARCDataset, build_dataloaders
from arc_it.data.rendering import (
    render_grid_to_rgb,
    render_canvas_to_rgb_224,
    batch_render_canvas_to_rgb_224,
)
