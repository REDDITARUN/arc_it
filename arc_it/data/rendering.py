"""Render ARC grids as RGB images for vision model input.

This module handles the conversion path:
    ARC grid (integers 0-11) → RGB image (3xHxW float tensor)

Two use cases:
1. render_grid_to_rgb: Raw grid → RGB numpy array (for visualization)
2. render_canvas_to_rgb_224: Padded canvas → 224x224 normalized tensor (for DINOv2/JEPA)

We use nearest-neighbor interpolation for upsampling to preserve
the sharp color boundaries that are critical for ARC pattern recognition.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from arc_it.data.canvas import ARC_COLORS_RGB

# ImageNet normalization (required by DINOv2)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def render_grid_to_rgb(
    grid: torch.Tensor,
    color_lut: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Convert a canvas tensor to an RGB float tensor.

    Args:
        grid: (H, W) integer tensor with values 0-11.
        color_lut: (12, 3) uint8 array of RGB colors. Defaults to ARC_COLORS_RGB.

    Returns:
        (3, H, W) float32 tensor with values in [0, 1].
    """
    if color_lut is None:
        color_lut = ARC_COLORS_RGB

    lut_tensor = torch.tensor(color_lut, dtype=torch.float32) / 255.0  # (12, 3)
    grid_long = grid.long().clamp(0, len(lut_tensor) - 1)
    rgb = lut_tensor[grid_long]  # (H, W, 3)
    return rgb.permute(2, 0, 1)  # (3, H, W)


def render_canvas_to_rgb_224(
    canvas: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a canvas to a 224x224 RGB tensor for DINOv2/JEPA input.

    Args:
        canvas: (H, W) integer tensor (e.g., 64x64 canvas with values 0-11).
        normalize: If True, apply ImageNet normalization.

    Returns:
        (3, 224, 224) float32 tensor, optionally normalized.
    """
    rgb = render_grid_to_rgb(canvas)  # (3, H, W)
    # Upsample to 224x224 with nearest-neighbor (preserves sharp edges)
    rgb_224 = F.interpolate(
        rgb.unsqueeze(0),
        size=(224, 224),
        mode="nearest",
    ).squeeze(0)  # (3, 224, 224)

    if normalize:
        mean = IMAGENET_MEAN.to(rgb_224.device)
        std = IMAGENET_STD.to(rgb_224.device)
        rgb_224 = (rgb_224 - mean) / std

    return rgb_224


def batch_render_canvas_to_rgb_224(
    canvases: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Batch-convert canvases to 224x224 RGB tensors.

    Args:
        canvases: (B, H, W) integer tensor.
        normalize: If True, apply ImageNet normalization.

    Returns:
        (B, 3, 224, 224) float32 tensor.
    """
    B = canvases.shape[0]
    lut_tensor = torch.tensor(ARC_COLORS_RGB, dtype=torch.float32, device=canvases.device) / 255.0
    canvases_long = canvases.long().clamp(0, len(lut_tensor) - 1)
    rgb = lut_tensor[canvases_long]  # (B, H, W, 3)
    rgb = rgb.permute(0, 3, 1, 2)   # (B, 3, H, W)

    rgb_224 = F.interpolate(rgb, size=(224, 224), mode="nearest")  # (B, 3, 224, 224)

    if normalize:
        mean = IMAGENET_MEAN.to(rgb_224.device)
        std = IMAGENET_STD.to(rgb_224.device)
        rgb_224 = (rgb_224 - mean) / std

    return rgb_224
