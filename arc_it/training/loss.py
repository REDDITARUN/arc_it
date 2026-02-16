"""Loss functions and metrics for ARC-IT training.

Primary loss: CrossEntropy on predicted vs ground truth colors per pixel,
with IGNORE_INDEX masking for padding and difficulty weighting for AGI-2.

Metrics tracked during training:
    - pixel_accuracy: fraction of correctly predicted valid pixels
    - grid_exact_match: fraction of grids predicted 100% correctly
"""

from typing import Dict

import torch
import torch.nn.functional as F

from arc_it.data.canvas import IGNORE_INDEX


def compute_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    difficulty: torch.Tensor = None,
) -> Dict[str, torch.Tensor]:
    """Compute cross-entropy loss with difficulty weighting.

    Args:
        logits: (B, C, H, W) raw class logits (C=12 colors).
        target: (B, H, W) ground truth integer labels (0-11, or IGNORE_INDEX).
        difficulty: (B,) per-sample difficulty weights. None = uniform.

    Returns:
        Dict with "loss" (scalar), "pixel_accuracy", "grid_exact_match".
    """
    B = logits.shape[0]

    # Per-pixel cross-entropy (unreduced)
    ce = F.cross_entropy(
        logits, target.long(),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )  # (B, H, W)

    # Apply difficulty weighting
    if difficulty is not None:
        ce = ce * difficulty[:, None, None]

    loss = ce.mean()

    # ─── Metrics ─────────────────────────────────────────────────
    with torch.no_grad():
        pred = logits.argmax(dim=1)  # (B, H, W)
        valid = target != IGNORE_INDEX  # (B, H, W)

        # Pixel accuracy
        correct_pixels = ((pred == target) & valid).sum()
        total_pixels = valid.sum().clamp(min=1)
        pixel_acc = correct_pixels.float() / total_pixels.float()

        # Grid exact match: a grid is correct only if ALL valid pixels match
        per_grid_correct = ((pred == target) | ~valid).all(dim=-1).all(dim=-1)  # (B,)
        grid_exact_match = per_grid_correct.float().mean()

    return {
        "loss": loss,
        "pixel_accuracy": pixel_acc,
        "grid_exact_match": grid_exact_match,
    }
