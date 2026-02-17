"""Spatial ARC Decoder: converts transformer patch tokens to 64x64 grid logits.

Architecture:
    Input:  (B, 256, hidden_size) output patch tokens from RuleApplier
    Reshape to spatial: (B, hidden_size, 16, 16)
    TransConv1: (B, ch1, 32, 32)
    TransConv2: (B, ch2, 64, 64)
    Conv1x1:    (B, 12, 64, 64) — 12-class logits per pixel
    Output: (B, 12, 64, 64)

Unlike global pooling (which destroys spatial information), this decoder
preserves full 2D structure through transposed convolutions. This is
critical for ARC where every pixel position matters.
"""

import math

import torch
import torch.nn as nn


class SpatialDecoder(nn.Module):
    """Upsample transformer patch tokens to full-resolution 12-class grid logits."""

    def __init__(
        self,
        hidden_size: int = 1152,
        num_patches: int = 256,
        canvas_size: int = 64,
        num_colors: int = 12,
        hidden_channels: tuple = (512, 256),
    ) -> None:
        """
        Args:
            hidden_size: Transformer's output dimension per patch.
            num_patches: Number of patches (must be perfect square).
            canvas_size: Target spatial resolution.
            num_colors: Number of output classes (10 ARC + IGNORE + PAD).
            hidden_channels: Channel sizes for transposed conv layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patches = num_patches
        self.grid_size = int(math.sqrt(num_patches))
        self.canvas_size = canvas_size
        self.num_colors = num_colors

        assert self.grid_size ** 2 == num_patches, \
            f"num_patches={num_patches} must be a perfect square"

        # Calculate number of 2x upsampling steps needed
        # 16 -> 32 -> 64 (2 steps for canvas_size=64)
        current_size = self.grid_size
        layers = []
        in_ch = hidden_size

        for out_ch in hidden_channels:
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(out_ch),
            ])
            in_ch = out_ch
            current_size *= 2

        # If we haven't reached canvas_size, add more layers
        while current_size < canvas_size:
            out_ch = max(in_ch // 2, num_colors)
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(out_ch),
            ])
            in_ch = out_ch
            current_size *= 2

        self.upsample = nn.Sequential(*layers)

        # Final 1x1 conv to produce per-pixel class logits
        self.head = nn.Conv2d(in_ch, num_colors, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.head.bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, N, C) where N=256, C=hidden_size.

        Returns:
            (B, num_colors, canvas_size, canvas_size) logits.
        """
        B, N, C = patches.shape
        # Reshape to spatial grid
        x = patches.reshape(B, self.grid_size, self.grid_size, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Upsample through transposed convolutions
        x = self.upsample(x)

        # Project to class logits
        logits = self.head(x)  # (B, num_colors, canvas_size, canvas_size)

        return logits

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return (
            f"{self.grid_size}x{self.grid_size} -> "
            f"{self.canvas_size}x{self.canvas_size}, "
            f"classes={self.num_colors}, "
            f"params={n_params:.1f}M"
        )
