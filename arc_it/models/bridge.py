"""Bridge module: maps frozen encoder features to Sana's hidden space.

Architecture:
    Input:  (B, 256, encoder_dim)  — patch embeddings from DINOv2/JEPA
    Layer1: Linear(encoder_dim → hidden_dim) + GELU + LayerNorm
    Layer2: Linear(hidden_dim → sana_dim) + LayerNorm
    PosEmb: Add learnable 2D positional embeddings (16x16 grid)
    Output: (B, 256, sana_dim)     — ready for Sana cross-attention

The two-layer design expands then compresses, creating a richer intermediate
representation. LayerNorm after each layer stabilizes training when bridging
between frozen encoder outputs and trainable Sana inputs.
"""

import math

import torch
import torch.nn as nn


class Bridge(nn.Module):
    """Learnable adapter from encoder feature space to Sana feature space."""

    def __init__(
        self,
        encoder_dim: int = 1024,
        hidden_dim: int = 2048,
        sana_dim: int = 1152,
        num_patches: int = 256,
        dropout: float = 0.1,
        use_2d_pos_embed: bool = True,
    ) -> None:
        """
        Args:
            encoder_dim: Input dimension from frozen encoder.
            hidden_dim: Intermediate expansion dimension.
            sana_dim: Output dimension matching Sana's hidden_size.
            num_patches: Number of spatial patches (256 for 16x16 grid).
            dropout: Dropout rate.
            use_2d_pos_embed: Add learnable 2D positional embeddings.
        """
        super().__init__()
        self.sana_dim = sana_dim
        self.num_patches = num_patches
        grid_size = int(math.sqrt(num_patches))
        self.grid_size = grid_size

        self.layer1 = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, sana_dim),
            nn.LayerNorm(sana_dim),
            nn.Dropout(dropout),
        )

        if use_2d_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, sana_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Map encoder features to Sana-compatible conditioning.

        Args:
            encoder_features: (B, num_patches, encoder_dim)

        Returns:
            (B, num_patches, sana_dim) conditioning for Sana cross-attention.
        """
        x = self.layer1(encoder_features)
        x = self.layer2(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        return x

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return f"params={n_params / 1e6:.1f}M, grid={self.grid_size}x{self.grid_size}"
