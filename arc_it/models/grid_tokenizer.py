"""Grid Tokenizer: embeds discrete ARC grids into continuous patch tokens.

Converts integer grids (values 0-11) into sequences of continuous patch
embeddings suitable for transformer processing. Used by both the Rule
Encoder (for demo grids) and the Rule Applier (for test grids).

Architecture:
    Input:  (B, 64, 64) discrete integer grid
    Step 1: Color embedding  — nn.Embedding(12, embed_dim)
    Step 2: Patch projection — Conv2d(embed_dim, hidden_size, patch_size, stride=patch_size)
    Step 3: Positional embed — learnable 2D positional embeddings (16x16)
    Output: (B, 256, hidden_size) patch token embeddings
"""

import torch
import torch.nn as nn


class GridTokenizer(nn.Module):
    """Tokenize discrete ARC grids into continuous patch embeddings.

    Shared across Rule Encoder and Rule Applier so that all grids
    (demo inputs, demo outputs, test inputs) live in the same
    embedding space.
    """

    def __init__(
        self,
        num_colors: int = 12,
        canvas_size: int = 64,
        hidden_size: int = 384,
        patch_size: int = 4,
    ) -> None:
        """
        Args:
            num_colors: Number of discrete colors (10 ARC + IGNORE + PAD).
            canvas_size: Fixed canvas dimension.
            hidden_size: Output embedding dimension per patch.
            patch_size: Spatial size of each patch (4 → 16x16 patches).
        """
        super().__init__()
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.grid_size = canvas_size // patch_size
        self.num_patches = self.grid_size ** 2

        embed_dim = hidden_size // 4

        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.patch_proj = nn.Conv2d(
            embed_dim, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Tokenize a discrete grid into patch embeddings.

        Args:
            grid: (B, H, W) integer grid with values 0-11.

        Returns:
            (B, num_patches, hidden_size) patch token embeddings.
        """
        x = self.color_embed(grid.long())                       # (B, H, W, embed_dim)
        x = x.permute(0, 3, 1, 2).contiguous()                 # (B, embed_dim, H, W)
        x = self.patch_proj(x)                                  # (B, hidden_size, gH, gW)
        x = x.flatten(2).transpose(1, 2).contiguous()           # (B, num_patches, hidden_size)
        x = x + self.pos_embed
        return x

    def extra_repr(self) -> str:
        n = sum(p.numel() for p in self.parameters()) / 1e6
        return (
            f"patch={self.patch_size}, grid={self.grid_size}x{self.grid_size}, "
            f"hidden={self.hidden_size}, params={n:.2f}M"
        )
