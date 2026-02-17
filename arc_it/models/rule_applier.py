"""Rule Applier: apply extracted transformation rule to test input.

Takes the test input tokens and the rule tokens extracted by the Rule
Encoder, and produces output tokens that encode the predicted output
grid. Uses cross-attention so the test tokens can "look up" what
transformation to apply from the rule representation.

Architecture per block:
    1. Self-attention on test tokens (spatial reasoning)
    2. Cross-attention: test tokens attend to rule tokens (apply rule)
    3. FFN (feature mixing)

Input:  test_tokens (B, N, C) + rule_tokens (B, M, C)
Output: output_tokens (B, N, C) ready for spatial decoder
"""

import torch
import torch.nn as nn

from arc_it.models.rule_encoder import StandardAttention, CrossAttention, FFN


class RuleApplierBlock(nn.Module):
    """Single Rule Applier block: self-attn + cross-attn(→rules) + FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 2.5,
    ) -> None:
        super().__init__()
        # Self-attention on test tokens
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = StandardAttention(hidden_size, num_heads)

        # Cross-attention: test tokens attend to rule tokens
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_rule = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads)

        # FFN
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FFN(hidden_size, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        rule_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) test input tokens.
            rule_tokens: (B, M, C) rule representation.
        Returns:
            (B, N, C) transformed tokens.
        """
        # Self-attend (spatial reasoning within the test grid)
        x = x + self.self_attn(self.norm1(x))
        # Cross-attend to rule tokens (apply the transformation)
        x = x + self.cross_attn(self.norm2(x), self.norm_rule(rule_tokens))
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class RuleApplier(nn.Module):
    """Apply extracted rule to test input via cross-attention.

    The Rule Applier is a standard transformer decoder that uses the
    rule tokens as its cross-attention context. Each layer allows the
    test tokens to progressively incorporate information from the rule
    representation, ultimately producing tokens that encode the
    predicted output grid.
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_layers: int = 4,
        num_heads: int = 8,
        num_patches: int = 256,
        mlp_ratio: float = 2.5,
    ) -> None:
        """
        Args:
            hidden_size: Transformer hidden dimension.
            num_layers: Number of applier blocks.
            num_heads: Attention heads.
            num_patches: Patches per grid (unused but kept for API symmetry).
            mlp_ratio: FFN expansion ratio.
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            RuleApplierBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        test_tokens: torch.Tensor,
        rule_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rule to test tokens.

        Args:
            test_tokens: (B, N, C) tokenized test input grid.
            rule_tokens: (B, M, C) rule representation from RuleEncoder.

        Returns:
            (B, N, C) output token embeddings for decoding.
        """
        x = test_tokens
        for block in self.blocks:
            x = block(x, rule_tokens)
        return self.final_norm(x)

    def extra_repr(self) -> str:
        n = sum(p.numel() for p in self.parameters()) / 1e6
        return f"layers={len(self.blocks)}, params={n:.1f}M"
