"""Rule Encoder: extract transformation rules from demonstration pairs.

The core novelty of the Rule-Conditioned Transformer. Instead of
implicitly learning rules during training, this module EXPLICITLY
extracts what transformation was applied by cross-attending output
features against input features for each demo pair, then aggregating
across pairs into a fixed-length rule representation.

Architecture:
    For each demo pair (input_k, output_k):
        1. Tokenize both grids → in_tokens, out_tokens (via shared GridTokenizer)
        2. Cross-attend: out_tokens attend to in_tokens → delta_tokens
           This captures "what changed between input and output"

    Across all K demo pairs:
        3. Add pair positional embeddings (distinguish pair 1, 2, 3, ...)
        4. Concatenate delta_tokens from all pairs → (B, K*N, C)
        5. Self-attend across pairs → shared understanding of the rule
        6. Learnable rule query tokens attend to all delta tokens
           → fixed-length rule embedding (B, M, C)

    Output: rule_tokens (B, num_rule_tokens, hidden_size)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Attention Primitives ────────────────────────────────────────────

class StandardAttention(nn.Module):
    """Standard multi-head self-attention with optional masking."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input tokens.
            mask: (B, N) boolean mask, True = valid token.
        Returns:
            (B, N, C) attended tokens.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)  # (B, H, N, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        if mask is not None:
            # mask shape: (B, N) → expand to (B, 1, 1, N) for key masking
            key_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)  # handle all-masked rows

        out = (attn @ v).permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Standard multi-head cross-attention with optional KV masking."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, hidden_size * 2)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) query tokens.
            context: (B, M, C) key/value tokens.
            kv_mask: (B, M) boolean mask for context, True = valid.
        Returns:
            (B, N, C) cross-attended tokens.
        """
        B, N, C = x.shape
        M = context.shape[1]

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, H, N, D)

        kv = self.kv_proj(context).reshape(B, M, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.permute(0, 2, 1, 3)  # (B, H, M, D)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, M)

        if kv_mask is not None:
            key_mask = kv_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, M)
            attn = attn.masked_fill(~key_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)

        out = (attn @ v).permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        return self.out_proj(out)


class FFN(nn.Module):
    """Standard feed-forward with GELU activation."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 2.5) -> None:
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ─── Encoder Blocks ─────────────────────────────────────────────────

class PairEncoderBlock(nn.Module):
    """Process a single demo pair: cross-attend output→input.

    This is the key operation that captures "what changed" between
    the input and output of a demonstration pair.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 2.5,
    ) -> None:
        super().__init__()
        # Cross-attention: output tokens attend to input tokens
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm_ctx = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads)

        # Self-attention within the delta tokens
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = StandardAttention(hidden_size, num_heads)

        # FFN
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FFN(hidden_size, mlp_ratio)

    def forward(
        self,
        out_tokens: torch.Tensor,
        in_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            out_tokens: (B, N, C) output grid token embeddings.
            in_tokens:  (B, N, C) input grid token embeddings.
        Returns:
            (B, N, C) delta tokens capturing the transformation.
        """
        # Cross-attend: what changed from input to output?
        out_tokens = out_tokens + self.cross_attn(
            self.norm1(out_tokens), self.norm_ctx(in_tokens)
        )
        # Self-attend within the pair
        out_tokens = out_tokens + self.self_attn(self.norm2(out_tokens))
        # FFN
        out_tokens = out_tokens + self.ffn(self.norm3(out_tokens))
        return out_tokens


class AggregatorBlock(nn.Module):
    """Self-attention across all demo pairs to build shared rule understanding."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 2.5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn = StandardAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FFN(hidden_size, mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, K*N, C) concatenated delta tokens from all pairs.
            mask: (B, K*N) boolean mask, True = valid token.
        """
        x = x + self.self_attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ─── Full Rule Encoder ──────────────────────────────────────────────

class RuleEncoder(nn.Module):
    """Extract a fixed-length rule representation from K demo pairs.

    The Rule Encoder is the core architectural novelty: it explicitly
    computes what transformation was applied by comparing output to
    input for each demo pair, then aggregates across pairs.

    Output is a fixed-length set of rule tokens regardless of how
    many demo pairs are provided (1-5).
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_pair_layers: int = 2,
        num_agg_layers: int = 2,
        num_heads: int = 8,
        num_rule_tokens: int = 64,
        max_demos: int = 5,
        num_patches: int = 256,
        mlp_ratio: float = 2.5,
    ) -> None:
        """
        Args:
            hidden_size: Transformer hidden dimension.
            num_pair_layers: Layers for per-pair cross-attention.
            num_agg_layers: Layers for cross-pair aggregation.
            num_heads: Attention heads.
            num_rule_tokens: Number of learnable rule query tokens (M).
            max_demos: Maximum number of demo pairs (for positional embeds).
            num_patches: Patches per grid (256 = 16x16 for canvas=64, patch=4).
            mlp_ratio: FFN expansion ratio.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_rule_tokens = num_rule_tokens
        self.max_demos = max_demos
        self.num_patches = num_patches

        # Per-pair encoder: cross-attend output → input
        self.pair_blocks = nn.ModuleList([
            PairEncoderBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(num_pair_layers)
        ])

        # Pair positional embeddings to distinguish demo 1, 2, 3, ...
        self.pair_pos_embed = nn.Parameter(
            torch.zeros(1, max_demos, 1, hidden_size)
        )
        nn.init.trunc_normal_(self.pair_pos_embed, std=0.02)

        # Cross-pair aggregation via self-attention
        self.agg_blocks = nn.ModuleList([
            AggregatorBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(num_agg_layers)
        ])

        # Learnable rule query tokens (Perceiver-style bottleneck)
        self.rule_queries = nn.Parameter(
            torch.zeros(1, num_rule_tokens, hidden_size)
        )
        nn.init.trunc_normal_(self.rule_queries, std=0.02)

        # Cross-attention: rule queries attend to aggregated delta tokens
        self.rule_cross_norm_q = nn.LayerNorm(hidden_size, eps=1e-6)
        self.rule_cross_norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.rule_cross_attn = CrossAttention(hidden_size, num_heads)
        self.rule_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        demo_in_tokens: torch.Tensor,
        demo_out_tokens: torch.Tensor,
        num_demos: torch.Tensor,
    ) -> torch.Tensor:
        """Extract rule tokens from demo pairs.

        Args:
            demo_in_tokens:  (B, K, N, C) tokenized demo input grids.
            demo_out_tokens: (B, K, N, C) tokenized demo output grids.
            num_demos:       (B,) number of valid demos per sample.

        Returns:
            rule_tokens: (B, num_rule_tokens, C) fixed-length rule embedding.
        """
        B, K, N, C = demo_in_tokens.shape

        # ── Step 1: Per-pair cross-attention ─────────────────────
        in_flat = demo_in_tokens.reshape(B * K, N, C)
        out_flat = demo_out_tokens.reshape(B * K, N, C)

        delta = out_flat
        for block in self.pair_blocks:
            delta = block(delta, in_flat)  # (B*K, N, C)

        # ── Step 2: Add pair positional embeddings ───────────────
        delta = delta.reshape(B, K, N, C)
        delta = delta + self.pair_pos_embed[:, :K]  # broadcast over N

        # ── Step 3: Flatten and build attention mask ─────────────
        delta_flat = delta.reshape(B, K * N, C)

        # Mask: True for valid tokens, False for padding pairs
        pair_valid = torch.arange(K, device=num_demos.device).unsqueeze(0) < num_demos.unsqueeze(1)
        token_mask = pair_valid.unsqueeze(2).expand(B, K, N).reshape(B, K * N)

        # ── Step 4: Cross-pair aggregation ───────────────────────
        for block in self.agg_blocks:
            delta_flat = block(delta_flat, mask=token_mask)

        # ── Step 5: Extract fixed-length rule tokens ─────────────
        queries = self.rule_queries.expand(B, -1, -1)  # (B, M, C)
        rule_tokens = queries + self.rule_cross_attn(
            self.rule_cross_norm_q(queries),
            self.rule_cross_norm_kv(delta_flat),
            kv_mask=token_mask,
        )
        rule_tokens = self.rule_norm(rule_tokens)

        return rule_tokens

    def extra_repr(self) -> str:
        n = sum(p.numel() for p in self.parameters()) / 1e6
        return (
            f"pair_layers={len(self.pair_blocks)}, "
            f"agg_layers={len(self.agg_blocks)}, "
            f"rule_tokens={self.num_rule_tokens}, "
            f"params={n:.1f}M"
        )
