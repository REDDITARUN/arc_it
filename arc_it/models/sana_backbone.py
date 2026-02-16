"""Sana Transformer Backbone for ARC-IT.

Reimplements the core Sana-0.6B architecture (linear attention + cross-attention
+ Mix-FFN with adaLN conditioning) without depending on the Sana repository.
This makes the code portable across Mac (dev) and H100 (training).

Architecture per SanaBlock:
    1. AdaLN-modulated self-attention (linear attention via LiteLA)
    2. Cross-attention to encoder conditioning (standard attention)
    3. AdaLN-modulated Mix-FFN (GLU + depthwise conv)

Key dimensions (Sana-0.6B):
    hidden_size=1152, depth=28, num_heads=16, linear_head_dim=32, mlp_ratio=2.5
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Timestep Embedding ─────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion timesteps into vectors.

    Sinusoidal embedding → MLP(hidden → hidden).
    """

    def __init__(self, hidden_size: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_dim = frequency_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar timesteps.
        Returns:
            (B, hidden_size) timestep embeddings.
        """
        emb = self._sinusoidal_embedding(t, self.frequency_dim)
        return self.mlp(emb)

    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ─── Linear Attention (LiteLA) ──────────────────────────────────────

class LinearAttention(nn.Module):
    """Linear attention with ReLU kernel (LiteLA from Sana).

    Complexity: O(N * D^2) instead of O(N^2 * D) for standard attention.
    Critical for fast TTT inference loops.

    Uses the kernel trick: attn(Q,K,V) = phi(Q) @ (phi(K)^T @ V)
    where phi(x) = ReLU(x)
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 36,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.qkv = nn.Linear(hidden_size, inner_dim * 3, bias=False)
        self.out_proj = nn.Linear(inner_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) input tokens.
        Returns:
            (B, N, C) attended tokens.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, N, H, D)

        # ReLU kernel
        q = F.relu(q)
        k = F.relu(k)

        # Linear attention: O(N * D^2)
        # Compute K^T @ V first: (B, H, D, D)
        k_t = k.permute(0, 2, 3, 1)  # (B, H, D, N)
        v_p = v.permute(0, 2, 1, 3)  # (B, H, N, D)
        kv = k_t @ v_p               # (B, H, D, D)

        # Q @ (K^T @ V): (B, H, N, D)
        q_p = q.permute(0, 2, 1, 3)  # (B, H, N, D)
        out = q_p @ kv               # (B, H, N, D)

        # Normalize by sum of keys
        k_sum = k.sum(dim=1, keepdim=True)      # (B, 1, H, D)
        k_sum = k_sum.permute(0, 2, 1, 3)       # (B, H, 1, D)
        denom = (q_p * k_sum).sum(dim=-1, keepdim=True) + 1e-6  # (B, H, N, 1)
        out = out / denom

        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, N, -1)  # (B, N, inner_dim)
        return self.out_proj(out)


# ─── Cross Attention ─────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """Standard multi-head cross-attention.

    Query from image tokens, Key/Value from conditioning (encoder features).
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 16,
    ) -> None:
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
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) query tokens (image/grid tokens).
            conditioning: (B, M, C) key/value tokens (encoder features).
        Returns:
            (B, N, C) cross-attended tokens.
        """
        B, N, C = x.shape
        M = conditioning.shape[1]

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        kv = self.kv_proj(conditioning).reshape(B, M, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().reshape(B, N, C)
        return self.out_proj(out)


# ─── Mix-FFN (GLU + Depthwise Conv) ─────────────────────────────────

class MixFFN(nn.Module):
    """Feed-forward with GLU gating and depthwise convolution.

    Matches Sana's 'glumbconv' FFN type. The 3x3 depthwise conv helps
    the transformer see local 2D patterns in grids.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        mlp_ratio: float = 2.5,
    ) -> None:
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        # GLU gate doubles the projection
        self.fc1 = nn.Linear(hidden_size, inner_dim * 2)
        self.dwconv = nn.Conv2d(
            inner_dim, inner_dim,
            kernel_size=3, padding=1, groups=inner_dim,
        )
        self.fc2 = nn.Linear(inner_dim, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) where N should be a perfect square (e.g., 256=16x16).
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        h = w = int(math.sqrt(N))

        # GLU: split into gate and value
        x_proj = self.fc1(x)
        gate, value = x_proj.chunk(2, dim=-1)
        x = F.silu(gate) * value  # (B, N, inner_dim)

        # Depthwise conv for local 2D patterns
        inner_dim = x.shape[-1]
        x = x.reshape(B, h, w, inner_dim).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous().reshape(B, N, inner_dim)  # (B, N, D)

        return self.fc2(x)


# ─── Sana Block ──────────────────────────────────────────────────────

class SanaBlock(nn.Module):
    """Single Sana transformer block with adaLN conditioning.

    Flow: adaLN-modulated self-attn → cross-attn → adaLN-modulated FFN
    Timestep conditioning controls scale/shift/gate via adaLN.
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        self_attn_heads: int = 36,
        self_attn_head_dim: int = 32,
        cross_attn_heads: int = 16,
        mlp_ratio: float = 2.5,
    ) -> None:
        super().__init__()

        # Self-attention (linear)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = LinearAttention(hidden_size, self_attn_heads, self_attn_head_dim)

        # Cross-attention
        self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, cross_attn_heads)

        # Mix-FFN
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffn = MixFFN(hidden_size, mlp_ratio)

        # adaLN parameters: 6 modulation vectors (shift, scale, gate for attn + ffn)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Learnable base scale/shift table
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) * 0.02)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) hidden states to transform.
            conditioning: (B, M, C) encoder conditioning for cross-attention.
            t_emb: (B, C) timestep embedding.
        Returns:
            (B, N, C) transformed hidden states.
        """
        B = x.shape[0]

        # Compute adaLN modulation from timestep
        ada = self.adaLN_modulation(t_emb)  # (B, 6*C)
        ada = ada.reshape(B, 6, -1)          # (B, 6, C)
        ada = self.scale_shift_table[None] + ada  # (B, 6, C)
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = ada.unbind(dim=1)

        # 1. Self-attention with adaLN
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(x_norm)

        # 2. Cross-attention (no adaLN, simple residual)
        x = x + self.cross_attn(self.norm_cross(x), conditioning)

        # 3. FFN with adaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
        x = x + gate_ff.unsqueeze(1) * self.ffn(x_norm)

        return x


# ─── Full Sana Backbone ─────────────────────────────────────────────

class SanaBackbone(nn.Module):
    """Stack of Sana transformer blocks with timestep and conditioning inputs.

    This is the core denoising network. It takes noisy patch embeddings,
    conditions on encoder features via cross-attention, and produces
    denoised patch embeddings.

    Input:  noisy patches (B, N, C) + conditioning (B, M, C) + timestep (B,)
    Output: denoised patches (B, N, C)
    """

    def __init__(
        self,
        hidden_size: int = 1152,
        depth: int = 28,
        self_attn_heads: int = 36,
        self_attn_head_dim: int = 32,
        cross_attn_heads: int = 16,
        mlp_ratio: float = 2.5,
        num_patches: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_patches = num_patches

        self.t_embedder = TimestepEmbedder(hidden_size)

        # Learnable positional embeddings for the patch sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            SanaBlock(
                hidden_size=hidden_size,
                self_attn_heads=self_attn_heads,
                self_attn_head_dim=self_attn_head_dim,
                cross_attn_heads=cross_attn_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following Sana conventions."""
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C) noisy patch embeddings.
            conditioning: (B, M, C) encoder conditioning.
            timestep: (B,) diffusion timestep (0 to T-1).
        Returns:
            (B, N, C) denoised patch embeddings.
        """
        # Add positional embeddings
        x = x + self.pos_embed

        # Embed timestep
        t_emb = self.t_embedder(timestep)  # (B, C)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, conditioning, t_emb)

        x = self.final_norm(x)
        return x

    def extra_repr(self) -> str:
        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        return f"depth={self.depth}, hidden={self.hidden_size}, params={n_params:.1f}M"
