"""ARC-IT: Full integrated model combining all components.

Architecture (direct prediction, no diffusion):
    Input Grid → [Render RGB 224] → FrozenEncoder(DINOv2) → Bridge → conditioning
    Input Grid → [ColorEmbed + PatchEmbed] → SanaBackbone(conditioning) → SpatialDecoder → logits

Training:
    1. Encode input grid with frozen DINOv2 → spatial features
    2. Bridge maps features to Sana conditioning space
    3. Embed the INPUT grid as patch tokens
    4. Sana transforms input tokens conditioned on encoder features
    5. Spatial decoder produces 12-class logits
    6. CrossEntropy loss against ground truth output grid

Inference:
    Same as training but without the loss computation.
    No noise, no denoising -- direct single forward pass.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_it.data.canvas import NUM_COLORS, IGNORE_INDEX
from arc_it.models.encoder import FrozenEncoder
from arc_it.models.bridge import Bridge
from arc_it.models.sana_backbone import SanaBackbone
from arc_it.models.decoder import SpatialDecoder


class InputEmbedder(nn.Module):
    """Embed the input grid into continuous patch embeddings for Sana."""

    def __init__(
        self,
        num_colors: int = 12,
        canvas_size: int = 64,
        hidden_size: int = 1152,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.grid_size = canvas_size // patch_size
        self.num_patches = self.grid_size ** 2

        embed_dim = hidden_size // 4
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.patch_proj = nn.Conv2d(
            embed_dim, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Embed a discrete grid into patch tokens.

        Args:
            grid: (B, H, W) integer grid with values 0-11.
        Returns:
            (B, num_patches, hidden_size) patch embeddings.
        """
        x = self.color_embed(grid.long())
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.patch_proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ARCITModel(nn.Module):
    """Full ARC-IT hybrid model (direct prediction, no diffusion).

    Combines frozen DINOv2 encoder + bridge + Sana conditional transformer + spatial decoder.
    """

    def __init__(
        self,
        # Encoder config
        encoder_name: str = "stub",
        encoder_dim: int = 1024,
        encoder_pretrained: bool = True,
        # Bridge config
        bridge_hidden: int = 2048,
        bridge_dropout: float = 0.1,
        # Sana config
        sana_hidden: int = 1152,
        sana_depth: int = 28,
        sana_self_attn_heads: int = 36,
        sana_self_attn_head_dim: int = 32,
        sana_cross_attn_heads: int = 16,
        sana_mlp_ratio: float = 2.5,
        # Decoder config
        canvas_size: int = 64,
        num_colors: int = 12,
        decoder_channels: tuple = (512, 256),
        # Input embedder
        input_patch_size: int = 4,
        # Performance
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.sana_hidden = sana_hidden
        self.canvas_size = canvas_size
        self.num_colors = num_colors

        # ─── Frozen Encoder ──────────────────────────────────────
        self.encoder = FrozenEncoder(
            encoder_name=encoder_name,
            pretrained=encoder_pretrained,
            embed_dim=encoder_dim,
        )
        actual_encoder_dim = self.encoder.embed_dim
        num_patches = self.encoder.num_patches

        # ─── Bridge ──────────────────────────────────────────────
        self.bridge = Bridge(
            encoder_dim=actual_encoder_dim,
            hidden_dim=bridge_hidden,
            sana_dim=sana_hidden,
            num_patches=num_patches,
            dropout=bridge_dropout,
        )

        # ─── Input Embedder ──────────────────────────────────────
        self.input_embedder = InputEmbedder(
            num_colors=num_colors,
            canvas_size=canvas_size,
            hidden_size=sana_hidden,
            patch_size=input_patch_size,
        )
        input_num_patches = self.input_embedder.num_patches

        # ─── Sana Backbone (direct transformer, no diffusion) ────
        self.sana = SanaBackbone(
            hidden_size=sana_hidden,
            depth=sana_depth,
            self_attn_heads=sana_self_attn_heads,
            self_attn_head_dim=sana_self_attn_head_dim,
            cross_attn_heads=sana_cross_attn_heads,
            mlp_ratio=sana_mlp_ratio,
            num_patches=input_num_patches,
            gradient_checkpointing=gradient_checkpointing,
        )

        # ─── Spatial Decoder ─────────────────────────────────────
        self.decoder = SpatialDecoder(
            hidden_size=sana_hidden,
            num_patches=input_num_patches,
            canvas_size=canvas_size,
            num_colors=num_colors,
            hidden_channels=decoder_channels,
        )

    def forward(
        self,
        input_rgb_224: torch.Tensor,
        input_canvas: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        difficulty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training (with target) or inference (without).

        Args:
            input_rgb_224: (B, 3, 224, 224) RGB rendering for DINOv2.
            input_canvas: (B, 64, 64) discrete input grid on canvas.
            target: (B, 64, 64) discrete target grid (None for inference).
            difficulty: (B,) per-sample difficulty weights.
        """
        # Encode input with frozen DINOv2
        encoder_features = self.encoder(input_rgb_224)
        conditioning = self.bridge(encoder_features)

        # Embed input grid as tokens
        input_tokens = self.input_embedder(input_canvas)

        # Sana transforms input tokens conditioned on encoder features
        output_tokens = self.sana(input_tokens, conditioning)

        # Decode to logits
        logits = self.decoder(output_tokens)

        # Build result
        result = {"logits": logits}

        if target is not None:
            # Compute loss
            valid_mask = (target != IGNORE_INDEX)
            ce = F.cross_entropy(
                logits,
                target.long(),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            )

            if difficulty is not None:
                ce = ce * difficulty[:, None, None]

            valid_pixels = valid_mask.sum().clamp(min=1)
            loss = ce.sum() / valid_pixels.float()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct = ((pred == target) & valid_mask).sum()
                pixel_acc = correct.float() / valid_pixels.float()

            result["loss"] = loss
            result["pixel_accuracy"] = pixel_acc

        # Always include prediction
        result["prediction"] = logits.argmax(dim=1)

        return result

    @classmethod
    def from_config(cls, config: dict) -> "ARCITModel":
        """Create model from a configuration dictionary."""
        model_cfg = config["model"]
        enc = model_cfg["encoder"]
        brg = model_cfg["bridge"]
        sana = model_cfg["sana"]
        dec = model_cfg["decoder"]
        data = config["data"]

        train_cfg = config.get("training", {})

        return cls(
            encoder_name=enc["name"],
            encoder_dim=enc["embed_dim"],
            encoder_pretrained=enc["pretrained"],
            bridge_hidden=brg["hidden_dim"],
            bridge_dropout=brg["dropout"],
            sana_hidden=sana["hidden_size"],
            sana_depth=sana["depth"],
            sana_self_attn_heads=sana["hidden_size"] // sana.get("linear_head_dim", 32),
            sana_self_attn_head_dim=sana.get("linear_head_dim", 32),
            sana_cross_attn_heads=sana["num_heads"],
            sana_mlp_ratio=sana["mlp_ratio"],
            canvas_size=data["canvas_size"],
            num_colors=data["num_colors"],
            decoder_channels=tuple(dec["hidden_channels"]),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        )

    def enable_torch_compile(self, mode: str = "reduce-overhead") -> None:
        """Compile trainable sub-modules with torch.compile for faster CUDA training.

        Only compiles bridge, input_embedder, sana, and decoder (the trainable
        parts). The frozen encoder is left uncompiled since it doesn't benefit
        and compilation overhead isn't worth it for a frozen module.

        Args:
            mode: Compile mode. "reduce-overhead" is best for fixed-shape inputs
                  (our case). "max-autotune" is slower to compile but can be
                  faster at runtime. "default" is the safest fallback.
        """
        self.bridge = torch.compile(self.bridge, mode=mode)
        self.input_embedder = torch.compile(self.input_embedder, mode=mode)
        self.sana = torch.compile(self.sana, mode=mode)
        self.decoder = torch.compile(self.decoder, mode=mode)

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Toggle gradient checkpointing on Sana backbone.

        Trades ~30% more compute for ~40% less VRAM, allowing larger batch
        sizes which often more than compensate for the extra compute.
        """
        self.sana.gradient_checkpointing = enable

    def get_trainable_params(self) -> list:
        """Return only trainable parameters (excludes frozen encoder)."""
        return [p for name, p in self.named_parameters() if p.requires_grad]

    def param_count(self) -> Dict[str, int]:
        """Count parameters per component."""
        counts = {}
        for component_name in ["encoder", "bridge", "input_embedder", "sana", "decoder"]:
            component = getattr(self, component_name)
            total = sum(p.numel() for p in component.parameters())
            trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
            counts[component_name] = {"total": total, "trainable": trainable}
        counts["_total"] = {
            "total": sum(c["total"] for c in counts.values()),
            "trainable": sum(c["trainable"] for c in counts.values()),
        }
        return counts
