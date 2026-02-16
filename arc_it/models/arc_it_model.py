"""ARC-IT: Full integrated model combining all components.

Architecture:
    Input Grid → [Render RGB 224] → FrozenEncoder → Bridge → conditioning
    Output Grid → [ColorEmbed + PatchEmbed + Noise] → SanaBackbone → SpatialDecoder → logits

Training:
    1. Encode input grid with frozen encoder → spatial features
    2. Bridge maps features to Sana conditioning space
    3. Embed + noise the output grid (diffusion forward process)
    4. Sana denoises conditioned on encoder features
    5. Spatial decoder produces 12-class logits
    6. CrossEntropy loss against ground truth output grid

Inference:
    1. Encode input grid (same as training)
    2. Start from pure noise
    3. Iteratively denoise for N steps
    4. Decode to logits → argmax → discrete grid
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_it.data.canvas import NUM_COLORS, IGNORE_INDEX
from arc_it.models.encoder import FrozenEncoder
from arc_it.models.bridge import Bridge
from arc_it.models.sana_backbone import SanaBackbone
from arc_it.models.decoder import SpatialDecoder
from arc_it.models.diffusion import DiffusionScheduler


class OutputEmbedder(nn.Module):
    """Embed the output grid into continuous patch embeddings for diffusion.

    Converts discrete grid → color embeddings → patch embeddings.
    This is the "input" side of the diffusion process.
    """

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

        # Learned color embeddings (like VARC)
        embed_dim = hidden_size // 4
        self.color_embed = nn.Embedding(num_colors, embed_dim)

        # Patch embedding: groups patch_size x patch_size pixels
        self.patch_proj = nn.Conv2d(
            embed_dim, hidden_size,
            kernel_size=patch_size, stride=patch_size,
        )

        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (B, H, W) integer tensor with values 0-11.

        Returns:
            (B, num_patches, hidden_size) patch embeddings.
        """
        B = grid.shape[0]
        # Color embed: (B, H, W) → (B, H, W, embed_dim)
        x = self.color_embed(grid.long())
        # Rearrange for conv: (B, embed_dim, H, W)
        x = x.permute(0, 3, 1, 2)
        # Patch projection: (B, hidden_size, grid_size, grid_size)
        x = self.patch_proj(x)
        # Flatten to sequence: (B, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        return x


class ARCITModel(nn.Module):
    """Full ARC-IT hybrid model.

    Combines frozen encoder + bridge + Sana diffusion transformer + spatial decoder.
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
        # Diffusion config
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        # Output embedder
        output_patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.sana_hidden = sana_hidden
        self.canvas_size = canvas_size
        self.num_colors = num_colors
        self.num_inference_steps = num_inference_steps

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

        # ─── Output Embedder (for diffusion) ─────────────────────
        self.output_embedder = OutputEmbedder(
            num_colors=num_colors,
            canvas_size=canvas_size,
            hidden_size=sana_hidden,
            patch_size=output_patch_size,
        )
        output_num_patches = self.output_embedder.num_patches

        # ─── Sana Backbone ───────────────────────────────────────
        self.sana = SanaBackbone(
            hidden_size=sana_hidden,
            depth=sana_depth,
            self_attn_heads=sana_self_attn_heads,
            self_attn_head_dim=sana_self_attn_head_dim,
            cross_attn_heads=sana_cross_attn_heads,
            mlp_ratio=sana_mlp_ratio,
            num_patches=output_num_patches,
        )

        # ─── Spatial Decoder ─────────────────────────────────────
        self.decoder = SpatialDecoder(
            hidden_size=sana_hidden,
            num_patches=output_num_patches,
            canvas_size=canvas_size,
            num_colors=num_colors,
            hidden_channels=decoder_channels,
        )

        # ─── Diffusion Scheduler ─────────────────────────────────
        self.scheduler = DiffusionScheduler(
            num_train_timesteps=num_train_timesteps,
        )

    def forward(
        self,
        input_rgb_224: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        difficulty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training (with target) or inference (without).

        Args:
            input_rgb_224: (B, 3, 224, 224) normalized RGB for encoder.
            target: (B, 64, 64) ground truth output grid (integers 0-11).
                    If None, runs inference mode.
            difficulty: (B,) difficulty weights for loss. Optional.

        Returns:
            Dict with "loss" (training) or "logits" and "prediction" (inference).
        """
        # ─── Encode input ────────────────────────────────────────
        encoder_features = self.encoder(input_rgb_224)      # (B, 256, encoder_dim)
        conditioning = self.bridge(encoder_features)         # (B, 256, sana_hidden)

        if target is not None:
            return self._forward_train(conditioning, target, difficulty)
        else:
            return self._forward_inference(conditioning)

    def _forward_train(
        self,
        conditioning: torch.Tensor,
        target: torch.Tensor,
        difficulty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward: add noise to output, denoise, compute loss."""
        B = conditioning.shape[0]
        device = conditioning.device

        # Embed the target output grid
        target_embed = self.output_embedder(target)  # (B, N, C)

        # Sample random timesteps and noise
        timesteps = self.scheduler.sample_timesteps(B, device)
        noise = torch.randn_like(target_embed)

        # Add noise (forward diffusion)
        noisy_embed = self.scheduler.add_noise(target_embed, noise, timesteps)

        # Sana denoises
        denoised = self.sana(noisy_embed, conditioning, timesteps)

        # Decode to logits
        logits = self.decoder(denoised)  # (B, 12, 64, 64)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits,
            target.long(),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        )  # (B, 64, 64)

        # Apply difficulty weighting
        if difficulty is not None:
            loss = loss * difficulty[:, None, None]

        loss = loss.mean()

        # Compute pixel accuracy (for logging)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            valid_mask = target != IGNORE_INDEX
            correct = ((pred == target) & valid_mask).sum()
            total = valid_mask.sum()
            pixel_acc = correct.float() / total.float().clamp(min=1)

        return {
            "loss": loss,
            "pixel_accuracy": pixel_acc,
            "logits": logits,
        }

    @torch.no_grad()
    def _forward_inference(
        self,
        conditioning: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference forward: denoise from pure noise."""
        B = conditioning.shape[0]
        N = self.output_embedder.num_patches
        C = self.sana_hidden

        def denoise_fn(x_t, cond, t):
            return self.sana(x_t, cond, t)

        denoised = self.scheduler.inference_loop(
            denoise_fn=denoise_fn,
            shape=(B, N, C),
            conditioning=conditioning,
            num_inference_steps=self.num_inference_steps,
            device=conditioning.device,
        )

        logits = self.decoder(denoised)  # (B, 12, 64, 64)
        prediction = logits.argmax(dim=1)  # (B, 64, 64)

        return {
            "logits": logits,
            "prediction": prediction,
        }

    @classmethod
    def from_config(cls, config: dict) -> "ARCITModel":
        """Create model from a configuration dictionary."""
        model_cfg = config["model"]
        enc = model_cfg["encoder"]
        brg = model_cfg["bridge"]
        sana = model_cfg["sana"]
        dec = model_cfg["decoder"]
        diff = model_cfg["diffusion"]
        data = config["data"]

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
            num_train_timesteps=diff["num_train_timesteps"],
            num_inference_steps=diff["num_inference_steps"],
        )

    def get_trainable_params(self) -> list:
        """Return only trainable parameters (excludes frozen encoder)."""
        params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def param_count(self) -> Dict[str, int]:
        """Count parameters per component."""
        counts = {}
        for component_name in ["encoder", "bridge", "output_embedder", "sana", "decoder"]:
            component = getattr(self, component_name)
            total = sum(p.numel() for p in component.parameters())
            trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
            counts[component_name] = {"total": total, "trainable": trainable}
        counts["scheduler"] = {"total": 0, "trainable": 0}
        counts["_total"] = {
            "total": sum(c["total"] for c in counts.values()),
            "trainable": sum(c["trainable"] for c in counts.values()),
        }
        return counts
