"""ARC-IT: Rule-Conditioned Transformer for ARC-AGI.

A novel architecture that explicitly extracts transformation rules from
demonstration pairs and applies them to new inputs. This mirrors how
humans solve ARC: look at what changed across examples, then apply
that pattern to the test input.

Architecture:
    Demo pairs:    (in₁,out₁), (in₂,out₂), ...
                        ↓
                  ┌────────────────┐
                  │ Grid Tokenizer │  ← shared tokenizer for all grids
                  └───────┬────────┘
                          ↓
                  ┌────────────────┐
                  │  Rule Encoder  │  ← cross-attend output↔input per pair
                  │   (4 layers)   │    then aggregate across pairs
                  └───────┬────────┘
                          │ rule_tokens (what's the transformation?)
                          ↓
    Test input → Grid Tokenizer → ┌────────────────┐
                                  │  Rule Applier   │  ← cross-attend to rule_tokens
                                  │   (4 layers)    │
                                  └───────┬─────────┘
                                          ↓
                                   Spatial Decoder → output grid logits

Key differences from prior ARC approaches:
    - VARC/TRM: encode only the test input, learn rules implicitly
    - LLMs: use text-based in-context learning
    - This: explicitly extract transformation rules as a first-class
      operation via paired cross-attention on demo pairs

Total: ~20M trainable parameters (no frozen encoder needed).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_it.data.canvas import NUM_COLORS, IGNORE_INDEX
from arc_it.models.grid_tokenizer import GridTokenizer
from arc_it.models.rule_encoder import RuleEncoder
from arc_it.models.rule_applier import RuleApplier
from arc_it.models.decoder import SpatialDecoder


class ARCITModel(nn.Module):
    """Rule-Conditioned Transformer for ARC-AGI.

    All ~20M parameters are trainable. No frozen encoder dependency.
    """

    def __init__(
        self,
        # Grid tokenizer
        num_colors: int = 12,
        canvas_size: int = 64,
        patch_size: int = 4,
        hidden_size: int = 384,
        # Rule encoder
        rule_encoder_pair_layers: int = 2,
        rule_encoder_agg_layers: int = 2,
        rule_encoder_heads: int = 8,
        num_rule_tokens: int = 64,
        max_demos: int = 5,
        # Rule applier
        rule_applier_layers: int = 4,
        rule_applier_heads: int = 8,
        # Decoder
        decoder_channels: tuple = (192, 96),
        # General
        mlp_ratio: float = 2.5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.canvas_size = canvas_size
        self.num_colors = num_colors
        self.max_demos = max_demos

        num_patches = (canvas_size // patch_size) ** 2

        # ─── Shared Grid Tokenizer ────────────────────────────────
        # Used for all grids: demo inputs, demo outputs, test input
        self.tokenizer = GridTokenizer(
            num_colors=num_colors,
            canvas_size=canvas_size,
            hidden_size=hidden_size,
            patch_size=patch_size,
        )

        # ─── Rule Encoder ─────────────────────────────────────────
        # Extracts transformation rule from demo pairs
        self.rule_encoder = RuleEncoder(
            hidden_size=hidden_size,
            num_pair_layers=rule_encoder_pair_layers,
            num_agg_layers=rule_encoder_agg_layers,
            num_heads=rule_encoder_heads,
            num_rule_tokens=num_rule_tokens,
            max_demos=max_demos,
            num_patches=num_patches,
            mlp_ratio=mlp_ratio,
        )

        # ─── Rule Applier ─────────────────────────────────────────
        # Applies extracted rule to test input
        self.rule_applier = RuleApplier(
            hidden_size=hidden_size,
            num_layers=rule_applier_layers,
            num_heads=rule_applier_heads,
            num_patches=num_patches,
            mlp_ratio=mlp_ratio,
        )

        # ─── Spatial Decoder ──────────────────────────────────────
        # Converts output tokens to 64x64 grid logits (12 classes)
        self.decoder = SpatialDecoder(
            hidden_size=hidden_size,
            num_patches=num_patches,
            canvas_size=canvas_size,
            num_colors=num_colors,
            hidden_channels=decoder_channels,
        )

    def forward(
        self,
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        query_input: torch.Tensor,
        num_demos: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        difficulty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training or inference.

        Args:
            demo_inputs:  (B, K, 64, 64) padded demo input canvases.
            demo_outputs: (B, K, 64, 64) padded demo output canvases.
            query_input:  (B, 64, 64) test input canvas.
            num_demos:    (B,) number of valid demos per sample.
            target:       (B, 64, 64) target output canvas (None for inference).
            difficulty:   (B,) per-sample difficulty weights (optional).

        Returns:
            Dict with keys: logits, prediction, and optionally loss + pixel_accuracy.
        """
        B, K = demo_inputs.shape[:2]
        CS = self.canvas_size

        # ── Tokenize all demo grids ──────────────────────────────
        demo_in_flat = demo_inputs.reshape(B * K, CS, CS)
        demo_out_flat = demo_outputs.reshape(B * K, CS, CS)

        demo_in_tokens = self.tokenizer(demo_in_flat)       # (B*K, N, C)
        demo_out_tokens = self.tokenizer(demo_out_flat)      # (B*K, N, C)

        N = demo_in_tokens.shape[1]
        C = self.hidden_size
        demo_in_tokens = demo_in_tokens.reshape(B, K, N, C)
        demo_out_tokens = demo_out_tokens.reshape(B, K, N, C)

        # ── Extract rule from demo pairs ─────────────────────────
        rule_tokens = self.rule_encoder(
            demo_in_tokens, demo_out_tokens, num_demos
        )  # (B, M, C)

        # ── Tokenize test input ──────────────────────────────────
        test_tokens = self.tokenizer(query_input)  # (B, N, C)

        # ── Apply rule to test input ─────────────────────────────
        output_tokens = self.rule_applier(test_tokens, rule_tokens)  # (B, N, C)

        # ── Decode to grid logits ────────────────────────────────
        logits = self.decoder(output_tokens)  # (B, 12, 64, 64)

        # ── Build result ─────────────────────────────────────────
        result = {"logits": logits}

        if target is not None:
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

        result["prediction"] = logits.argmax(dim=1)

        return result

    @classmethod
    def from_config(cls, config: dict) -> "ARCITModel":
        """Create model from a configuration dictionary."""
        model_cfg = config["model"]
        data = config["data"]

        tok = model_cfg.get("tokenizer", {})
        re = model_cfg.get("rule_encoder", {})
        ra = model_cfg.get("rule_applier", {})
        dec = model_cfg.get("decoder", {})

        return cls(
            num_colors=data.get("num_colors", 12),
            canvas_size=data.get("canvas_size", 64),
            patch_size=tok.get("patch_size", 4),
            hidden_size=model_cfg.get("hidden_size", 384),
            rule_encoder_pair_layers=re.get("pair_layers", 2),
            rule_encoder_agg_layers=re.get("agg_layers", 2),
            rule_encoder_heads=re.get("num_heads", 8),
            num_rule_tokens=re.get("num_rule_tokens", 64),
            max_demos=data.get("max_demos", 5),
            rule_applier_layers=ra.get("num_layers", 4),
            rule_applier_heads=ra.get("num_heads", 8),
            decoder_channels=tuple(dec.get("hidden_channels", [192, 96])),
            mlp_ratio=model_cfg.get("mlp_ratio", 2.5),
        )

    def get_trainable_params(self) -> list:
        """Return all trainable parameters (all params are trainable)."""
        return [p for p in self.parameters() if p.requires_grad]

    def param_count(self) -> Dict[str, int]:
        """Count parameters per component."""
        counts = {}
        for name in ["tokenizer", "rule_encoder", "rule_applier", "decoder"]:
            comp = getattr(self, name)
            total = sum(p.numel() for p in comp.parameters())
            trainable = sum(
                p.numel() for p in comp.parameters() if p.requires_grad
            )
            counts[name] = {"total": total, "trainable": trainable}
        counts["_total"] = {
            "total": sum(c["total"] for c in counts.values()),
            "trainable": sum(c["trainable"] for c in counts.values()),
        }
        return counts
