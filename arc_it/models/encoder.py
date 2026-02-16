"""Frozen I-JEPA/DINOv2 encoder wrapper for ARC grid feature extraction.

Extracts spatial patch tokens (NOT CLS token) from a pretrained vision
transformer. All parameters are frozen to preserve pretrained representations.

Supported encoders:
    - ijepa_vith14_1k:  I-JEPA ViT-H/14 (ImageNet-1K), embed_dim=1280, 256 patches
    - ijepa_vith14_22k: I-JEPA ViT-H/14 (ImageNet-22K), embed_dim=1280, 256 patches
    - dinov2_vits14:    DINOv2-S, embed_dim=384, 22M params (fast dev)
    - dinov2_vitb14:    DINOv2-B, embed_dim=768, 86M params
    - dinov2_vitl14:    DINOv2-L, embed_dim=1024, 304M params
    - dinov2_vitg14:    DINOv2-G, embed_dim=1536, 1.1B params
    - stub:             Random weights matching I-JEPA interface (unit tests)
"""

import torch
import torch.nn as nn


class FrozenEncoder(nn.Module):
    """Frozen vision encoder that extracts patch-level spatial features.

    Input:  (B, 3, 224, 224) RGB images (ImageNet-normalized)
    Output: (B, num_patches, embed_dim) patch token embeddings

    For ViT with patch_size=14 and input 224x224:
        num_patches = (224 / 14)^2 = 256
    """

    def __init__(
        self,
        encoder_name: str = "ijepa_vith14_1k",
        pretrained: bool = True,
        embed_dim: int = 1280,
        num_patches: int = 256,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        if encoder_name == "stub":
            self.encoder = _StubEncoder(embed_dim, num_patches)
        elif encoder_name.startswith("ijepa_"):
            self.encoder = _load_ijepa(encoder_name, pretrained)
            self.embed_dim = 1280  # ViT-H
            self.num_patches = (224 // 14) ** 2  # 256
        elif encoder_name.startswith("dinov2_"):
            self.encoder = _load_dinov2(encoder_name, pretrained)
            self.embed_dim = self.encoder.embed_dim
            self.num_patches = (224 // self.encoder.patch_size) ** 2
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

        self._freeze()

    def _freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def train(self, mode: bool = True) -> "FrozenEncoder":
        """Override train to keep encoder always in eval mode."""
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from input images.

        Args:
            x: (B, 3, 224, 224) normalized RGB tensor.

        Returns:
            (B, num_patches, embed_dim) patch embeddings.
        """
        if self.encoder_name == "stub":
            return self.encoder(x)

        if self.encoder_name.startswith("ijepa_"):
            return self._forward_ijepa(x)

        return self._forward_dinov2(x)

    def _forward_ijepa(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through I-JEPA (HuggingFace transformers)."""
        outputs = self.encoder(pixel_values=x)
        tokens = outputs.last_hidden_state  # (B, seq_len, embed_dim)

        # Remove CLS token if present (seq_len == num_patches + 1)
        if tokens.shape[1] == self.num_patches + 1:
            tokens = tokens[:, 1:]

        return tokens

    def _forward_dinov2(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through DINOv2 (torch.hub)."""
        features = self.encoder.forward_features(x)

        if isinstance(features, dict):
            patch_tokens = features.get(
                "x_norm_patchtokens",
                features.get("x_prenorm", None),
            )
            if patch_tokens is None:
                raise RuntimeError(
                    f"Could not find patch tokens in DINOv2 output. "
                    f"Available keys: {list(features.keys())}"
                )
        else:
            patch_tokens = features[:, 1:, :]

        return patch_tokens

    def extra_repr(self) -> str:
        return (
            f"encoder={self.encoder_name}, "
            f"embed_dim={self.embed_dim}, "
            f"num_patches={self.num_patches}, "
            f"frozen=True"
        )


# ─── Internal helpers ───────────────────────────────────────────────

def _load_ijepa(model_name: str, pretrained: bool) -> nn.Module:
    """Load I-JEPA from HuggingFace transformers."""
    from transformers import AutoModel

    hf_model_id = f"facebook/{model_name}"
    print(f"Loading I-JEPA encoder: {hf_model_id}")
    model = AutoModel.from_pretrained(hf_model_id)
    return model


def _load_dinov2(model_name: str, pretrained: bool) -> nn.Module:
    """Load DINOv2 from torch.hub."""
    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        pretrained=pretrained,
    )
    return model


class _StubEncoder(nn.Module):
    """Lightweight stub matching the I-JEPA interface for testing."""

    def __init__(self, embed_dim: int = 1280, num_patches: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.proj = nn.Linear(3 * 14 * 14, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.unfold(2, 14, 14).unfold(3, 14, 14)  # (B, 3, 16, 16, 14, 14)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, 16, 16, 3, 14, 14)
        x = x.view(B, self.num_patches, -1)  # (B, 256, 588)
        return self.proj(x)  # (B, 256, embed_dim)
