"""Load pretrained Sana-0.6B weights into our SanaBackbone.

Downloads the official Sana-0.6B checkpoint from HuggingFace and maps
the weight keys to our reimplementation. The mapping is derived from
NVlabs/Sana/tools/convert_scripts/convert_sana_to_diffusers.py.

Transferable weights (self-attention, cross-attention, FFN, scale_shift_table)
cover ~95% of parameters. Non-transferable weights (per-block adaLN, norms)
are initialized randomly and learned during training.
"""

from typing import Dict

import torch
import torch.nn as nn


SANA_600M_REPO = "Efficient-Large-Model/Sana_600M_1024px"
SANA_600M_CKPT = "checkpoints/Sana_600M_1024px_MultiLing.pth"
SANA_DEPTH = 28


def load_sana_pretrained(backbone: nn.Module) -> Dict[str, int]:
    """Load pretrained Sana-0.6B weights into our SanaBackbone.

    Downloads the official checkpoint from HuggingFace, maps key names,
    and loads matching weights. Non-matching keys are left at their
    random initialization.

    Args:
        backbone: Our SanaBackbone instance.

    Returns:
        Dict with loading statistics: matched, skipped, total.
    """
    from huggingface_hub import hf_hub_download

    print(f"Downloading Sana-0.6B checkpoint from {SANA_600M_REPO}...")
    ckpt_path = hf_hub_download(
        repo_id=SANA_600M_REPO,
        filename=SANA_600M_CKPT,
    )

    print(f"Loading checkpoint: {ckpt_path}")
    all_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    src_sd = all_state_dict.get("state_dict", all_state_dict)

    our_sd = backbone.state_dict()
    matched = 0
    skipped = 0
    mismatched_shape = 0

    for depth in range(min(SANA_DEPTH, backbone.depth)):
        # Self-attention: qkv (combined) and output projection
        _transfer(src_sd, our_sd, f"blocks.{depth}.attn.qkv.weight",
                  f"blocks.{depth}.self_attn.qkv.weight", matched_count=None)
        # Output projection
        _transfer(src_sd, our_sd, f"blocks.{depth}.attn.proj.weight",
                  f"blocks.{depth}.self_attn.out_proj.weight")
        _transfer(src_sd, our_sd, f"blocks.{depth}.attn.proj.bias",
                  f"blocks.{depth}.self_attn.out_proj.bias")

        # Cross-attention: Q, KV, output
        _transfer(src_sd, our_sd, f"blocks.{depth}.cross_attn.q_linear.weight",
                  f"blocks.{depth}.cross_attn.q_proj.weight")
        _transfer(src_sd, our_sd, f"blocks.{depth}.cross_attn.q_linear.bias",
                  f"blocks.{depth}.cross_attn.q_proj.bias")
        _transfer(src_sd, our_sd, f"blocks.{depth}.cross_attn.kv_linear.weight",
                  f"blocks.{depth}.cross_attn.kv_proj.weight")
        _transfer(src_sd, our_sd, f"blocks.{depth}.cross_attn.kv_linear.bias",
                  f"blocks.{depth}.cross_attn.kv_proj.bias")
        _transfer(src_sd, our_sd, f"blocks.{depth}.cross_attn.proj.weight",
                  f"blocks.{depth}.cross_attn.out_proj.weight")
        _transfer(src_sd, our_sd, f"blocks.{depth}.cross_attn.proj.bias",
                  f"blocks.{depth}.cross_attn.out_proj.bias")

        # FFN: inverted_conv → fc1 (Conv2d k=1 → Linear, needs squeeze)
        _transfer_conv_to_linear(src_sd, our_sd,
                                 f"blocks.{depth}.mlp.inverted_conv.conv.weight",
                                 f"blocks.{depth}.ffn.fc1.weight")
        _transfer(src_sd, our_sd, f"blocks.{depth}.mlp.inverted_conv.conv.bias",
                  f"blocks.{depth}.ffn.fc1.bias")

        # FFN: depth_conv → dwconv (Conv2d k=3, direct match)
        _transfer(src_sd, our_sd, f"blocks.{depth}.mlp.depth_conv.conv.weight",
                  f"blocks.{depth}.ffn.dwconv.weight")
        _transfer(src_sd, our_sd, f"blocks.{depth}.mlp.depth_conv.conv.bias",
                  f"blocks.{depth}.ffn.dwconv.bias")

        # FFN: point_conv → fc2 (Conv2d k=1 → Linear, needs squeeze)
        _transfer_conv_to_linear(src_sd, our_sd,
                                 f"blocks.{depth}.mlp.point_conv.conv.weight",
                                 f"blocks.{depth}.ffn.fc2.weight")

        # Scale-shift table (direct match)
        _transfer(src_sd, our_sd, f"blocks.{depth}.scale_shift_table",
                  f"blocks.{depth}.scale_shift_table")

    # Timestep embedder (direct match)
    _transfer(src_sd, our_sd, "t_embedder.mlp.0.weight", "t_embedder.mlp.0.weight")
    _transfer(src_sd, our_sd, "t_embedder.mlp.0.bias", "t_embedder.mlp.0.bias")
    _transfer(src_sd, our_sd, "t_embedder.mlp.2.weight", "t_embedder.mlp.2.weight")
    _transfer(src_sd, our_sd, "t_embedder.mlp.2.bias", "t_embedder.mlp.2.bias")

    # Count statistics
    for key in our_sd:
        if key in _transferred_keys:
            matched += 1
        else:
            skipped += 1

    backbone.load_state_dict(our_sd)

    stats = {
        "matched": matched,
        "skipped": skipped,
        "total": len(our_sd),
        "mismatched_shape": mismatched_shape,
    }
    print(f"Sana pretrained: loaded {matched}/{len(our_sd)} weights "
          f"({skipped} randomly initialized)")

    # Clean up
    _transferred_keys.clear()
    del all_state_dict, src_sd
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return stats


# ─── Internal transfer helpers ──────────────────────────────────────

_transferred_keys: set = set()


def _transfer(
    src: Dict[str, torch.Tensor],
    dst: Dict[str, torch.Tensor],
    src_key: str,
    dst_key: str,
    matched_count=None,
) -> bool:
    """Transfer a weight from source to destination if shapes match."""
    if src_key not in src or dst_key not in dst:
        return False
    if src[src_key].shape != dst[dst_key].shape:
        return False
    dst[dst_key] = src[src_key].clone()
    _transferred_keys.add(dst_key)
    return True


def _transfer_conv_to_linear(
    src: Dict[str, torch.Tensor],
    dst: Dict[str, torch.Tensor],
    src_key: str,
    dst_key: str,
) -> bool:
    """Transfer Conv2d(k=1) weight to Linear weight (squeeze spatial dims)."""
    if src_key not in src or dst_key not in dst:
        return False
    conv_weight = src[src_key]
    if conv_weight.dim() == 4:
        linear_weight = conv_weight.squeeze(-1).squeeze(-1)
    else:
        linear_weight = conv_weight
    if linear_weight.shape != dst[dst_key].shape:
        return False
    dst[dst_key] = linear_weight.clone()
    _transferred_keys.add(dst_key)
    return True
