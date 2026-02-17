"""Hugging Face Hub integration for ARC-IT.

Upload trained model checkpoints to HF Hub as model repositories
with config, model card, and weights.

Usage:
    from arc_it.utils.hf_upload import upload_checkpoint_to_hf
    upload_checkpoint_to_hf("checkpoints/best_stage2.pt", "REDDITARUN/arc-it")
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


MODEL_CARD_TEMPLATE = """---
tags:
  - arc-agi
  - abstract-reasoning
  - jepa
  - sana
  - vision-transformer
license: mit
---

# ARC-IT: JEPA + Sana Hybrid for ARC-AGI

A hybrid neural architecture that solves abstract reasoning tasks (ARC-AGI) by combining:

- **JEPA/DINOv2 Encoder (Frozen)** -- Pretrained spatial feature extractor
- **Bridge Module (Trainable)** -- Maps encoder features to transformer space
- **Sana Transformer (Trainable)** -- Linear-attention conditional transformer
- **Spatial Decoder (Trainable)** -- Converts transformer output to discrete ARC grids

## Architecture

```
Input Grid -> Pad to 64x64 Canvas -> Render RGB -> Upsample to 224x224
  -> DINOv2-L/14 Encoder (FROZEN) -> Bridge -> Sana Transformer (denoising)
  -> Spatial Decoder -> 12-class logits -> argmax -> Predicted Grid
```

## Training

- **3-stage training**: Bridge Alignment -> Full Training -> Hard Focus
- **Test-Time Training (TTT)**: Per-task fine-tuning on demonstration examples

## Model Details

{model_details}

## Usage

```python
import torch
from arc_it.models.arc_it_model import ARCITModel

model = ARCITModel.from_config(config)
ckpt = torch.load("model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
```

## Links

- **Repository**: [github.com/REDDITARUN/arc_it](https://github.com/REDDITARUN/arc_it)
- **ARC-AGI**: [arcprize.org](https://arcprize.org)
"""


def upload_checkpoint_to_hf(
    checkpoint_path: str,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    commit_message: str = "Upload ARC-IT model checkpoint",
    private: bool = False,
) -> str:
    """Upload a trained ARC-IT checkpoint to Hugging Face Hub.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        repo_id: HF repo ID (e.g., "REDDITARUN/arc-it").
                 Falls back to HF_REPO_ID env var.
        token: HF API token with write access.
               Falls back to HF_TOKEN env var.
        commit_message: Commit message for the upload.
        private: Whether to create a private repo.

    Returns:
        URL of the uploaded model on HF Hub.

    Raises:
        ValueError: If repo_id or token are not provided.
        ImportError: If huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError(
            "huggingface-hub is required for uploading. "
            "Install with: pip install huggingface-hub"
        )

    repo_id = repo_id or os.environ.get("HF_REPO_ID")
    token = token or os.environ.get("HF_TOKEN")

    if not repo_id:
        raise ValueError(
            "repo_id must be provided or set HF_REPO_ID in .env"
        )
    if not token:
        raise ValueError(
            "token must be provided or set HF_TOKEN in .env"
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Uploading {checkpoint_path} to hf.co/{repo_id}...")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
    except Exception as e:
        print(f"  Note: {e}")

    # Load checkpoint to extract config for model card
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    global_step = ckpt.get("global_step", "unknown")
    best_val_acc = ckpt.get("best_val_acc", "unknown")

    model_details = _format_model_details(config, global_step, best_val_acc)

    # Create a temporary directory with all upload artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy checkpoint
        shutil.copy2(checkpoint_path, tmpdir / "model.pt")

        # Write config
        if config:
            import json
            with open(tmpdir / "config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)

        # Write model card
        readme_content = MODEL_CARD_TEMPLATE.format(model_details=model_details)
        with open(tmpdir / "README.md", "w") as f:
            f.write(readme_content)

        # Upload entire folder
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            commit_message=commit_message,
        )

    url = f"https://huggingface.co/{repo_id}"
    print(f"  Uploaded successfully to {url}")
    return url


def _format_model_details(
    config: Dict[str, Any],
    global_step: Any,
    best_val_acc: Any,
) -> str:
    """Format model details for the HF model card."""
    lines = []
    lines.append(f"- **Training step**: {global_step}")
    lines.append(f"- **Best validation accuracy**: {best_val_acc}")

    model_cfg = config.get("model", {})
    if model_cfg:
        enc = model_cfg.get("encoder", {})
        sana = model_cfg.get("sana", {})
        lines.append(f"- **Encoder**: {enc.get('name', 'unknown')} (dim={enc.get('embed_dim', '?')})")
        lines.append(f"- **Sana backbone**: hidden={sana.get('hidden_size', '?')}, depth={sana.get('depth', '?')}")
        lines.append(f"- **Canvas size**: {config.get('data', {}).get('canvas_size', 64)}")

    return "\n".join(lines)
