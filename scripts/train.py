#!/usr/bin/env python3
"""ARC-IT Training Script.

Usage:
    # Mac (auto-detects, uses small batches):
    python scripts/train.py

    # H100 (auto-detects CUDA, uses full config):
    python scripts/train.py

    # With custom config:
    python scripts/train.py --config configs/default.yaml

    # With W&B logging:
    python scripts/train.py --wandb

    # Push best checkpoint to HF Hub after training:
    python scripts/train.py --wandb --push-to-hf
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

import torch
from arc_it.utils.config import load_config
from arc_it.utils.device import device_info
from arc_it.data.dataset import build_dataloaders
from arc_it.models.arc_it_model import ARCITModel
from arc_it.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train ARC-IT model")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--override", default=None, help="Override config (e.g., configs/mac_dev.yaml)")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--push-to-hf", action="store_true", help="Push best checkpoint to HF Hub after training")
    args = parser.parse_args()

    # ─── Setup ───────────────────────────────────────────────────
    config = load_config(args.config, args.override)
    info = device_info()

    print("=" * 60)
    print("ARC-IT Training")
    print("=" * 60)
    print(f"Device:     {info['device']}")
    if info.get("gpu_name"):
        print(f"GPU:        {info['gpu_name']} ({info['gpu_memory_gb']}GB)")
    print(f"Dtype:      {info['dtype']}")
    print(f"AMP:        {info['amp_enabled']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print()

    # ─── Data ────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds, train_loader, val_ds, val_loader = build_dataloaders(config)
    print(f"Train: {len(train_ds)} samples, {train_ds.num_tasks} tasks")
    if val_ds:
        print(f"Val:   {len(val_ds)} samples")
    print()

    # ─── Model ───────────────────────────────────────────────────
    print("Building model...")
    model = ARCITModel.from_config(config)
    counts = model.param_count()
    print(f"Total params:     {counts['_total']['total'] / 1e6:.1f}M")
    print(f"Trainable params: {counts['_total']['trainable'] / 1e6:.1f}M")
    for name, c in counts.items():
        if name != "_total":
            print(f"  {name:20s}: {c['total']/1e6:>8.1f}M total, {c['trainable']/1e6:>8.1f}M trainable")
    print()

    # ─── Resume ──────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=config["training"].get("checkpoint_dir", "checkpoints"),
        use_wandb=args.wandb,
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # ─── Train ───────────────────────────────────────────────────
    trainer.train()

    # ─── Push to HF Hub ──────────────────────────────────────────
    if args.push_to_hf:
        from arc_it.utils.hf_upload import upload_checkpoint_to_hf

        best_ckpt = trainer.checkpoint_dir / f"best_stage3.pt"
        if not best_ckpt.exists():
            best_ckpt = trainer.checkpoint_dir / "best_stage2.pt"
        if not best_ckpt.exists():
            best_ckpt = trainer.checkpoint_dir / "best_stage1.pt"

        if best_ckpt.exists():
            url = upload_checkpoint_to_hf(str(best_ckpt))
            print(f"Model uploaded to: {url}")
        else:
            print("No best checkpoint found to upload.")


if __name__ == "__main__":
    main()
