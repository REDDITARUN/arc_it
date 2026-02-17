#!/usr/bin/env python3
"""ARC-IT Training Script (Rule-Conditioned Transformer).

Usage:
    # Mac (auto-detects, uses small batches):
    python scripts/train.py

    # H100/A100 (auto-detects CUDA, uses full config):
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import torch
from arc_it.utils.config import load_config
from arc_it.utils.device import device_info
from arc_it.models.arc_it_model import ARCITModel
from arc_it.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train ARC-IT model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--push-to-hf", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    info = device_info()

    print("=" * 60)
    print("ARC-IT Training (Rule-Conditioned Transformer)")
    print("=" * 60)
    print(f"Device:     {info['device']}")
    if info.get("gpu_name"):
        print(f"GPU:        {info['gpu_name']} ({info['gpu_memory_gb']}GB)")
    print(f"Dtype:      {info['dtype']}")
    print(f"AMP:        {info['amp_enabled']}")
    print(f"Batch size: {config['training']['batch_size']}")
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

    # ─── Training stages ──────────────────────────────────────────
    train_cfg = config.get("training", {})
    s1 = train_cfg.get("stage1", {})
    s2 = train_cfg.get("stage2", {})
    s3 = train_cfg.get("stage3", {})
    print("Training plan:")
    print(f"  Stage 1 ({s1.get('name', 'pretrain')}): "
          f"{s1.get('epochs', 30)} epochs, LR={s1.get('lr', 3e-4)}, "
          f"data={s1.get('data_sources', ['re_arc', 'agi1'])}")
    print(f"  Stage 2 ({s2.get('name', 'finetune')}): "
          f"{s2.get('epochs', 20)} epochs, LR={s2.get('lr', 1e-4)}, "
          f"data={s2.get('data_sources', ['agi1', 'agi2'])}")
    print(f"  Stage 3 ({s3.get('name', 'hard_focus')}): "
          f"{s3.get('epochs', 10)} epochs, LR={s3.get('lr', 3e-5)}, "
          f"data={s3.get('data_sources', ['agi1', 'agi2'])}, "
          f"agi2_weight={s3.get('agi2_oversample', 2.0)}x")
    print()

    # ─── Trainer ─────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        config=config,
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
        use_wandb=args.wandb,
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # ─── Train ───────────────────────────────────────────────────
    trainer.train()

    # ─── Push to HF Hub ──────────────────────────────────────────
    if args.push_to_hf:
        from arc_it.utils.hf_upload import upload_checkpoint_to_hf

        best_ckpt = trainer.checkpoint_dir / "best_stage3.pt"
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
