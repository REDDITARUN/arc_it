"""Three-stage Trainer for ARC-IT Rule-Conditioned Transformer.

Stage 1 (Pretrain):    RE-ARC + AGI-1 — bulk pretraining on synthetic data.
Stage 2 (Finetune):    AGI-1 + AGI-2  — fine-tune on real ARC tasks.
Stage 3 (Hard Focus):  AGI-1 + AGI-2  — lower LR, oversample AGI-2.

Each stage builds its own dataloaders based on config data_sources.
Validation always runs on AGI-1 + AGI-2 test split.

Features:
    - Per-stage data source selection
    - tqdm progress bars for training and validation
    - Automatic mixed precision (AMP) on CUDA
    - Gradient clipping
    - Checkpoint saving
    - Weights & Biases logging (optional)
    - Device-adaptive batch sizes and dtypes
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_it.models.arc_it_model import ARCITModel
from arc_it.training.loss import compute_loss
from arc_it.data.dataset import build_dataloaders, build_eval_dataloader
from arc_it.utils.device import get_device, get_dtype, get_amp_enabled


class Trainer:
    """Manages the multi-stage training process."""

    def __init__(
        self,
        model: ARCITModel,
        config: Dict[str, Any],
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
    ) -> None:
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        self.amp_enabled = get_amp_enabled(self.device)

        self.model = model.to(self.device)
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.wandb_run = None

        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)

        self.global_step = 0
        self.best_val_acc = 0.0

        train_cfg = self.config.get("training", {})
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.log_every = train_cfg.get("log_every_n_steps", 100)
        self.save_every = train_cfg.get("save_every_n_epochs", 5)

        # Build val loader once (always on real ARC)
        self.val_loader = None
        self._build_val_loader()

    # ─── Public API ──────────────────────────────────────────────

    def train(self) -> None:
        """Run the full 3-stage training pipeline."""
        if self.use_wandb:
            self._init_wandb()

        train_cfg = self.config.get("training", {})

        # Stage 1: Pretrain on RE-ARC + AGI-1
        s1 = train_cfg.get("stage1", {})
        s1_sources = s1.get("data_sources", ["re_arc", "agi1"])
        print(f"\n{'='*60}")
        print(f"STAGE 1: {s1.get('name', 'Pretrain')}")
        print(f"{'='*60}")
        print(f"  Data sources: {s1_sources}")
        self._print_trainable_info()
        train_loader = self._build_train_loader(s1_sources)
        optimizer, scheduler = self._build_optimizer(
            lr=s1.get("lr", 3e-4),
            epochs=s1.get("epochs", 30),
            steps_per_epoch=len(train_loader),
        )
        self._train_epochs(
            train_loader, optimizer, scheduler,
            epochs=s1.get("epochs", 30),
            stage_name="stage1",
        )

        # Stage 2: Fine-tune on AGI-1 + AGI-2
        s2 = train_cfg.get("stage2", {})
        s2_sources = s2.get("data_sources", ["agi1", "agi2"])
        print(f"\n{'='*60}")
        print(f"STAGE 2: {s2.get('name', 'Finetune')}")
        print(f"{'='*60}")
        print(f"  Data sources: {s2_sources}")
        self._print_trainable_info()
        train_loader = self._build_train_loader(s2_sources)
        optimizer, scheduler = self._build_optimizer(
            lr=s2.get("lr", 1e-4),
            epochs=s2.get("epochs", 20),
            steps_per_epoch=len(train_loader),
        )
        self._train_epochs(
            train_loader, optimizer, scheduler,
            epochs=s2.get("epochs", 20),
            stage_name="stage2",
        )

        # Stage 3: Hard Focus (AGI-2 oversampling)
        s3 = train_cfg.get("stage3", {})
        s3_sources = s3.get("data_sources", ["agi1", "agi2"])
        agi2_weight = s3.get("agi2_oversample", 2.0)
        print(f"\n{'='*60}")
        print(f"STAGE 3: {s3.get('name', 'Hard Focus')}")
        print(f"{'='*60}")
        print(f"  Data sources: {s3_sources}")
        print(f"  AGI-2 difficulty multiplier: {agi2_weight}x")
        self._print_trainable_info()
        train_loader = self._build_train_loader(s3_sources)
        optimizer, scheduler = self._build_optimizer(
            lr=s3.get("lr", 3e-5),
            epochs=s3.get("epochs", 10),
            steps_per_epoch=len(train_loader),
        )
        self._train_epochs(
            train_loader, optimizer, scheduler,
            epochs=s3.get("epochs", 10),
            stage_name="stage3",
            difficulty_multiplier=agi2_weight,
        )

        print("\nTraining complete!")
        if self.wandb_run:
            self.wandb_run.finish()

    # ─── Dataloader builders ─────────────────────────────────────

    def _build_train_loader(self, data_sources: List[str]) -> DataLoader:
        """Build a training DataLoader for the given data sources."""
        train_ds, train_loader, _, _ = build_dataloaders(
            self.config, data_sources=data_sources
        )
        print(f"  Train: {len(train_ds)} samples, {train_ds.num_tasks} tasks")
        return train_loader

    def _build_val_loader(self) -> None:
        """Build validation DataLoader (always on real ARC data)."""
        try:
            val_ds, val_loader = build_eval_dataloader(self.config)
            self.val_loader = val_loader
            if val_ds:
                print(f"  Val: {len(val_ds)} samples")
        except (RuntimeError, FileNotFoundError):
            print("  Warning: Could not build val loader, skipping validation.")
            self.val_loader = None

    # ─── Info ────────────────────────────────────────────────────

    def _print_trainable_info(self) -> None:
        """Print trainable parameter counts."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M params")

    # ─── Optimizer ───────────────────────────────────────────────

    def _build_optimizer(
        self, lr: float, epochs: int, steps_per_epoch: int
    ) -> tuple:
        """Build AdamW optimizer and cosine scheduler."""
        opt_cfg = self.config.get("training", {}).get("optimizer", {})
        sched_cfg = self.config.get("training", {}).get("scheduler", {})

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        total_steps = epochs * steps_per_epoch
        warmup_steps = int(total_steps * sched_cfg.get("warmup_ratio", 0.1))

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_steps / max(total_steps, 1),
            anneal_strategy="cos",
        )

        return optimizer, scheduler

    # ─── Training Loop ───────────────────────────────────────────

    def _train_epochs(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epochs: int,
        stage_name: str,
        difficulty_multiplier: float = 1.0,
    ) -> None:
        """Run training for a given number of epochs."""
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_steps = 0

            pbar = tqdm(
                train_loader,
                desc=f"  {stage_name} epoch {epoch+1}/{epochs}",
                leave=True,
                dynamic_ncols=True,
            )

            for batch in pbar:
                metrics = self._train_step(
                    batch, optimizer, scheduler, difficulty_multiplier
                )
                epoch_loss += metrics["loss"]
                epoch_acc += metrics["pixel_accuracy"]
                epoch_steps += 1
                self.global_step += 1

                pbar.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    acc=f"{metrics['pixel_accuracy']:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

                if self.global_step % self.log_every == 0:
                    self._log({
                        f"{stage_name}/loss": metrics["loss"],
                        f"{stage_name}/pixel_accuracy": metrics["pixel_accuracy"],
                        f"{stage_name}/lr": optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    })

            pbar.close()

            avg_loss = epoch_loss / max(epoch_steps, 1)
            avg_acc = epoch_acc / max(epoch_steps, 1)
            print(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"pixel_acc={avg_acc:.3f}"
            )

            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
                print(
                    f"    val_loss={val_metrics['loss']:.4f} | "
                    f"val_pixel_acc={val_metrics['pixel_accuracy']:.3f} | "
                    f"val_grid_match={val_metrics['grid_exact_match']:.3f}"
                )
                self._log({
                    f"{stage_name}/val_{k}": v for k, v in val_metrics.items()
                })

                if val_metrics["pixel_accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["pixel_accuracy"]
                    self._save_checkpoint(f"best_{stage_name}.pt")

            # Periodic save
            if (epoch + 1) % self.save_every == 0 or (epoch + 1) == epochs:
                self._save_checkpoint(f"{stage_name}_epoch{epoch + 1}.pt")

    def _train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        difficulty_multiplier: float = 1.0,
    ) -> Dict[str, float]:
        """Single training step with AMP and gradient clipping."""
        demo_inputs = batch["demo_inputs"].to(self.device)
        demo_outputs = batch["demo_outputs"].to(self.device)
        query_input = batch["query_input"].to(self.device)
        num_demos = batch["num_demos"].to(self.device)
        target = batch["target"].to(self.device)
        difficulty = batch["difficulty"].to(self.device)

        if difficulty_multiplier != 1.0:
            difficulty = difficulty * difficulty_multiplier

        optimizer.zero_grad()

        amp_device = "cuda" if self.amp_enabled else "cpu"
        with torch.amp.autocast(
            device_type=amp_device, dtype=self.dtype, enabled=self.amp_enabled
        ):
            result = self.model(
                demo_inputs, demo_outputs, query_input, num_demos,
                target=target, difficulty=difficulty,
            )
            loss = result["loss"]

        self.scaler.scale(loss).backward()

        if self.gradient_clip > 0:
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        self.scaler.step(optimizer)
        self.scaler.update()
        scheduler.step()

        return {
            "loss": loss.item(),
            "pixel_accuracy": result["pixel_accuracy"].item(),
        }

    # ─── Validation ──────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation with progress bar."""
        self.model.eval()
        total_loss = 0.0
        total_pixel_acc = 0.0
        total_grid_match = 0.0
        n_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc="    Validating",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            demo_inputs = batch["demo_inputs"].to(self.device)
            demo_outputs = batch["demo_outputs"].to(self.device)
            query_input = batch["query_input"].to(self.device)
            num_demos = batch["num_demos"].to(self.device)
            target = batch["target"].to(self.device)

            result = self.model(
                demo_inputs, demo_outputs, query_input, num_demos, target=target
            )
            metrics = compute_loss(result["logits"], target)

            total_loss += metrics["loss"].item()
            total_pixel_acc += metrics["pixel_accuracy"].item()
            total_grid_match += metrics["grid_exact_match"].item()
            n_batches += 1

            pbar.set_postfix(val_acc=f"{total_pixel_acc / n_batches:.3f}")

        pbar.close()

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "pixel_accuracy": total_pixel_acc / n,
            "grid_exact_match": total_grid_match / n,
        }

    # ─── Checkpointing ───────────────────────────────────────────

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }, path)
        print(f"    Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Loaded checkpoint from {path} (step {self.global_step})")

    # ─── Logging ─────────────────────────────────────────────────

    def _init_wandb(self) -> None:
        try:
            import os
            import wandb
            self.wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "arc-it"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                config=self.config,
            )
        except ImportError:
            print("wandb not installed, logging to stdout only")
            self.use_wandb = False

    def _log(self, metrics: dict) -> None:
        if self.wandb_run:
            self.wandb_run.log(metrics, step=self.global_step)
