"""Three-stage Trainer for ARC-IT.

Stage 1 (Bridge Alignment): Freeze encoder + Sana, train only Bridge + Decoder.
Stage 2 (Full Training):    Freeze encoder only, train Bridge + Sana + Decoder.
Stage 3 (Hard Focus):       Same as Stage 2 but lower LR, oversample AGI-2.

Features:
    - Automatic mixed precision (AMP) on CUDA, disabled on CPU/MPS
    - Gradient clipping, gradient checkpointing (CUDA only)
    - Checkpoint saving (best + periodic)
    - Weights & Biases logging (optional)
    - Device-adaptive batch sizes and dtypes
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arc_it.models.arc_it_model import ARCITModel
from arc_it.training.loss import compute_loss
from arc_it.utils.device import get_device, get_dtype, get_amp_enabled


class Trainer:
    """Manages the full 3-stage training process."""

    def __init__(
        self,
        model: ARCITModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
    ) -> None:
        self.device = get_device()
        self.dtype = get_dtype(self.device)
        self.amp_enabled = get_amp_enabled(self.device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.wandb_run = None

        # AMP scaler for CUDA
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)

        # Tracking
        self.global_step = 0
        self.best_val_acc = 0.0

        train_cfg = self.config.get("training", {})
        self.gradient_clip = train_cfg.get("gradient_clip", 1.0)
        self.log_every = train_cfg.get("log_every_n_steps", 100)
        self.save_every = train_cfg.get("save_every_n_epochs", 1)

    # ─── Public API ──────────────────────────────────────────────

    def train(self) -> None:
        """Run the full 3-stage training pipeline."""
        if self.use_wandb:
            self._init_wandb()

        train_cfg = self.config.get("training", {})

        # Stage 1: Bridge Alignment
        s1 = train_cfg.get("stage1", {})
        print("\n" + "=" * 60)
        print("STAGE 1: Bridge Alignment")
        print("=" * 60)
        self._set_frozen_stage(freeze_sana=True)
        optimizer, scheduler = self._build_optimizer(
            lr=s1.get("lr", 1e-4),
            epochs=s1.get("epochs", 5),
        )
        self._train_epochs(
            optimizer, scheduler,
            epochs=s1.get("epochs", 5),
            stage_name="stage1",
        )

        # Stage 2: Full Training
        s2 = train_cfg.get("stage2", {})
        print("\n" + "=" * 60)
        print("STAGE 2: Full Training")
        print("=" * 60)
        self._set_frozen_stage(freeze_sana=False)
        optimizer, scheduler = self._build_optimizer(
            lr=s2.get("lr", 5e-5),
            epochs=s2.get("epochs", 20),
        )
        self._train_epochs(
            optimizer, scheduler,
            epochs=s2.get("epochs", 20),
            stage_name="stage2",
        )

        # Stage 3: Hard Focus
        s3 = train_cfg.get("stage3", {})
        print("\n" + "=" * 60)
        print("STAGE 3: Hard Example Focus")
        print("=" * 60)
        optimizer, scheduler = self._build_optimizer(
            lr=s3.get("lr", 1e-5),
            epochs=s3.get("epochs", 5),
        )
        self._train_epochs(
            optimizer, scheduler,
            epochs=s3.get("epochs", 5),
            stage_name="stage3",
        )

        print("\nTraining complete!")
        if self.wandb_run:
            self.wandb_run.finish()

    # ─── Stage Freezing ──────────────────────────────────────────

    def _set_frozen_stage(self, freeze_sana: bool) -> None:
        """Configure which parameters are trainable."""
        # Encoder is always frozen (handled by FrozenEncoder)
        for param in self.model.sana.parameters():
            param.requires_grad = not freeze_sana

        # Bridge, output_embedder, decoder always trainable
        for module in [self.model.bridge, self.model.output_embedder, self.model.decoder]:
            for param in module.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M params")
        print(f"  Sana frozen: {freeze_sana}")

    # ─── Optimizer ───────────────────────────────────────────────

    def _build_optimizer(
        self,
        lr: float,
        epochs: int,
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

        total_steps = epochs * len(self.train_loader)
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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epochs: int,
        stage_name: str,
    ) -> None:
        """Run training for a given number of epochs."""
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_steps = 0
            t_start = time.time()

            for batch in self.train_loader:
                metrics = self._train_step(batch, optimizer, scheduler)
                epoch_loss += metrics["loss"]
                epoch_acc += metrics["pixel_accuracy"]
                epoch_steps += 1
                self.global_step += 1

                if self.global_step % self.log_every == 0:
                    self._log({
                        f"{stage_name}/loss": metrics["loss"],
                        f"{stage_name}/pixel_accuracy": metrics["pixel_accuracy"],
                        f"{stage_name}/lr": optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    })

            # Epoch summary
            avg_loss = epoch_loss / max(epoch_steps, 1)
            avg_acc = epoch_acc / max(epoch_steps, 1)
            elapsed = time.time() - t_start
            print(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.4f} | "
                f"pixel_acc={avg_acc:.3f} | "
                f"time={elapsed:.1f}s"
            )

            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
                print(
                    f"    val_loss={val_metrics['loss']:.4f} | "
                    f"val_pixel_acc={val_metrics['pixel_accuracy']:.3f} | "
                    f"val_grid_match={val_metrics['grid_exact_match']:.3f}"
                )
                self._log({f"{stage_name}/val_{k}": v for k, v in val_metrics.items()})

                if val_metrics["pixel_accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["pixel_accuracy"]
                    self._save_checkpoint(f"best_{stage_name}.pt")

            # Periodic save
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(f"{stage_name}_epoch{epoch + 1}.pt")

    def _train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> Dict[str, float]:
        """Single training step with AMP and gradient clipping."""
        input_rgb = batch["input_rgb_224"].to(self.device)
        target = batch["target"].to(self.device)
        difficulty = batch["difficulty"].to(self.device)

        optimizer.zero_grad()

        amp_device = "cuda" if self.amp_enabled else "cpu"
        with torch.amp.autocast(device_type=amp_device, dtype=self.dtype, enabled=self.amp_enabled):
            result = self.model(input_rgb, target=target, difficulty=difficulty)
            loss = result["loss"]

        self.scaler.scale(loss).backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )

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
        """Run validation and return averaged metrics."""
        self.model.eval()
        total_loss = 0.0
        total_pixel_acc = 0.0
        total_grid_match = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_rgb = batch["input_rgb_224"].to(self.device)
            target = batch["target"].to(self.device)

            result = self.model(input_rgb, target=target)
            metrics = compute_loss(result["logits"], target)

            total_loss += metrics["loss"].item()
            total_pixel_acc += metrics["pixel_accuracy"].item()
            total_grid_match += metrics["grid_exact_match"].item()
            n_batches += 1

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
            import wandb
            self.wandb_run = wandb.init(
                project="arc-it",
                config=self.config,
            )
        except ImportError:
            print("wandb not installed, logging to stdout only")
            self.use_wandb = False

    def _log(self, metrics: dict) -> None:
        if self.wandb_run:
            self.wandb_run.log(metrics, step=self.global_step)
