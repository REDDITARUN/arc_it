"""Test-Time Training (TTT) for ARC-IT.

Per-task adaptation: for each evaluation task, fine-tune the model
on the task's demonstration examples (2-5 input/output pairs) before
making predictions. This is the key technique used by ALL top ARC
solutions (VARC, ARChitects, NVARC).

Workflow per task:
    1. Snapshot model weights
    2. Augment demonstration examples (geometric + color perms)
    3. Fine-tune last N Sana layers + Bridge + Decoder for K steps
    4. Generate predictions (optionally with multi-sample)
    5. Restore original weights for next task

Key hyperparameters (defaults from VARC):
    steps=100, lr=1e-4, batch_size=8, layers_to_update=4
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from arc_it.data.augmentation import (
    get_geometric_augmentations,
    random_color_permutation,
    permute_colors,
    get_inverse_geometric,
    inverse_color_permutation,
)
from arc_it.data.canvas import (
    IGNORE_INDEX,
    pad_grid_to_canvas,
    crop_prediction_from_canvas,
    random_offset,
    resolution_scale,
    random_scale_factor,
)
from arc_it.data.rendering import render_canvas_to_rgb_224
from arc_it.models.arc_it_model import ARCITModel
from arc_it.utils.device import get_device


class TestTimeTrainer:
    """Per-task test-time training and prediction."""

    def __init__(
        self,
        model: ARCITModel,
        ttt_steps: int = 100,
        ttt_lr: float = 1e-4,
        ttt_batch_size: int = 8,
        num_layers_to_update: int = 4,
        num_candidates: int = 32,
        num_denoising_steps: int = 50,
        canvas_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model.to(device or get_device())
        self.ttt_steps = ttt_steps
        self.ttt_lr = ttt_lr
        self.ttt_batch_size = ttt_batch_size
        self.num_layers_to_update = num_layers_to_update
        self.num_candidates = num_candidates
        self.num_denoising_steps = num_denoising_steps
        self.canvas_size = canvas_size
        self.device = device or get_device()

    def predict_task(
        self,
        task: Dict[str, Any],
        num_attempts: int = 2,
    ) -> List[List[List[int]]]:
        """Predict test output(s) for a single ARC task with TTT.

        Args:
            task: ARC task dict with "train" (demos) and "test" (inputs).
            num_attempts: Number of predictions to return (ARC allows 2).

        Returns:
            List of predicted grids (each a 2D list of integers).
        """
        # 1. Snapshot weights
        snapshot = self._snapshot_weights()

        # 2. Prepare augmented training data from demonstrations
        train_data = self._prepare_ttt_data(task["train"])

        # 3. Fine-tune
        self._fine_tune(train_data)

        # 4. Generate predictions
        predictions = []
        for test_example in task["test"]:
            candidates = self._generate_candidates(
                test_example["input"],
                num_candidates=self.num_candidates,
            )
            # Score and pick top predictions
            scored = self._score_candidates(candidates)
            top = scored[:num_attempts]
            predictions.extend(top)

        # 5. Restore weights
        self._restore_weights(snapshot)

        return predictions[:num_attempts]

    # ─── TTT Data Preparation ────────────────────────────────────

    def _prepare_ttt_data(
        self,
        train_examples: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Create augmented training batches from demonstration examples.

        Applies geometric + color augmentation to each demo pair.
        """
        rng = np.random.RandomState(42)
        geos = get_geometric_augmentations()

        all_input_rgb = []
        all_targets = []

        for example in train_examples:
            inp = example["input"]
            out = example["output"]

            for geo_name, geo_fn in geos:
                # Apply geometric augmentation
                aug_inp = geo_fn(inp)
                aug_out = geo_fn(out)

                # Apply color permutation
                perm = random_color_permutation(rng, keep_background=True)
                aug_inp = permute_colors(aug_inp, perm)
                aug_out = permute_colors(aug_out, perm)

                # Compute canvas dimensions
                h_in, w_in = len(aug_inp), len(aug_inp[0])
                h_out, w_out = len(aug_out), len(aug_out[0])
                max_h, max_w = max(h_in, h_out), max(w_in, w_out)

                x_off, y_off = random_offset(max_h, max_w, self.canvas_size, rng)

                # Create canvases
                input_canvas, _, _, _ = pad_grid_to_canvas(
                    aug_inp, self.canvas_size, x_off, y_off, mark_boundary=False
                )
                output_canvas, output_mask, _, _ = pad_grid_to_canvas(
                    aug_out, self.canvas_size, x_off, y_off, mark_boundary=True
                )
                target = output_canvas.clone()
                target[output_mask == 0] = IGNORE_INDEX

                input_rgb = render_canvas_to_rgb_224(input_canvas, normalize=True)
                all_input_rgb.append(input_rgb)
                all_targets.append(target)

        return {
            "input_rgb_224": torch.stack(all_input_rgb),
            "target": torch.stack(all_targets),
        }

    # ─── Fine-tuning ─────────────────────────────────────────────

    def _fine_tune(self, train_data: Dict[str, torch.Tensor]) -> None:
        """Fine-tune model on augmented demonstration data."""
        self.model.train()

        # Only update last N Sana layers + bridge + decoder
        self._set_ttt_trainable()

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.ttt_lr, weight_decay=0)

        input_rgb = train_data["input_rgb_224"].to(self.device)
        targets = train_data["target"].to(self.device)
        n_samples = input_rgb.shape[0]

        for step in range(self.ttt_steps):
            # Sample a mini-batch
            indices = torch.randint(0, n_samples, (min(self.ttt_batch_size, n_samples),))
            batch_rgb = input_rgb[indices]
            batch_target = targets[indices]

            optimizer.zero_grad()
            result = self.model(batch_rgb, target=batch_target)
            result["loss"].backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        self.model.eval()

    def _set_ttt_trainable(self) -> None:
        """Freeze everything except last N Sana layers + bridge + decoder."""
        # Freeze all
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze bridge
        for param in self.model.bridge.parameters():
            param.requires_grad = True

        # Unfreeze decoder
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        # Unfreeze output embedder
        for param in self.model.output_embedder.parameters():
            param.requires_grad = True

        # Unfreeze last N Sana blocks
        n_blocks = len(self.model.sana.blocks)
        start = max(0, n_blocks - self.num_layers_to_update)
        for block in self.model.sana.blocks[start:]:
            for param in block.parameters():
                param.requires_grad = True

    # ─── Candidate Generation ────────────────────────────────────

    @torch.no_grad()
    def _generate_candidates(
        self,
        input_grid: List[List[int]],
        num_candidates: int = 32,
    ) -> List[List[List[int]]]:
        """Generate multiple candidate predictions using diffusion stochasticity."""
        self.model.eval()

        # Prepare input
        input_canvas, _, _, _ = pad_grid_to_canvas(
            input_grid, self.canvas_size, x_offset=1, y_offset=1, mark_boundary=False
        )
        input_rgb = render_canvas_to_rgb_224(input_canvas, normalize=True)
        input_rgb = input_rgb.unsqueeze(0).to(self.device)

        candidates = []
        for _ in range(num_candidates):
            result = self.model(input_rgb, target=None)
            pred = result["prediction"][0].cpu()
            grid = crop_prediction_from_canvas(pred, x_offset=1, y_offset=1)
            candidates.append(grid)

        return candidates

    # ─── Candidate Scoring ───────────────────────────────────────

    def _score_candidates(
        self,
        candidates: List[List[List[int]]],
    ) -> List[List[List[int]]]:
        """Score and rank candidates using heuristics. Return sorted list."""
        scored = []
        for grid in candidates:
            score = self._compute_grid_score(grid)
            scored.append((score, grid))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate (prefer unique predictions)
        seen = set()
        unique = []
        for score, grid in scored:
            key = str(grid)
            if key not in seen:
                seen.add(key)
                unique.append(grid)

        return unique

    @staticmethod
    def _compute_grid_score(grid: List[List[int]]) -> float:
        """Score a predicted grid using heuristic quality metrics.

        Higher score = more likely to be correct.
        """
        if not grid or not grid[0]:
            return 0.0

        arr = np.array(grid)
        h, w = arr.shape
        score = 0.0

        # 1. Color parsimony: fewer distinct colors is better (weight=0.3)
        n_colors = len(set(arr.flatten()))
        color_score = max(0, 1.0 - (n_colors - 2) / 8.0)
        score += 0.3 * color_score

        # 2. Symmetry: check H/V symmetry (weight=0.3)
        h_sym = np.array_equal(arr, np.fliplr(arr))
        v_sym = np.array_equal(arr, np.flipud(arr))
        sym_score = (int(h_sym) + int(v_sym)) / 2.0
        score += 0.3 * sym_score

        # 3. Non-trivial: penalize all-same grids (weight=0.2)
        if n_colors > 1:
            score += 0.2

        # 4. Reasonable size: penalize tiny or huge grids (weight=0.2)
        if 1 <= h <= 30 and 1 <= w <= 30:
            score += 0.2

        return score

    # ─── Weight Snapshot/Restore ─────────────────────────────────

    def _snapshot_weights(self) -> Dict[str, torch.Tensor]:
        """Save current model weights."""
        return copy.deepcopy(self.model.state_dict())

    def _restore_weights(self, snapshot: Dict[str, torch.Tensor]) -> None:
        """Restore model to a previous weight snapshot."""
        self.model.load_state_dict(snapshot)
        self.model.eval()
