"""Test-Time Training (TTT) for the Rule-Conditioned Transformer.

Per-task adaptation: for each evaluation task, fine-tune the model
on augmented versions of the task's demonstration examples before
making predictions.

The Rule-Conditioned architecture makes TTT more natural: the demo
pairs are already part of the forward pass, so TTT fine-tunes the
rule encoder to better extract rules from THIS task's specific demos.

Workflow per task:
    1. Snapshot model weights
    2. Create augmented training data (leave-one-out on demo pairs)
    3. Fine-tune rule_encoder + rule_applier + decoder for K steps
    4. Generate predictions with augmentation voting
    5. Restore original weights for next task
"""

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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
)
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
        num_candidates: int = 32,
        canvas_size: int = 64,
        max_demos: int = 5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model.to(device or get_device())
        self.ttt_steps = ttt_steps
        self.ttt_lr = ttt_lr
        self.ttt_batch_size = ttt_batch_size
        self.num_candidates = num_candidates
        self.canvas_size = canvas_size
        self.max_demos = max_demos
        self.device = device or get_device()

    def predict_task(
        self,
        task: Dict[str, Any],
        num_attempts: int = 2,
    ) -> List[List[List[int]]]:
        """Predict test output(s) for a single ARC task with TTT."""
        snapshot = self._snapshot_weights()

        train_data = self._prepare_ttt_data(task["train"])
        self._fine_tune(train_data)

        predictions = []
        for test_example in task["test"]:
            candidates = self._generate_candidates(
                task["train"],
                test_example["input"],
                num_candidates=self.num_candidates,
            )
            scored = self._score_candidates(candidates)
            top = scored[:num_attempts]
            predictions.extend(top)

        self._restore_weights(snapshot)
        return predictions[:num_attempts]

    # ─── TTT Data Preparation ────────────────────────────────────

    def _prepare_ttt_data(
        self,
        train_examples: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Create augmented training batches using leave-one-out.

        For each training example, hold it out as the query and use
        the rest as demos. Apply geometric + color augmentation.
        """
        rng = np.random.RandomState(42)
        geos = get_geometric_augmentations()
        K = len(train_examples)

        all_demo_inputs = []
        all_demo_outputs = []
        all_query_inputs = []
        all_targets = []
        all_num_demos = []

        for hold_out_idx in range(K):
            query = train_examples[hold_out_idx]
            demos = [ex for i, ex in enumerate(train_examples) if i != hold_out_idx]

            for geo_name, geo_fn in geos:
                perm = random_color_permutation(rng, keep_background=True)

                # Augment demos
                aug_demos = []
                for d in demos:
                    aug_in = permute_colors(geo_fn(d["input"]), perm)
                    aug_out = permute_colors(geo_fn(d["output"]), perm)
                    aug_demos.append({"input": aug_in, "output": aug_out})

                # Augment query
                aug_q_in = permute_colors(geo_fn(query["input"]), perm)
                aug_q_out = permute_colors(geo_fn(query["output"]), perm)

                # Find max dims
                all_grids = []
                for d in aug_demos:
                    all_grids.extend([d["input"], d["output"]])
                all_grids.extend([aug_q_in, aug_q_out])
                max_h = max(len(g) for g in all_grids)
                max_w = max(max(len(r) for r in g) for g in all_grids)

                x_off, y_off = random_offset(max_h, max_w, self.canvas_size, rng)

                # Place demos on canvas
                demo_in_list = []
                demo_out_list = []
                for d in aug_demos[:self.max_demos]:
                    d_in, _, _, _ = pad_grid_to_canvas(
                        d["input"], self.canvas_size, x_off, y_off,
                        mark_boundary=False,
                    )
                    d_out, _, _, _ = pad_grid_to_canvas(
                        d["output"], self.canvas_size, x_off, y_off,
                        mark_boundary=True,
                    )
                    demo_in_list.append(d_in)
                    demo_out_list.append(d_out)

                nd = len(demo_in_list)
                while len(demo_in_list) < self.max_demos:
                    demo_in_list.append(
                        torch.full((self.canvas_size, self.canvas_size),
                                   IGNORE_INDEX, dtype=torch.long)
                    )
                    demo_out_list.append(
                        torch.full((self.canvas_size, self.canvas_size),
                                   IGNORE_INDEX, dtype=torch.long)
                    )

                # Place query on canvas
                q_in, _, _, _ = pad_grid_to_canvas(
                    aug_q_in, self.canvas_size, x_off, y_off,
                    mark_boundary=False,
                )
                q_out, q_mask, _, _ = pad_grid_to_canvas(
                    aug_q_out, self.canvas_size, x_off, y_off,
                    mark_boundary=True,
                )
                q_target = q_out.clone()
                q_target[q_mask == 0] = IGNORE_INDEX

                all_demo_inputs.append(torch.stack(demo_in_list))
                all_demo_outputs.append(torch.stack(demo_out_list))
                all_query_inputs.append(q_in)
                all_targets.append(q_target)
                all_num_demos.append(torch.tensor(nd, dtype=torch.long))

        return {
            "demo_inputs": torch.stack(all_demo_inputs),
            "demo_outputs": torch.stack(all_demo_outputs),
            "query_input": torch.stack(all_query_inputs),
            "target": torch.stack(all_targets),
            "num_demos": torch.stack(all_num_demos),
        }

    # ─── Fine-tuning ─────────────────────────────────────────────

    def _fine_tune(self, train_data: Dict[str, torch.Tensor]) -> None:
        """Fine-tune model on augmented task data."""
        self.model.train()
        self._set_ttt_trainable()

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.ttt_lr, weight_decay=0)

        demo_inputs = train_data["demo_inputs"].to(self.device)
        demo_outputs = train_data["demo_outputs"].to(self.device)
        query_inputs = train_data["query_input"].to(self.device)
        targets = train_data["target"].to(self.device)
        num_demos = train_data["num_demos"].to(self.device)
        n_samples = demo_inputs.shape[0]

        pbar = tqdm(
            range(self.ttt_steps),
            desc="    TTT fine-tune",
            leave=False,
            dynamic_ncols=True,
        )

        for step in pbar:
            indices = torch.randint(
                0, n_samples, (min(self.ttt_batch_size, n_samples),)
            )
            optimizer.zero_grad()
            result = self.model(
                demo_inputs[indices],
                demo_outputs[indices],
                query_inputs[indices],
                num_demos[indices],
                target=targets[indices],
            )
            result["loss"].backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if step % 20 == 0:
                pbar.set_postfix(loss=f"{result['loss'].item():.4f}")

        pbar.close()
        self.model.eval()

    def _set_ttt_trainable(self) -> None:
        """Make all parameters trainable for TTT."""
        for param in self.model.parameters():
            param.requires_grad = True

    # ─── Candidate Generation ────────────────────────────────────

    @torch.no_grad()
    def _generate_candidates(
        self,
        train_examples: List[Dict],
        input_grid: List[List[int]],
        num_candidates: int = 32,
    ) -> List[List[List[int]]]:
        """Generate candidate predictions via augmentation voting.

        Apply geometric + color augmentations to both demos and test
        input, predict, then invert the augmentation on the output.
        """
        self.model.eval()
        rng = np.random.RandomState(123)
        geos = get_geometric_augmentations()

        candidates = []
        pbar = tqdm(
            range(num_candidates),
            desc="    Generating candidates",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            geo_name, geo_fn = geos[i % len(geos)]
            inv_fn = get_inverse_geometric(geo_name)

            if i >= len(geos):
                color_perm = random_color_permutation(rng, keep_background=True)
                inv_color = inverse_color_permutation(color_perm)
            else:
                color_perm = None
                inv_color = None

            # Augment demos
            aug_demos = []
            for ex in train_examples:
                aug_in = geo_fn(ex["input"])
                aug_out = geo_fn(ex["output"])
                if color_perm:
                    aug_in = permute_colors(aug_in, color_perm)
                    aug_out = permute_colors(aug_out, color_perm)
                aug_demos.append({"input": aug_in, "output": aug_out})

            # Augment test input
            aug_test = geo_fn(input_grid)
            if color_perm:
                aug_test = permute_colors(aug_test, color_perm)

            # Find max dims
            all_grids = []
            for d in aug_demos:
                all_grids.extend([d["input"], d["output"]])
            all_grids.append(aug_test)
            max_h = max(len(g) for g in all_grids)
            max_w = max(max(len(r) for r in g) for g in all_grids)

            x_off, y_off = 1, 1

            # Place on canvas
            demo_in_list = []
            demo_out_list = []
            for d in aug_demos[:self.max_demos]:
                d_in, _, _, _ = pad_grid_to_canvas(
                    d["input"], self.canvas_size, x_off, y_off,
                    mark_boundary=False,
                )
                d_out, _, _, _ = pad_grid_to_canvas(
                    d["output"], self.canvas_size, x_off, y_off,
                    mark_boundary=True,
                )
                demo_in_list.append(d_in)
                demo_out_list.append(d_out)

            nd = len(demo_in_list)
            while len(demo_in_list) < self.max_demos:
                demo_in_list.append(
                    torch.full((self.canvas_size, self.canvas_size),
                               IGNORE_INDEX, dtype=torch.long)
                )
                demo_out_list.append(
                    torch.full((self.canvas_size, self.canvas_size),
                               IGNORE_INDEX, dtype=torch.long)
                )

            q_in, _, _, _ = pad_grid_to_canvas(
                aug_test, self.canvas_size, x_off, y_off,
                mark_boundary=False,
            )

            # Forward pass
            di = torch.stack(demo_in_list).unsqueeze(0).to(self.device)
            do = torch.stack(demo_out_list).unsqueeze(0).to(self.device)
            qi = q_in.unsqueeze(0).to(self.device)
            nd_t = torch.tensor([nd], dtype=torch.long, device=self.device)

            result = self.model(di, do, qi, nd_t)
            pred = result["prediction"][0].cpu()
            grid = crop_prediction_from_canvas(pred, x_offset=x_off, y_offset=y_off)

            # Invert augmentations
            if inv_color is not None:
                grid = permute_colors(grid, inv_color)
            grid = inv_fn(grid)

            candidates.append(grid)

        pbar.close()
        return candidates

    # ─── Candidate Scoring ───────────────────────────────────────

    def _score_candidates(
        self,
        candidates: List[List[List[int]]],
    ) -> List[List[List[int]]]:
        """Score and rank candidates by frequency (majority voting)."""
        counts = {}
        for grid in candidates:
            key = str(grid)
            if key not in counts:
                counts[key] = {"grid": grid, "count": 0, "score": 0.0}
            counts[key]["count"] += 1

        for key, info in counts.items():
            freq_score = info["count"] / len(candidates)
            quality_score = self._compute_grid_score(info["grid"])
            info["score"] = 0.7 * freq_score + 0.3 * quality_score

        ranked = sorted(counts.values(), key=lambda x: x["score"], reverse=True)
        return [item["grid"] for item in ranked]

    @staticmethod
    def _compute_grid_score(grid: List[List[int]]) -> float:
        """Heuristic quality metrics for a predicted grid."""
        if not grid or not grid[0]:
            return 0.0

        arr = np.array(grid)
        h, w = arr.shape
        score = 0.0

        n_colors = len(set(arr.flatten()))
        color_score = max(0, 1.0 - (n_colors - 2) / 8.0)
        score += 0.3 * color_score

        h_sym = np.array_equal(arr, np.fliplr(arr))
        v_sym = np.array_equal(arr, np.flipud(arr))
        sym_score = (int(h_sym) + int(v_sym)) / 2.0
        score += 0.3 * sym_score

        if n_colors > 1:
            score += 0.2

        if 1 <= h <= 30 and 1 <= w <= 30:
            score += 0.2

        return score

    # ─── Weight Snapshot/Restore ─────────────────────────────────

    def _snapshot_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.model.state_dict())

    def _restore_weights(self, snapshot: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(snapshot)
        self.model.eval()
