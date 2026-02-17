"""Evaluation pipeline for ARC-IT.

Evaluates model on ARC-AGI tasks with metrics:
    - pixel_accuracy: fraction of correctly predicted valid pixels
    - grid_exact_match: fraction of tasks solved perfectly
    - per_difficulty: breakdown by AGI-1 vs AGI-2

Can run with or without test-time training (TTT).
Includes tqdm progress bars for all evaluation loops.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_it.data.canvas import IGNORE_INDEX, crop_prediction_from_canvas
from arc_it.data.dataset import ARCDataset, collate_fn
from arc_it.models.arc_it_model import ARCITModel
from arc_it.utils.device import get_device


class Evaluator:
    """Evaluate ARC-IT model on ARC-AGI benchmarks."""

    def __init__(
        self,
        model: ARCITModel,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: ARCDataset,
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate model on an entire dataset with progress bar."""
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        all_results = []
        total_correct_pixels = 0
        total_pixels = 0
        total_grids = 0
        total_exact_match = 0
        total_time = 0.0

        pbar = tqdm(loader, desc="Evaluating", dynamic_ncols=True)

        for batch in pbar:
            t_start = time.time()

            input_rgb = batch["input_rgb_224"].to(self.device)
            input_canvas = batch["input_canvas"].to(self.device)
            target = batch["target"].to(self.device)

            result = self.model(input_rgb, input_canvas)
            prediction = result["prediction"]

            t_elapsed = time.time() - t_start

            for i in range(prediction.shape[0]):
                pred_i = prediction[i]
                target_i = target[i]
                valid = target_i != IGNORE_INDEX

                correct = ((pred_i == target_i) & valid).sum().item()
                total_valid = valid.sum().item()
                exact_match = ((pred_i == target_i) | ~valid).all().item()

                total_correct_pixels += correct
                total_pixels += total_valid
                total_grids += 1
                total_exact_match += int(exact_match)

                offset = batch["offset"][i]
                scale = batch["scale_factor"][i].item()
                pred_grid = crop_prediction_from_canvas(
                    pred_i.cpu(), offset[0].item(), offset[1].item(), scale
                )

                all_results.append({
                    "task_name": batch["task_name"][i],
                    "example_index": batch["example_index"][i].item(),
                    "pixel_accuracy": correct / max(total_valid, 1),
                    "exact_match": exact_match,
                    "prediction": pred_grid,
                    "time": t_elapsed / prediction.shape[0],
                })

            total_time += t_elapsed
            pixel_acc_so_far = total_correct_pixels / max(total_pixels, 1)
            grid_match_so_far = total_exact_match / max(total_grids, 1)
            pbar.set_postfix(
                px_acc=f"{pixel_acc_so_far:.3f}",
                grid_match=f"{grid_match_so_far:.3f}",
                grids=total_grids,
            )

        pbar.close()

        pixel_acc = total_correct_pixels / max(total_pixels, 1)
        grid_match = total_exact_match / max(total_grids, 1)

        return {
            "pixel_accuracy": pixel_acc,
            "grid_exact_match": grid_match,
            "total_grids": total_grids,
            "total_exact_match": total_exact_match,
            "avg_time_per_grid": total_time / max(total_grids, 1),
            "total_time": total_time,
            "per_task": all_results,
        }

    @torch.no_grad()
    def predict_single(
        self,
        input_rgb_224: torch.Tensor,
        input_canvas: torch.Tensor,
    ) -> torch.Tensor:
        """Predict on a single input grid."""
        if input_rgb_224.dim() == 3:
            input_rgb_224 = input_rgb_224.unsqueeze(0)
        if input_canvas.dim() == 2:
            input_canvas = input_canvas.unsqueeze(0)
        input_rgb_224 = input_rgb_224.to(self.device)
        input_canvas = input_canvas.to(self.device)
        result = self.model(input_rgb_224, input_canvas)
        return result["prediction"][0].cpu()

    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """Save evaluation results to JSON."""
        serializable = {
            k: v for k, v in results.items() if k != "per_task"
        }
        serializable["per_task_summary"] = [
            {
                "task_name": r["task_name"],
                "example_index": r["example_index"],
                "pixel_accuracy": r["pixel_accuracy"],
                "exact_match": r["exact_match"],
            }
            for r in results.get("per_task", [])
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {path}")
