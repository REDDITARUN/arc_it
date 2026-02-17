"""Evaluation pipeline for ARC-IT Rule-Conditioned Transformer.

Evaluates the model on ARC-AGI tasks using the task-level dataset.
Each evaluation sample includes demo pairs + query input.

Metrics:
    - pixel_accuracy: fraction of correctly predicted valid pixels
    - grid_exact_match: fraction of tasks solved perfectly
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_it.data.canvas import IGNORE_INDEX, crop_prediction_from_canvas
from arc_it.data.dataset import ARCTaskDataset, collate_fn
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
        dataset: ARCTaskDataset,
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate model on an entire dataset."""
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

            demo_inputs = batch["demo_inputs"].to(self.device)
            demo_outputs = batch["demo_outputs"].to(self.device)
            query_input = batch["query_input"].to(self.device)
            num_demos = batch["num_demos"].to(self.device)
            target = batch["target"].to(self.device)

            result = self.model(
                demo_inputs, demo_outputs, query_input, num_demos
            )
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
                    "pixel_accuracy": correct / max(total_valid, 1),
                    "exact_match": exact_match,
                    "prediction": pred_grid,
                    "time": t_elapsed / prediction.shape[0],
                })

            total_time += t_elapsed
            px_acc = total_correct_pixels / max(total_pixels, 1)
            grid_match = total_exact_match / max(total_grids, 1)
            pbar.set_postfix(
                px_acc=f"{px_acc:.3f}",
                grid_match=f"{grid_match:.3f}",
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
        demo_inputs: torch.Tensor,
        demo_outputs: torch.Tensor,
        query_input: torch.Tensor,
        num_demos: torch.Tensor,
    ) -> torch.Tensor:
        """Predict on a single task instance.

        Args:
            demo_inputs:  (K, 64, 64) demo input canvases.
            demo_outputs: (K, 64, 64) demo output canvases.
            query_input:  (64, 64) query input canvas.
            num_demos:    scalar tensor.

        Returns:
            (64, 64) prediction tensor.
        """
        if demo_inputs.dim() == 3:
            demo_inputs = demo_inputs.unsqueeze(0)
        if demo_outputs.dim() == 3:
            demo_outputs = demo_outputs.unsqueeze(0)
        if query_input.dim() == 2:
            query_input = query_input.unsqueeze(0)
        if num_demos.dim() == 0:
            num_demos = num_demos.unsqueeze(0)

        demo_inputs = demo_inputs.to(self.device)
        demo_outputs = demo_outputs.to(self.device)
        query_input = query_input.to(self.device)
        num_demos = num_demos.to(self.device)

        result = self.model(demo_inputs, demo_outputs, query_input, num_demos)
        return result["prediction"][0].cpu()

    def save_results(self, results: Dict[str, Any], path: str) -> None:
        """Save evaluation results to JSON."""
        serializable = {k: v for k, v in results.items() if k != "per_task"}
        serializable["per_task_summary"] = [
            {
                "task_name": r["task_name"],
                "pixel_accuracy": r["pixel_accuracy"],
                "exact_match": r["exact_match"],
            }
            for r in results.get("per_task", [])
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {path}")
