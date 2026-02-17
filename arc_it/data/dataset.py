"""PyTorch Dataset for ARC-AGI tasks with on-the-fly augmentation.

Supports both ARC-AGI-1 and ARC-AGI-2 datasets with:
- On-the-fly geometric + color augmentation
- Resolution scaling augmentation
- Translation augmentation within a fixed canvas
- Difficulty-weighted sampling (AGI-2 gets higher weight)

Each sample returns both the canvas representation (for Sana/decoder)
and the 224x224 RGB rendering (for JEPA/DINOv2 encoder).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from arc_it.data.augmentation import (
    get_geometric_augmentations,
    random_color_permutation,
    permute_colors,
)
from arc_it.data.canvas import (
    IGNORE_INDEX,
    pad_grid_to_canvas,
    random_offset,
    random_scale_factor,
    resolution_scale,
)
from arc_it.data.rendering import render_canvas_to_rgb_224


class ARCDataset(Dataset):
    """Dataset for ARC-AGI training and evaluation.

    Each sample is a single (input_grid, output_grid) pair from a task.
    Augmentations are applied on-the-fly per __getitem__ call.

    The `repeat_factor` multiplies the effective dataset size. Since
    augmentation is random on each access, every repeated sample gets
    a different geometric/color augmentation, effectively expanding
    the dataset.
    """

    def __init__(
        self,
        data_roots: List[str],
        split: str = "training",
        subset: str = "train",
        canvas_size: int = 64,
        enable_augmentation: bool = True,
        enable_translation: bool = True,
        enable_resolution: bool = True,
        num_color_perms: int = 10,
        keep_background: bool = True,
        difficulty_labels: Optional[Dict[str, float]] = None,
        seed: int = 42,
        repeat_factor: int = 1,
    ) -> None:
        """
        Args:
            data_roots: List of paths to ARC dataset roots (e.g.,
                        ["References/ARC-AGI", "References/ARC-AGI-2"]).
            split: Dataset split directory name ("training" or "evaluation").
            subset: Which examples to use ("train" = demonstration pairs,
                    "test" = test pairs for evaluation).
            canvas_size: Fixed canvas size for padding.
            enable_augmentation: Enable geometric + color augmentation.
            enable_translation: Enable random translation in canvas.
            enable_resolution: Enable random resolution scaling.
            num_color_perms: Number of color permutations per sample.
            keep_background: Keep color 0 fixed during permutation.
            difficulty_labels: Optional dict mapping source path patterns
                              to difficulty weights.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.canvas_size = canvas_size
        self.enable_augmentation = enable_augmentation
        self.enable_translation = enable_translation
        self.enable_resolution = enable_resolution
        self.num_color_perms = num_color_perms
        self.keep_background = keep_background
        self.rng = np.random.RandomState(seed)

        self.samples: List[Dict[str, Any]] = []
        self.task_lookup: Dict[str, int] = {}

        # Load tasks from all data roots
        for root_path in data_roots:
            root = Path(root_path)
            split_dir = root / "data" / split
            if not split_dir.exists():
                print(f"Warning: {split_dir} does not exist, skipping.")
                continue

            # Determine difficulty (AGI-2 = 1.5, AGI-1 = 1.0)
            difficulty = 1.0
            if difficulty_labels:
                for pattern, weight in difficulty_labels.items():
                    if pattern in str(root):
                        difficulty = weight
                        break
            elif "AGI-2" in str(root):
                difficulty = 1.5

            files = sorted(split_dir.glob("*.json"))
            examples_key = "train" if subset == "train" else "test"

            for file_path in files:
                task_name = file_path.stem
                if task_name not in self.task_lookup:
                    self.task_lookup[task_name] = len(self.task_lookup)
                task_index = self.task_lookup[task_name]

                with file_path.open("r") as fh:
                    task_data = json.load(fh)

                examples = task_data.get(examples_key, [])
                for ex_idx, example in enumerate(examples):
                    h_in = len(example["input"])
                    w_in = len(example["input"][0]) if h_in > 0 else 0
                    h_out = len(example.get("output", [])) if "output" in example else 0
                    w_out = len(example["output"][0]) if h_out > 0 else 0

                    max_h = max(h_in, h_out)
                    max_w = max(w_in, w_out)

                    # Skip grids that exceed max size
                    if max_h > 30 or max_w > 30:
                        continue

                    self.samples.append({
                        "example": example,
                        "task_index": task_index,
                        "task_name": task_name,
                        "example_index": ex_idx,
                        "difficulty": difficulty,
                        "source": str(root),
                    })

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}', subset='{subset}' "
                f"in roots: {data_roots}"
            )

        self.num_tasks = len(self.task_lookup)
        self.repeat_factor = max(1, repeat_factor)
        raw_count = len(self.samples)
        effective = raw_count * self.repeat_factor
        print(f"Loaded {raw_count} samples from {self.num_tasks} tasks "
              f"(repeat={self.repeat_factor}x → {effective} effective)")

    def __len__(self) -> int:
        return len(self.samples) * self.repeat_factor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = idx % len(self.samples)
        sample = self.samples[real_idx]
        example = sample["example"]
        input_grid = example["input"]
        output_grid = example.get("output")

        # ─── On-the-fly augmentation ─────────────────────────────
        color_perm = None
        geometric_name = "identity"

        if self.enable_augmentation:
            # Random geometric transform (1 of 8)
            geos = get_geometric_augmentations()
            geo_idx = self.rng.randint(0, len(geos))
            geometric_name, geo_fn = geos[geo_idx]
            input_grid = geo_fn(input_grid)
            if output_grid is not None:
                output_grid = geo_fn(output_grid)

            # Random color permutation
            color_perm = random_color_permutation(self.rng, self.keep_background)
            input_grid = permute_colors(input_grid, color_perm)
            if output_grid is not None:
                output_grid = permute_colors(output_grid, color_perm)

        # ─── Resolution scaling ──────────────────────────────────
        h_in = len(input_grid)
        w_in = len(input_grid[0]) if h_in > 0 else 0
        h_out = len(output_grid) if output_grid else 0
        w_out = len(output_grid[0]) if h_out > 0 else 0
        max_h = max(h_in, h_out)
        max_w = max(w_in, w_out)

        if self.enable_resolution:
            scale = random_scale_factor(max_h, max_w, self.canvas_size, self.rng)
        else:
            scale = 1

        if scale > 1:
            input_grid = resolution_scale(input_grid, scale)
            if output_grid is not None:
                output_grid = resolution_scale(output_grid, scale)
            max_h *= scale
            max_w *= scale

        # ─── Canvas placement ────────────────────────────────────
        if self.enable_translation:
            x_off, y_off = random_offset(max_h, max_w, self.canvas_size, self.rng)
        else:
            x_off, y_off = 1, 1

        input_canvas, input_mask, _, _ = pad_grid_to_canvas(
            input_grid, self.canvas_size, x_off, y_off, mark_boundary=False,
        )

        if output_grid is not None:
            output_canvas, output_mask, out_h, out_w = pad_grid_to_canvas(
                output_grid, self.canvas_size, x_off, y_off, mark_boundary=True,
            )
        else:
            output_canvas = torch.full(
                (self.canvas_size, self.canvas_size), IGNORE_INDEX, dtype=torch.long
            )
            output_mask = torch.zeros(
                (self.canvas_size, self.canvas_size), dtype=torch.long
            )
            out_h, out_w = 0, 0

        # Mask non-valid pixels in target (for loss masking)
        target = output_canvas.clone()
        target[output_mask == 0] = IGNORE_INDEX

        # ─── RGB rendering for JEPA encoder ──────────────────────
        input_rgb_224 = render_canvas_to_rgb_224(input_canvas, normalize=True)

        return {
            "input_canvas": input_canvas,           # (64, 64) integers 0-11
            "input_rgb_224": input_rgb_224,          # (3, 224, 224) normalized float
            "input_mask": input_mask,                # (64, 64) binary
            "target": target,                        # (64, 64) integers 0-11, IGNORE_INDEX for padding
            "output_mask": output_mask,              # (64, 64) binary
            "task_id": torch.tensor(sample["task_index"], dtype=torch.long),
            "task_name": sample["task_name"],
            "example_index": torch.tensor(sample["example_index"], dtype=torch.long),
            "difficulty": torch.tensor(sample["difficulty"], dtype=torch.float32),
            "target_shape": torch.tensor([out_h, out_w], dtype=torch.long),
            "offset": torch.tensor([x_off, y_off], dtype=torch.long),
            "scale_factor": torch.tensor(scale, dtype=torch.long),
            "geometric": geometric_name,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collation for variable metadata types."""
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], str):
            result[key] = values
        else:
            result[key] = values
    return result


def build_dataloaders(
    config: Dict[str, Any],
) -> Tuple[ARCDataset, DataLoader, Optional[ARCDataset], Optional[DataLoader]]:
    """Build train and optional eval dataloaders from config.

    Args:
        config: Full configuration dict (from load_config).

    Returns:
        (train_dataset, train_loader, eval_dataset, eval_loader)
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    data_roots = [data_cfg["arc_agi1_path"], data_cfg["arc_agi2_path"]]
    canvas_size = data_cfg["canvas_size"]
    aug_cfg = data_cfg["augmentation"]

    repeat_factor = data_cfg.get("repeat_factor", 1)

    train_dataset = ARCDataset(
        data_roots=data_roots,
        split="training",
        subset="train",
        canvas_size=canvas_size,
        enable_augmentation=aug_cfg["geometric"],
        enable_translation=aug_cfg["translation"],
        enable_resolution=aug_cfg["resolution_scaling"],
        num_color_perms=aug_cfg["num_color_perms"],
        keep_background=aug_cfg["keep_background"],
        repeat_factor=repeat_factor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    eval_dataset = None
    eval_loader = None

    eval_cfg = config.get("evaluation", {})
    if eval_cfg:
        eval_dataset = ARCDataset(
            data_roots=data_roots,
            split="training",
            subset="test",
            canvas_size=canvas_size,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg["num_workers"],
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=True,
        )

    return train_dataset, train_loader, eval_dataset, eval_loader
