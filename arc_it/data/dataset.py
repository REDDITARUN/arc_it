"""PyTorch Dataset for ARC-AGI tasks — task-level sampling.

The Rule-Conditioned Transformer processes FULL TASKS, not individual
examples. Each sample contains:
    - K demonstration pairs (input, output)
    - 1 query pair (input, target)

Supports two data formats:
    1. ARC-AGI (standard): JSON with {"train": [...], "test": [...]}
    2. RE-ARC (synthetic):  JSON with a flat list of {"input", "output"} dicts

Training mode (subset="train"):
    ARC-AGI:  Leave-one-out — hold out each training example as the query.
    RE-ARC:   Random K+1 selection from the example pool.

Evaluation mode (subset="test"):
    Use ALL training examples as demos, predict each test example.

Augmentation is applied CONSISTENTLY across all grids in a task
(same geometric transform, same color permutation) to preserve
the underlying transformation rule.
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


# ─── Data source name → config key mapping ────────────────────────
SOURCE_KEY_MAP = {
    "agi1": "arc_agi1_path",
    "agi2": "arc_agi2_path",
    "re_arc": "re_arc_path",
}


class ARCTaskDataset(Dataset):
    """Task-level dataset for the Rule-Conditioned Transformer.

    Each sample is a full task context: K demo pairs + 1 query pair.
    Augmentations are applied on-the-fly, consistently across all grids.
    """

    def __init__(
        self,
        data_roots: List[str],
        split: str = "training",
        subset: str = "train",
        canvas_size: int = 64,
        max_demos: int = 5,
        enable_augmentation: bool = True,
        enable_translation: bool = True,
        enable_resolution: bool = True,
        num_color_perms: int = 10,
        keep_background: bool = True,
        difficulty_labels: Optional[Dict[str, float]] = None,
        seed: int = 42,
        repeat_factor: int = 1,
        re_arc_samples_per_task: int = 25,
    ) -> None:
        """
        Args:
            data_roots: Paths to ARC dataset roots.
            split: "training" or "evaluation".
            subset: "train" (leave-one-out / random) or "test" (full demos → test).
            canvas_size: Fixed canvas dimension.
            max_demos: Maximum demo pairs (pads shorter, truncates longer).
            enable_augmentation: Enable geometric + color augmentation.
            enable_translation: Enable random canvas offset.
            enable_resolution: Enable random resolution scaling.
            num_color_perms: Number of color permutation variants.
            keep_background: Keep color 0 fixed during permutation.
            difficulty_labels: Optional difficulty weight overrides.
            seed: Random seed.
            repeat_factor: Multiply effective dataset size.
            re_arc_samples_per_task: Samples per RE-ARC task (random selection).
        """
        super().__init__()
        self.canvas_size = canvas_size
        self.max_demos = max_demos
        self.subset = subset
        self.enable_augmentation = enable_augmentation
        self.enable_translation = enable_translation
        self.enable_resolution = enable_resolution
        self.keep_background = keep_background
        self.rng = np.random.RandomState(seed)
        self.re_arc_samples_per_task = re_arc_samples_per_task

        # Task storage
        self.tasks: List[Dict[str, Any]] = []
        self.samples: List[Tuple[int, int, str]] = []

        for root_path in data_roots:
            root = Path(root_path)

            # Determine difficulty for this source
            difficulty = 1.0
            if difficulty_labels:
                for pattern, weight in difficulty_labels.items():
                    if pattern in str(root):
                        difficulty = weight
                        break
            elif "AGI-2" in str(root):
                difficulty = 1.5

            # Detect format and load
            re_arc_dir = root / "tasks"
            arc_agi_dir = root / "data" / split

            if re_arc_dir.exists() and re_arc_dir.is_dir():
                self._load_re_arc(re_arc_dir, difficulty, str(root))
            elif arc_agi_dir.exists():
                self._load_arc_agi(arc_agi_dir, difficulty, str(root))
            else:
                print(f"Warning: {root} has no recognized format, skipping.")

        if not self.samples:
            raise RuntimeError(
                f"No samples found for split='{split}', subset='{subset}' "
                f"in roots: {data_roots}"
            )

        self.num_tasks = len(self.tasks)
        self.repeat_factor = max(1, repeat_factor)
        raw_count = len(self.samples)
        effective = raw_count * self.repeat_factor
        re_arc_tasks = sum(1 for t in self.tasks if t["format"] == "re_arc")
        agi_tasks = self.num_tasks - re_arc_tasks
        print(
            f"Loaded {raw_count} samples from {self.num_tasks} tasks "
            f"({agi_tasks} ARC-AGI + {re_arc_tasks} RE-ARC) "
            f"(repeat={self.repeat_factor}x → {effective} effective)"
        )

    # ─── Loaders ──────────────────────────────────────────────────

    def _load_arc_agi(
        self, split_dir: Path, difficulty: float, source: str
    ) -> None:
        """Load standard ARC-AGI format: {train: [...], test: [...]}."""
        files = sorted(split_dir.glob("*.json"))
        for file_path in files:
            with file_path.open("r") as fh:
                task_data = json.load(fh)

            task_name = file_path.stem
            task_idx = len(self.tasks)

            # Validate grid sizes
            all_examples = task_data.get("train", []) + task_data.get("test", [])
            if self._has_oversized_grids(all_examples):
                continue

            self.tasks.append({
                "task": task_data,
                "name": task_name,
                "difficulty": difficulty,
                "source": source,
                "format": "agi",
            })

            if self.subset == "train":
                train_examples = task_data.get("train", [])
                if len(train_examples) >= 2:
                    for i in range(len(train_examples)):
                        self.samples.append((task_idx, i, "loo"))
            else:
                test_examples = task_data.get("test", [])
                for i in range(len(test_examples)):
                    self.samples.append((task_idx, i, "test"))

    def _load_re_arc(
        self, tasks_dir: Path, difficulty: float, source: str
    ) -> None:
        """Load RE-ARC format: flat list of {input, output} dicts per task."""
        if self.subset != "train":
            return

        files = sorted(tasks_dir.glob("*.json"))
        for file_path in files:
            with file_path.open("r") as fh:
                examples = json.load(fh)

            if not isinstance(examples, list) or len(examples) < 2:
                continue

            task_name = f"re_arc_{file_path.stem}"
            task_idx = len(self.tasks)

            # Validate grid sizes
            if self._has_oversized_grids(examples):
                continue

            self.tasks.append({
                "examples": examples,
                "name": task_name,
                "difficulty": difficulty,
                "source": source,
                "format": "re_arc",
            })

            for i in range(self.re_arc_samples_per_task):
                self.samples.append((task_idx, i, "re_arc"))

    def _has_oversized_grids(self, examples: list) -> bool:
        """Check if any grid exceeds the maximum allowed size."""
        for ex in examples:
            for key in ("input", "output"):
                grid = ex.get(key, [])
                h = len(grid)
                w = len(grid[0]) if h > 0 else 0
                if h > 30 or w > 30:
                    return True
        return False

    # ─── Dataset interface ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples) * self.repeat_factor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = idx % len(self.samples)
        task_idx, query_idx, mode = self.samples[real_idx]
        task_info = self.tasks[task_idx]

        # ─── Build demo pairs and query ──────────────────────────
        if mode == "loo":
            all_train = task_info["task"]["train"]
            query_example = all_train[query_idx]
            demo_examples = [
                ex for i, ex in enumerate(all_train) if i != query_idx
            ]
        elif mode == "re_arc":
            all_examples = task_info["examples"]
            n = len(all_examples)
            k = min(self.max_demos + 1, n)
            chosen = self.rng.choice(n, size=k, replace=False)
            query_example = all_examples[chosen[-1]]
            demo_examples = [all_examples[chosen[i]] for i in range(k - 1)]
        else:
            demo_examples = task_info["task"]["train"]
            query_example = task_info["task"]["test"][query_idx]

        # Deep copy to avoid mutating the original data
        demos = [
            {"input": [r[:] for r in ex["input"]],
             "output": [r[:] for r in ex["output"]]}
            for ex in demo_examples
        ]
        query_in = [r[:] for r in query_example["input"]]
        query_out = (
            [r[:] for r in query_example["output"]]
            if "output" in query_example
            else None
        )

        # ─── Consistent task-level augmentation ──────────────────
        if self.enable_augmentation:
            geos = get_geometric_augmentations()
            geo_idx = self.rng.randint(0, len(geos))
            _, geo_fn = geos[geo_idx]
            color_perm = random_color_permutation(self.rng, self.keep_background)

            for d in demos:
                d["input"] = permute_colors(geo_fn(d["input"]), color_perm)
                d["output"] = permute_colors(geo_fn(d["output"]), color_perm)
            query_in = permute_colors(geo_fn(query_in), color_perm)
            if query_out is not None:
                query_out = permute_colors(geo_fn(query_out), color_perm)

        # ─── Find max grid dimensions across entire task ─────────
        all_grids = []
        for d in demos:
            all_grids.extend([d["input"], d["output"]])
        all_grids.append(query_in)
        if query_out is not None:
            all_grids.append(query_out)

        max_h = max(len(g) for g in all_grids)
        max_w = max(max(len(r) for r in g) for g in all_grids)

        # ─── Resolution scaling (same for all grids) ─────────────
        if self.enable_resolution:
            scale = random_scale_factor(max_h, max_w, self.canvas_size, self.rng)
        else:
            scale = 1

        if scale > 1:
            for d in demos:
                d["input"] = resolution_scale(d["input"], scale)
                d["output"] = resolution_scale(d["output"], scale)
            query_in = resolution_scale(query_in, scale)
            if query_out is not None:
                query_out = resolution_scale(query_out, scale)
            max_h *= scale
            max_w *= scale

        # ─── Canvas placement (same offset for all grids) ────────
        if self.enable_translation:
            x_off, y_off = random_offset(max_h, max_w, self.canvas_size, self.rng)
        else:
            x_off, y_off = 1, 1

        # ─── Place demo grids on canvas ──────────────────────────
        demo_input_canvases = []
        demo_output_canvases = []

        for d in demos[:self.max_demos]:
            d_in, _, _, _ = pad_grid_to_canvas(
                d["input"], self.canvas_size, x_off, y_off, mark_boundary=False
            )
            d_out, _, _, _ = pad_grid_to_canvas(
                d["output"], self.canvas_size, x_off, y_off, mark_boundary=True
            )
            demo_input_canvases.append(d_in)
            demo_output_canvases.append(d_out)

        num_demos = len(demo_input_canvases)

        # Pad to max_demos with IGNORE_INDEX canvases
        while len(demo_input_canvases) < self.max_demos:
            demo_input_canvases.append(
                torch.full((self.canvas_size, self.canvas_size),
                           IGNORE_INDEX, dtype=torch.long)
            )
            demo_output_canvases.append(
                torch.full((self.canvas_size, self.canvas_size),
                           IGNORE_INDEX, dtype=torch.long)
            )

        # ─── Place query grids on canvas ─────────────────────────
        q_in_canvas, _, _, _ = pad_grid_to_canvas(
            query_in, self.canvas_size, x_off, y_off, mark_boundary=False
        )

        if query_out is not None:
            q_out_canvas, q_mask, q_h, q_w = pad_grid_to_canvas(
                query_out, self.canvas_size, x_off, y_off, mark_boundary=True
            )
            q_target = q_out_canvas.clone()
            q_target[q_mask == 0] = IGNORE_INDEX
        else:
            q_target = torch.full(
                (self.canvas_size, self.canvas_size),
                IGNORE_INDEX, dtype=torch.long,
            )
            q_h, q_w = 0, 0

        return {
            "demo_inputs": torch.stack(demo_input_canvases),    # (K, 64, 64)
            "demo_outputs": torch.stack(demo_output_canvases),  # (K, 64, 64)
            "query_input": q_in_canvas,                         # (64, 64)
            "target": q_target,                                 # (64, 64)
            "num_demos": torch.tensor(num_demos, dtype=torch.long),
            "difficulty": torch.tensor(task_info["difficulty"], dtype=torch.float32),
            "task_name": task_info["name"],
            "offset": torch.tensor([x_off, y_off], dtype=torch.long),
            "scale_factor": torch.tensor(scale, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collation for task-level samples."""
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


def _resolve_data_roots(
    data_cfg: Dict[str, Any],
    data_sources: List[str],
) -> List[str]:
    """Resolve data source names to filesystem paths, skipping missing."""
    roots = []
    for source in data_sources:
        key = SOURCE_KEY_MAP.get(source)
        if key and data_cfg.get(key):
            path = data_cfg[key]
            if Path(path).exists():
                roots.append(path)
            else:
                print(f"Warning: {source} path '{path}' not found, skipping.")
    return roots


def build_dataloaders(
    config: Dict[str, Any],
    data_sources: Optional[List[str]] = None,
) -> Tuple["ARCTaskDataset", DataLoader, Optional["ARCTaskDataset"], Optional[DataLoader]]:
    """Build train and optional eval dataloaders from config.

    Args:
        config: Full config dict.
        data_sources: Optional list of source names (e.g. ["re_arc", "agi1"]).
                      If None, uses ["agi1", "agi2"] as default.

    Returns:
        (train_dataset, train_loader, eval_dataset, eval_loader)
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    # Resolve data roots
    if data_sources is None:
        data_sources = ["agi1", "agi2"]

    data_roots = _resolve_data_roots(data_cfg, data_sources)
    if not data_roots:
        raise RuntimeError(
            f"No valid data roots found for sources: {data_sources}"
        )

    canvas_size = data_cfg["canvas_size"]
    max_demos = data_cfg.get("max_demos", 5)
    aug_cfg = data_cfg["augmentation"]
    repeat_factor = data_cfg.get("repeat_factor", 1)
    re_arc_spt = data_cfg.get("re_arc_samples_per_task", 25)

    train_dataset = ARCTaskDataset(
        data_roots=data_roots,
        split="training",
        subset="train",
        canvas_size=canvas_size,
        max_demos=max_demos,
        enable_augmentation=aug_cfg["geometric"],
        enable_translation=aug_cfg["translation"],
        enable_resolution=aug_cfg["resolution_scaling"],
        num_color_perms=aug_cfg["num_color_perms"],
        keep_background=aug_cfg["keep_background"],
        repeat_factor=repeat_factor,
        re_arc_samples_per_task=re_arc_spt,
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

    # Validation always on real ARC data
    eval_dataset = None
    eval_loader = None

    eval_cfg = config.get("evaluation", {})
    if eval_cfg:
        val_sources = eval_cfg.get("val_data_sources", ["agi1", "agi2"])
        val_roots = _resolve_data_roots(data_cfg, val_sources)
        if val_roots:
            eval_dataset = ARCTaskDataset(
                data_roots=val_roots,
                split="training",
                subset="test",
                canvas_size=canvas_size,
                max_demos=max_demos,
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


def build_eval_dataloader(
    config: Dict[str, Any],
    data_sources: Optional[List[str]] = None,
) -> Tuple[Optional["ARCTaskDataset"], Optional[DataLoader]]:
    """Build only the eval dataloader (no train dataset created).

    Args:
        config: Full config dict.
        data_sources: Source names for eval data. Defaults to ["agi1", "agi2"].

    Returns:
        (eval_dataset, eval_loader) or (None, None) if no data found.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    if data_sources is None:
        eval_cfg = config.get("evaluation", {})
        data_sources = eval_cfg.get("val_data_sources", ["agi1", "agi2"])

    val_roots = _resolve_data_roots(data_cfg, data_sources)
    if not val_roots:
        return None, None

    canvas_size = data_cfg["canvas_size"]
    max_demos = data_cfg.get("max_demos", 5)

    eval_dataset = ARCTaskDataset(
        data_roots=val_roots,
        split="training",
        subset="test",
        canvas_size=canvas_size,
        max_demos=max_demos,
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

    return eval_dataset, eval_loader
