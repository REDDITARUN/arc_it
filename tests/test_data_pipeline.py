"""Smoke tests for the ARC-IT data pipeline.

Run with: python -m pytest tests/test_data_pipeline.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from arc_it.data.canvas import (
    IGNORE_INDEX,
    PAD_INDEX,
    NUM_COLORS,
    pad_grid_to_canvas,
    random_offset,
    resolution_scale,
    crop_prediction_from_canvas,
)
from arc_it.data.augmentation import (
    rotate_grid,
    flip_grid,
    transpose_grid,
    permute_colors,
    random_color_permutation,
    inverse_color_permutation,
    get_geometric_augmentations,
    augment_task,
    generate_all_augmentations,
)
from arc_it.data.rendering import (
    render_grid_to_rgb,
    render_canvas_to_rgb_224,
    batch_render_canvas_to_rgb_224,
)


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_grid():
    """A simple 3x4 ARC grid."""
    return [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 0, 1],
    ]


@pytest.fixture
def sample_task():
    """A minimal ARC task with 3 train + 1 test example."""
    return {
        "train": [
            {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
            {"input": [[1, 0], [3, 2]], "output": [[2, 3], [0, 1]]},
            {"input": [[4, 5], [6, 7]], "output": [[7, 6], [5, 4]]},
        ],
        "test": [
            {"input": [[0, 2], [1, 3]], "output": [[3, 1], [2, 0]]},
        ],
    }


@pytest.fixture
def arc_data_dir(sample_task):
    """Create a temporary ARC dataset directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = Path(tmpdir) / "data" / "training"
        train_dir.mkdir(parents=True)
        with open(train_dir / "test_task_001.json", "w") as f:
            json.dump(sample_task, f)
        with open(train_dir / "test_task_002.json", "w") as f:
            json.dump(sample_task, f)

        eval_dir = Path(tmpdir) / "data" / "evaluation"
        eval_dir.mkdir(parents=True)
        with open(eval_dir / "test_task_001.json", "w") as f:
            json.dump(sample_task, f)

        yield tmpdir


# ─── Canvas Tests ───────────────────────────────────────────────────

class TestCanvas:
    def test_pad_basic(self, sample_grid):
        canvas, mask, h, w = pad_grid_to_canvas(sample_grid, canvas_size=64)
        assert canvas.shape == (64, 64)
        assert mask.shape == (64, 64)
        assert h == 3
        assert w == 4
        assert canvas.dtype == torch.long
        assert canvas[1, 1].item() == 0
        assert canvas[1, 2].item() == 1
        assert canvas[2, 1].item() == 4

    def test_pad_with_boundary(self, sample_grid):
        canvas, mask, h, w = pad_grid_to_canvas(
            sample_grid, canvas_size=64, mark_boundary=True
        )
        assert canvas[1, 5].item() == PAD_INDEX
        assert canvas[4, 1].item() == PAD_INDEX

    def test_pad_ignore_fills(self, sample_grid):
        canvas, _, _, _ = pad_grid_to_canvas(sample_grid, canvas_size=64)
        assert canvas[0, 0].item() == IGNORE_INDEX
        assert canvas[63, 63].item() == IGNORE_INDEX

    def test_resolution_scale(self):
        grid = [[0, 1], [2, 3]]
        scaled = resolution_scale(grid, 2)
        assert len(scaled) == 4
        assert len(scaled[0]) == 4
        assert scaled[0][0] == 0
        assert scaled[0][1] == 0
        assert scaled[1][0] == 0
        assert scaled[0][2] == 1

    def test_crop_prediction(self):
        canvas = torch.full((64, 64), IGNORE_INDEX, dtype=torch.long)
        canvas[3, 2] = 1
        canvas[3, 3] = 2
        canvas[3, 4] = 3
        canvas[4, 2] = 4
        canvas[4, 3] = 5
        canvas[4, 4] = 6
        canvas[3, 5] = PAD_INDEX
        canvas[4, 5] = PAD_INDEX
        canvas[5, 2] = PAD_INDEX
        canvas[5, 3] = PAD_INDEX
        canvas[5, 4] = PAD_INDEX
        canvas[5, 5] = PAD_INDEX

        result = crop_prediction_from_canvas(canvas, x_offset=2, y_offset=3)
        assert result == [[1, 2, 3], [4, 5, 6]]


# ─── Augmentation Tests ────────────────────────────────────────────

class TestAugmentation:
    def test_rotate_90(self, sample_grid):
        rotated = rotate_grid(sample_grid, k=1)
        assert len(rotated) == 4
        assert len(rotated[0]) == 3

    def test_rotate_360_is_identity(self, sample_grid):
        rotated = rotate_grid(sample_grid, k=4)
        assert rotated == sample_grid

    def test_flip_horizontal(self, sample_grid):
        flipped = flip_grid(sample_grid, axis=1)
        assert flipped[0] == [3, 2, 1, 0]

    def test_flip_vertical(self, sample_grid):
        flipped = flip_grid(sample_grid, axis=0)
        assert flipped[0] == [8, 9, 0, 1]

    def test_transpose(self):
        grid = [[1, 2], [3, 4]]
        transposed = transpose_grid(grid)
        assert transposed == [[1, 3], [2, 4]]

    def test_all_geometrics_produce_valid_grids(self, sample_grid):
        geos = get_geometric_augmentations()
        assert len(geos) == 8
        for name, fn in geos:
            result = fn(sample_grid)
            assert isinstance(result, list)
            assert len(result) > 0
            assert len(result[0]) > 0

    def test_color_permutation(self):
        grid = [[0, 1, 2], [3, 4, 5]]
        perm = [0, 5, 4, 3, 2, 1, 6, 7, 8, 9]
        result = permute_colors(grid, perm)
        assert result[0][1] == 5
        assert result[0][2] == 4
        assert result[1][0] == 3

    def test_color_perm_inverse(self):
        perm = random_color_permutation(np.random.RandomState(42))
        inv = inverse_color_permutation(perm)
        for i in range(10):
            assert inv[perm[i]] == i

    def test_keep_background(self):
        for _ in range(20):
            perm = random_color_permutation(np.random.RandomState(_), keep_background=True)
            assert perm[0] == 0

    def test_augment_task_preserves_structure(self, sample_task):
        from arc_it.data.augmentation import identity
        aug = augment_task(sample_task, identity, None)
        assert len(aug["train"]) == len(sample_task["train"])
        assert len(aug["test"]) == len(sample_task["test"])
        assert "input" in aug["train"][0]
        assert "output" in aug["train"][0]

    def test_generate_all_augmentations(self, sample_task):
        results = generate_all_augmentations(sample_task, num_color_perms=3)
        assert len(results) == 8 * 3
        for aug_task, meta in results:
            assert "train" in aug_task
            assert "geometric" in meta
            assert "color_perm" in meta


# ─── Rendering Tests ───────────────────────────────────────────────

class TestRendering:
    def test_render_grid_to_rgb(self, sample_grid):
        canvas, _, _, _ = pad_grid_to_canvas(sample_grid, canvas_size=64)
        rgb = render_grid_to_rgb(canvas)
        assert rgb.shape == (3, 64, 64)
        assert rgb.dtype == torch.float32
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_render_canvas_to_rgb_224(self, sample_grid):
        canvas, _, _, _ = pad_grid_to_canvas(sample_grid, canvas_size=64)
        rgb_224 = render_canvas_to_rgb_224(canvas, normalize=True)
        assert rgb_224.shape == (3, 224, 224)
        assert rgb_224.dtype == torch.float32

    def test_render_canvas_to_rgb_224_unnormalized(self, sample_grid):
        canvas, _, _, _ = pad_grid_to_canvas(sample_grid, canvas_size=64)
        rgb_224 = render_canvas_to_rgb_224(canvas, normalize=False)
        assert rgb_224.min() >= 0.0
        assert rgb_224.max() <= 1.0

    def test_batch_render(self, sample_grid):
        canvas, _, _, _ = pad_grid_to_canvas(sample_grid, canvas_size=64)
        batch = torch.stack([canvas, canvas, canvas])
        rgb_batch = batch_render_canvas_to_rgb_224(batch, normalize=True)
        assert rgb_batch.shape == (3, 3, 224, 224)


# ─── Dataset Integration Test ──────────────────────────────────────

class TestDatasetIntegration:
    def test_dataset_loads(self, arc_data_dir):
        from arc_it.data.dataset import ARCTaskDataset
        ds = ARCTaskDataset(
            data_roots=[arc_data_dir],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=True,
            enable_translation=True,
            enable_resolution=False,
        )
        assert len(ds) > 0
        assert ds.num_tasks == 2

    def test_dataset_getitem(self, arc_data_dir):
        from arc_it.data.dataset import ARCTaskDataset
        ds = ARCTaskDataset(
            data_roots=[arc_data_dir],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
        )
        sample = ds[0]
        assert sample["demo_inputs"].shape == (5, 64, 64)
        assert sample["demo_outputs"].shape == (5, 64, 64)
        assert sample["query_input"].shape == (64, 64)
        assert sample["target"].shape == (64, 64)
        assert sample["num_demos"].dtype == torch.long
        assert sample["difficulty"].dtype == torch.float32
        assert sample["scale_factor"].item() == 1

    def test_dataloader_batching(self, arc_data_dir):
        from arc_it.data.dataset import ARCTaskDataset, collate_fn
        ds = ARCTaskDataset(
            data_roots=[arc_data_dir],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=2, collate_fn=collate_fn
        )
        batch = next(iter(loader))
        assert batch["demo_inputs"].shape == (2, 5, 64, 64)
        assert batch["demo_outputs"].shape == (2, 5, 64, 64)
        assert batch["query_input"].shape == (2, 64, 64)
        assert batch["target"].shape == (2, 64, 64)
        assert batch["difficulty"].shape == (2,)

    def test_target_has_ignore_index(self, arc_data_dir):
        from arc_it.data.dataset import ARCTaskDataset
        ds = ARCTaskDataset(
            data_roots=[arc_data_dir],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
        )
        sample = ds[0]
        target = sample["target"]
        ignore_count = (target == IGNORE_INDEX).sum().item()
        assert ignore_count > 64 * 64 - 20


# ─── RE-ARC Dataset Tests ─────────────────────────────────────────

class TestREARCDataset:
    @pytest.fixture
    def re_arc_dir(self):
        """Create a temporary RE-ARC directory with flat-list format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_dir = Path(tmpdir) / "tasks"
            tasks_dir.mkdir()

            for i in range(3):
                examples = [
                    {"input": [[j, j+1], [j+2, j+3]],
                     "output": [[j+3, j+2], [j+1, j]]}
                    for j in range(10)
                ]
                with open(tasks_dir / f"task_{i:03d}.json", "w") as f:
                    json.dump(examples, f)

            yield tmpdir

    def test_re_arc_loads(self, re_arc_dir):
        from arc_it.data.dataset import ARCTaskDataset
        ds = ARCTaskDataset(
            data_roots=[re_arc_dir],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
            re_arc_samples_per_task=5,
        )
        assert ds.num_tasks == 3
        assert len(ds) == 15  # 3 tasks × 5 samples each

    def test_re_arc_sample_shape(self, re_arc_dir):
        from arc_it.data.dataset import ARCTaskDataset
        ds = ARCTaskDataset(
            data_roots=[re_arc_dir],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
            re_arc_samples_per_task=3,
        )
        sample = ds[0]
        assert sample["demo_inputs"].shape == (5, 64, 64)
        assert sample["demo_outputs"].shape == (5, 64, 64)
        assert sample["query_input"].shape == (64, 64)
        assert sample["target"].shape == (64, 64)
        assert sample["num_demos"].item() >= 1

    def test_re_arc_mixed_with_agi(self, re_arc_dir, arc_data_dir):
        from arc_it.data.dataset import ARCTaskDataset
        ds = ARCTaskDataset(
            data_roots=[re_arc_dir, arc_data_dir],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
            re_arc_samples_per_task=5,
        )
        # 3 RE-ARC tasks × 5 + 2 AGI tasks × 3 (leave-one-out)
        assert ds.num_tasks == 5
        assert len(ds) == 15 + 6

    def test_re_arc_ignored_for_eval(self, re_arc_dir):
        """RE-ARC should be skipped for evaluation subset."""
        from arc_it.data.dataset import ARCTaskDataset
        with pytest.raises(RuntimeError, match="No samples found"):
            ARCTaskDataset(
                data_roots=[re_arc_dir],
                split="training",
                subset="test",
                canvas_size=64,
                max_demos=5,
                enable_augmentation=False,
            )

    def test_re_arc_task_name_prefix(self, re_arc_dir):
        from arc_it.data.dataset import ARCTaskDataset
        ds = ARCTaskDataset(
            data_roots=[re_arc_dir],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
            re_arc_samples_per_task=2,
        )
        sample = ds[0]
        assert sample["task_name"].startswith("re_arc_")
