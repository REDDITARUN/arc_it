"""Tests for training, evaluation, and TTT modules.

Run with: python -m pytest tests/test_training_and_inference.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from arc_it.data.canvas import IGNORE_INDEX
from arc_it.data.dataset import ARCTaskDataset, collate_fn
from arc_it.models.arc_it_model import ARCITModel
from arc_it.training.loss import compute_loss
from arc_it.inference.evaluate import Evaluator
from arc_it.inference.ttt import TestTimeTrainer


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def small_model():
    """Small model for fast testing."""
    return ARCITModel(
        num_colors=12,
        canvas_size=64,
        patch_size=4,
        hidden_size=64,
        rule_encoder_pair_layers=1,
        rule_encoder_agg_layers=1,
        rule_encoder_heads=2,
        num_rule_tokens=8,
        max_demos=5,
        rule_applier_layers=1,
        rule_applier_heads=2,
        decoder_channels=(32, 16),
        mlp_ratio=2.0,
    )


@pytest.fixture
def sample_task():
    return {
        "train": [
            {"input": [[0, 1, 2], [3, 4, 5]], "output": [[5, 4, 3], [2, 1, 0]]},
            {"input": [[1, 0], [3, 2]], "output": [[2, 3], [0, 1]]},
            {"input": [[4, 5], [6, 7]], "output": [[7, 6], [5, 4]]},
        ],
        "test": [
            {"input": [[0, 2], [1, 3]], "output": [[3, 1], [2, 0]]},
        ],
    }


@pytest.fixture
def dummy_dataset(sample_task):
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = Path(tmpdir) / "data" / "training"
        train_dir.mkdir(parents=True)
        for i in range(3):
            with open(train_dir / f"task_{i:03d}.json", "w") as f:
                json.dump(sample_task, f)

        eval_dir = Path(tmpdir) / "data" / "evaluation"
        eval_dir.mkdir(parents=True)
        with open(eval_dir / "task_000.json", "w") as f:
            json.dump(sample_task, f)

        yield tmpdir


# ─── Loss Function Tests ────────────────────────────────────────────

class TestLoss:
    def test_compute_loss_basic(self):
        logits = torch.randn(2, 12, 64, 64)
        target = torch.randint(0, 10, (2, 64, 64))
        result = compute_loss(logits, target)
        assert "loss" in result
        assert "pixel_accuracy" in result
        assert "grid_exact_match" in result
        assert torch.isfinite(result["loss"])

    def test_ignore_index_excluded(self):
        logits = torch.randn(1, 12, 64, 64)
        target = torch.full((1, 64, 64), IGNORE_INDEX, dtype=torch.long)
        target[0, 0, 0] = 0
        result = compute_loss(logits, target)
        assert torch.isfinite(result["loss"])

    def test_difficulty_weighting(self):
        logits = torch.randn(2, 12, 64, 64)
        target = torch.randint(0, 10, (2, 64, 64))
        r1 = compute_loss(logits, target, difficulty=None)
        r2 = compute_loss(logits, target, difficulty=torch.tensor([5.0, 5.0]))
        assert r2["loss"] > r1["loss"]

    def test_perfect_prediction(self):
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        logits = torch.zeros(1, 12, 64, 64)
        logits[0, 0, :, :] = 100.0
        result = compute_loss(logits, target)
        assert result["pixel_accuracy"].item() > 0.99
        assert result["grid_exact_match"].item() == 1.0


# ─── Dataset Tests ──────────────────────────────────────────────────

class TestARCTaskDataset:
    def test_basic_loading(self, dummy_dataset):
        ds = ARCTaskDataset(
            data_roots=[dummy_dataset],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
        )
        assert len(ds) > 0

    def test_sample_shape(self, dummy_dataset):
        ds = ARCTaskDataset(
            data_roots=[dummy_dataset],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
        )
        sample = ds[0]
        assert sample["demo_inputs"].shape == (5, 64, 64)
        assert sample["demo_outputs"].shape == (5, 64, 64)
        assert sample["query_input"].shape == (64, 64)
        assert sample["target"].shape == (64, 64)
        assert sample["num_demos"].item() >= 1

    def test_leave_one_out(self, dummy_dataset):
        ds = ARCTaskDataset(
            data_roots=[dummy_dataset],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
        )
        # With 3 tasks, each having 3 training examples, we get 3*3=9 samples
        assert len(ds) == 9

    def test_collation(self, dummy_dataset):
        ds = ARCTaskDataset(
            data_roots=[dummy_dataset],
            split="training",
            subset="train",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=2, collate_fn=collate_fn,
        )
        batch = next(iter(loader))
        assert batch["demo_inputs"].shape == (2, 5, 64, 64)
        assert batch["demo_outputs"].shape == (2, 5, 64, 64)
        assert batch["query_input"].shape == (2, 64, 64)
        assert batch["num_demos"].shape == (2,)


class TestREARCDataset:
    """Test RE-ARC flat-list format loading and sampling."""

    @pytest.fixture
    def re_arc_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_dir = Path(tmpdir) / "tasks"
            tasks_dir.mkdir()
            for i in range(2):
                examples = [
                    {"input": [[j % 10, (j+1) % 10], [(j+2) % 10, (j+3) % 10]],
                     "output": [[(j+3) % 10, (j+2) % 10], [(j+1) % 10, j % 10]]}
                    for j in range(20)
                ]
                with open(tasks_dir / f"task_{i:03d}.json", "w") as f:
                    json.dump(examples, f)
            yield tmpdir

    def test_re_arc_loads(self, re_arc_dir):
        ds = ARCTaskDataset(
            data_roots=[re_arc_dir],
            split="training", subset="train",
            canvas_size=64, max_demos=5,
            enable_augmentation=False,
            re_arc_samples_per_task=10,
        )
        assert ds.num_tasks == 2
        assert len(ds) == 20

    def test_re_arc_sample_valid(self, re_arc_dir):
        ds = ARCTaskDataset(
            data_roots=[re_arc_dir],
            split="training", subset="train",
            canvas_size=64, max_demos=5,
            enable_augmentation=False,
            re_arc_samples_per_task=3,
        )
        sample = ds[0]
        assert sample["demo_inputs"].shape == (5, 64, 64)
        assert sample["target"].shape == (64, 64)
        assert sample["task_name"].startswith("re_arc_")


# ─── Evaluator Tests ────────────────────────────────────────────────

class TestEvaluator:
    def test_evaluate_dataset(self, small_model, dummy_dataset):
        dataset = ARCTaskDataset(
            data_roots=[dummy_dataset],
            split="training",
            subset="test",
            canvas_size=64,
            max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
        )
        evaluator = Evaluator(small_model, device=torch.device("cpu"))
        results = evaluator.evaluate_dataset(dataset, batch_size=2)
        assert "pixel_accuracy" in results
        assert "grid_exact_match" in results
        assert results["total_grids"] > 0
        assert 0.0 <= results["pixel_accuracy"] <= 1.0

    def test_predict_single(self, small_model):
        evaluator = Evaluator(small_model, device=torch.device("cpu"))
        di = torch.randint(0, 12, (5, 64, 64))
        do = torch.randint(0, 12, (5, 64, 64))
        qi = torch.randint(0, 12, (64, 64))
        nd = torch.tensor(3)
        pred = evaluator.predict_single(di, do, qi, nd)
        assert pred.shape == (64, 64)
        assert pred.min() >= 0
        assert pred.max() < 12


# ─── TTT Tests ──────────────────────────────────────────────────────

class TestTTT:
    def test_predict_task(self, small_model, sample_task):
        ttt = TestTimeTrainer(
            model=small_model,
            ttt_steps=3,
            ttt_lr=1e-4,
            ttt_batch_size=4,
            num_candidates=2,
            canvas_size=64,
            max_demos=5,
            device=torch.device("cpu"),
        )
        predictions = ttt.predict_task(sample_task, num_attempts=2)
        assert len(predictions) > 0
        assert len(predictions) <= 2
        for pred in predictions:
            assert isinstance(pred, list)
            assert len(pred) > 0

    def test_weights_restored_after_ttt(self, small_model, sample_task):
        before = {k: v.clone() for k, v in small_model.state_dict().items()}

        ttt = TestTimeTrainer(
            model=small_model,
            ttt_steps=3,
            num_candidates=1,
            canvas_size=64,
            max_demos=5,
            device=torch.device("cpu"),
        )
        ttt.predict_task(sample_task, num_attempts=1)

        after = small_model.state_dict()
        for key in before:
            assert torch.allclose(before[key], after[key]), f"Weight changed: {key}"

    def test_ttt_data_preparation(self, small_model, sample_task):
        ttt = TestTimeTrainer(
            model=small_model,
            canvas_size=64,
            max_demos=5,
            device=torch.device("cpu"),
        )
        data = ttt._prepare_ttt_data(sample_task["train"])
        assert "demo_inputs" in data
        assert "demo_outputs" in data
        assert "query_input" in data
        assert "target" in data
        assert "num_demos" in data
        # 3 examples * 8 geometric augmentations = 24 leave-one-out samples
        assert data["demo_inputs"].shape[0] == 24
        assert data["demo_inputs"].shape[1] == 5  # max_demos
        assert data["demo_inputs"].shape[2:] == (64, 64)
        assert data["query_input"].shape[1:] == (64, 64)

    def test_candidate_scoring(self, small_model):
        ttt = TestTimeTrainer(
            model=small_model, canvas_size=64, max_demos=5,
            device=torch.device("cpu"),
        )
        sym_grid = [[1, 2, 1], [3, 4, 3], [1, 2, 1]]
        random_grid = [[0, 5, 3], [8, 1, 7], [4, 9, 2]]
        sym_score = ttt._compute_grid_score(sym_grid)
        rand_score = ttt._compute_grid_score(random_grid)
        assert sym_score > rand_score


# ─── Integration: Train → Evaluate Cycle ─────────────────────────────

class TestTrainEvalCycle:
    def test_train_then_evaluate(self, small_model, dummy_dataset):
        train_ds = ARCTaskDataset(
            data_roots=[dummy_dataset],
            split="training", subset="train",
            canvas_size=64, max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
        )
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=2, collate_fn=collate_fn,
        )
        small_model.train()
        optimizer = torch.optim.AdamW(small_model.get_trainable_params(), lr=1e-3)

        batch = next(iter(loader))
        for _ in range(3):
            optimizer.zero_grad()
            result = small_model(
                batch["demo_inputs"],
                batch["demo_outputs"],
                batch["query_input"],
                batch["num_demos"],
                target=batch["target"],
            )
            result["loss"].backward()
            optimizer.step()

        eval_ds = ARCTaskDataset(
            data_roots=[dummy_dataset],
            split="training", subset="test",
            canvas_size=64, max_demos=5,
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
        )
        evaluator = Evaluator(small_model, device=torch.device("cpu"))
        results = evaluator.evaluate_dataset(eval_ds, batch_size=2)
        assert results["total_grids"] > 0
        assert 0.0 <= results["pixel_accuracy"] <= 1.0
