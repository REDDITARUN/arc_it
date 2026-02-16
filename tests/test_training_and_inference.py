"""Tests for training, evaluation, and TTT modules.

Run with: python -m pytest tests/test_training_and_inference.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from arc_it.data.canvas import IGNORE_INDEX
from arc_it.data.dataset import ARCDataset, collate_fn
from arc_it.models.arc_it_model import ARCITModel
from arc_it.training.loss import compute_loss
from arc_it.inference.evaluate import Evaluator
from arc_it.inference.ttt import TestTimeTrainer


# ─── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def small_model():
    """Small model for fast testing."""
    return ARCITModel(
        encoder_name="stub",
        encoder_dim=1024,
        encoder_pretrained=False,
        sana_hidden=128,
        sana_depth=1,
        sana_self_attn_heads=4,
        sana_self_attn_head_dim=32,
        sana_cross_attn_heads=2,
        sana_mlp_ratio=2.0,
        canvas_size=64,
        num_colors=12,
        decoder_channels=(64, 32),
        num_train_timesteps=50,
        num_inference_steps=3,
        output_patch_size=4,
    )


@pytest.fixture
def sample_task():
    return {
        "train": [
            {"input": [[0, 1, 2], [3, 4, 5]], "output": [[5, 4, 3], [2, 1, 0]]},
            {"input": [[1, 0], [3, 2]], "output": [[2, 3], [0, 1]]},
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
        # All targets are IGNORE_INDEX except one pixel
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
        # Create logits that strongly predict the target
        target = torch.zeros(1, 64, 64, dtype=torch.long)
        logits = torch.zeros(1, 12, 64, 64)
        logits[0, 0, :, :] = 100.0  # Very high confidence for class 0
        result = compute_loss(logits, target)
        assert result["pixel_accuracy"].item() > 0.99
        assert result["grid_exact_match"].item() == 1.0


# ─── Evaluator Tests ────────────────────────────────────────────────

class TestEvaluator:
    def test_evaluate_dataset(self, small_model, dummy_dataset):
        dataset = ARCDataset(
            data_roots=[dummy_dataset],
            split="training",
            subset="train",
            canvas_size=64,
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
        input_rgb = torch.randn(3, 224, 224)
        pred = evaluator.predict_single(input_rgb)
        assert pred.shape == (64, 64)
        assert pred.min() >= 0
        assert pred.max() < 12


# ─── TTT Tests ──────────────────────────────────────────────────────

class TestTTT:
    def test_predict_task(self, small_model, sample_task):
        ttt = TestTimeTrainer(
            model=small_model,
            ttt_steps=3,         # Very few for speed
            ttt_lr=1e-4,
            ttt_batch_size=4,
            num_layers_to_update=1,
            num_candidates=2,    # Few candidates for speed
            canvas_size=64,
            device=torch.device("cpu"),
        )
        predictions = ttt.predict_task(sample_task, num_attempts=2)
        assert len(predictions) > 0
        assert len(predictions) <= 2
        for pred in predictions:
            assert isinstance(pred, list)
            assert len(pred) > 0

    def test_weights_restored_after_ttt(self, small_model, sample_task):
        """Model weights should be identical before and after TTT."""
        before = {k: v.clone() for k, v in small_model.state_dict().items()}

        ttt = TestTimeTrainer(
            model=small_model,
            ttt_steps=3,
            num_candidates=1,
            canvas_size=64,
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
            device=torch.device("cpu"),
        )
        data = ttt._prepare_ttt_data(sample_task["train"])
        assert "input_rgb_224" in data
        assert "target" in data
        # 2 examples * 8 geometric augmentations = 16 samples
        assert data["input_rgb_224"].shape[0] == 16
        assert data["target"].shape[0] == 16
        assert data["input_rgb_224"].shape[1:] == (3, 224, 224)
        assert data["target"].shape[1:] == (64, 64)

    def test_candidate_scoring(self, small_model):
        ttt = TestTimeTrainer(model=small_model, canvas_size=64, device=torch.device("cpu"))

        # Symmetric grid should score higher
        sym_grid = [[1, 2, 1], [3, 4, 3], [1, 2, 1]]
        random_grid = [[0, 5, 3], [8, 1, 7], [4, 9, 2]]

        sym_score = ttt._compute_grid_score(sym_grid)
        rand_score = ttt._compute_grid_score(random_grid)
        assert sym_score > rand_score


# ─── Integration: Train → Evaluate Cycle ─────────────────────────────

class TestTrainEvalCycle:
    def test_train_then_evaluate(self, small_model, dummy_dataset):
        """Full cycle: train a few steps, then evaluate."""
        # Train
        dataset = ARCDataset(
            data_roots=[dummy_dataset],
            split="training", subset="train",
            canvas_size=64, enable_augmentation=False,
            enable_translation=False, enable_resolution=False,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, collate_fn=collate_fn,
        )
        small_model.train()
        optimizer = torch.optim.AdamW(small_model.get_trainable_params(), lr=1e-3)

        batch = next(iter(loader))
        for _ in range(3):
            optimizer.zero_grad()
            result = small_model(batch["input_rgb_224"], target=batch["target"])
            result["loss"].backward()
            optimizer.step()

        # Evaluate
        evaluator = Evaluator(small_model, device=torch.device("cpu"))
        results = evaluator.evaluate_dataset(dataset, batch_size=2)
        assert results["total_grids"] > 0
        assert 0.0 <= results["pixel_accuracy"] <= 1.0
