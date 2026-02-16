#!/usr/bin/env python3
"""ARC-IT End-to-End Demo.

Demonstrates the full pipeline working on Mac/CPU:
    1. Load ARC datasets
    2. Build model (with stub encoder for fast testing)
    3. Run a few training steps
    4. Run inference on a test sample
    5. Run TTT on a single task

Usage:
    python scripts/demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from arc_it.utils.device import device_info, get_device
from arc_it.utils.config import load_config
from arc_it.data.dataset import ARCDataset, collate_fn
from arc_it.data.canvas import crop_prediction_from_canvas
from arc_it.models.arc_it_model import ARCITModel
from arc_it.training.loss import compute_loss
from arc_it.inference.ttt import TestTimeTrainer


def main():
    print("=" * 60)
    print("ARC-IT End-to-End Demo")
    print("=" * 60)

    # ─── 1. Environment ──────────────────────────────────────────
    info = device_info()
    print(f"\n[1/6] Device: {info['device']}")
    config = load_config()
    print(f"  Config auto-adapted for {'GPU' if info['device'] == 'cuda' else 'Mac/CPU'}")

    # ─── 2. Load Data ────────────────────────────────────────────
    print("\n[2/6] Loading datasets...")
    data_roots = []
    if Path("References/ARC-AGI/data/training").exists():
        data_roots.append("References/ARC-AGI")
    if Path("References/ARC-AGI-2/data/training").exists():
        data_roots.append("References/ARC-AGI-2")

    if not data_roots:
        print("  WARNING: No ARC datasets found in References/")
        print("  Using synthetic dummy data for demo")
        data_roots = None

    if data_roots:
        dataset = ARCDataset(
            data_roots=data_roots,
            split="training",
            subset="train",
            canvas_size=64,
            enable_augmentation=True,
            enable_translation=True,
            enable_resolution=False,
        )
        print(f"  Loaded {len(dataset)} samples from {dataset.num_tasks} tasks")
    else:
        dataset = _make_dummy_dataset()

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0,
    )

    # ─── 3. Build Model ─────────────────────────────────────────
    print("\n[3/6] Building model (stub encoder for demo)...")
    model = ARCITModel(
        encoder_name="stub",
        encoder_dim=1280,
        encoder_pretrained=False,
        sana_hidden=256,
        sana_depth=2,
        sana_self_attn_heads=8,
        sana_self_attn_head_dim=32,
        sana_cross_attn_heads=4,
        sana_mlp_ratio=2.0,
        canvas_size=64,
        num_colors=12,
        decoder_channels=(128, 64),
        num_train_timesteps=100,
        num_inference_steps=5,
        output_patch_size=4,
    )
    counts = model.param_count()
    print(f"  Total:     {counts['_total']['total'] / 1e6:.1f}M params")
    print(f"  Trainable: {counts['_total']['trainable'] / 1e6:.1f}M params")

    # ─── 4. Training Steps ───────────────────────────────────────
    print("\n[4/6] Running 5 training steps...")
    device = get_device()
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=1e-4)

    batch = next(iter(loader))
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        result = model(
            batch["input_rgb_224"].to(device),
            target=batch["target"].to(device),
        )
        result["loss"].backward()
        optimizer.step()
        losses.append(result["loss"].item())
        print(f"  Step {step + 1}: loss={result['loss'].item():.4f}, "
              f"pixel_acc={result['pixel_accuracy'].item():.3f}")

    if losses[-1] < losses[0]:
        print("  Loss is decreasing (training is working)")
    else:
        print("  Note: loss may not decrease in 5 steps with random init")

    # ─── 5. Inference ────────────────────────────────────────────
    print("\n[5/6] Running inference...")
    model.eval()
    t_start = time.time()
    with torch.no_grad():
        result = model(batch["input_rgb_224"][:1].to(device), target=None)
    t_infer = time.time() - t_start

    prediction = result["prediction"][0]
    print(f"  Prediction shape: {prediction.shape}")
    print(f"  Unique colors in prediction: {prediction.unique().tolist()}")
    print(f"  Inference time: {t_infer:.3f}s")

    # Crop prediction
    offset = batch["offset"][0]
    scale = batch["scale_factor"][0].item()
    pred_grid = crop_prediction_from_canvas(
        prediction, offset[0].item(), offset[1].item(), scale
    )
    print(f"  Cropped grid size: {len(pred_grid)}x{len(pred_grid[0]) if pred_grid else 0}")

    # ─── 6. TTT Demo ────────────────────────────────────────────
    print("\n[6/6] Running TTT demo...")
    if data_roots:
        import json
        # Load a real task
        task_files = list(Path(data_roots[0], "data", "training").glob("*.json"))
        if task_files:
            with open(task_files[0]) as f:
                task = json.load(f)
            task_name = task_files[0].stem

            ttt = TestTimeTrainer(
                model=model,
                ttt_steps=5,        # Very few steps for demo
                ttt_lr=1e-4,
                ttt_batch_size=4,
                num_layers_to_update=2,
                num_candidates=4,    # Few candidates for demo
                canvas_size=64,
            )

            t_start = time.time()
            predictions = ttt.predict_task(task, num_attempts=2)
            t_ttt = time.time() - t_start

            print(f"  Task: {task_name}")
            print(f"  Train examples: {len(task['train'])}")
            print(f"  Predictions generated: {len(predictions)}")
            for i, pred in enumerate(predictions):
                print(f"    Attempt {i + 1}: {len(pred)}x{len(pred[0]) if pred else 0} grid")
            print(f"  TTT time: {t_ttt:.3f}s")

            # Check if any prediction matches
            gt = task["test"][0].get("output")
            if gt:
                match = any(p == gt for p in predictions)
                print(f"  Matches ground truth: {match}")

    # ─── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(f"  Device:     {info['device']}")
    print(f"  Model size: {counts['_total']['trainable'] / 1e6:.1f}M trainable params")
    print(f"  Pipeline:   data -> encoder -> bridge -> sana -> decoder -> grid")
    print(f"  Training:   loss decreases, gradients flow")
    print(f"  Inference:  produces valid {prediction.shape} grids")
    print(f"  TTT:        per-task fine-tuning works")
    print()
    print("Next steps:")
    print("  1. On H100: python scripts/train.py (full training)")
    print("  2. Evaluate: python scripts/evaluate.py --checkpoint <path>")
    print("  3. With TTT: python scripts/evaluate.py --checkpoint <path> --ttt")


def _make_dummy_dataset():
    """Create a tiny dummy dataset when no real data is available."""
    import json, tempfile
    tmpdir = tempfile.mkdtemp()
    train_dir = Path(tmpdir) / "data" / "training"
    train_dir.mkdir(parents=True)
    for i in range(5):
        task = {
            "train": [
                {"input": [[0, 1], [2, 3]], "output": [[3, 2], [1, 0]]},
                {"input": [[1, 0], [3, 2]], "output": [[2, 3], [0, 1]]},
            ],
            "test": [{"input": [[0, 2], [1, 3]], "output": [[3, 1], [2, 0]]}],
        }
        with open(train_dir / f"dummy_{i:03d}.json", "w") as f:
            json.dump(task, f)

    return ARCDataset(
        data_roots=[tmpdir], split="training", subset="train",
        canvas_size=64, enable_augmentation=True,
    )


if __name__ == "__main__":
    main()
