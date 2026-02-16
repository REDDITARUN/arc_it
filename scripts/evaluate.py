#!/usr/bin/env python3
"""ARC-IT Evaluation Script.

Usage:
    # Evaluate without TTT:
    python scripts/evaluate.py --checkpoint checkpoints/best_stage2.pt

    # Evaluate with TTT:
    python scripts/evaluate.py --checkpoint checkpoints/best_stage2.pt --ttt

    # Evaluate on specific split:
    python scripts/evaluate.py --checkpoint checkpoints/best_stage2.pt --split evaluation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import json
import torch
from arc_it.utils.config import load_config
from arc_it.utils.device import device_info
from arc_it.data.dataset import ARCDataset
from arc_it.models.arc_it_model import ARCITModel
from arc_it.inference.evaluate import Evaluator
from arc_it.inference.ttt import TestTimeTrainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC-IT model")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="evaluation", help="Dataset split to evaluate")
    parser.add_argument("--ttt", action="store_true", help="Enable test-time training")
    parser.add_argument("--output", default="outputs/eval_results.json", help="Output path")
    args = parser.parse_args()

    config = load_config(args.config)
    info = device_info()
    print(f"Evaluating on {info['device']}")

    # Load model
    model = ARCITModel.from_config(config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    data_cfg = config["data"]
    data_roots = [data_cfg["arc_agi1_path"], data_cfg["arc_agi2_path"]]

    if args.ttt:
        # TTT mode: evaluate per-task with fine-tuning
        ttt_cfg = config.get("ttt", {})
        ttt = TestTimeTrainer(
            model=model,
            ttt_steps=ttt_cfg.get("steps", 100),
            ttt_lr=ttt_cfg.get("lr", 1e-4),
            ttt_batch_size=ttt_cfg.get("batch_size", 8),
            num_layers_to_update=ttt_cfg.get("num_layers_to_update", 4),
            num_candidates=ttt_cfg.get("num_candidates", 32),
            canvas_size=data_cfg["canvas_size"],
        )

        # Load raw tasks for TTT
        results = evaluate_with_ttt(ttt, data_roots, args.split)
    else:
        # Standard evaluation
        dataset = ARCDataset(
            data_roots=data_roots,
            split=args.split,
            subset="test",
            canvas_size=data_cfg["canvas_size"],
            enable_augmentation=False,
            enable_translation=False,
            enable_resolution=False,
        )
        evaluator = Evaluator(model)
        results = evaluator.evaluate_dataset(dataset)

    print(f"\n{'='*60}")
    print(f"Results ({args.split}):")
    print(f"  Pixel Accuracy:   {results['pixel_accuracy']:.3f}")
    print(f"  Grid Exact Match: {results['grid_exact_match']:.3f}")
    print(f"  Total Grids:      {results['total_grids']}")
    print(f"  Total Time:       {results['total_time']:.1f}s")
    print(f"{'='*60}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "per_task"}, f, indent=2)
    print(f"Saved to {args.output}")


def evaluate_with_ttt(ttt, data_roots, split):
    """Evaluate with TTT: load raw tasks and predict each."""
    from pathlib import Path
    import time

    total_correct = 0
    total_tasks = 0
    t_start = time.time()

    for root in data_roots:
        split_dir = Path(root) / "data" / split
        if not split_dir.exists():
            continue
        for task_file in sorted(split_dir.glob("*.json")):
            with open(task_file) as f:
                task = json.load(f)

            predictions = ttt.predict_task(task, num_attempts=2)

            # Check if any prediction matches ground truth
            for test_ex in task.get("test", []):
                gt = test_ex.get("output")
                if gt is None:
                    continue
                total_tasks += 1
                for pred in predictions:
                    if pred == gt:
                        total_correct += 1
                        break

    total_time = time.time() - t_start
    return {
        "pixel_accuracy": 0.0,  # N/A for TTT mode
        "grid_exact_match": total_correct / max(total_tasks, 1),
        "total_grids": total_tasks,
        "total_exact_match": total_correct,
        "total_time": total_time,
        "avg_time_per_grid": total_time / max(total_tasks, 1),
    }


if __name__ == "__main__":
    main()
