# ARC-IT: DINOv2 + Sana Hybrid for ARC-AGI

A hybrid neural architecture that solves abstract reasoning tasks (ARC-AGI) by combining:

- **DINOv2 Encoder (Frozen)** - Meta's self-supervised ViT-L/14 for visual feature extraction
- **Bridge Module (Trainable)** - Maps encoder features to diffusion space
- **Sana-0.6B Transformer (Trainable)** - Linear-attention diffusion backbone trained from scratch
- **Spatial Decoder (Trainable)** - Converts transformer output to discrete ARC grids
- **Test-Time Training (TTT)** - Per-task adaptation for each puzzle

## Architecture

```
Input Grid (variable size, up to 30x30)
    |
    v
Pad to 64x64 Canvas → Render as RGB → Upsample to 224x224
    |
    v
DINOv2 ViT-L/14 Encoder (FROZEN) → Patch features [B, 256, 1024]
    |
    v
Bridge Module (TRAINABLE) → [B, 256, 1152]
    |
    v
Sana-0.6B Transformer (TRAINABLE, random init) → x_0 prediction, conditioned on encoder
    |
    v
Spatial Decoder (TRAINABLE) → [B, 12, 64, 64] logits
    |
    v
argmax → Crop from canvas → Predicted Grid (original size)
```

## Project Structure

```
arc_it/
├── arc_it/                  # Main package
│   ├── data/                # Data loading, augmentation, canvas, rendering
│   ├── models/              # Encoder, Bridge, Sana backbone, Decoder
│   ├── training/            # Loss functions, training loops, optimization
│   ├── inference/           # TTT, multi-sample generation, evaluation
│   └── utils/               # Device detection, config, visualization
├── configs/                 # YAML configurations
│   ├── default.yaml         # H100 training defaults
│   └── mac_dev.yaml         # Mac development overrides
├── scripts/                 # Entry point scripts
├── tests/                   # Pytest test suite
├── References/              # (gitignored) Reference repositories
└── requirements.txt
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/REDDITARUN/arc_it.git
cd arc_it
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download reference datasets

The ARC-AGI datasets are required but not included in the repo. Use the setup script:

```bash
bash scripts/setup_data.sh
```

Or manually:

```bash
mkdir -p References && cd References
git clone --depth 1 https://github.com/fchollet/ARC-AGI.git
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git
cd ..
```

### 5. Configure API keys (for training + model upload)

```bash
cp .env.example .env
# Edit .env with your WANDB_API_KEY and HF_TOKEN
```

### 6. (Optional) Download reference repositories for study

```bash
cd References

# VARC - Vision ARC baseline
git clone https://github.com/arc-community/VARC.git

# Sana - NVIDIA diffusion transformer
git clone https://github.com/NVlabs/Sana.git

# JEPA World Models - Meta
git clone https://github.com/facebookresearch/jepa-wms.git

# NVARC - NVIDIA ARC solution
git clone https://github.com/NVIDIA/NVARC.git

# ARChitects solution
git clone https://github.com/ARChitectsDev/ARC2025_Solution_by_the_ARChitects.git

cd ..
```

## Cloud Quickstart (H100 / Colab)

One-shot setup on a fresh cloud machine:

```bash
git clone https://github.com/REDDITARUN/arc_it.git && cd arc_it
pip install -r requirements.txt
bash scripts/setup_data.sh
cp .env.example .env
# Edit .env with your WANDB_API_KEY and HF_TOKEN
python scripts/train.py --wandb --push-to-hf
```

## Quick Start

### Run tests (Mac/CPU)

```bash
python -m pytest tests/ -v
```

### End-to-End Demo

Runs the full pipeline (data loading, model build, training steps, inference, TTT) on Mac/CPU:

```bash
python scripts/demo.py
```

### Train (H100)

```bash
# Full 3-stage training (Bridge Alignment → Full Training → Hard Focus)
python scripts/train.py

# With W&B logging
python scripts/train.py --wandb

# Resume from checkpoint
python scripts/train.py --checkpoint checkpoints/best_stage2.pt
```

### Evaluate

```bash
# Standard evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_stage2.pt

# With test-time training (per-task fine-tuning)
python scripts/evaluate.py --checkpoint checkpoints/best_stage2.pt --ttt
```

### Load datasets and inspect

```python
from arc_it.data.dataset import ARCDataset

ds = ARCDataset(
    data_roots=["References/ARC-AGI", "References/ARC-AGI-2"],
    split="training",
    subset="train",
    canvas_size=64,
)
print(f"Loaded {len(ds)} samples from {ds.num_tasks} tasks")

sample = ds[0]
print(f"Input canvas: {sample['input_canvas'].shape}")     # (64, 64)
print(f"RGB for JEPA: {sample['input_rgb_224'].shape}")     # (3, 224, 224)
print(f"Target:       {sample['target'].shape}")            # (64, 64)
```

### Device auto-detection

```python
from arc_it.utils.device import device_info
print(device_info())
# Mac:  {'device': 'mps', 'dtype': 'torch.float32', ...}
# H100: {'device': 'cuda', 'gpu_name': 'NVIDIA H100', 'gpu_memory_gb': 80.0, ...}
```

## Development Workflow

- **Mac (local)**: Develop, test, iterate. Small batches, CPU/MPS. Config auto-adapts.
- **H100 (remote)**: `git clone` → `pip install` → run training scripts. Full batch sizes, bf16, multi-GPU.

## Training Strategy

**3-Stage Training (H100 Blitz)**:

| Stage | Duration | What | LR |
|-------|----------|------|----|
| 1. Bridge Alignment | ~45 min | Freeze Sana, train Bridge + Decoder only | 1e-4 |
| 2. Full Training | ~2.5 hrs | Unfreeze Sana, train on full augmented dataset | 5e-5 |
| 3. Hard Focus | ~1 hr | Lower LR, oversample AGI-2 tasks | 1e-5 |

## Test-Time Training (TTT)

For each evaluation task:
1. Snapshot model weights
2. Augment the 3-5 demonstration examples (geometric + color perms)
3. Fine-tune last N Sana layers for K steps on augmented demos
4. Generate multiple candidate predictions via diffusion stochasticity
5. Score candidates with heuristics (symmetry, color parsimony)
6. Restore original weights for next task

## Data Pipeline

- **ARC-AGI-1**: 400 training + 400 evaluation tasks (difficulty weight: 1.0)
- **ARC-AGI-2**: 1000 training + 120 evaluation tasks (difficulty weight: 1.5)
- **Augmentation**: 8 geometric (D4 symmetry) x 10 color permutations = 80x per example
- **Canvas**: 64x64 fixed size with translation + resolution scaling on-the-fly

## License

Research use only. See individual reference repositories for their respective licenses.
