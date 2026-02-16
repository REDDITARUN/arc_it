# ARC-IT: JEPA + Sana Hybrid for ARC-AGI

A hybrid neural architecture that solves abstract reasoning tasks (ARC-AGI) by combining:

- **JEPA/DINOv2 Encoder (Frozen)** - Pretrained spatial feature extractor
- **Bridge Module (Trainable)** - Maps encoder features to diffusion space
- **Sana Transformer (Trainable)** - Linear-attention diffusion backbone for iterative refinement
- **Spatial Decoder (Trainable)** - Converts transformer output to discrete ARC grids

## Architecture

```
Input Grid (variable size, up to 30x30)
    |
    v
Pad to 64x64 Canvas → Render as RGB → Upsample to 224x224
    |
    v
DINOv2-L/14 Encoder (FROZEN) → Patch features [B, 256, 1024]
    |
    v
Bridge Module (TRAINABLE) → [B, 256, 1152]
    |
    v
Sana Transformer (TRAINABLE) → Iterative denoising, conditioned on encoder features
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

The ARC-AGI datasets are required but not included in the repo:

```bash
mkdir -p References
cd References

# ARC-AGI-1 (800 tasks)
git clone https://github.com/fchollet/ARC-AGI.git

# ARC-AGI-2 (1120 tasks)
git clone https://github.com/arcprize/ARC-AGI-2.git

cd ..
```

### 5. (Optional) Download reference repositories for study

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

## Quick Start

### Run tests (Mac/CPU)

```bash
python -m pytest tests/ -v
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

## Data Pipeline

- **ARC-AGI-1**: 400 training + 400 evaluation tasks (difficulty weight: 1.0)
- **ARC-AGI-2**: 1000 training + 120 evaluation tasks (difficulty weight: 1.5)
- **Augmentation**: 8 geometric (D4 symmetry) x 10 color permutations = 80x per example
- **Canvas**: 64x64 fixed size with translation + resolution scaling on-the-fly

## License

Research use only. See individual reference repositories for their respective licenses.
