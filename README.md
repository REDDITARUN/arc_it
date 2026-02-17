# ARC-IT: Rule-Conditioned Transformer for ARC-AGI

A novel architecture that solves abstract reasoning tasks (ARC-AGI) by explicitly extracting transformation rules from demonstration pairs, then applying those rules to new inputs — mirroring how humans solve ARC puzzles.

- **GridTokenizer** - Embeds discrete ARC grids (0-11) into continuous patch tokens
- **RuleEncoder** - Extracts transformation rules from demo input/output pairs via cross-attention
- **RuleApplier** - Applies learned rules to test inputs via cross-attention
- **SpatialDecoder** - Converts output tokens to 64x64 grid logits
- **Test-Time Training (TTT)** - Per-task adaptation for each puzzle

## Architecture

```
Demo Pairs (input, output) × K
    |
    v
GridTokenizer → Patch tokens per grid
    |
    v
RuleEncoder:
    Per-pair cross-attention (output queries input → delta tokens)
    Cross-pair aggregation (self-attention over all deltas)
    Rule query attention → Fixed-length Rule Tokens [B, 64, 384]
    |
    v
Test Input
    |
    v
GridTokenizer → Test patch tokens [B, 256, 384]
    |
    v
RuleApplier:
    Self-attention on test tokens
    Cross-attention to Rule Tokens
    → Output tokens [B, 256, 384]
    |
    v
SpatialDecoder → [B, 12, 64, 64] logits → argmax → Crop → Predicted Grid
```

## Key Design Choices

- **No pretrained encoder** — works directly on discrete grids, avoiding lossy RGB rendering
- **Discrete-to-discrete** — integer grid in, integer grid out via learned patch embeddings
- **Explicit rule extraction** — paired cross-attention captures what changed between demo I/O
- **Lightweight** — ~20M parameters, trainable from scratch on a single A100/H100

## Project Structure

```
arc_it/
├── arc_it/                  # Main package
│   ├── data/                # Data loading, augmentation, canvas, rendering
│   ├── models/              # GridTokenizer, RuleEncoder, RuleApplier, Decoder
│   ├── training/            # Loss functions, training loops
│   ├── inference/           # TTT, evaluation
│   └── utils/               # Device detection, config, visualization
├── configs/                 # YAML configurations
│   ├── default.yaml         # A100/H100 training defaults
│   └── mac_dev.yaml         # Mac development overrides
├── scripts/                 # Entry point scripts
├── tests/                   # Pytest test suite (65 tests)
├── References/              # (gitignored) ARC-AGI datasets
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

### 4. Download ARC-AGI datasets

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

### 5. Configure API keys (optional, for W&B + HF upload)

```bash
cp .env.example .env
# Edit .env with your WANDB_API_KEY and HF_TOKEN
```

## Cloud Quickstart (A100 / H100)

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

### Run tests

```bash
python -m pytest tests/ -v
```

### End-to-End Demo

Runs the full pipeline (data loading, model build, training steps, inference, TTT) on CPU:

```bash
python scripts/demo.py
```

### Train

```bash
# Full 2-stage training (Full Training → Hard Focus)
python scripts/train.py

# With W&B logging
python scripts/train.py --wandb

# Resume from checkpoint
python scripts/train.py --checkpoint checkpoints/best_stage1.pt
```

### Evaluate

```bash
# Standard evaluation
python scripts/evaluate.py --checkpoint checkpoints/best_stage1.pt

# With test-time training (per-task fine-tuning)
python scripts/evaluate.py --checkpoint checkpoints/best_stage1.pt --ttt
```

### Load dataset

```python
from arc_it.data.dataset import ARCTaskDataset

ds = ARCTaskDataset(
    data_roots=["References/ARC-AGI", "References/ARC-AGI-2"],
    split="training",
    subset="train",
    canvas_size=64,
    max_demos=5,
)
print(f"Loaded {len(ds)} samples from {ds.num_tasks} tasks")

sample = ds[0]
print(f"Demo inputs:  {sample['demo_inputs'].shape}")   # (5, 64, 64)
print(f"Demo outputs: {sample['demo_outputs'].shape}")   # (5, 64, 64)
print(f"Query input:  {sample['query_input'].shape}")    # (64, 64)
print(f"Target:       {sample['target'].shape}")         # (64, 64)
```

## Training Strategy

**2-Stage Training**:

| Stage | Duration | What | LR |
|-------|----------|------|----|
| 1. Full Training | ~1.5 hrs | Train all 20M params on augmented dataset | 3e-4 |
| 2. Hard Focus | ~30 min | Lower LR, oversample AGI-2 tasks (2x weight) | 3e-5 |

## Test-Time Training (TTT)

For each evaluation task:
1. Snapshot model weights
2. Augment the 3-5 demonstration examples (geometric + color permutations)
3. Fine-tune all model parameters for K steps on augmented leave-one-out samples
4. Generate multiple candidate predictions via augmentation voting
5. Score candidates with majority voting
6. Restore original weights for next task

## Data Pipeline

- **ARC-AGI-1**: 400 training + 400 evaluation tasks (difficulty weight: 1.0)
- **ARC-AGI-2**: 1000 training + 120 evaluation tasks (difficulty weight: 1.5)
- **Task-centric sampling**: Leave-one-out — each train example becomes a query, others are demos
- **Augmentation**: 8 geometric (D4 symmetry) x 10 color permutations = 80x per task (on-the-fly)
- **Canvas**: 64x64 fixed size with translation + resolution scaling

## License

Research use only. See individual reference repositories for their respective licenses.
