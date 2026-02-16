#!/bin/bash
# Download ARC-AGI datasets into References/
# Run this once on any new machine before training.
#
# Usage:
#   bash scripts/setup_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REF_DIR="$PROJECT_DIR/References"

echo "============================================================"
echo "ARC-IT Dataset Setup"
echo "============================================================"
echo "Project root: $PROJECT_DIR"
echo "References:   $REF_DIR"
echo ""

mkdir -p "$REF_DIR"
cd "$REF_DIR"

# ARC-AGI-1 (400 training + 400 evaluation tasks)
if [ -d "ARC-AGI/data/training" ]; then
    echo "[OK] ARC-AGI-1 already present"
else
    echo "[>>] Cloning ARC-AGI-1..."
    git clone --depth 1 https://github.com/fchollet/ARC-AGI.git
    echo "[OK] ARC-AGI-1 downloaded"
fi

# ARC-AGI-2 (1000 training + 120 evaluation tasks)
if [ -d "ARC-AGI-2/data/training" ]; then
    echo "[OK] ARC-AGI-2 already present"
else
    echo "[>>] Cloning ARC-AGI-2..."
    git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git
    echo "[OK] ARC-AGI-2 downloaded"
fi

# Verify
echo ""
echo "Verifying datasets..."
AGI1_COUNT=$(ls "$REF_DIR/ARC-AGI/data/training/"*.json 2>/dev/null | wc -l | tr -d ' ')
AGI2_COUNT=$(ls "$REF_DIR/ARC-AGI-2/data/training/"*.json 2>/dev/null | wc -l | tr -d ' ')
echo "  ARC-AGI-1 training tasks: $AGI1_COUNT"
echo "  ARC-AGI-2 training tasks: $AGI2_COUNT"

if [ "$AGI1_COUNT" -gt 0 ] && [ "$AGI2_COUNT" -gt 0 ]; then
    echo ""
    echo "Setup complete! You can now run:"
    echo "  python scripts/train.py --wandb"
else
    echo ""
    echo "WARNING: Some datasets may be missing. Check the output above."
    exit 1
fi
