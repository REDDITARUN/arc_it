#!/bin/bash
# Download ARC-AGI datasets and RE-ARC into References/
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

# RE-ARC (400 task concepts × 1000 generated examples each)
if [ -d "RE-ARC/tasks" ]; then
    echo "[OK] RE-ARC already present"
else
    echo "[>>] Downloading RE-ARC..."
    git clone --depth 1 https://github.com/michaelhodel/re-arc.git re-arc-repo
    mkdir -p RE-ARC
    cd re-arc-repo

    # Extract the pre-generated dataset from the zip
    if [ -f "re_arc.zip" ]; then
        echo "  Extracting re_arc.zip..."
        # Use Python's zipfile (always available) as fallback for unzip
        python3 -c "
import zipfile, shutil, os
with zipfile.ZipFile('re_arc.zip', 'r') as z:
    z.extractall('../RE-ARC-tmp')
# Find and move the tasks/ directory
for root, dirs, files in os.walk('../RE-ARC-tmp'):
    if 'tasks' in dirs:
        shutil.move(os.path.join(root, 'tasks'), '../RE-ARC/tasks')
        break
shutil.rmtree('../RE-ARC-tmp', ignore_errors=True)
print(f'  Extracted {len(os.listdir(\"../RE-ARC/tasks\"))} task files')
"
    else
        echo "  Warning: re_arc.zip not found, generating dataset..."
        echo "  This requires Python and may take a while."
        python main.py 2>/dev/null || echo "  Generation failed, install deps first."
        if [ -d "re_arc/tasks" ]; then
            mv re_arc/tasks ../RE-ARC/tasks
        fi
    fi

    cd "$REF_DIR"
    rm -rf re-arc-repo
    echo "[OK] RE-ARC downloaded"
fi

# Verify
echo ""
echo "Verifying datasets..."
AGI1_COUNT=$(ls "$REF_DIR/ARC-AGI/data/training/"*.json 2>/dev/null | wc -l | tr -d ' ')
AGI2_COUNT=$(ls "$REF_DIR/ARC-AGI-2/data/training/"*.json 2>/dev/null | wc -l | tr -d ' ')
REARC_COUNT=$(ls "$REF_DIR/RE-ARC/tasks/"*.json 2>/dev/null | wc -l | tr -d ' ')
echo "  ARC-AGI-1 training tasks: $AGI1_COUNT"
echo "  ARC-AGI-2 training tasks: $AGI2_COUNT"
echo "  RE-ARC task concepts:     $REARC_COUNT"

if [ "$AGI1_COUNT" -gt 0 ] && [ "$AGI2_COUNT" -gt 0 ]; then
    echo ""
    echo "Setup complete! You can now run:"
    echo "  python scripts/train.py --wandb"
    if [ "$REARC_COUNT" -eq 0 ]; then
        echo ""
        echo "  Note: RE-ARC not available. Stage 1 will fall back to AGI-1 only."
        echo "  To add RE-ARC manually, place task JSONs in References/RE-ARC/tasks/"
    fi
else
    echo ""
    echo "WARNING: Some datasets may be missing. Check the output above."
    exit 1
fi
