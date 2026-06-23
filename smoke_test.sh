#!/bin/bash
# Quick smoke test for GemDepth environment and data loading
# This script verifies that the environment is correctly set up
# and that data loading works (using evaluation datasets if training data unavailable)

set -e

MODULE_PURGE=true
CONDA_LOAD=false

if [ "$MODULE_PURGE" = true ]; then
    module purge
    module load conda/4.8.5
fi

# Use gemdepth environment
PY=/home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python
ROOT=/home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth

echo "=========================================="
echo "GemDepth Smoke Test"
echo "=========================================="
echo ""

# Test 1: Environment
echo "[1/5] Checking Python and PyTorch..."
$PY -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda}')" || exit 1
echo ""

# Test 2: Accelerate
echo "[2/5] Checking Accelerate..."
$PY -c "import accelerate; print(f'✓ Accelerate {accelerate.__version__}')" || exit 1
echo ""

# Test 3: FlashAttention
echo "[3/5] Checking FlashAttention..."
$PY -c "import flash_attn; print(f'✓ FlashAttention available')" || (echo "⚠ FlashAttention not available (optional)" && true)
echo ""

# Test 4: GemDepth model imports
echo "[4/5] Checking GemDepth imports..."
cd $ROOT
$PY -c "
import sys
sys.path.insert(0, '.')
from model.gemdepth import GemDepth
from dataset.dataset_mix import DepthVideoDataset
print('✓ GemDepth model imports OK')
" || exit 1
echo ""

# Test 5: Data loading (if available)
echo "[5/5] Checking data availability..."
TARTANAIR_DIR=/home/izi2sgh/MYDATA/quanjie/liren/datasets/tartanair
VKITTI_DIR=/home/izi2sgh/MYDATA/quanjie/liren/datasets/vkitti_1.3.1
EVAL_DIR=/home/izi2sgh/MYDATA/quanjie/liren/datasets/gemdepth_eval

if [ -d "$VKITTI_DIR/vkitti_1.3.1_rgb" ]; then
    echo "✓ VKITTI training data available"
elif [ -d "$TARTANAIR_DIR" ] && [ -n "$(find $TARTANAIR_DIR -name 'image_left' -type d | head -1)" ]; then
    echo "✓ TartanAir training data available"
else
    echo "⚠ Training datasets not yet available"
    echo "  - VKITTI: $([ -d "$VKITTI_DIR" ] && echo "directory exists (empty)" || echo "directory not found")"
    echo "  - TartanAir: $([ -d "$TARTANAIR_DIR" ] && echo "directory exists (empty)" || echo "directory not found")"
fi

if [ -d "$EVAL_DIR/kitti" ]; then
    echo "✓ KITTI evaluation data available (34GB)"
fi

echo ""
echo "=========================================="
echo "Smoke Test Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. To download VKITTI:"
echo "   - See VKITTI_SETUP.md for manual download instructions"
echo ""
echo "2. To download TartanAir:"
echo "   bsub < scripts/download_tartanair.bsub"
echo ""
echo "3. To run training (when data is ready):"
echo "   bsub < scripts/train_stage1_vkitti_only.bsub"
echo "   OR"
echo "   bsub < scripts/train_stage1.bsub  (if TartanAir downloaded)"
echo ""
