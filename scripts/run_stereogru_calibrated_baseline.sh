#!/usr/bin/env bash
# NEW bounded calibration pilot only. No legacy checkpoint writes or method arms.
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/data/PROJECT_CHEN/data/train/vkitti2/vkitti}
PYTHON=${PYTHON:-/usr/local/bin/python}
GPU=${GPU:-4}
BACKBONE_WEIGHTS=${BACKBONE_WEIGHTS:-$RUN_ROOT/checkpoint/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
[[ -x "$PYTHON" ]] || { echo "Missing interpreter: $PYTHON"; exit 2; }
[[ -d "$VKITTI_ROOT" ]] || { echo "Missing data: $VKITTI_ROOT"; exit 2; }
[[ -f "$BACKBONE_WEIGHTS" ]] || { echo "Set BACKBONE_WEIGHTS to the CLEAN official ConvNeXt-S export (not a trained costvol checkpoint): $BACKBONE_WEIGHTS"; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing"; exit 2; }
nvidia-smi --query-gpu=index,name,memory.used --format=csv
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
if [[ "$USED" -gt 2048 ]] || grep -Fq "$GPU_UUID" <<< "$PROCESSES"; then
    echo "GPU=$GPU busy; no training launched. Choose an idle GPU."
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$GPU"
OUT=$(mktemp -d "$RUN_ROOT/stereogru_calibrated_XXXXXXXX")
echo "NEW 200-step volume-only calibration pilot: $OUT"
echo "GT camera / training scenes / frozen clean backbone / no GEM / no GRU."
echo "Legacy runs and checkpoints are read-only; no method will be launched."
cd "$SOURCE_ROOT"
"$PYTHON" -u scripts/stereogru_calibrated_baseline.py \
    --backbone-weights "$BACKBONE_WEIGHTS" --vkitti-root "$VKITTI_ROOT" --output "$OUT/run" \
    2>&1 | tee "$OUT/pilot.log"