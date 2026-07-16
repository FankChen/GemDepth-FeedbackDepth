#!/bin/bash
# Early quality gate for a v2 training checkpoint. This does not train.
# Example:
#   CHECKPOINT=checkpoint/scratch_ed_gt_error_rgbfeat_v2/checkpoint_500.pth \
#   GPU_ID=0 bash scripts/run_dpt_warp_v2_visualization.sh
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-/usr/local/bin/python3}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/workspace/vkitti/vkitti}
CONFIG=${CONFIG:-config/scratch_ed_gt_error_rgbfeat_v2.yaml}
CHECKPOINT=${CHECKPOINT:-checkpoint/scratch_ed_gt_error_rgbfeat_v2/checkpoint_500.pth}
GPU_ID=${GPU_ID:-0}
SAMPLE_IDX=${SAMPLE_IDX:-0}
FRAME_IDX=${FRAME_IDX:-1}
OUTPUT=${OUTPUT:-results/dpt_warp_v2_visualization/sample${SAMPLE_IDX}}
LOG=${LOG:-jobs/dpt_warp_v2_visualization_sample${SAMPLE_IDX}.log}

cd "$ROOT"
mkdir -p jobs "$OUTPUT"
for required in "$PY" "$VKITTI_ROOT" "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: missing $required" >&2
    exit 1
  fi
done

selected_gpu_pids=$(nvidia-smi -i "$GPU_ID" \
  --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$selected_gpu_pids" ]]; then
  echo "ERROR: GPU $GPU_ID already has compute processes" >&2
  nvidia-smi -i "$GPU_ID" >&2
  exit 1
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" \
  evaluation/visualization/viz_dpt_warp_v2.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --data_dir "$VKITTI_ROOT" \
  --sample_idx "$SAMPLE_IDX" \
  --frame_idx "$FRAME_IDX" \
  --output "$OUTPUT" \
  2>&1 | tee "$LOG"

echo "PASS: all p4/p3/p2/p1 depth/warp rows are finite and have non-zero validity"
echo "figure=$OUTPUT/four_level_depth_warp.png"
echo "summary=$OUTPUT/summary.json"
