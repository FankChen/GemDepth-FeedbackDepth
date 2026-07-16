#!/bin/bash
# Fastest diagnostic path: one deterministic Scene20 clip, no retraining.
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-/usr/local/bin/python3}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/workspace/vkitti/vkitti}
GPU_ID=${GPU_ID:-0}
SAMPLE_IDX=${SAMPLE_IDX:-0}
FRAME_IDX=${FRAME_IDX:-1}
OUTPUT=${OUTPUT:-results/gt_error_visualization/sample${SAMPLE_IDX}}
LOG=${LOG:-jobs/gt_error_visualization_sample${SAMPLE_IDX}.log}
PID_FILE=${PID_FILE:-jobs/gt_error_visualization_sample${SAMPLE_IDX}.pid}

cd "$ROOT"
mkdir -p jobs "$OUTPUT"

for required in \
  "$PY" \
  "$VKITTI_ROOT" \
  checkpoint/scratch_ed_gt_error_baseline/final_model.pth \
  checkpoint/scratch_ed_gt_error_rgbfeat/final_model.pth
do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: missing $required" >&2
    exit 1
  fi
done

selected_gpu_pids=$(nvidia-smi -i "$GPU_ID" \
  --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$selected_gpu_pids" ]]; then
  echo "ERROR: GPU $GPU_ID already has compute processes:" >&2
  nvidia-smi -i "$GPU_ID" >&2
  exit 1
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}

if [[ -e "$LOG" ]]; then
  mv "$LOG" "${LOG}.$(date +%Y%m%d_%H%M%S).bak"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup "$PY" \
  evaluation/visualization/viz_gt_error_layers.py \
  --data_dir "$VKITTI_ROOT" \
  --sample_idx "$SAMPLE_IDX" \
  --frame_idx "$FRAME_IDX" \
  --output "$OUTPUT" \
  > "$LOG" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "visualization RUNNING pid=$pid GPU=$GPU_ID"
echo "log=$LOG"
echo "output=$OUTPUT"
