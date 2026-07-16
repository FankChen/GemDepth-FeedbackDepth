#!/bin/bash
# Continuable pilot for the repaired metric-depth branch. It uses the full 20k
# scheduler from the start, but saves every 250 steps for an early health gate.
# If the gate passes, leave it running; no restart or scheduler change is needed.
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-/usr/local/bin/python3}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/workspace/vkitti/vkitti}
GPU_IDS=${GPU_IDS:-0,1}
PORT=${PORT:-29671}
STEPS=${STEPS:-20000}
RUN_NAME=${RUN_NAME:-scratch_ed_gt_error_rgbfeat_metricfix_pilot}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoint/$RUN_NAME}
LOG_DIR=${LOG_DIR:-logs/$RUN_NAME}
LOG=${LOG:-jobs/${RUN_NAME}.log}
PID_FILE=${PID_FILE:-jobs/${RUN_NAME}.pid}
export VKITTI_ROOT
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

cd "$ROOT"
mkdir -p jobs logs
if [[ -L checkpoint && ! -e checkpoint ]]; then
  echo "Replacing dangling checkpoint symlink: checkpoint -> $(readlink checkpoint)"
  rm checkpoint
fi
mkdir -p checkpoint

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python executable not found: $PY" >&2
  exit 1
fi
if [[ ! -d "$VKITTI_ROOT" ]]; then
  echo "ERROR: VKITTI_ROOT does not exist: $VKITTI_ROOT" >&2
  exit 1
fi
if [[ ! "$GPU_IDS" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "ERROR: GPU_IDS must contain exactly two comma-separated indices" >&2
  exit 1
fi
if [[ -s "$CHECKPOINT_DIR/final_model.pth" ]]; then
  echo "ERROR: pilot already complete: $CHECKPOINT_DIR/final_model.pth" >&2
  exit 1
fi
if pgrep -af "train.py.*scratch_ed_gt_error_rgbfeat_metricfix" >/dev/null; then
  echo "ERROR: metric-depth pilot/full training is already active:" >&2
  pgrep -af "train.py.*scratch_ed_gt_error_rgbfeat_metricfix" >&2
  exit 1
fi

selected_gpu_pids=$(nvidia-smi -i "$GPU_IDS" \
  --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$selected_gpu_pids" ]]; then
  echo "ERROR: selected GPUs $GPU_IDS already have compute processes:" >&2
  nvidia-smi -i "$GPU_IDS" >&2
  exit 1
fi

if [[ -e "$LOG" ]]; then
  mv "$LOG" "${LOG}.$(date +%Y%m%d_%H%M%S).bak"
fi

echo "[$(date)] launch continuable metric-depth log-parameterization pilot"
echo "GPUs=$GPU_IDS steps=$STEPS checkpoint=$CHECKPOINT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_IDS" nohup "$PY" -m accelerate.commands.launch \
  --num_processes 2 --main_process_port "$PORT" --mixed_precision bf16 \
  train.py --config-name scratch_ed_gt_error_rgbfeat_metricfix \
  "total_step=$STEPS" \
  "training.checkpoint_dir=$CHECKPOINT_DIR" \
  "training.log_dir=$LOG_DIR" \
  "training.save_freq=250" \
  "project_name=$RUN_NAME" \
  > "$LOG" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "pilot RUNNING pid=$pid log=$LOG"
