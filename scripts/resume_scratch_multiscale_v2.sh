#!/bin/bash
# Resume scratch DINOv2 ViT-L + multiscale + T=4 from the last clean v2
# checkpoint after the SSI scale/shift numerical-stability fix.
# Usage: GPU_IDS=2,3 PY=/usr/local/bin/python3 VKITTI_ROOT=/path/to/vkitti \
#        bash scripts/resume_scratch_multiscale_v2.sh
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-/usr/local/bin/python3}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/workspace/vkitti/vkitti}
GPU_IDS=${GPU_IDS:-2,3}
PORT=${PORT:-29661}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoint/scratch_ed_dinov2_multiscale_v2}
RESUME_CKPT=${RESUME_CKPT:-$CHECKPOINT_DIR/checkpoint_12000.pth}
LOG=${LOG:-jobs/ed_dinov2_multiscale_v2_resume.log}
PID_FILE=${PID_FILE:-jobs/ed_dinov2_multiscale_v2_resume.pid}
export VKITTI_ROOT
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

cd "$ROOT"
mkdir -p jobs logs

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python executable not found: $PY" >&2
  exit 1
fi
if [[ ! -d "$VKITTI_ROOT" ]]; then
  echo "ERROR: VKITTI_ROOT does not exist: $VKITTI_ROOT" >&2
  exit 1
fi
if [[ ! -s "$RESUME_CKPT" ]]; then
  echo "ERROR: recovery checkpoint missing: $RESUME_CKPT" >&2
  exit 1
fi
if [[ -s "$CHECKPOINT_DIR/final_model.pth" ]]; then
  echo "ERROR: experiment already complete: $CHECKPOINT_DIR/final_model.pth" >&2
  exit 1
fi
if pgrep -af 'train.py.*scratch_ed_dinov2_multiscale' >/dev/null; then
  echo "ERROR: a scratch multiscale training process is already active:" >&2
  pgrep -af 'train.py.*scratch_ed_dinov2_multiscale' >&2
  exit 1
fi
if [[ ! "$GPU_IDS" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo "ERROR: GPU_IDS must contain exactly two comma-separated indices" >&2
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

echo "[$(date)] resume scratch multiscale v2 from $RESUME_CKPT"
echo "GPUs=$GPU_IDS port=$PORT output=$CHECKPOINT_DIR"
CUDA_VISIBLE_DEVICES="$GPU_IDS" nohup "$PY" -m accelerate.commands.launch \
  --num_processes 2 --main_process_port "$PORT" --mixed_precision bf16 \
  train.py --config-name scratch_ed_dinov2_multiscale \
  "resume=$RESUME_CKPT" \
  "training.checkpoint_dir=$CHECKPOINT_DIR" \
  training.log_dir=./logs/scratch_ed_dinov2_multiscale_v2 \
  project_name=scratch_ed_dinov2_multiscale_v2 \
  training.save_freq=500 \
  > "$LOG" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "launcher pid=$pid log=$LOG"
