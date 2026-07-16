#!/bin/bash
# Full 20k-step baseline-anchored DPT + warp v2 training.
# Usage:
#   GPU_IDS=0,1 PY=/usr/local/bin/python3 VKITTI_ROOT=/mnt/workspace/vkitti/vkitti \
#     bash scripts/launch_dpt_warp_v2_train.sh
# Resume after interruption:
#   RESUME=true GPU_IDS=0,1 bash scripts/launch_dpt_warp_v2_train.sh
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-/usr/local/bin/python3}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/workspace/vkitti/vkitti}
GPU_IDS=${GPU_IDS:-0,1}
PORT=${PORT:-29681}
RUN_NAME=${RUN_NAME:-scratch_ed_gt_error_rgbfeat_v2}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoint/$RUN_NAME}
LOG_DIR=${LOG_DIR:-logs/$RUN_NAME}
LOG=${LOG:-jobs/${RUN_NAME}.log}
PID_FILE=${PID_FILE:-jobs/${RUN_NAME}.pid}
RESUME=${RESUME:-false}
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
if [[ "$RESUME" != true && "$RESUME" != false ]]; then
  echo "ERROR: RESUME must be true or false" >&2
  exit 1
fi
if [[ -s "$CHECKPOINT_DIR/final_model.pth" ]]; then
  echo "ERROR: experiment already complete: $CHECKPOINT_DIR/final_model.pth" >&2
  exit 1
fi
if [[ "$RESUME" == false ]] && compgen -G "$CHECKPOINT_DIR/checkpoint_*.pth" >/dev/null; then
  echo "ERROR: checkpoints already exist in $CHECKPOINT_DIR; use RESUME=true" >&2
  exit 1
fi
if pgrep -af 'train.py.*scratch_ed_gt_error_rgbfeat_v2' >/dev/null; then
  echo "ERROR: DPT+warp v2 training is already active:" >&2
  pgrep -af 'train.py.*scratch_ed_gt_error_rgbfeat_v2' >&2
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

overrides=(
  "training.checkpoint_dir=$CHECKPOINT_DIR"
  "training.log_dir=$LOG_DIR"
  "training.save_freq=500"
  "project_name=$RUN_NAME"
)
if [[ "$RESUME" == true ]]; then
  overrides+=("resume=true")
fi

echo "[$(date)] launch full DPT+warp v2 training"
echo "GPUs=$GPU_IDS port=$PORT resume=$RESUME"
echo "checkpoint=$CHECKPOINT_DIR log=$LOG"
CUDA_VISIBLE_DEVICES="$GPU_IDS" nohup "$PY" -m accelerate.commands.launch \
  --num_processes 2 --num_machines 1 \
  --main_process_port "$PORT" --mixed_precision bf16 --dynamo_backend no \
  train.py --config-name scratch_ed_gt_error_rgbfeat_v2 \
  "${overrides[@]}" \
  > "$LOG" 2>&1 &

pid=$!
echo "$pid" > "$PID_FILE"
echo "DPT+warp v2 RUNNING pid=$pid"
echo "status: kill -0 $pid && tail -n 20 $LOG"
echo "early visualization after checkpoint_500.pth:"
echo "  CHECKPOINT=$CHECKPOINT_DIR/checkpoint_500.pth GPU_ID=<free_gpu> bash scripts/run_dpt_warp_v2_visualization.sh"
