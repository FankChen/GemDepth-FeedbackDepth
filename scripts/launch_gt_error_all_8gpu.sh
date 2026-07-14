#!/bin/bash
# Launch all five fair GT-camera oracle arms on an 8-GPU node.
# Four 2-GPU arms run immediately; RGB+feature is queued behind baseline and
# automatically reuses GPUs 0,1 only after baseline completes successfully.
# Usage: PY=/usr/local/bin/python3 VKITTI_ROOT=/path/to/vkitti bash scripts/launch_gt_error_all_8gpu.sh
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-/usr/local/bin/python3}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/workspace/vkitti/vkitti}
export PY VKITTI_ROOT
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export NO_ALBUMENTATIONS_UPDATE=${NO_ALBUMENTATIONS_UPDATE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

cd "$ROOT"
mkdir -p jobs logs checkpoint

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python executable not found: $PY" >&2
  exit 1
fi
if [[ ! -d "$VKITTI_ROOT" ]]; then
  echo "ERROR: VKITTI_ROOT does not exist: $VKITTI_ROOT" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable" >&2
  exit 1
fi
gpu_count=$(nvidia-smi -L | wc -l)
if (( gpu_count < 8 )); then
  echo "ERROR: five-arm schedule requires at least 8 GPUs; found $gpu_count" >&2
  exit 1
fi

active_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$active_gpu_pids" ]]; then
  echo "ERROR: GPU compute processes already exist; refusing to oversubscribe:" >&2
  nvidia-smi >&2
  exit 1
fi

for arm in baseline rgb feat geom rgbfeat; do
  final_model="checkpoint/scratch_ed_gt_error_${arm}/final_model.pth"
  if [[ -e "$final_model" ]]; then
    echo "ERROR: existing result would be overwritten: $final_model" >&2
    exit 1
  fi
done

bash scripts/launch_gt_error_8gpu.sh

for arm in baseline rgb feat geom; do
  pid_file="jobs/gt_error_${arm}.pid"
  if [[ ! -s "$pid_file" ]]; then
    echo "ERROR: launcher did not create $pid_file" >&2
    exit 1
  fi
  pid=$(cat "$pid_file")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: $arm launcher PID $pid exited during startup" >&2
    tail -n 100 "jobs/gt_error_${arm}.log" >&2 || true
    exit 1
  fi
  echo "$arm RUNNING pid=$pid"
done

nohup env ROOT="$ROOT" PY="$PY" \
  bash scripts/dispatch_gt_error_rgbfeat.sh \
  > jobs/gt_error_rgbfeat_dispatch.log 2>&1 &
dispatch_pid=$!
echo "$dispatch_pid" > jobs/gt_error_rgbfeat_dispatch.pid

echo "rgbfeat QUEUED dispatcher_pid=$dispatch_pid"
echo "All five arms are running or queued."
echo "Wave 1: baseline=GPU0,1 rgb=GPU2,3 feat=GPU4,5 geom=GPU6,7"
echo "Wave 2: rgbfeat automatically reuses GPU0,1 after baseline succeeds"
