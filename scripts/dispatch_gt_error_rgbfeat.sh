#!/bin/bash
# Wait for the held-out baseline to finish successfully, then reuse GPUs 0,1
# for the fifth (RGB+feature) GT-camera error-channel arm.
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-/usr/local/bin/python3}
cd "$ROOT"

mkdir -p jobs
baseline_pid_file=jobs/gt_error_baseline.pid
baseline_log=jobs/gt_error_baseline.log
baseline_model=checkpoint/scratch_ed_gt_error_baseline/final_model.pth

if [[ ! -s "$baseline_pid_file" ]]; then
  echo "[$(date)] ERROR: missing $baseline_pid_file" >&2
  exit 1
fi

baseline_pid=$(cat "$baseline_pid_file")
if [[ ! "$baseline_pid" =~ ^[0-9]+$ ]]; then
  echo "[$(date)] ERROR: invalid baseline PID: $baseline_pid" >&2
  exit 1
fi

echo "[$(date)] waiting for baseline PID $baseline_pid before reusing GPUs 0,1"
if kill -0 "$baseline_pid" 2>/dev/null; then
  tail --pid="$baseline_pid" --sleep-interval=30 -f /dev/null || true
fi

if [[ ! -s "$baseline_model" ]]; then
  echo "[$(date)] ERROR: baseline ended without $baseline_model" >&2
  tail -n 100 "$baseline_log" 2>/dev/null || true
  exit 1
fi

fatal_pattern='Traceback|CUDA out of memory|Non-finite|marked ready twice|ChildFailedError'
if grep -qE "$fatal_pattern" "$baseline_log"; then
  echo "[$(date)] ERROR: baseline log contains a fatal error; RGB+feature will not start" >&2
  grep -nE "$fatal_pattern" "$baseline_log" | tail -20 >&2
  exit 1
fi

echo "[$(date)] baseline completed; launching RGB+feature on GPUs 0,1"
CUDA_VISIBLE_DEVICES=0,1 "$PY" -m accelerate.commands.launch \
  --num_processes 2 --main_process_port 29635 --mixed_precision bf16 \
  train.py --config-name scratch_ed_gt_error_rgbfeat \
  > jobs/gt_error_rgbfeat.log 2>&1 &

rgbfeat_pid=$!
echo "$rgbfeat_pid" > jobs/gt_error_rgbfeat.pid
echo "[$(date)] RGB+feature launcher PID $rgbfeat_pid"

set +e
wait "$rgbfeat_pid"
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "[$(date)] ERROR: RGB+feature exited with status $status" >&2
  tail -n 100 jobs/gt_error_rgbfeat.log >&2 || true
  exit "$status"
fi

if [[ ! -s checkpoint/scratch_ed_gt_error_rgbfeat/final_model.pth ]]; then
  echo "[$(date)] ERROR: RGB+feature exited cleanly but final_model.pth is missing" >&2
  exit 1
fi

echo "[$(date)] RGB+feature completed successfully"
