#!/bin/bash
# One-shot status report for all GT-camera oracle arms.
set -u

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT" || exit 1

date
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
  --format=csv,noheader 2>/dev/null || true

fatal_pattern='Traceback|CUDA out of memory|Non-finite|marked ready twice|ChildFailedError'
for arm in baseline rgb feat geom rgbfeat; do
  echo
  echo "========== $arm =========="
  pid_file="jobs/gt_error_${arm}.pid"
  log_file="jobs/gt_error_${arm}.log"
  final_model="checkpoint/scratch_ed_gt_error_${arm}/final_model.pth"

  if [[ -s "$final_model" ]]; then
    echo "STATUS=COMPLETED"
  elif [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "STATUS=RUNNING PID=$(cat "$pid_file")"
  elif [[ "$arm" == rgbfeat ]] && [[ -s jobs/gt_error_rgbfeat_dispatch.pid ]] \
      && kill -0 "$(cat jobs/gt_error_rgbfeat_dispatch.pid)" 2>/dev/null; then
    echo "STATUS=QUEUED"
  elif [[ -e "$pid_file" || -e "$log_file" ]]; then
    echo "STATUS=STOPPED_OR_FAILED"
  else
    echo "STATUS=NOT_LAUNCHED"
  fi

  if [[ -s "$log_file" ]]; then
    grep -nE "$fatal_pattern" "$log_file" | tail -10 || true
    tail -n 3 "$log_file"
  fi
done

echo
echo "========== RGBFEAT DISPATCHER =========="
if [[ -s jobs/gt_error_rgbfeat_dispatch.log ]]; then
  tail -n 20 jobs/gt_error_rgbfeat_dispatch.log
else
  echo "No dispatcher log"
fi
