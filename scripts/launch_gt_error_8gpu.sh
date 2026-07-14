#!/bin/bash
# Wave 1 on 8 GPUs: held-out-scene baseline + three primary GT-camera channels.
# RGBFEAT is intentionally Wave 2: a fair baseline is more important than the
# fourth channel in the first 8-GPU allocation.
# Usage: PY=/usr/local/bin/python3 VKITTI_ROOT=/path/to/vkitti bash scripts/launch_gt_error_8gpu.sh
set -euo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-python}
cd "$ROOT"
mkdir -p jobs logs checkpoint

# Fail closed: do not silently collide with existing scratch/GT-error jobs.
if pgrep -af 'train.py.*(scratch_ed_gt_error|scratch_ed_dinov2_multiscale)' >/dev/null; then
  echo "Refusing to launch: a GT-error or multiscale scratch training process already exists:" >&2
  pgrep -af 'train.py.*(scratch_ed_gt_error|scratch_ed_dinov2_multiscale)' >&2
  exit 1
fi

launch() {
  local gpus=$1 port=$2 cfg=$3 log=$4
  echo "[$(date)] launch cfg=$cfg GPUs=$gpus port=$port log=$log"
  CUDA_VISIBLE_DEVICES="$gpus" nohup "$PY" -m accelerate.commands.launch \
    --num_processes 2 --main_process_port "$port" --mixed_precision bf16 \
    train.py --config-name "$cfg" > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "${log%.log}.pid"
  echo "  launcher pid=$pid"
}

launch 0,1 29631 scratch_ed_gt_error_baseline jobs/gt_error_baseline.log
launch 2,3 29632 scratch_ed_gt_error_rgb      jobs/gt_error_rgb.log
launch 4,5 29633 scratch_ed_gt_error_feat     jobs/gt_error_feat.log
launch 6,7 29634 scratch_ed_gt_error_geom     jobs/gt_error_geom.log

echo "All four arms launched. Check after 2 minutes:"
echo "  for f in jobs/gt_error_*.log; do echo ====\$f====; grep -nE '\[params\]|Traceback|out of memory|Non-finite' \$f | tail -3; tail -n1 \$f; done"
