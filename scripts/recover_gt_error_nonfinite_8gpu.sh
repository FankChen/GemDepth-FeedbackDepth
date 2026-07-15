#!/bin/bash
# Recover the four unfinished GT-camera study arms after the scale/shift
# numerical-stability fix. RGB is already complete and is deliberately untouched.
# GPU allocation: baseline=0,1 feat=2,3 geom=4,5 rgbfeat=6,7.
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
if [[ ! -s checkpoint/scratch_ed_gt_error_rgb/final_model.pth ]]; then
  echo "ERROR: completed RGB control is missing; refusing an ambiguous recovery" >&2
  exit 1
fi

if pgrep -af 'train.py.*scratch_ed_gt_error' >/dev/null; then
  echo "ERROR: a GT-error training process is already active:" >&2
  pgrep -af 'train.py.*scratch_ed_gt_error' >&2
  exit 1
fi
active_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d' || true)
if [[ -n "$active_gpu_pids" ]]; then
  echo "ERROR: GPU compute processes already exist; refusing to oversubscribe:" >&2
  nvidia-smi >&2
  exit 1
fi

for arm in baseline feat geom rgbfeat; do
  if [[ -s "checkpoint/scratch_ed_gt_error_${arm}/final_model.pth" ]]; then
    echo "ERROR: $arm is already complete; refusing to overwrite it" >&2
    exit 1
  fi
done
if ! compgen -G 'checkpoint/scratch_ed_gt_error_feat/checkpoint_*.pth' >/dev/null; then
  echo "ERROR: feature-arm recovery checkpoint is missing" >&2
  exit 1
fi
if ! compgen -G 'checkpoint/scratch_ed_gt_error_geom/checkpoint_*.pth' >/dev/null; then
  echo "ERROR: geometry-arm recovery checkpoint is missing" >&2
  exit 1
fi

stamp=$(date +%Y%m%d_%H%M%S)
archive="jobs/failed_nonfinite_${stamp}"
mkdir -p "$archive"
for arm in baseline feat geom rgbfeat; do
  for suffix in log pid; do
    old="jobs/gt_error_${arm}.${suffix}"
    [[ -e "$old" ]] && mv "$old" "$archive/"
  done
done
for file in jobs/gt_error_rgbfeat_dispatch.log jobs/gt_error_rgbfeat_dispatch.pid; do
  [[ -e "$file" ]] && mv "$file" "$archive/"
done

echo "Archived failed-run logs under $archive"
echo "Feature resume: $(find checkpoint/scratch_ed_gt_error_feat -maxdepth 1 -name 'checkpoint_*.pth' -printf '%f\n' | sort -V | tail -1)"
echo "Geometry resume: $(find checkpoint/scratch_ed_gt_error_geom -maxdepth 1 -name 'checkpoint_*.pth' -printf '%f\n' | sort -V | tail -1)"

launch() {
  local gpus=$1 port=$2 config=$3 arm=$4 resume_value=$5
  local log="jobs/gt_error_${arm}.log"
  echo "[$(date)] launch $arm on GPUs $gpus (resume=$resume_value)"
  CUDA_VISIBLE_DEVICES="$gpus" nohup "$PY" -m accelerate.commands.launch \
    --num_processes 2 --main_process_port "$port" --mixed_precision bf16 \
    train.py --config-name "$config" \
    "resume=$resume_value" training.save_freq=500 \
    > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "jobs/gt_error_${arm}.pid"
  echo "  launcher pid=$pid log=$log"
}

launch 0,1 29651 scratch_ed_gt_error_baseline baseline false
launch 2,3 29652 scratch_ed_gt_error_feat     feat     true
launch 4,5 29653 scratch_ed_gt_error_geom     geom     true
launch 6,7 29654 scratch_ed_gt_error_rgbfeat  rgbfeat  false

echo "Recovery wave launched: baseline+rgbfeat restart; feat+geom resume latest checkpoints."
echo "Run: bash scripts/status_gt_error_all.sh"
