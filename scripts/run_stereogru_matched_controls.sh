#!/usr/bin/env bash
# Run ONE matched arm. No pull/reset, legacy writes, auto next arm or GRU.
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/data/PROJECT_CHEN/data/train/vkitti2/vkitti}
PYTHON=${PYTHON:-/usr/local/bin/python}
GPU=${GPU:-4}
ARM=${ARM:-C0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NO_ALBUMENTATIONS_UPDATE=1
[[ -x "$PYTHON" ]] || { echo "Missing interpreter: $PYTHON"; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable"; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout is required for the hard per-arm safety limit"; exit 2; }
require_idle_gpu() {
    local uuid used processes
    nvidia-smi --query-gpu=index,name,memory.used --format=csv
    uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
    if [[ "$used" -gt 2048 ]] || grep -Fq "$uuid" <<< "$processes"; then
        echo "GPU=$GPU busy; no job launched. Choose an idle GPU."
        exit 3
    fi
}
require_idle_gpu
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$SOURCE_ROOT"
if [[ -z "${MATCHED_RUN:-}" ]]; then
    : "${BACKBONE_WEIGHTS:?For a new experiment, set BACKBONE_WEIGHTS to the verified clean export}"
    [[ -f "$BACKBONE_WEIGHTS" ]] || { echo "Clean weights missing: $BACKBONE_WEIGHTS"; exit 2; }
    [[ -d "$VKITTI_ROOT" ]] || { echo "Data missing: $VKITTI_ROOT"; exit 2; }
    MATCHED_RUN=$(mktemp -d "$RUN_ROOT/stereogru_matched_XXXXXXXX")
    # Logs live outside the prepared experiment, whose directory must be empty.
    "$PYTHON" -u scripts/stereogru_matched_controls.py prepare \
        --backbone-weights "$BACKBONE_WEIGHTS" --vkitti-root "$VKITTI_ROOT" \
        --output "$MATCHED_RUN/experiment" 2>&1 | tee "$MATCHED_RUN/prepare.log"
    MATCHED_RUN="$MATCHED_RUN/experiment"
fi
echo "Matched experiment: $MATCHED_RUN"
require_idle_gpu
echo "Running ONLY $ARM; 1000 fixed updates, 3600s process cap. Prerequisites are verified before training."
LOG=$(mktemp "$RUN_ROOT/stereogru_${ARM}_XXXXXXXX.log")
timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -u scripts/stereogru_matched_controls.py train \
    --experiment "$MATCHED_RUN" --arm "$ARM" 2>&1 | tee "$LOG"
echo "Finished $ARM. Experiment: $MATCHED_RUN"
echo "No next arm launched. Preserve this path for the matched follow-up. Log: $LOG"