#!/usr/bin/env bash
# Read-only model/data probe. New reports only; no optimiser/checkpoint writes.
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/data/PROJECT_CHEN/data/train/vkitti2/vkitti}
PYTHON=${PYTHON:-/usr/local/bin/python}
GPU=${GPU:-4}
: "${PILOT_RUN:?Set PILOT_RUN to the completed pilot run directory}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NO_ALBUMENTATIONS_UPDATE=1
[[ -x "$PYTHON" ]] || { echo "Missing interpreter: $PYTHON"; exit 2; }
for name in config.json manifest.json provenance.json summary.json baseline_head.pth; do
    [[ -f "$PILOT_RUN/$name" ]] || { echo "Missing pilot artifact: $PILOT_RUN/$name"; exit 2; }
done
command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing"; exit 2; }
nvidia-smi --query-gpu=index,name,memory.used --format=csv
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
if [[ "$USED" -gt 2048 ]] || grep -Fq "$GPU_UUID" <<< "$PROCESSES"; then
    echo "GPU=$GPU busy; no readout launched. Choose an idle GPU."
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$GPU"
OUT=$(mktemp -d "$RUN_ROOT/stereogru_readout_XXXXXXXX")
echo "Read-only pilot dependency check: $OUT"
echo "Replay saved pilot, then fixed-weight raw-volume interventions; NO training."
cd "$SOURCE_ROOT"
"$PYTHON" -u scripts/diagnose_stereogru_pilot.py --pilot-run "$PILOT_RUN" \
    --vkitti-root "$VKITTI_ROOT" --output "$OUT/results" 2>&1 | tee "$OUT/readout.log"