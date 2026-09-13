#!/usr/bin/env bash
# Read-only failed-run audit; never invokes train/gate/prepare or writes completion.
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
PYTHON=${PYTHON:-/usr/local/bin/python}
GPU=${GPU:-5}
: "${SEQUENCE_RUN:?Set SEQUENCE_RUN to the retained failed experiment}"
RUN_ROOT=$(cd -- "$RUN_ROOT" && pwd -P)
SEQUENCE_RUN=$(cd -- "$SEQUENCE_RUN" && pwd -P)
if [[ "$RUN_ROOT" == "$SOURCE_ROOT" || "$RUN_ROOT" == "$SOURCE_ROOT/"* || "$RUN_ROOT" == "$SEQUENCE_RUN" || "$RUN_ROOT" == "$SEQUENCE_RUN/"* ]]; then
    echo "Audit output root must not be inside frozen code or the failed experiment."
    exit 2
fi
[[ -x "$PYTHON" ]] || { echo "Interpreter missing"; exit 2; }
command -v timeout >/dev/null
command -v nvidia-smi >/dev/null
uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
if [[ "$used" -gt 2048 ]] || grep -Fq "$uuid" <<< "$processes"; then
    echo "GPU=$GPU busy; audit not started. Do not kill other jobs."
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONDONTWRITEBYTECODE=1
OUT=$(mktemp -d "$RUN_ROOT/sequence_audit_XXXXXXXX")
echo "READ_ONLY_AUDIT=$OUT; no formal training, no completion, no automatic method launch."
cd "$SOURCE_ROOT"
timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -B -u diagnostics/sequence_replay_audit.py \
    --experiment "$SEQUENCE_RUN" --report "$OUT/report" --seed 0 2>&1 | tee "$OUT/audit.log"