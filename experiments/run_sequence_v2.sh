#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
PYTHON=${PYTHON:-/usr/local/bin/python}
GPU=${GPU:-5}
OLD="$RUN_ROOT/stereogru_sequence_wcIm3t5e/run"
AUDIT="$RUN_ROOT/sequence_audit_XR5svFMg/report"
OUT="$RUN_ROOT/sequence_v2_wcIm3t5e"
exec 9>"$OUT.lock"
flock -n 9 || { echo 'ALREADY_RUNNING: no duplicate training launched'; exit 3; }
trap 'r=$?; if [[ $r -ne 0 ]]; then echo "V2_STOPPED exit=$r; no next phase; preserve $OUT"; fi' EXIT
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1 CUDA_VISIBLE_DEVICES="$GPU"
idle() {
    local uuid used processes
    uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
    if [[ "$used" -gt 2048 ]] || grep -Fq "$uuid" <<< "$processes"; then
        echo "GPU_BUSY=$GPU; no training started for next phase"; exit 3
    fi
}
cd "$SOURCE_ROOT"
idle
if [[ ! -e "$OUT" ]]; then
    echo 'PREPARING_ACCEPTANCE_V2: no optimizer updates yet; retaining B0 seed0, not restarting it'
    timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -B -u experiments/sequence_acceptance_v2.py prepare \
        --source "$OLD" --audit "$AUDIT" --output "$OUT"
fi
while true; do
    echo "VERIFY_NEXT: training inactive during certificate checks; output=$OUT"
    NEXT=$(timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -B experiments/sequence_acceptance_v2.py next --output "$OUT")
    if [[ "$NEXT" == 'V2_NEXT summary - -' ]]; then
        timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -B -u experiments/sequence_acceptance_v2.py summary --output "$OUT"
        break
    fi
    [[ "$NEXT" =~ ^V2_NEXT\ (formal|gate)\ (B0|F1|S1)\ ([012])$ ]] || { echo "INVALID_ACTION=$NEXT"; exit 2; }
    STAGE=${BASH_REMATCH[1]} ARM=${BASH_REMATCH[2]} SEED=${BASH_REMATCH[3]}
    idle
    [[ ! -e "$OUT/${STAGE}_${ARM}_${SEED}.log" ]] || { echo 'Existing phase log: inspect failure, no blind retry'; exit 2; }
    echo "LAUNCHING_TRAINING stage=$STAGE arm=$ARM seed=$SEED GPU=$GPU; wait for TRAIN step to confirm updates"
    timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -B -u experiments/sequence_acceptance_v2.py "$STAGE" \
        --output "$OUT" --arm "$ARM" --seed "$SEED" 2>&1 | tee "$OUT/${STAGE}_${ARM}_${SEED}.log"
done