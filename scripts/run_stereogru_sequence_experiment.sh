#!/usr/bin/env bash
# New B0 -> final-only GRU F1 -> sequence-supervised GRU S1; never resume weights.
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
PYTHON=${PYTHON:-/usr/local/bin/python}
GPU=${GPU:-5}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NO_ALBUMENTATIONS_UPDATE=1
[[ -x "$PYTHON" ]] || { echo "Interpreter missing: $PYTHON"; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing"; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout required"; exit 2; }
require_idle() {
    local uuid used processes
    uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
    used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    processes=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
    if [[ "$used" -gt 2048 ]] || grep -Fq "$uuid" <<< "$processes"; then
        echo "GPU=$GPU busy; no next process launched. Do not kill other jobs."
        exit 3
    fi
}
require_idle
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$SOURCE_ROOT"
if [[ -z "${SEQUENCE_RUN:-}" ]]; then
    : "${REPETITIONS:?Set REPETITIONS to the completed three-seed C0/C1 runs directory}"
    OUT=$(mktemp -d "$RUN_ROOT/stereogru_sequence_XXXXXXXX")
    SEQUENCE_RUN="$OUT/run"
    echo "NEW SEQUENCE_RUN=$SEQUENCE_RUN"
    echo "WILL run B0/F1/S1 x seeds0/1/2 x1000 formal updates, plus 6 discarded 200-step gates."
    CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -u \
        scripts/stereogru_sequence_experiment.py prepare --repetitions "$REPETITIONS" --output "$SEQUENCE_RUN" \
        2>&1 | tee "$OUT/prepare.log"
else
    [[ -f "$SEQUENCE_RUN/experiment.json" ]] || { echo "Missing prepared experiment: $SEQUENCE_RUN"; exit 2; }
    OUT=$(dirname -- "$SEQUENCE_RUN")
    echo "VERIFY existing SEQUENCE_RUN=$SEQUENCE_RUN; only NOT-STARTED phases may continue."
fi
while true; do
    echo "VERIFY next phase (read-only source/cache/certificate checks); SEQUENCE_RUN=$SEQUENCE_RUN"
    NEXT=$(timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" \
        scripts/stereogru_sequence_experiment.py next --output "$SEQUENCE_RUN")
    if [[ "$NEXT" == 'SEQUENCE_NEXT summarize - -' ]]; then
        STAGE=summarize
    elif [[ "$NEXT" =~ ^SEQUENCE_NEXT\ (train|gate)\ (B0|F1|S1)\ ([012])$ ]]; then
        STAGE=${BASH_REMATCH[1]}
        ARM=${BASH_REMATCH[2]}
        SEED=${BASH_REMATCH[3]}
    else
        echo "Invalid next-stage protocol (no training launched): $NEXT" >&2
        exit 2
    fi
    if [[ "$STAGE" == "summarize" ]]; then
        timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -u \
            scripts/stereogru_sequence_experiment.py summarize --output "$SEQUENCE_RUN" \
            2>&1 | tee "$OUT/summary.log"
        break
    fi
    [[ "$STAGE" == "train" || "$STAGE" == "gate" ]] || { echo "Invalid next stage: $NEXT"; exit 2; }
    echo "NEXT=$NEXT; SEQUENCE_RUN=$SEQUENCE_RUN"
    require_idle
    LOG="$OUT/${STAGE}_${ARM}_seed${SEED}.log"
    [[ ! -e "$LOG" ]] || { echo "Stage log already exists; inspect partial run, never retry blindly: $LOG"; exit 2; }
    timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -u \
        scripts/stereogru_sequence_experiment.py "$STAGE" --output "$SEQUENCE_RUN" --arm "$ARM" --seed "$SEED" \
        2>&1 | tee "$LOG"
done