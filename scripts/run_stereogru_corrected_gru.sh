#!/usr/bin/env bash
# Corrected G1: completed C1 checks -> disposable gates -> RESET -> three formal seeds.
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
PYTHON=${PYTHON:-/usr/local/bin/python}
GPU=${GPU:-5}
: "${REPETITIONS:?Set REPETITIONS to the completed three-seed C0/C1 runs directory}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NO_ALBUMENTATIONS_UPDATE=1
[[ -x "$PYTHON" ]] || { echo "Interpreter missing: $PYTHON"; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing"; exit 2; }
command -v timeout >/dev/null || { echo "GNU timeout required"; exit 2; }
require_idle() {
    local uuid used processes
    nvidia-smi --query-gpu=index,name,memory.used --format=csv
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
OUT=$(mktemp -d "$RUN_ROOT/stereogru_corrected_gru_XXXXXXXX")
echo "G1 output: $OUT/run"
echo "WILL train disposable 200-step gates, then RESET and train seeds0/1/2 for 1000 steps each."
echo "GT metric cameras, raw+GEV lookup, bounded index variant, same final-only C1 objective."
cd "$SOURCE_ROOT"
"$PYTHON" -u scripts/stereogru_corrected_gru.py prepare --repetitions "$REPETITIONS" --output "$OUT/run" 2>&1 | tee "$OUT/prepare.log"
# All gates must pass before spending the formal-run budget; no dev tuning or relaxed retries.
for SEED in 0 1 2; do
    require_idle
    timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -u scripts/stereogru_corrected_gru.py gate \
        --output "$OUT/run" --seed "$SEED" 2>&1 | tee "$OUT/gate_seed$SEED.log"
done
for SEED in 0 1 2; do
    require_idle
    timeout --signal=TERM --kill-after=30s 3600 "$PYTHON" -u scripts/stereogru_corrected_gru.py train \
        --output "$OUT/run" --seed "$SEED" 2>&1 | tee "$OUT/train_seed$SEED.log"
done
"$PYTHON" -u scripts/stereogru_corrected_gru.py summarize --output "$OUT/run" 2>&1 | tee "$OUT/summary.log"