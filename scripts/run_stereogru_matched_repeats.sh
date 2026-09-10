#!/usr/bin/env bash
# Explicit four-run repetition plan: seed1 C0->C1, then seed2 C0->C1. NO GRU.
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
PYTHON=${PYTHON:-/usr/local/bin/python}
GPU=${GPU:-4}
: "${SOURCE_EXPERIMENT:?Set SOURCE_EXPERIMENT to the completed seed-0 C0/C1 experiment}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NO_ALBUMENTATIONS_UPDATE=1
[[ -x "$PYTHON" ]] || { echo "Missing interpreter: $PYTHON"; exit 2; }
OUT=$(mktemp -d "$RUN_ROOT/stereogru_seed_repeats_XXXXXXXX")
echo "PAIRED REPEATS: seed1 C0->C1, seed2 C0->C1; each 1000 steps, same data/loss/model."
echo "This command WILL train four runs. Each C1 waits for its matched C0 certificate. No GRU."
echo "Repetition outputs: $OUT/runs"
cd "$SOURCE_ROOT"
"$PYTHON" -u scripts/stereogru_matched_repeats.py prepare --source-experiment "$SOURCE_EXPERIMENT" \
    --output "$OUT/runs" 2>&1 | tee "$OUT/prepare.log"
for SEED in 1 2; do
    for ARM in C0 C1; do
        echo "NEXT: seed=$SEED arm=$ARM"
        MATCHED_RUN="$OUT/runs/seed$SEED/experiment" ARM="$ARM" GPU="$GPU" RUN_ROOT="$RUN_ROOT" PYTHON="$PYTHON" \
            bash "$SOURCE_ROOT/scripts/run_stereogru_matched_controls.sh"
    done
done
"$PYTHON" -u scripts/stereogru_matched_repeats.py summarize --output "$OUT/runs" 2>&1 | tee "$OUT/summary.log"