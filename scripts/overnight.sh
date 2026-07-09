#!/bin/bash -l
# Overnight auto-fill: keep every idle GPU busy from a long queue (skips arms that are RUNNING or
# already DONE), then TAE-eval everything at the end. Leaves current runs untouched.
# nohup and walk away:
#   PY=/usr/local/bin/python3 VKITTI_ROOT=/mnt/workspace/vkitti2/vkitti \
#     nohup bash scripts/overnight.sh > jobs/overnight.log 2>&1 &
set -u
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"
export PY=${PY:-python}
export VKITTI_ROOT=${VKITTI_ROOT:-/mnt/workspace/vkitti2/vkitti}
mkdir -p jobs

# Batch 1: batlin warp-fix re-runs (were silently no-op before the geometry fix).
# Batch 2: perlayer non_backbone deepening (warp vs no-warp).
QUEUE="${QUEUE:-batlin_4scale batlin_rgbfeat batlin_hog batlin_cycle batlin_cycle_4scale batlin_cycle_o12 perlayer_nb perlayer_nowarp_nb}"
LOOPS=${LOOPS:-33}     # 33 * 20min ≈ 11h
SLEEP=${SLEEP:-1200}

echo "=== [overnight] start $(date); queue: $QUEUE ==="
for ((k = 0; k < LOOPS; k++)); do
    echo "--- [overnight loop $k @ $(date)] fill idle GPUs ---"
    ARMS="$QUEUE" bash scripts/launch_free_gpus.sh || true
    sleep "$SLEEP"
done

echo "=== [overnight] loops done, TAE eval on the least-busy GPU @ $(date) ==="
gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
CUDA_VISIBLE_DEVICES="${gpu:-0}" bash scripts/eval_tae_all.sh > jobs/tae_final.log 2>&1 || true
echo "=== [overnight] finished $(date). See jobs/tae_final.log and check_progress.sh ==="
