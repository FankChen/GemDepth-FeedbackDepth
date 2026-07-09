#!/bin/bash -l
# Fill every IDLE GPU with a different training arm (over-night ablation sweep).
# One command to saturate the node before you leave:
#   PY=/usr/local/bin/python3 VKITTI_ROOT=/mnt/workspace/vkitti2/vkitti bash scripts/launch_free_gpus.sh
#
# A GPU whose used-memory < THRESH_MB (default 2000) is considered free; running arms (e.g. the
# ones already on GPU2/GPU7) are left untouched. Each arm gets one GPU via CUDA_VISIBLE_DEVICES,
# runs under nohup, and logs to jobs/<arm>.log. Override the queue with ARMS="a b c".
set -u
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"
PY=${PY:-python}
THRESH_MB=${THRESH_MB:-2000}
export VKITTI_ROOT=${VKITTI_ROOT:-/mnt/workspace/vkitti2/vkitti}
mkdir -p jobs

# Ablation queue (highest-value first). Each name must exist in train_single_a100.sh.
ARMS=(${ARMS:-batlin_cycle baseline_cycle batlin_4scale batlin_rgbfeat batlin_hog batlin_cycle_4scale batlin_cycle_o12})

mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F',' -v t="$THRESH_MB" '{gsub(/ /,"",$2); if ($2+0 < t) print $1}')

echo "VKITTI_ROOT=$VKITTI_ROOT"
echo "Free GPUs (< ${THRESH_MB}MB used): ${FREE[*]:-none}"
echo "Queue: ${ARMS[*]}"
echo ""

i=0
for gpu in "${FREE[@]}"; do
    [ $i -lt ${#ARMS[@]} ] || { echo "(no more arms in queue; $((${#FREE[@]}-i)) GPU(s) left idle)"; break; }
    arm=${ARMS[$i]}
    log="jobs/${arm}.log"
    echo "[GPU$gpu] launch '$arm'  ->  $log"
    CUDA_VISIBLE_DEVICES=$gpu PY=$PY nohup bash scripts/train_single_a100.sh "$arm" > "$log" 2>&1 &
    echo "    pid=$!  (CUDA_VISIBLE_DEVICES=$gpu)"
    i=$((i + 1))
    sleep 8   # stagger so they don't hit CPFS / checkpoint load simultaneously
done

echo ""
echo "Launched $i arm(s). Verify in ~1 min:"
echo "  PY=$PY bash scripts/check_progress.sh"
echo "  tail -f jobs/${ARMS[0]}.log"
