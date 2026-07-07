#!/bin/bash -l
# Sweep the BASELINE (temporal DPT head) VKITTI head-only finetune over learning rates.
#
# WHY: em_single(rgb) got 0.088 vs the known baseline 0.0677. This sweep reruns the
# ORIGINAL temporal head on Aliyun across several lrs to answer:
#   - if some lr reproduces ~0.0677  -> env/data are fine, em_single's gap is a METHOD/head issue
#   - if ALL lrs also land at ~0.08x -> it's a SETTING issue (data/env/lr), not em_single-specific
#
# head_only training => head params live in the optimizer "other" group, so we sweep
# optimizer.other_lr. Each lr runs on its own GPU in parallel (nohup, survives logout).
#
#   export VKITTI_ROOT=/mnt/workspace/vkitti2/vkitti
#   PY=/usr/local/bin/python3 GPUS="3 4 5 6" ./scripts/sweep_baseline_lr.sh
#
# Env overrides: PY, GPUS (space list), LRS (space list), STEPS, VKITTI_ROOT, CONFIG.
set -e
PY=${PY:-/usr/local/bin/python3}
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"

CONFIG=${CONFIG:-single_a100_baseline}
LRS=${LRS:-"1e-4 5e-5 1e-5 1e-6"}
GPUS=${GPUS:-"3 4 5 6"}
STEPS=${STEPS:-10000}

if [ -z "${VKITTI_ROOT:-}" ]; then
  echo "WARNING: VKITTI_ROOT not set; config uses its default path. (export VKITTI_ROOT=/mnt/workspace/vkitti2/vkitti)" >&2
fi

read -ra LR_ARR <<< "$LRS"
read -ra GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -lt "${#LR_ARR[@]}" ]; then
  echo "ERROR: need >= ${#LR_ARR[@]} GPUs (got ${#GPU_ARR[@]}: '$GPUS')" >&2
  exit 1
fi

echo "=== baseline lr sweep: config=$CONFIG  LRS='$LRS'  GPUS='$GPUS'  STEPS=$STEPS ==="
mkdir -p logs
i=0
for LR in "${LR_ARR[@]}"; do
  GPU=${GPU_ARR[$i]}
  TAG="baseline_lr${LR}"
  CKDIR="./checkpoint/sweep_${TAG}"
  LOGDIR="./logs/sweep_${TAG}"
  LOG="train_sweep_${TAG}.log"
  echo "[$(date +%H:%M:%S)] launch $TAG on GPU $GPU -> $LOG"
  CUDA_VISIBLE_DEVICES=$GPU nohup $PY -m accelerate.commands.launch \
      --num_processes 1 --mixed_precision bf16 \
      train.py --config-name "$CONFIG" \
      optimizer.other_lr="$LR" \
      training.checkpoint_dir="$CKDIR" \
      training.log_dir="$LOGDIR" \
      total_step="$STEPS" \
      project_name="sweep_${TAG}" \
      > "$LOG" 2>&1 &
  echo "    PID $!"
  i=$((i + 1))
done

echo ""
echo "all $i jobs launched (nohup, survive logout). monitor with:"
echo "  tail -f train_sweep_baseline_lr*.log"
echo "  ls -lt checkpoint/sweep_baseline_lr*/*.pth"
echo ""
echo "tomorrow, eval each with:"
echo "  for lr in $LRS; do BENCH=/mnt/workspace/gemdepth_eval PY=$PY \\"
echo "    ./scripts/eval_all.sh checkpoint/sweep_baseline_lr\$lr/final_model.pth temporal rgb; done"
