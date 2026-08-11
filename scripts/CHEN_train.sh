#!/usr/bin/env bash
# ============================================================================
# 直接操作版 · 训练脚本（阿里云 8×GPU）
# ----------------------------------------------------------------------------
# 用法：
#   bash scripts/CHEN_train.sh temporal              # baseline，先跑这个
#   bash scripts/CHEN_train.sh multiscale            # method，baseline 出来后再跑
#   TAG=pr8 bash scripts/CHEN_train.sh temporal      # 写到独立目录，不碰旧结果
#
# 说明：两个 arm 用的是同一份 6 数据集混训配方（同数据/同预算/同 backbone/同 batch/
#       同步数），唯一差别是模型头（multiscale vs 单尺度 temporal）——这是公平对照。
#       ⚠ 两个 arm 必须用同一份数据和同一个 TAG，否则不构成对照。
#       后台运行，日志在 runlogs/<name>[_<tag>].log，用 tail -f 看进度。
# ============================================================================
set -euo pipefail

cd /mnt/data/PROJECT_CHEN/code/PP-DPT

ARM="${1:-temporal}"
TAG="${TAG:-}"
case "$ARM" in
  multiscale) CFG=scratch/dinov3_convnext/scratch_ed_dinov3convnext_ms_mixdata_8gpu ;    NAME=scratch_ed_dinov3convnext_ms_mixdata_8gpu ;;
  temporal)   CFG=scratch/dinov3_convnext/scratch_ed_dinov3convnext_temporal_mixdata_8gpu ; NAME=scratch_ed_dinov3convnext_temporal_mixdata_8gpu ;;
  *) echo "用法: [TAG=xxx] bash scripts/CHEN_train.sh [multiscale|temporal]"; exit 1 ;;
esac

# config 必须存在
[ -f "config/${CFG}.yaml" ] || { echo "❌ 找不到 config/${CFG}.yaml"; exit 1; }

# TAG 非空时写到独立的 checkpoint/log 目录，避免覆盖已完成的训练
RUN="${NAME}${TAG:+_$TAG}"
CKPT_DIR="./checkpoint/${RUN}"
LOG_DIR="./logs/${RUN}"

mkdir -p runlogs
LOG="runlogs/${RUN}.log"

echo "==============================================="
echo " 训练 arm : $ARM"
echo " config   : config/${CFG}.yaml"
echo " tag      : ${TAG:-<none>}"
echo " 日志     : $LOG"
echo " checkpoint 存到: ${CKPT_DIR}/"
echo "==============================================="

NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_DEBUG=WARN OPENCV_IO_ENABLE_OPENEXR=1 \
NUM_PROC=8 nohup python -m accelerate.commands.launch \
  --num_processes 8 --mixed_precision bf16 \
  train.py --config-name="$CFG" \
  training.checkpoint_dir="$CKPT_DIR" \
  training.log_dir="$LOG_DIR" > "$LOG" 2>&1 &

echo "✅ 已后台启动，PID=$!"
echo "   看进度:  tail -f $LOG"
echo "   停止:    pkill -f 'config-name=$CFG'"
echo "   评测:    TAG=${TAG:-} bash scripts/CHEN_eval.sh $ARM"
