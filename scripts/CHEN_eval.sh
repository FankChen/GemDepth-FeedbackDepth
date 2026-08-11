#!/usr/bin/env bash
# ============================================================================
# 直接操作版 · 四集评测脚本（kitti / sintel / bonn / scannet）
# ----------------------------------------------------------------------------
# 用法：
#   bash scripts/CHEN_eval.sh multiscale                 # 评已训完的 multiscale final.pth（四集全）
#   bash scripts/CHEN_eval.sh temporal                   # 评 temporal baseline final.pth（四集全）
#   bash scripts/CHEN_eval.sh multiscale final.pth kitti # 只评 kitti（冒烟/快速）
#   bash scripts/CHEN_eval.sh temporal checkpoint_10000.pth "kitti bonn"
#
# 参数（都可省，有默认）：
#   $1 arm      : multiscale | temporal      (默认 multiscale)
#   $2 ckpt名   : final.pth / checkpoint_N.pth (默认 final.pth)
#   $3 数据集   : "kitti sintel bonn scannet" (默认四集全)
#
# 输出：output_eval/<config>_<ckpt>/  下的 npy + results.txt + 末尾 SUMMARY 汇总表。
# 评测口径：视差空间 lstsq 对齐（与官方 GemDepth 一致），multiscale/temporal 都不取倒数。
# ============================================================================
set -euo pipefail

cd /mnt/data/PROJECT_CHEN/code/PP-DPT

ARM="${1:-multiscale}"
CKPT_NAME="${2:-final.pth}"
DS="${3:-kitti sintel bonn scannet}"

case "$ARM" in
  multiscale) CFG=scratch/dinov3_convnext/scratch_ed_dinov3convnext_ms_mixdata_8gpu ;    NAME=scratch_ed_dinov3convnext_ms_mixdata_8gpu ;;
  temporal)   CFG=scratch/dinov3_convnext/scratch_ed_dinov3convnext_temporal_mixdata_8gpu ; NAME=scratch_ed_dinov3convnext_temporal_mixdata_8gpu ;;
  *) echo "用法: bash scripts/CHEN_eval.sh [multiscale|temporal] [ckpt] [datasets]"; exit 1 ;;
esac

CKPT="checkpoint/${NAME}/${CKPT_NAME}"
if [ ! -f "$CKPT" ]; then
  echo "❌ 找不到 checkpoint: $CKPT"
  echo "   该 arm 现有的 checkpoint:"
  ls -t "checkpoint/${NAME}/"*.pth 2>/dev/null | head || echo "   (无)"
  exit 1
fi

OUT="output_eval/${NAME}_${CKPT_NAME%.pth}"

echo "==============================================="
echo " 评测 arm : $ARM"
echo " config   : config/${CFG}.yaml"
echo " ckpt     : $CKPT"
echo " 数据集   : $DS"
echo " bench    : /mnt/data/PROJECT_CHEN/data/eval"
echo " 输出     : $OUT"
echo "==============================================="

CONFIG="config/${CFG}.yaml" \
CKPT="$CKPT" \
BENCH="/mnt/data/PROJECT_CHEN/data/eval" \
OUT="$OUT" \
DATASETS="$DS" \
GPU=0 \
bash scripts/eval_gemdepth_4bench.sh
