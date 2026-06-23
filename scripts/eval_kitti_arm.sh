#!/bin/bash -l
# Evaluate a trained arm on KITTI (inference -> metrics).
#
#   ./scripts/eval_kitti_arm.sh baseline
#   ./scripts/eval_kitti_arm.sh errormap
#   ./scripts/eval_kitti_arm.sh baseline ./checkpoint/single_a100_baseline/checkpoint_8000.pth
#
# Produces per-arm npy depths and a results.txt under output_eval/<arm>/.
set -e

ARM=${1:-baseline}
CKPT_OVERRIDE=${2:-}

case "$ARM" in
    baseline) HEAD_TYPE=temporal; CKPT=./checkpoint/single_a100_baseline/final_model.pth ;;
    errormap) HEAD_TYPE=errormap; CKPT=./checkpoint/single_a100_errormap/final_model.pth ;;
    *) echo "Unknown arm: $ARM (use 'baseline' or 'errormap')" && exit 1 ;;
esac
[ -n "$CKPT_OVERRIDE" ] && CKPT="$CKPT_OVERRIDE"

PY=/home/izi2sgh/MYDATA/quanjie/liren/envs/gemdepth/bin/python
ROOT=/home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth
EVAL_ROOT=/home/izi2sgh/MYDATA/quanjie/liren/datasets/gemdepth_eval
OUT_ROOT=$ROOT/output_eval/$ARM
JSON=$EVAL_ROOT/kitti/kitti_video.json

cd "$ROOT"
mkdir -p "$OUT_ROOT" jobs

echo "[$(date)] eval arm=$ARM head_type=$HEAD_TYPE ckpt=$CKPT"
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT" && exit 1; }

$PY evaluation/inference/infer.py \
    --infer_path "$OUT_ROOT" \
    --json_file "$JSON" \
    --datasets kitti \
    --input_size 518 \
    --encoder vitl \
    --head_type "$HEAD_TYPE" \
    --ckpt "$CKPT"

$PY evaluation/eval/eval.py \
    --infer_path "$OUT_ROOT" \
    --benchmark_path "$EVAL_ROOT" \
    --datasets kitti

echo "[$(date)] eval done: results at $OUT_ROOT/results.txt"
cat "$OUT_ROOT/results.txt" 2>/dev/null || true
