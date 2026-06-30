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
ERROR_MODALITIES=""
WARP_SIGNAL=""

case "$ARM" in
    baseline)    HEAD_TYPE=temporal;        CKPT=./checkpoint/single_a100_baseline/final_model.pth ;;
    errormap)    HEAD_TYPE=errormap;        CKPT=./checkpoint/single_a100_errormap/final_model.pth ;;
    em_rgb)      HEAD_TYPE=errormap_coattn; ERROR_MODALITIES=rgb;     CKPT=./checkpoint/single_a100_em_rgb/final_model.pth ;;
    em_feat)     HEAD_TYPE=errormap_coattn; ERROR_MODALITIES=feat;    CKPT=./checkpoint/single_a100_em_feat/final_model.pth ;;
    em_hog)      HEAD_TYPE=errormap_coattn; ERROR_MODALITIES=hog;     CKPT=./checkpoint/single_a100_em_hog/final_model.pth ;;
    em_rgbfeat)  HEAD_TYPE=errormap_coattn; ERROR_MODALITIES=rgbfeat; CKPT=./checkpoint/single_a100_em_rgbfeat/final_model.pth ;;
    em_refine)   HEAD_TYPE=errormap_refine; CKPT=./checkpoint/single_a100_em_refine/final_model.pth ;;
    em_single)   HEAD_TYPE=errormap_single; WARP_SIGNAL=rgb; CKPT=./checkpoint/single_a100_em_single/final_model.pth ;;
    *) echo "Unknown arm: $ARM (baseline|errormap|em_rgb|em_feat|em_hog|em_rgbfeat|em_refine|em_single)" && exit 1 ;;
esac
[ -n "$CKPT_OVERRIDE" ] && CKPT="$CKPT_OVERRIDE"

# Portable: PY defaults to `python` on PATH (activate env first; override with PY=...).
# ROOT is script-relative; EVAL_ROOT (benchmark data) overridable via env var.
PY=${PY:-python}
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
EVAL_ROOT=${EVAL_ROOT:-"$ROOT/../../datasets/gemdepth_eval"}
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
    ${ERROR_MODALITIES:+--error_modalities "$ERROR_MODALITIES"} \
    ${WARP_SIGNAL:+--warp_signal "$WARP_SIGNAL"} \
    --ckpt "$CKPT"

$PY evaluation/eval/eval.py \
    --infer_path "$OUT_ROOT" \
    --benchmark_path "$EVAL_ROOT" \
    --datasets kitti

echo "[$(date)] eval done: results at $OUT_ROOT/results.txt"
cat "$OUT_ROOT/results.txt" 2>/dev/null || true
