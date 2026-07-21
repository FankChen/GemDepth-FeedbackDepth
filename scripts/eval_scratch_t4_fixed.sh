#!/bin/bash
# Re-evaluate scratch T=4 models with the training-matched non-overlapping
# clip protocol and fail-fast finite/stable SSI checks.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_scratch_t4_fixed.sh temporal
#   CUDA_VISIBLE_DEVICES=1 bash scripts/eval_scratch_t4_fixed.sh multiscale
set -euo pipefail

ARM=${1:-}
PY=${PY:-/usr/local/bin/python3}
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
BENCH=${BENCH:-/mnt/workspace/gemdepth_eval}

case "$ARM" in
  temporal)
    CONFIG=config/scratch_ed_dinov2_temporal.yaml
    CKPT=checkpoint/scratch_ed_dinov2_temporal/final_model.pth
    NAME=scratch_ed_dinov2_temporal_t4_fixed
    ;;
  multiscale)
    CONFIG=config/scratch_ed_dinov2_multiscale.yaml
    CKPT=checkpoint/scratch_ed_dinov2_multiscale_v2/final_model.pth
    NAME=scratch_ed_dinov2_multiscale_v2_t4_fixed
    ;;
  *)
    echo "Usage: $0 {temporal|multiscale}" >&2
    exit 2
    ;;
esac

cd "$ROOT"
OUT="output_eval/$NAME"
JSON="$BENCH/kitti/kitti_video.json"
mkdir -p jobs

for required in "$PY" "$CONFIG" "$CKPT" "$JSON"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: missing $required" >&2
    exit 1
  fi
done

# Old NPY files were generated with the wrong default 32-frame protocol and must
# never be mixed into this run.
rm -rf "$OUT"
mkdir -p "$OUT"

echo "[$(date)] infer arm=$ARM config=$CONFIG ckpt=$CKPT"
"$PY" evaluation/inference/infer.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --json_file "$JSON" \
  --datasets kitti \
  --input_size 518 \
  --infer_path "$OUT"

echo "[$(date)] evaluate arm=$ARM"
"$PY" evaluation/eval/eval.py \
  --infer_path "$OUT" \
  --benchmark_path "$BENCH" \
  --datasets kitti

echo "[$(date)] completed $ARM"
cat "$OUT/results.txt"
