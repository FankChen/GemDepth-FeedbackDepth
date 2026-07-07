#!/bin/bash -l
# One-shot evaluation of a trained checkpoint across all standard benchmarks.
#
#   ./scripts/eval_all.sh <ckpt> [head_type] [warp_signal]
#
# Example:
#   ./scripts/eval_all.sh checkpoint/single_a100_em_single_hog/final_model.pth errormap_single hog
#   ./scripts/eval_all.sh checkpoint/gemdepth_baseline/final_model.pth temporal rgb
#
# Runs infer + eval on KITTI / Sintel / Bonn / ScanNet using the SAME protocol as the
# baseline (kitti=110, sintel, bonn_500, scannet_500), writes predictions and a combined
# results.txt under output_eval/<tag>/.
#
# Env overrides:
#   PY        python (default /usr/local/bin/python3  -- Aliyun system python)
#   BENCH     gemdepth_eval root (default /mnt/workspace/gemdepth_eval)
#   OUT       output dir (default output_eval/<ckpt-parent-dir-name>)
#   DATASETS  subset to run (default "kitti sintel bonn scannet")
set -e

CKPT=${1:?"usage: eval_all.sh <ckpt> [head_type] [warp_signal]"}
HEAD_TYPE=${2:-errormap_single}
WARP_SIGNAL=${3:-rgb}

PY=${PY:-/usr/local/bin/python3}
BENCH=${BENCH:-/mnt/workspace/gemdepth_eval}
TAG=$(basename "$(dirname "$CKPT")")
OUT=${OUT:-output_eval/$TAG}
DATASETS=${DATASETS:-"kitti sintel bonn scannet"}
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"

# infer uses the json key; eval.py uses its own dataset key (some are *_500 long-seq).
json_of() { case "$1" in
  kitti)   echo "kitti/kitti_video.json" ;;
  sintel)  echo "sintel/sintel_video.json" ;;
  bonn)    echo "bonn/bonn_video_500.json" ;;
  scannet) echo "scannet/scannet_video_500.json" ;;
esac ; }
evalkey_of() { case "$1" in
  kitti)   echo "kitti" ;;
  sintel)  echo "sintel" ;;
  bonn)    echo "bonn_500" ;;
  scannet) echo "scannet_500" ;;
esac ; }

if [ ! -f "$CKPT" ]; then echo "ERROR: ckpt not found: $CKPT" >&2; exit 1; fi
echo "=== eval_all: ckpt=$CKPT  head=$HEAD_TYPE  signal=$WARP_SIGNAL ==="
echo "=== bench=$BENCH  out=$OUT  datasets='$DATASETS' ==="
mkdir -p "$OUT"
: > "$OUT/results.txt"   # reset combined results (eval.py appends)

for ds in $DATASETS; do
  JS="$BENCH/$(json_of "$ds")"
  if [ ! -f "$JS" ]; then echo "--- skip $ds: missing json $JS ---"; continue; fi
  echo ""
  echo "===== [$ds] infer ====="
  $PY evaluation/inference/infer.py \
      --json_file "$JS" --datasets "$ds" \
      --ckpt "$CKPT" --head_type "$HEAD_TYPE" --warp_signal "$WARP_SIGNAL" \
      --infer_path "$OUT"
  echo "===== [$ds] eval ($(evalkey_of "$ds")) ====="
  $PY evaluation/eval/eval.py \
      --infer_path "$OUT" --benchmark_path "$BENCH" --datasets "$(evalkey_of "$ds")"
done

echo ""
echo "########## COMBINED SUMMARY ($TAG) ##########"
cat "$OUT/results.txt"
echo "##############################################"
echo "results file: $OUT/results.txt"
