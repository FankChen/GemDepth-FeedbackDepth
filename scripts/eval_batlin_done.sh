#!/bin/bash -l
# Evaluate the already-trained BATLin arms on KITTI/Sintel/Bonn/ScanNet (single-frame AbsRel).
#
# Why a dedicated script: eval_all.sh does NOT forward `--scales`, but the batlin head is
# built with per-arm scales (the *_4scale arms use [p4 p3 p2 p1]; the rest use [p2 p1]).
# Evaluating with the wrong scales => head structure mismatch => checkpoint load fails.
# This script passes the correct scales per arm and reuses infer.py + eval.py.
#
# Usage (Aliyun):
#   CUDA_VISIBLE_DEVICES=2 PY=/usr/local/bin/python3 BENCH=/mnt/workspace/gemdepth_eval \
#     bash scripts/eval_batlin_done.sh 2>&1 | tee jobs/eval_batlin.log
#
# Env overrides: PY, BENCH, DATASETS ("kitti sintel bonn scannet"), ARMS (list below).
# Skips arms without final_model.pth and datasets without a benchmark json.
set -u
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}; cd "$ROOT"
PY=${PY:-/usr/local/bin/python3}
BENCH=${BENCH:-/mnt/workspace/gemdepth_eval}
DATASETS=${DATASETS:-"kitti sintel bonn scannet"}

# "checkpoint_dir : scales"  (all batlin/feat). single_a100_batlin (base) is added once it
# finishes training; add it here as:  single_a100_batlin : p2 p1
ARMS=${ARMS:-'
single_a100_batlin_cycle : p2 p1
single_a100_batlin_cycle_o12 : p2 p1
single_a100_batlin_4scale : p4 p3 p2 p1
single_a100_batlin_cycle_4scale : p4 p3 p2 p1
'}

json_of() { case "$1" in
  kitti)   echo kitti/kitti_video.json ;;
  sintel)  echo sintel/sintel_video.json ;;
  bonn)    echo bonn/bonn_video_500.json ;;
  scannet) echo scannet/scannet_video_500.json ;;
  vkitti)  echo vkitti/vkitti_video.json ;;
esac; }
evalkey_of() { case "$1" in
  kitti)   echo kitti ;;
  sintel)  echo sintel ;;
  bonn)    echo bonn_500 ;;
  scannet) echo scannet_500 ;;
  vkitti)  echo vkitti ;;
esac; }

printf '%s\n' "$ARMS" | while IFS= read -r line; do
  [ -n "${line// /}" ] || continue
  arm="$(echo "${line%%:*}" | xargs)"
  scales="$(echo "${line#*:}" | xargs)"
  [ -n "$arm" ] || continue
  ckpt="checkpoint/$arm/final_model.pth"
  if [ ! -f "$ckpt" ]; then echo "[skip] $arm (no final_model.pth)"; continue; fi
  OUT="output_eval/$arm"; mkdir -p "$OUT"; : > "$OUT/results.txt"
  echo ""; echo "########## $arm  (batlin/feat, scales=[$scales]) ##########"
  for ds in $DATASETS; do
    JS="$BENCH/$(json_of "$ds")"
    if [ ! -f "$JS" ]; then echo "--- skip $ds: missing json $JS ---"; continue; fi
    echo "===== [$ds] infer ====="
    $PY evaluation/inference/infer.py --json_file "$JS" --datasets "$ds" \
        --ckpt "$ckpt" --head_type batlin --warp_signal feat --scales $scales --infer_path "$OUT" || true
    echo "===== [$ds] eval ($(evalkey_of "$ds")) ====="
    $PY evaluation/eval/eval.py --infer_path "$OUT" --benchmark_path "$BENCH" --datasets "$(evalkey_of "$ds")" || true
  done
  echo "########## results: $arm ##########"; cat "$OUT/results.txt"
done

echo ""; echo "================= BATLin SUMMARY (single-frame AbsRel / delta1) ================="
printf '%s\n' "$ARMS" | while IFS= read -r line; do
  [ -n "${line// /}" ] || continue
  arm="$(echo "${line%%:*}" | xargs)"; [ -n "$arm" ] || continue
  f="output_eval/$arm/results.txt"
  [ -f "$f" ] || continue
  echo "--- $arm ---"; grep -E "start |abs_relative_difference|delta1_acc" "$f" 2>/dev/null
done
