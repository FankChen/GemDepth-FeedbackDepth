#!/usr/bin/env bash
# Eval all multiscale-ablation experiments on KITTI, SAME protocol as the 0.170 / temporal
# baselines (--datasets kitti, kitti_video.json). Per experiment: infer -> (invert if metric) -> eval.
#
# Run (from anywhere; cd's to repo root):  bash scripts/eval_all_ablation.sh
# Pick a free GPU:                          GPU=7 bash scripts/eval_all_ablation.sh
#
# invert=1 (A/B/D, metric depth output): NPYs are taken 1/D before eval, because eval.py treats the
#          prediction as inverse depth (disparity) and aligns in disparity space.
# invert=0 (C, video/disparity output):  no inversion.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1
PY="${PY:-python}"
BENCH="${BENCH:-/mnt/workspace/gemdepth_eval}"
GPU="${GPU:-0}"
JSON="$BENCH/kitti/kitti_video.json"

# name : invert
JOBS=(
  "scratch_ed_dinov3vits_ms_A_orig_l2:1"
  "scratch_ed_dinov3vits_ms_C_native_video:0"
  "scratch_ed_dinov3vits_ms_D_allgt_gamma08:1"
  "scratch_ed_dinov3convnext_ms_A_orig_l2:1"
  "scratch_ed_dinov3convnext_ms_B_native_l2:1"
  "scratch_ed_dinov3convnext_ms_C_native_video:0"
  "scratch_ed_dinov3convnext_ms_D_allgt_gamma08:1"
)

for job in "${JOBS[@]}"; do
  cfg="${job%%:*}"; inv="${job##*:}"
  ckpt="checkpoint/${cfg}/final_model.pth"
  out="output_eval/${cfg}"
  echo "==================== ${cfg}  (invert=${inv}) ===================="
  if [ ! -f "$ckpt" ]; then echo "  [skip] 无 $ckpt (还没训完?)"; continue; fi
  # 1) infer -> per-frame NPYs
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" evaluation/inference/infer.py \
    --config "config/${cfg}.yaml" --ckpt "$ckpt" \
    --json_file "$JSON" --datasets kitti --infer_path "$out" \
    || { echo "  [infer 失败]"; continue; }
  # 2) metric output -> take 1/D (infer just wrote metric depth; eval expects disparity)
  if [ "$inv" = "1" ]; then
    "$PY" -c "import numpy as np,glob;[np.save(f,1.0/np.clip(np.load(f),1e-3,None)) for f in glob.glob('$out/kitti/**/*.npy',recursive=True)]"
    echo "  [invert] 已对 $out 取倒数(米制->视差)"
  fi
  # 3) eval (eval.py derives the json from --datasets)
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" evaluation/eval/eval.py \
    --infer_path "$out" --benchmark_path "$BENCH" --datasets kitti
done

echo; echo "==================== 汇总 (AbsRel / RMSE / delta1) ===================="
for job in "${JOBS[@]}"; do
  cfg="${job%%:*}"
  echo "----- $cfg -----"
  grep -E 'abs_relative|rmse|delta1' "output_eval/${cfg}/results.txt" 2>/dev/null | tail -3
done
