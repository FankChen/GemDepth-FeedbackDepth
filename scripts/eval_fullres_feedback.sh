#!/usr/bin/env bash
# Eval the fullres / depth-feedback multiscale experiments on KITTI, SAME protocol as the 0.0946 C
# and temporal baselines (--datasets kitti, kitti_video.json). Per experiment: infer -> eval.
# All of these are ConvNeXt + video loss -> disparity output -> invert=0 (NO 1/D inversion), like C.
#
# Run (from anywhere; cd's to repo root):  bash scripts/eval_fullres_feedback.sh
# Pick a free GPU:                          GPU=7 bash scripts/eval_fullres_feedback.sh
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1
PY="${PY:-python}"
BENCH="${BENCH:-/mnt/workspace/gemdepth_eval}"
GPU="${GPU:-0}"
JSON="$BENCH/kitti/kitti_video.json"

# config name (checkpoint dir == config name; all invert=0 / video loss / disparity output)
CFGS=(
  scratch_ed_dinov3convnext_ms_C_fullres             # Method A, finest scale full-res
  scratch_ed_dinov3convnext_ms_E1_fullres_all        # all scales full-res
  scratch_ed_dinov3convnext_ms_E2a_feedback          # native + depth feedback
  scratch_ed_dinov3convnext_ms_E2b_fullres_feedback  # all full-res + depth feedback
  scratch_ed_dinov3convnext_ms_E3_convfull_nativeout # conv full-res, native output
)

for cfg in "${CFGS[@]}"; do
  ckpt="checkpoint/${cfg}/final_model.pth"
  out="output_eval/${cfg}"
  echo "==================== ${cfg} ===================="
  if [ ! -f "$ckpt" ]; then echo "  [skip] 无 $ckpt (还没训完?)"; continue; fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" evaluation/inference/infer.py \
    --config "config/${cfg}.yaml" --ckpt "$ckpt" \
    --json_file "$JSON" --datasets kitti --infer_path "$out" \
    || { echo "  [infer 失败]"; continue; }
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" evaluation/eval/eval.py \
    --infer_path "$out" --benchmark_path "$BENCH" --datasets kitti
done

echo; echo "==================== 汇总 (AbsRel / RMSE / delta1) ===================="
echo "参照基线: ConvNeXt temporal 0.0891 | 多尺度 C(native video) 0.0946"
for cfg in scratch_ed_dinov3convnext_ms_C_native_video "${CFGS[@]}"; do
  echo "----- $cfg -----"
  grep -E 'abs_relative|rmse|delta1' "output_eval/${cfg}/results.txt" 2>/dev/null | tail -3 \
    || echo "  (无 results.txt)"
done
