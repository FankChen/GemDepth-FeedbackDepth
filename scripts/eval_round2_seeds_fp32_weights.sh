#!/usr/bin/env bash
# Eval round 2: E2a multi-seed reproduction, E4 (=E3+depth feedback), FP32-head controls
# (E2a/E4 baseline vs +fp32), and per-scale loss-weight sweeps (coarse-heavy / fine-heavy /
# ends-heavy), all on top of the E2a (native + depth feedback) base. SAME protocol as C /
# eval_fullres_feedback.sh: KITTI kitti_video.json, ConvNeXt + video loss -> disparity output ->
# invert=0 (NO 1/D inversion). Per experiment: infer -> eval.
#
# Run (from anywhere; cd's to repo root):  bash scripts/eval_round2_seeds_fp32_weights.sh
# Pick a free GPU:                          GPU=7 bash scripts/eval_round2_seeds_fp32_weights.sh
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
  scratch_ed_dinov3convnext_ms_E2a_feedback_s1          # E2a seed=1 (multi-seed repro)
  scratch_ed_dinov3convnext_ms_E2a_feedback_s2          # E2a seed=2 (multi-seed repro)
  scratch_ed_dinov3convnext_ms_E4_convfull_feedback     # E3 + depth feedback (all_native+fb)
  scratch_ed_dinov3convnext_ms_E4_convfull_feedback_fp32 # E4 + fp32_head control
  scratch_ed_dinov3convnext_ms_E2a_fp32head             # E2a + fp32_head control
  scratch_ed_dinov3convnext_ms_E2a_wcoarse              # E2a + weights [0.4,0.3,0.2,0.1] coarse-heavy
  scratch_ed_dinov3convnext_ms_E2a_wfine                # E2a + weights [0.1,0.2,0.3,0.4] fine-heavy
  scratch_ed_dinov3convnext_ms_E2a_wends                # E2a + weights [0.4,0.1,0.1,0.4] ends-heavy
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
echo "参照基线: ConvNeXt temporal 0.0891 | 多尺度 C(native video) 0.0946 | E2a(第一批) 0.0915 | E3 0.0919"
for cfg in scratch_ed_dinov3convnext_ms_C_native_video scratch_ed_dinov3convnext_ms_E2a_feedback \
           scratch_ed_dinov3convnext_ms_E3_convfull_nativeout "${CFGS[@]}"; do
  echo "----- $cfg -----"
  grep -E 'abs_relative|rmse|delta1' "output_eval/${cfg}/results.txt" 2>/dev/null | tail -3 \
    || echo "  (无 results.txt)"
done
