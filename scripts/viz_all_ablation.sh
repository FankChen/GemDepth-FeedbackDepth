#!/usr/bin/env bash
# Generate side-by-side error-map comparisons (RGB | GT | A depth | B depth | A AbsRel | B AbsRel)
# for the ablation runs, same figure style as the earlier temporal-vs-native comparison.
#
# All NPYs on disk are DISPARITY (the metric ones A/B/D were inverted 1/D during eval), so NO
# --invert flags are needed -> every panel uses the same eval-aligned pipeline, fully comparable.
#
# Run:  bash scripts/viz_all_ablation.sh            (PNGs -> runlogs/viz/)
#       NF=6 bash scripts/viz_all_ablation.sh       (6 frames per figure)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || exit 1
PY="${PY:-python}"
BENCH="${BENCH:-/mnt/workspace/gemdepth_eval}"
OE="${OE:-output_eval}"
OUT="runlogs/viz"; mkdir -p "$OUT"
NF="${NF:-4}"

# a_dir : a_label : b_dir : b_label : out.png     (dirs/labels/png must contain NO colon)
COMPARES=(
  "scratch_ed_dinov3vits_temporal_t4_fixed:temporal 0.105:scratch_ed_dinov3vits_ms_C_native_video:C video 0.115:vits_temporal_vs_C.png"
  "scratch_ed_dinov3vits_multiscale_native:B L2 0.170:scratch_ed_dinov3vits_ms_C_native_video:C video 0.115:vits_B_L2_vs_C_video.png"
  "scratch_ed_dinov3vits_ms_A_orig_l2:A L2 0.190:scratch_ed_dinov3vits_ms_D_allgt_gamma08:D L2+g 0.180:vits_A_vs_D_gamma.png"
  "scratch_ed_dinov3convnext_temporal_t4_fixed:temporal 0.089:scratch_ed_dinov3convnext_ms_C_native_video:C video 0.095:convnext_temporal_vs_C.png"
  "scratch_ed_dinov3convnext_ms_A_orig_l2:A L2 0.104:scratch_ed_dinov3convnext_ms_C_native_video:C video 0.095:convnext_A_L2_vs_C_video.png"
  "scratch_ed_dinov3convnext_ms_A_orig_l2:A L2 0.104:scratch_ed_dinov3convnext_ms_D_allgt_gamma08:D L2+g 0.126:convnext_A_vs_D_gamma.png"
)

for row in "${COMPARES[@]}"; do
  IFS=':' read -r ad al bd bl png <<< "$row"
  if [ ! -d "$OE/$ad/kitti" ]; then echo "[skip] 无 $OE/$ad/kitti (那份 NPY 不在?)"; continue; fi
  if [ ! -d "$OE/$bd/kitti" ]; then echo "[skip] 无 $OE/$bd/kitti"; continue; fi
  echo "[viz] $png    ($al  vs  $bl)"
  "$PY" evaluation/eval/visualize_compare.py \
    --benchmark_path "$BENCH" \
    --pred_a "$OE/$ad" --label_a "$al" \
    --pred_b "$OE/$bd" --label_b "$bl" \
    --num_frames "$NF" --out "$OUT/$png" || echo "  [失败] $png"
done

echo; echo "==== 出图完成，在 $OUT/ ===="; ls -1 "$OUT"/*.png 2>/dev/null
