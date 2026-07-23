#!/usr/bin/env bash
# eval_vkitti_all.sh — VKITTI2 held-out DENSE eval across the loss x backbone grid.
#
# Reuses evaluation/inference/eval_vkitti_dense.py once per experiment, collects
# AbsRel / RMSE / delta1 into runlogs/vkitti_dense_summary.csv, and prints a table.
# Also drops a per-experiment DENSE viz under runlogs/vkitti_dense/<stem>.png.
#
# Usage:
#   bash scripts/eval_vkitti_all.sh            # all: ConvNeXt-s FIRST, then ViT-S+
#   bash scripts/eval_vkitti_all.sh convnext   # only ConvNeXt-s (better backbone)
#   bash scripts/eval_vkitti_all.sh vits       # only ViT-S+
#
# invert rule: temporal + C (video loss -> disparity output) => NO invert;
#              A / B / D (l2 loss -> metric output)          => --invert.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python}
FILTER="${1:-all}"
OUT=runlogs/vkitti_dense
mkdir -p "$OUT"
CSV=runlogs/vkitti_dense_summary.csv
echo "backbone,exp,loss,AbsRel,RMSE,delta1" > "$CSV"

run () {  # backbone exp loss stem invert(0/1)
  local backbone="$1" exp="$2" loss="$3" stem="$4" invert="$5"
  [[ "$FILTER" != "all" && "$FILTER" != "$backbone" ]] && return 0
  local ckpt="checkpoint/${stem}/final_model.pth"
  if [[ ! -f "$ckpt" ]]; then echo "[skip] $stem (no ckpt at $ckpt)"; return 0; fi
  local flag=""; [[ "$invert" == "1" ]] && flag="--invert"
  local log="$OUT/${stem}.log"
  echo; echo "=== [$backbone / $exp]  $stem  (invert=$invert) ==="
  $PY evaluation/inference/eval_vkitti_dense.py \
      --config "config/${stem}.yaml" --ckpt "$ckpt" $flag \
      --label1 "${backbone}-${exp}" --out_viz "$OUT/${stem}.png" 2>&1 | tee "$log"
  local line a r d
  line=$(grep -oE '\[vkitti DENSE.*' "$log" | tail -1)
  a=$(echo "$line" | grep -oE 'AbsRel=[0-9.]+' | cut -d= -f2)
  r=$(echo "$line" | grep -oE 'RMSE=[0-9.]+'   | cut -d= -f2)
  d=$(echo "$line" | grep -oE 'delta1=[0-9.]+' | cut -d= -f2)
  echo "${backbone},${exp},${loss},${a:-NA},${r:-NA},${d:-NA}" >> "$CSV"
}

# ---- ConvNeXt-s FIRST (stronger backbone) ----
run convnext temporal       video scratch_ed_dinov3convnext_temporal           0
run convnext C_native_video video scratch_ed_dinov3convnext_ms_C_native_video  0
run convnext A_allgt_l2     l2    scratch_ed_dinov3convnext_ms_A_orig_l2       1
run convnext B_native_l2    l2    scratch_ed_dinov3convnext_ms_B_native_l2     1
run convnext D_gamma08      l2    scratch_ed_dinov3convnext_ms_D_allgt_gamma08 1

# ---- ViT-S+ ----
run vits temporal       video scratch_ed_dinov3vits_temporal           0
run vits C_native_video video scratch_ed_dinov3vits_ms_C_native_video  0
run vits A_allgt_l2     l2    scratch_ed_dinov3vits_ms_A_orig_l2       1
run vits B_native_l2    l2    scratch_ed_dinov3vits_ms_B_native_l2     1
run vits D_gamma08      l2    scratch_ed_dinov3vits_ms_D_allgt_gamma08 1

echo; echo "======== VKITTI2 held-out DENSE  (loss x backbone) ========"
column -t -s, "$CSV"
echo "csv -> $CSV ; per-exp viz -> $OUT/<stem>.png"
