#!/bin/bash
# Deterministic Scene20-held-out evaluation for baseline + GT-camera channel arms.
# Usage: PY=/usr/local/bin/python3 VKITTI_ROOT=/path/to/vkitti CUDA_VISIBLE_DEVICES=0 bash scripts/eval_gt_error_all.sh
set -euo pipefail
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
PY=${PY:-python}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/workspace/vkitti/vkitti/}
cd "$ROOT"
mkdir -p results/gt_error_scene20

arms=(baseline rgb feat rgbfeat geom)
for arm in "${arms[@]}"; do
  cfg="config/scratch_ed_gt_error_${arm}.yaml"
  ckpt="checkpoint/scratch_ed_gt_error_${arm}/final_model.pth"
  out="results/gt_error_scene20/${arm}.json"
  if [[ ! -f "$ckpt" ]]; then
    echo "[skip] $arm: missing $ckpt"
    continue
  fi
  echo "[$(date)] evaluate $arm"
  "$PY" evaluation/eval/eval_vkitti_gt_error.py \
    --config "$cfg" --ckpt "$ckpt" --data_dir "$VKITTI_ROOT" \
    --output "$out" --batch_size 1 --num_workers 0
 done

echo "=== Scene20 GT-error summary ==="
"$PY" - <<'PY'
import json
from pathlib import Path
for p in sorted(Path('results/gt_error_scene20').glob('*.json')):
    d=json.loads(p.read_text())
    print(f"{p.stem:9s} AbsRel={d['abs_relative_difference']:.6f} "
          f"RMSE={d['rmse_linear']:.6f} delta1={d['delta1_acc']:.6f} "
          f"clips={d['clips']}")
PY
