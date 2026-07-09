#!/bin/bash -l
# Compare TAE (temporal alignment error) across arms on VKITTI. No retraining.
#   PY=/usr/local/bin/python3 VKITTI_ROOT=/mnt/workspace/vkitti2/vkitti bash scripts/eval_tae_all.sh
# Each line: "checkpoint_subpath head_type warp_signal use_warp". Only runs arms whose ckpt exists.
set -u
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"
PY=${PY:-python}
DATA=${VKITTI_ROOT:-/mnt/workspace/vkitti2/vkitti}
NUM=${NUM:-40}
SEQ=${SEQ:-16}

ARMS=(
  "sweep_baseline_lr1e-5/final_model.pth temporal feat true"
  "single_a100_em_single/final_model.pth errormap_single rgb true"
  "single_a100_em_single_feat/final_model.pth errormap_single feat true"
  "single_a100_em_single_hog/final_model.pth errormap_single hog true"
  "single_a100_em_single_rgbfeat/final_model.pth errormap_single rgbfeat true"
  "single_a100_perlayer/final_model.pth perlayer feat true"
  "single_a100_perlayer_nowarp/final_model.pth perlayer feat false"
  "single_a100_perlayer_rgb/final_model.pth perlayer rgb true"
  "single_a100_perlayer_hog/final_model.pth perlayer hog true"
  "single_a100_perlayer_rgbfeat/final_model.pth perlayer rgbfeat true"
  "single_a100_perlayer_ds05/final_model.pth perlayer feat true"
)

echo "TAE comparison (VKITTI, seq_len=$SEQ, num_seqs=$NUM, data=$DATA)"
for row in "${ARMS[@]}"; do
    set -- $row
    ckpt="checkpoint/$1"; head="$2"; sig="$3"; warp="$4"
    if [ ! -f "$ckpt" ]; then
        echo "  [skip] $ckpt (not found)"
        continue
    fi
    $PY scripts/eval_tae.py --ckpt "$ckpt" --head_type "$head" --warp_signal "$sig" \
        --use_warp "$warp" --data_dirs "$DATA" --seq_len "$SEQ" --num_seqs "$NUM"
done
