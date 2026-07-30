#!/usr/bin/env bash
# One-shot 4-benchmark evaluation (kitti / sintel / bonn / scannet) for a single
# checkpoint. Runs inference (writes per-frame depth NPYs) then eval (affine
# disparity-space alignment + AbsRel/RMSE/delta1), and prints a summary table.
#
# Works for BOTH multiscale and temporal checkpoints: the model is rebuilt from
# --config via model/factory.py so inference matches training exactly, and the
# checkpoint loader accepts training dicts ({model_state_dict,...}) as well as
# bare state_dicts.
#
# Environment-agnostic: pass BENCH / OUT explicitly.
#   Aliyun: BENCH=/mnt/workspace/gemdepth_eval
#   Bosch : BENCH=/home/izi2sgh/MYDATA/quanjie/liren/datasets/gemdepth_eval
#
# Usage (from repo root):
#   CONFIG=config/scratch_ed_dinov3convnext_ms_mixdata_8gpu.yaml \
#   CKPT=checkpoint/scratch_ed_dinov3convnext_ms_mixdata_8gpu/checkpoint_4000.pth \
#   BENCH=/mnt/workspace/gemdepth_eval \
#   OUT=output_eval/mixdata_step4000 \
#   bash scripts/eval_gemdepth_4bench.sh
#
# Optional env: DATASETS="kitti sintel bonn scannet"  GPU=0  INPUT_SIZE=518
#
set -u

CONFIG=${CONFIG:?set CONFIG=path/to/experiment.yaml}
CKPT=${CKPT:?set CKPT=path/to/checkpoint.pth}
BENCH=${BENCH:?set BENCH=path/to/gemdepth_eval}
OUT=${OUT:?set OUT=output_eval/<name>}
DATASETS=${DATASETS:-"kitti sintel bonn scannet"}
INPUT_SIZE=${INPUT_SIZE:-518}
PY=${PY:-python}
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}

for f in "$CONFIG" "$CKPT"; do
    [ -e "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
[ -d "$BENCH" ] || { echo "ERROR: benchmark dir not found: $BENCH"; exit 1; }

mkdir -p "$OUT"
# Fresh results file (eval.py appends).
: > "$OUT/results.txt"

echo "=========================================================="
echo " 4-bench eval"
echo "   config : $CONFIG"
echo "   ckpt   : $CKPT"
echo "   bench  : $BENCH"
echo "   out    : $OUT"
echo "   sets   : $DATASETS"
echo "   gpu    : $CUDA_VISIBLE_DEVICES   input_size=$INPUT_SIZE"
echo "=========================================================="

# ---- 1) inference: one call per dataset (each uses its own json) ----
for ds in $DATASETS; do
    json="$BENCH/$ds/${ds}_video.json"
    if [ ! -f "$json" ]; then
        echo "[infer] SKIP $ds (missing $json)"
        continue
    fi
    echo "[infer] $ds ..."
    $PY evaluation/inference/infer.py \
        --config "$CONFIG" \
        --ckpt "$CKPT" \
        --json_file "$json" \
        --datasets "$ds" \
        --input_size "$INPUT_SIZE" \
        --infer_path "$OUT" || { echo "[infer] $ds FAILED"; continue; }
done

# ---- 2) eval: single call over all datasets (appends to results.txt) ----
echo "[eval] scoring $DATASETS ..."
$PY evaluation/eval/eval.py \
    --benchmark_path "$BENCH" \
    --infer_path "$OUT" \
    --datasets $DATASETS

# ---- 3) summary table parsed from results.txt ----
echo ""
echo "==================== SUMMARY ===================="
$PY - "$OUT/results.txt" <<'PYEOF'
import re, sys
txt = open(sys.argv[1]).read()
# Blocks look like:  <---- kitti start ---->  ... metric: value ...  <---- kitti finish ---->
rows = []
cur = None
for line in txt.splitlines():
    m = re.search(r'<-+\s*(\w+)\s+start', line)
    if m:
        cur = {'name': m.group(1)}
        continue
    m = re.match(r'\s*(abs_relative_difference|rmse_linear|delta1_acc):\s*([0-9.]+)', line)
    if m and cur is not None:
        cur[m.group(1)] = float(m.group(2))
    if 'finish' in line and cur is not None:
        rows.append(cur); cur = None
hdr = f"{'dataset':<10}{'AbsRel':>10}{'RMSE':>10}{'delta1':>10}"
print(hdr); print('-'*len(hdr))
for r in rows:
    print(f"{r.get('name',''):<10}"
          f"{r.get('abs_relative_difference',float('nan')):>10.4f}"
          f"{r.get('rmse_linear',float('nan')):>10.4f}"
          f"{r.get('delta1_acc',float('nan')):>10.4f}")
PYEOF
echo "================================================="
echo "raw: $OUT/results.txt   npys: $OUT/<dataset>/"
