#!/bin/bash -l
# One-shot FULL evaluation: for every DONE arm, run single-frame AbsRel (KITTI[+VKITTI], via
# eval_all.sh) and temporal TAE (VKITTI, via eval_tae.py), then print ONE combined table.
#
#   CUDA_VISIBLE_DEVICES=3 PY=/usr/local/bin/python3 BENCH=/mnt/workspace/gemdepth_eval \
#     VKITTI_ROOT=/mnt/workspace/vkitti2/vkitti bash scripts/eval_full.sh 2>&1 | tee jobs/eval_full.log
#
# Env: ABSREL_DATASETS (default "kitti vkitti"), NUM_TAE (40), SEQ (16), ARMS (override list).
# Skips arms without final_model.pth. VKITTI AbsRel needs $BENCH/vkitti/vkitti_video.json
# (generate once with scripts/gen_vkitti_val_json.py); if missing it's just left blank.
set -u
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"
PY=${PY:-/usr/local/bin/python3}
BENCH=${BENCH:-/mnt/workspace/gemdepth_eval}
DATA=${VKITTI_ROOT:-/mnt/workspace/vkitti2/vkitti}
ABSREL_DATASETS=${ABSREL_DATASETS:-"kitti vkitti"}
NUM_TAE=${NUM_TAE:-40}
SEQ=${SEQ:-16}

# "checkpoint_dir head_type warp_signal use_warp"
DEFAULT_ARMS='
sweep_baseline_lr1e-5 temporal feat true
single_a100_perlayer perlayer feat true
single_a100_perlayer_nowarp perlayer feat false
single_a100_perlayer_refine perlayer_refine feat true
single_a100_perlayer_rgb perlayer rgb true
single_a100_perlayer_hog perlayer hog true
single_a100_perlayer_ds05 perlayer feat true
single_a100_baseline_unfreeze temporal feat true
single_a100_scratch temporal feat true
single_a100_em_single_feat errormap_single feat true
'
ARMS_LIST=${ARMS:-$DEFAULT_ARMS}

if [ ! -f "$BENCH/vkitti/vkitti_video.json" ]; then
  echo "[warn] $BENCH/vkitti/vkitti_video.json missing -> VKITTI AbsRel will be blank."
  echo "       generate once: $PY scripts/gen_vkitti_val_json.py  (then re-run)"
fi

printf '%s\n' "$ARMS_LIST" | while read -r dir head sig warp; do
  [ -n "${dir:-}" ] || continue
  ckpt="checkpoint/$dir/final_model.pth"
  if [ ! -f "$ckpt" ]; then echo "[skip] $dir (no final_model.pth)"; continue; fi
  mkdir -p "output_eval/$dir"
  echo ""; echo "########## AbsRel: $dir ($head/$sig) ##########"
  OUT="output_eval/$dir" DATASETS="$ABSREL_DATASETS" bash scripts/eval_all.sh "$ckpt" "$head" "$sig" || true
  echo "########## TAE: $dir ##########"
  $PY scripts/eval_tae.py --ckpt "$ckpt" --head_type "$head" --warp_signal "$sig" --use_warp "$warp" \
      --data_dirs "$DATA" --seq_len "$SEQ" --num_seqs "$NUM_TAE" --tag "$dir" 2>/dev/null \
      | grep "TAE\[" | tee "output_eval/$dir/tae.txt" || true
done

echo ""
echo "================= FULL SUMMARY (AbsRel single-frame + TAE temporal) ================="
printf '%s\n' "$ARMS_LIST" | awk 'NF{print $1}' | $PY - <<'PYEOF'
import sys, os, re
arms = [l.strip() for l in sys.stdin if l.strip()]

def parse_results(path):
    d, cur = {}, None
    if not os.path.exists(path):
        return d
    for line in open(path):
        m = re.search(r'<[^>]*\s([A-Za-z0-9_]+)\sstart', line)
        if m:
            cur = m.group(1); continue
        m = re.search(r'<[^>]*\s([A-Za-z0-9_]+)\sfinish', line)
        if m:
            cur = None; continue
        if cur:
            mm = re.match(r'([A-Za-z0-9_]+):\s*([\d.]+)', line.strip())
            if mm:
                d[(cur, mm.group(1))] = float(mm.group(2))
    return d

def tae_of(dir):
    p = f"output_eval/{dir}/tae.txt"
    if not os.path.exists(p):
        return None
    m = re.search(r'mean=([\d.]+)', open(p).read())
    return float(m.group(1)) if m else None

def cell(x, n=4):
    return f"{x:.{n}f}" if x is not None else "   -  "

hdr = f"{'arm':<30}{'KITTI_AbsRel':>13}{'KITTI_d1':>10}{'VKITTI_AbsRel':>15}{'VKITTI_d1':>10}{'TAE':>11}"
print(hdr)
print('-' * len(hdr))
rows = []
for dir in arms:
    r = parse_results(f"output_eval/{dir}/results.txt")
    ka = r.get(('kitti', 'abs_relative_difference'))
    kd = r.get(('kitti', 'delta1_acc'))
    va = r.get(('vkitti', 'abs_relative_difference'))
    vd = r.get(('vkitti', 'delta1_acc'))
    tae = tae_of(dir)
    print(f"{dir:<30}{cell(ka):>13}{cell(kd):>10}{cell(va):>15}{cell(vd):>10}{cell(tae,5):>11}")
    rows.append([dir, ka, kd, va, vd, tae])

# also dump csv for the paper table
with open('output_eval/eval_full_summary.csv', 'w') as f:
    f.write("arm,kitti_absrel,kitti_d1,vkitti_absrel,vkitti_d1,tae\n")
    for row in rows:
        f.write(",".join("" if v is None else (v if isinstance(v, str) else f"{v:.6f}") for v in row) + "\n")
print("\ncsv -> output_eval/eval_full_summary.csv")
PYEOF
echo "===================================================================================="
