#!/usr/bin/env bash
# Diagnostic-only runner. No training, installations, checkpoint writes or worktree edits.
# May run from a git-archive snapshot; weights/data/output paths remain explicit.
set -euo pipefail

SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
CKPT_ROOT=${CKPT_ROOT:-$RUN_ROOT/checkpoint}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/data/PROJECT_CHEN/data/train/vkitti2/vkitti}
PYTHON=${PYTHON:-/usr/local/bin/python}
GPU=${GPU:-2}
MAX_BATCHES=${MAX_BATCHES:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export VKITTI_ROOT

[[ -x "$PYTHON" ]] || { echo "Interpreter missing: $PYTHON"; exit 2; }
[[ -d "$VKITTI_ROOT" ]] || { echo "Data root missing: $VKITTI_ROOT"; exit 2; }
for arm in vkitti_costvol vkitti_costvol_d3 vkitti_ms_gem; do
    [[ -f "$CKPT_ROOT/$arm/final_model.pth" ]] || {
        echo "Missing checkpoint: $CKPT_ROOT/$arm/final_model.pth"
        echo "Set CKPT_ROOT to the directory containing the historical runs; do not move checkpoints."
        exit 2
    }
done

OUT=$(mktemp -d "$RUN_ROOT/stereogru_audit_XXXXXXXX")
export OUT CKPT_ROOT SOURCE_ROOT
echo "Audit outputs: $OUT"
echo "Source snapshot: $SOURCE_ROOT"
echo "No training or checkpoint writes."
cd "$SOURCE_ROOT"

{
    printf 'source=%s\nweights=%s\ndata=%s\n' "$SOURCE_ROOT" "$CKPT_ROOT" "$VKITTI_ROOT"
    git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null || true
    sha256sum model/dpt_cost_volume_convnext.py model/util/cost_volume.py \
        model/util/warp.py loss/videoloss.py scripts/diagnose_gem_camera.py \
        scripts/diagnose_cost_volume_oracle.py
    for arm in vkitti_costvol vkitti_costvol_d3 vkitti_ms_gem; do
        sha256sum "$CKPT_ROOT/$arm/final_model.pth"
    done
} > "$OUT/provenance.txt"

# Do not assume GPU 2 is free just because the old baseline used GPUs 0/1.
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable"; exit 2; }
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
USED=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
PROCESSES=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader)
if [[ "$USED" -gt 2048 ]] || grep -Fq "$GPU_UUID" <<< "$PROCESSES"; then
    echo "GPU=$GPU is busy; no job launched. Choose an idle GPU and rerun."
    printf '%s\n' "$PROCESSES"
    exit 3
fi
export CUDA_VISIBLE_DEVICES="$GPU"

"$PYTHON" -u - <<'PY' > "$OUT/config_summary.json"
import json
import os
import sys
sys.path[:0] = [os.environ['SOURCE_ROOT'], os.path.join(os.environ['SOURCE_ROOT'], 'evaluation/inference')]
import torch
from protocol import load_experiment_config, resolve_inference_clip_len
from omegaconf import OmegaConf
from loss.objective_video import resolve_scale_weights

results = {'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(),
           'bf16_supported': torch.cuda.is_bf16_supported(),
           'config_origin': 'Source snapshot; old model-only exports do not embed training config',
           'runs': {}}
if not results['cuda_available'] or not results['bf16_supported']:
    raise RuntimeError('These checkpoints use GEM CUDA bf16 attention; select a compatible idle H20')
for arm in ('vkitti_costvol', 'vkitti_costvol_d3', 'vkitti_ms_gem'):
    cfg = load_experiment_config(f'config/vkitti/{arm}.yaml')
    weights = resolve_scale_weights(OmegaConf.select(cfg, 'multiscale_scale_weights'),
                                    OmegaConf.select(cfg, 'multiscale_gamma'),
                                    OmegaConf.select(cfg, 'multiscale_scales', default=4))
    if weights and OmegaConf.select(cfg, 'multiscale_normalize_scale_weights', default=True):
        total = sum(weights)
        weights = [value / total for value in weights]
    results['runs'][arm] = {
        'decoder': cfg.model.decoder, 'decoder_kwargs': OmegaConf.to_container(
            OmegaConf.select(cfg, 'model.decoder_kwargs', default=OmegaConf.create({})), resolve=True),
        'pose_flag': bool(cfg.pose_flag), 'use_astt': bool(cfg.model.use_astt),
        'camera_weight_focal': OmegaConf.select(cfg, 'loss.kwargs.camera_weight_focal', default=0.),
        'clip_len': resolve_inference_clip_len(cfg), 'weights': weights,
        'budget': int(cfg.total_step), 'seed': int(cfg.training.seed),
    }
print(json.dumps(results, indent=2))
PY

run_diagnostic() {
    local label=$1
    shift
    echo "Running $label (details in $OUT/$label.log)"
    if "$PYTHON" -u "$@" > "$OUT/$label.log" 2>&1; then
        echo "Completed $label"
    else
        local code=$?
        tail -60 "$OUT/$label.log"
        echo "Failed $label; stopping, exit=$code. Send this log, not a replacement result."
        exit "$code"
    fi
}

for arm in vkitti_costvol vkitti_costvol_d3; do
    run_diagnostic "${arm}_oracle" scripts/diagnose_cost_volume_oracle.py \
        --config "config/vkitti/$arm.yaml" --ckpt "$CKPT_ROOT/$arm/final_model.pth" \
        --vkitti-root "$VKITTI_ROOT" --max-batches "$MAX_BATCHES" --trace-head \
        --output "$OUT/${arm}_oracle.json"
done
run_diagnostic vkitti_costvol_d3_backbone scripts/diagnose_cost_volume_oracle.py \
    --config config/vkitti/vkitti_costvol_d3.yaml \
    --ckpt "$CKPT_ROOT/vkitti_costvol_d3/final_model.pth" --vkitti-root "$VKITTI_ROOT" \
    --max-batches "$MAX_BATCHES" --descriptor backbone --output "$OUT/vkitti_costvol_d3_backbone.json"
run_diagnostic vkitti_ms_gem_camera scripts/diagnose_gem_camera.py \
    --config config/vkitti/vkitti_ms_gem.yaml --ckpt "$CKPT_ROOT/vkitti_ms_gem/final_model.pth" \
    --vkitti-root "$VKITTI_ROOT" --max-batches "$MAX_BATCHES" --output "$OUT/vkitti_ms_gem_camera.json"

"$PYTHON" - <<'PY' | tee "$OUT/summary.txt"
import json
import math
import os
from pathlib import Path

root = Path(os.environ['OUT'])
def fmt(value):
    return f'{value:.4f}' if value is not None and math.isfinite(value) else 'N/A'

for path in sorted(root.glob('*oracle.json')) + sorted(root.glob('*backbone.json')):
    data = json.loads(path.read_text())
    print('\n###', path.name, 'clips=', data['batches'], 'descriptor=', data['descriptor'])
    print('mode                        support in-range top1  chance top3  margin  depth-std')
    for mode, item in data['modes'].items():
        coverage = item['in_range_pixels'] / max(item['target_pixels'], 1)
        print(f"{mode:27s} {fmt(item['valid_fraction'])} {fmt(coverage)} "
              f"{fmt(item['top1'])} {fmt(item['top1_chance'])} {fmt(item['top3'])} "
              f"{fmt(item['true_bin_margin'])} {fmt(item['score_std'])}")
    shared = data['shared_support_modes']['gt_metric']['eligible_pixels']
    print('shared eligible pixels:', shared, '(zero means full eight-way comparison is unavailable)')
    print('paired support against GT camera (mode top1 / GT top1 / eligible):')
    for name, pair in data['paired_gt_support_modes'].items():
        a, b = pair['mode'], pair['gt_metric']
        print(name, fmt(a['top1']), fmt(b['top1']), a['eligible_pixels'])
    for clip in data['clips']:
        trace = clip.get('head_trace')
        if trace:
            print('clip', clip['sample_index'], 'unit_m=', clip['translation_unit_metres'],
                  'nonfinite_K_frames=', clip['nonfinite_intrinsic_frames'],
                  'raw_zero=', fmt(trace['raw_volume']['zero_pixel_fraction']),
                  'GEV_entropy=', fmt(trace['geometry_logits']['normalized_entropy']),
                  'lookup_outside=', [round(x['outside_volume_fraction'], 4) for x in trace['lookups']],
                  'delta_mean=', [round(x['mean'], 4) if 'mean' in x else None for x in trace['updates']],
                  'output_floor=', fmt(trace['final_raw_output']['training_floor_fraction']))
    if data['failure_examples']:
        print('ERRORS:', data['failure_examples'])

camera = json.loads((root / 'vkitti_ms_gem_camera.json').read_text())
print('\n### ms_gem camera (its own auxiliary-task baseline, not costvol camera weights)')
for key in ('focal_relative_error', 'focal_relative_error_finite_only',
            'focal_relative_error_valid_frames', 'focal_finite_fraction',
            'focal_valid_frame_fraction', 'principal_point_error_pixels', 'rotation_error_degrees',
            'translation_direction_cosine', 'translation_magnitude_ratio', 'translation_unit_metres',
            'nonfinite_intrinsics_fraction', 'nonfinite_extrinsics_fraction',
            'gt_warp_valid', 'pred_consistent_warp_valid'):
    print(key, fmt(camera[key]))
print('focal_nonfinite_by_axis', camera['focal_nonfinite_by_axis'])
print(camera['focal_aggregation_caveat'])
print('\nSend summary.txt plus config_summary.json and provenance.txt; full JSONs retain per-clip evidence.')
print('Output directory:', root)
PY