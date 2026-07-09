#!/bin/bash -l
# Single-GPU (A100) launcher for the two-arm controlled experiment.
#
#   ./scripts/train_single_a100.sh baseline            # control arm (original DPT head)
#   ./scripts/train_single_a100.sh errormap            # method arm (error-map DPT head)
#   ./scripts/train_single_a100.sh baseline -resume    # auto-resume from latest checkpoint
#
# Both arms freeze everything except the DPT head and fine-tune on VKITTI from the
# pretrained GemDepth weights. Only the head differs between arms.
set -e

ARM=${1:-baseline}
RESUME_FLAG=${2:-}

case "$ARM" in
    baseline)          CONFIG=single_a100_baseline ;;
    errormap)          CONFIG=single_a100_errormap ;;
    em_single)         CONFIG=single_a100_em_single ;;
    em_single_rgb)     CONFIG=single_a100_em_single_rgb ;;
    em_single_feat)    CONFIG=single_a100_em_single_feat ;;
    em_single_rgbfeat) CONFIG=single_a100_em_single_rgbfeat ;;
    em_single_hog)     CONFIG=single_a100_em_single_hog ;;
    batlin)            CONFIG=single_a100_batlin ;;
    batlin_cycle)      CONFIG=single_a100_batlin_cycle ;;
    baseline_cycle)    CONFIG=single_a100_baseline_cycle ;;
    batlin_4scale)     CONFIG=single_a100_batlin_4scale ;;
    batlin_rgbfeat)    CONFIG=single_a100_batlin_rgbfeat ;;
    batlin_hog)        CONFIG=single_a100_batlin_hog ;;
    batlin_cycle_4scale) CONFIG=single_a100_batlin_cycle_4scale ;;
    batlin_cycle_o12)  CONFIG=single_a100_batlin_cycle_o12 ;;
    perlayer)          CONFIG=single_a100_perlayer ;;
    perlayer_refine)   CONFIG=single_a100_perlayer_refine ;;
    perlayer_nb)       CONFIG=single_a100_perlayer_nb ;;
    perlayer_nowarp)   CONFIG=single_a100_perlayer_nowarp ;;
    perlayer_nowarp_nb) CONFIG=single_a100_perlayer_nowarp_nb ;;
    perlayer_rgb)      CONFIG=single_a100_perlayer_rgb ;;
    perlayer_hog)      CONFIG=single_a100_perlayer_hog ;;
    perlayer_rgbfeat)  CONFIG=single_a100_perlayer_rgbfeat ;;
    perlayer_ds05)     CONFIG=single_a100_perlayer_ds05 ;;
    baseline_unfreeze) CONFIG=single_a100_baseline_unfreeze ;;
    scratch)           CONFIG=single_a100_scratch ;;
    *) echo "Unknown arm: $ARM (baseline|errormap|em_single|em_single_{rgb,feat,rgbfeat,hog}|batlin|batlin_cycle|baseline_cycle|batlin_4scale|batlin_rgbfeat|batlin_hog|batlin_cycle_4scale|batlin_cycle_o12|perlayer|baseline_unfreeze|scratch)" && exit 1 ;;
esac

# Portable: PY defaults to the `python` on PATH (activate your env first); override
# with PY=/path/to/python. ROOT is resolved relative to this script's location.
PY=${PY:-python}
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"
mkdir -p jobs logs checkpoint

CLI_ARGS=(--config-name "$CONFIG")
if [ "$RESUME_FLAG" == "-resume" ] || [ "$RESUME_FLAG" == "--resume" ]; then
    CLI_ARGS+=(resume=true)
    echo "[config] Auto-resume enabled"
fi

echo "[$(date)] start single-A100 training: arm=$ARM config=$CONFIG"
nvidia-smi | head -15 || true

$PY -m accelerate.commands.launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    train.py \
    "${CLI_ARGS[@]}"

echo "[$(date)] done: arm=$ARM"
