#!/bin/bash -l
# Launcher for legacy single-GPU arms and the two-GPU scratch encoder/decoder experiments.
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
    baseline) CONFIG=single_a100_baseline ;;
    errormap) CONFIG=single_a100_errormap ;;
    multiscale) CONFIG=single_a100_multiscale ;;
    multiscale_fix) CONFIG=single_a100_multiscale_fix ;;
    ed_dinov2_static)     CONFIG=scratch_ed_dinov2_static; DEFAULT_NUM_PROC=2 ;;
    ed_dinov2_temporal)   CONFIG=scratch_ed_dinov2_temporal; DEFAULT_NUM_PROC=2 ;;
    ed_dinov2_multiscale) CONFIG=scratch_ed_dinov2_multiscale; DEFAULT_NUM_PROC=2 ;;
    ed_dinov3vits_static) CONFIG=scratch_ed_dinov3vits_static; DEFAULT_NUM_PROC=2 ;;
    ed_dinov3convnext_static) CONFIG=scratch_ed_dinov3convnext_static; DEFAULT_NUM_PROC=2 ;;
    *) echo "Unknown arm: $ARM (baseline|errormap|multiscale|multiscale_fix|ed_dinov2_{static,temporal,multiscale}|ed_dinov3vits_static|ed_dinov3convnext_static)" && exit 1 ;;
esac
DEFAULT_NUM_PROC=${DEFAULT_NUM_PROC:-1}

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

echo "[$(date)] start training: arm=$ARM config=$CONFIG processes=${NUM_PROC:-$DEFAULT_NUM_PROC}"
# nvidia-smi | head -15 || true

$PY -m accelerate.commands.launch \
    --num_processes ${NUM_PROC:-$DEFAULT_NUM_PROC} \
    --mixed_precision bf16 \
    train.py \
    "${CLI_ARGS[@]}"

echo "[$(date)] done: arm=$ARM"
