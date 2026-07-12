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
    baseline) CONFIG=single_a100_baseline ;;
    errormap) CONFIG=single_a100_errormap ;;
    multiscale) CONFIG=single_a100_multiscale ;;
    multiscale_fix) CONFIG=single_a100_multiscale_fix ;;
    *) echo "Unknown arm: $ARM (use 'baseline', 'errormap', 'multiscale' or 'multiscale_fix')" && exit 1 ;;
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
# nvidia-smi | head -15 || true

$PY -m accelerate.commands.launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    train.py \
    "${CLI_ARGS[@]}"

echo "[$(date)] done: arm=$ARM"
