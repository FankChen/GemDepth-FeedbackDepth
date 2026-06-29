#!/bin/bash -l
# Single-GPU (A100) launcher for the baseline arm.
#
#   ./scripts/train_single_a100.sh baseline            # original temporal DPT head
#   ./scripts/train_single_a100.sh baseline -resume    # auto-resume from latest checkpoint
#
# Freezes everything except the DPT head and fine-tunes on VKITTI from the
# pretrained GemDepth weights.
set -e

ARM=${1:-baseline}
RESUME_FLAG=${2:-}

case "$ARM" in
    baseline)    CONFIG=single_a100_baseline ;;
    *) echo "Unknown arm: $ARM (baseline)" && exit 1 ;;
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
