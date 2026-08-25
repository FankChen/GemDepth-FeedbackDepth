#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: bash scripts/vkitti.sh {train|test} [additional arguments]" >&2
  exit 2
fi
shift

PYTHON="${PYTHON:-python}"
CONFIG_NAME="${CONFIG_NAME:-vkitti/vkitti}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
# Every accelerate launch grabs the same rendezvous port by default, so a second
# concurrent run on other GPUs dies with "address already in use". Arms are meant
# to run side by side, so give each one its own port.
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
VKITTI_ROOT="${VKITTI_ROOT:-/mnt/workspace/vkitti/vkitti}"

if [[ ! -d "$VKITTI_ROOT" ]]; then
  echo "VKITTI_ROOT does not exist: $VKITTI_ROOT" >&2
  exit 2
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON" >&2
  exit 2
fi

export VKITTI_ROOT
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT"

case "$MODE" in
  train)
    mkdir -p checkpoint logs
    TRAIN_ARGS=(--config-name "$CONFIG_NAME" "num_gpus=$NUM_PROCESSES")
    if [[ "${RESUME:-false}" == "true" ]]; then
      TRAIN_ARGS+=(resume=true)
    fi
    exec "$PYTHON" -m accelerate.commands.launch \
      --num_processes "$NUM_PROCESSES" \
      --main_process_port "$MAIN_PROCESS_PORT" \
      --mixed_precision bf16 \
      train.py \
      "${TRAIN_ARGS[@]}" \
      "$@"
    ;;
  test)
    CONFIG_PATH="${CONFIG_PATH:-config/${CONFIG_NAME}.yaml}"
    CKPT="${CKPT:-checkpoint/vkitti/final_model.pth}"
    OUTPUT="${OUTPUT:-outputs/vkitti/vkitti_dense.png}"
    if [[ ! -f "$CONFIG_PATH" ]]; then
      echo "Config file not found: $CONFIG_PATH" >&2
      exit 2
    fi
    if [[ ! -f "$CKPT" ]]; then
      echo "Checkpoint not found: $CKPT" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$OUTPUT")"
    exec "$PYTHON" evaluation/inference/eval_vkitti_dense.py \
      --config "$CONFIG_PATH" \
      --ckpt "$CKPT" \
      --vkitti_root "$VKITTI_ROOT" \
      --out_viz "$OUTPUT" \
      "$@"
    ;;
  *)
    echo "Unknown mode: $MODE (expected train or test)" >&2
    exit 2
    ;;
esac
