#!/usr/bin/env bash
# Explicit quota-v2 NEW C0 run. Original v1 config, trainers and failed outputs stay intact.
set -euo pipefail
SOURCE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/mnt/data/PROJECT_CHEN/code/PP-DPT}
VKITTI_ROOT=${VKITTI_ROOT:-/mnt/data/PROJECT_CHEN/data/train/vkitti2/vkitti}
PYTHON=${PYTHON:-/usr/local/bin/python}
: "${BACKBONE_WEIGHTS:?Set BACKBONE_WEIGHTS to the previously verified clean export}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export NO_ALBUMENTATIONS_UPDATE=1
[[ -x "$PYTHON" ]] || { echo "Missing interpreter: $PYTHON"; exit 2; }
[[ -f "$BACKBONE_WEIGHTS" ]] || { echo "Clean weights missing: $BACKBONE_WEIGHTS"; exit 2; }
[[ -d "$VKITTI_ROOT" ]] || { echo "Data missing: $VKITTI_ROOT"; exit 2; }
OUT=$(mktemp -d "$RUN_ROOT/stereogru_matched30_XXXXXXXX")
echo "QUOTA_V2: Scene01/02=15+15 train, Scene18=16 dev; unchanged gap8, motion .5-5m, 1000 steps/arm."
echo "New experiment directory: $OUT/experiment"
cd "$SOURCE_ROOT"
# CPU preparation only. Still fails closed if the revised fixed quota is infeasible.
"$PYTHON" -u scripts/stereogru_matched_controls.py prepare \
    --config "$SOURCE_ROOT/config/stereogru/matched_c0_c1_30train.yaml" \
    --backbone-weights "$BACKBONE_WEIGHTS" --vkitti-root "$VKITTI_ROOT" \
    --output "$OUT/experiment" 2>&1 | tee "$OUT/prepare.log"
# Original launcher keeps the idle-GPU check, time limit and completed-baseline gate.
# Explicit MATCHED_RUN prevents it from preparing the default v1 config again.
MATCHED_RUN="$OUT/experiment" ARM=C0 RUN_ROOT="$RUN_ROOT" PYTHON="$PYTHON" GPU="${GPU:-4}" \
    bash "$SOURCE_ROOT/scripts/run_stereogru_matched_controls.sh"