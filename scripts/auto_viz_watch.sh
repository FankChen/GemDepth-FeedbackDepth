#!/usr/bin/env bash
# Auto-visualise every new training checkpoint.
#
# Polls CKPT_DIR; whenever a checkpoint_<step>.pth appears that has not been
# rendered yet (and is stable, i.e. not mid-write), it runs the dense VKITTI
# eval to produce runlogs/viz/viz_step<step>.png and appends the AbsRel / RMSE /
# delta1 numbers to runlogs/viz/metrics.csv so you can watch the trend.
#
# Usage (from repo root):
#   nohup bash scripts/auto_viz_watch.sh > runlogs/auto_viz_watch.out 2>&1 &
#
# Override defaults via env, e.g.:
#   INTERVAL=60 CKPT_DIR=checkpoint/xxx CONFIG=config/xxx.yaml bash scripts/auto_viz_watch.sh
#
# Stop with:  pkill -f auto_viz_watch.sh

set -u

CONFIG=${CONFIG:-config/scratch_ed_dinov3convnext_ms_mixdata_8gpu.yaml}
CKPT_DIR=${CKPT_DIR:-checkpoint/scratch_ed_dinov3convnext_ms_mixdata_8gpu}
OUT_DIR=${OUT_DIR:-runlogs/viz}
INTERVAL=${INTERVAL:-120}          # seconds between polls
MIN_AGE=${MIN_AGE:-20}             # skip files modified within the last N sec (still writing)
# CUDA_VISIBLE_DEVICES is inherited; leave unset to use cuda:0. To pin a card:
#   export CUDA_VISIBLE_DEVICES=0   before launching.

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/auto_viz.log"
CSV="$OUT_DIR/metrics.csv"
[ -f "$CSV" ] || echo "step,absrel,rmse,delta1" > "$CSV"

declare -A DONE
# Pre-mark already-rendered steps so a restart does not redo them.
for f in "$OUT_DIR"/viz_step*.png; do
    [ -e "$f" ] || continue
    s=$(basename "$f" | sed -E 's/viz_step([0-9]+)\.png/\1/')
    DONE[$s]=1
done

echo "[auto_viz] watching $CKPT_DIR every ${INTERVAL}s -> $OUT_DIR (config=$CONFIG)"

while true; do
    for ckpt in "$CKPT_DIR"/checkpoint_*.pth; do
        [ -e "$ckpt" ] || continue
        step=$(basename "$ckpt" | sed -E 's/checkpoint_([0-9]+)\.pth/\1/')
        [ -n "${DONE[$step]:-}" ] && continue
        # skip if the file is still being written (recently modified)
        if [ -n "$(find "$ckpt" -mmin -"$(awk "BEGIN{print $MIN_AGE/60}")" 2>/dev/null)" ]; then
            continue
        fi
        out="$OUT_DIR/viz_step${step}.png"
        echo "[auto_viz] $(date '+%F %T') rendering step $step ..."
        if python evaluation/inference/eval_vkitti_dense.py \
                --config "$CONFIG" --ckpt "$ckpt" --out_viz "$out" >> "$LOG" 2>&1; then
            DONE[$step]=1
            line=$(grep "DENSE model1" "$LOG" | tail -1)
            absrel=$(echo "$line" | sed -nE 's/.*AbsRel=([0-9.]+).*/\1/p')
            rmse=$(echo "$line"  | sed -nE 's/.*RMSE=([0-9.]+).*/\1/p')
            d1=$(echo "$line"    | sed -nE 's/.*delta1=([0-9.]+).*/\1/p')
            echo "${step},${absrel},${rmse},${d1}" >> "$CSV"
            echo "[auto_viz] step $step done: AbsRel=${absrel} RMSE=${rmse} delta1=${d1} -> $out"
        else
            echo "[auto_viz] step $step FAILED (see $LOG)"
        fi
    done
    sleep "$INTERVAL"
done
