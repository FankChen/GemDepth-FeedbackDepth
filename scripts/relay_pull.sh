#!/bin/bash -l
# Aliyun side: pull a relay branch, reassemble + md5-verify + extract into gemdepth_eval.
# Usage:  bash scripts/relay_pull.sh sintel   (or bonn / scannet)
set -e
ds=${1:?"usage: relay_pull.sh <sintel|bonn|scannet|kitti>"}
BENCH=${BENCH:-/mnt/workspace/gemdepth_eval}
REPO=${REPO:-/mnt/workspace/liren/GemDepth-FeedbackDepth}
REMOTE=${REMOTE:-origin}
TMP=/tmp/pull_$ds

cd "$REPO"
echo "[fetch] $REMOTE relay-$ds"
git fetch "$REMOTE" "relay-$ds"
rm -rf "$TMP"; mkdir -p "$TMP"
git archive "$REMOTE/relay-$ds" | tar -x -C "$TMP"

cd "$TMP"
echo "[merge] cat parts -> $ds.tar"
cat part_* > "$ds.tar"
echo "[verify] md5"
md5sum -c MD5.txt || { echo "!! MD5 MISMATCH — re-pull"; exit 1; }

mkdir -p "$BENCH"
echo "[extract] -> $BENCH"
tar xf "$ds.tar" -C "$BENCH"
cp ./*.json "$BENCH/$ds/" 2>/dev/null || true
echo "[done] $ds ready:"
ls "$BENCH/$ds" | head -6
ls "$BENCH/$ds"/*.json 2>/dev/null || true
echo "tip: after all datasets pulled, delete relay branches on remote to save space."
