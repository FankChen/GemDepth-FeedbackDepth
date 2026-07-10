#!/bin/bash -l
# Bosch side: pack a gemdepth_eval dataset and push it to a relay branch on `feedback`,
# split into <100MB parts and pushed in <2GB batches (GitHub limits). Reuses /tmp/<ds>.tar
# if already present. Usage:  bash scripts/relay_push.sh sintel   (or bonn / scannet)
set -e
ds=${1:?"usage: relay_push.sh <sintel|bonn|scannet|kitti>"}
BENCH=${BENCH:-/home/izi2sgh/MYDATA/quanjie/liren/datasets/gemdepth_eval}
REPO=${REPO:-/home/izi2sgh/MYDATA/quanjie/liren/depth_baselines/GemDepth}
URL=$(cd "$REPO" && git remote get-url feedback)
TMP=/tmp/relay_$ds
BATCH=${BATCH:-15}   # parts per push (~1.4G at 95M each)

echo "[pack] $ds  (bench=$BENCH)"
[ -f "/tmp/$ds.tar" ] || tar cf "/tmp/$ds.tar" -C "$BENCH" "$ds"
MD5=$(md5sum "/tmp/$ds.tar" | cut -d' ' -f1)
echo "[pack] /tmp/$ds.tar md5=$MD5 size=$(du -h /tmp/$ds.tar | cut -f1)"

rm -rf "$TMP"; mkdir -p "$TMP"; cd "$TMP"
split -b 95M -d --suffix-length=3 "/tmp/$ds.tar" part_
cp "$BENCH/$ds"/*.json . 2>/dev/null || true
printf '%s  %s.tar\n' "$MD5" "$ds" > MD5.txt

git init -q
git checkout -q -b "relay-$ds"
git remote add fb "$URL"
n=0; b=0
for p in part_*; do
  git add "$p"; n=$((n + 1))
  if [ $((n % BATCH)) -eq 0 ]; then
    git commit -q -m "relay $ds batch $b"
    echo "[push] batch $b ($n parts so far) ..."
    git push -q fb "relay-$ds"
    b=$((b + 1))
  fi
done
git add MD5.txt *.json 2>/dev/null || git add MD5.txt
git commit -q -m "relay $ds final ($n parts, md5 $MD5)"
git push -q fb "relay-$ds"
echo "[done] pushed relay-$ds : $n parts, md5=$MD5"
echo "       Aliyun:  bash scripts/relay_pull.sh $ds"
