#!/bin/bash -l
# One-shot progress dashboard for all training arms.
#   PY=/usr/local/bin/python3 bash scripts/check_progress.sh        # Aliyun
#   PY=/path/to/env/python    bash scripts/check_progress.sh        # Bosch
# Shows: GPU usage, live train.py processes, latest checkpoint step per arm,
# and latest tensorboard train/loss. Works from any machine on its own checkpoints/logs.
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"
PY=${PY:-python}

echo "======== GPU ========"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
  awk -F',' '{printf "  GPU%s: used=%sMB free=%sMB util=%s%%\n",$1,$2,$3,$4}' || echo "  (no nvidia-smi here)"

echo ""
echo "======== train.py processes ========"
found=$(ps -eo pid,etime,cmd 2>/dev/null | grep -E "train\.py|accelerate.commands.launch" | grep -v grep)
if [ -n "$found" ]; then
  echo "$found" | awk '{printf "  pid=%-7s elapsed=%-10s %s\n",$1,$2,$3" "$4" "$5" "$6}'
else
  echo "  (no train.py running on THIS machine)"
fi

echo ""
echo "======== Checkpoint progress ========"
for d in checkpoint/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    if [ -f "$d/final.pth" ] || [ -f "$d/final_model.pth" ]; then
        f="$d/final_model.pth"; [ -f "$f" ] || f="$d/final.pth"
        ts=$(date -r "$f" "+%m-%d %H:%M" 2>/dev/null)
        printf "  %-40s DONE            %s\n" "$name" "$ts"
        continue
    fi
    last=$(ls -t "$d"checkpoint_*.pth 2>/dev/null | head -1)
    if [ -n "$last" ]; then
        step=$(basename "$last" | sed -E 's/checkpoint_([0-9]+)\.pth/\1/')
        ts=$(date -r "$last" "+%m-%d %H:%M" 2>/dev/null)
        printf "  %-40s step=%-8s %s\n" "$name" "$step" "$ts"
    else
        printf "  %-40s (no checkpoint yet)\n" "$name"
    fi
done

echo ""
echo "======== Latest tensorboard train/loss ========"
$PY - <<'PYEOF'
import glob, os
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:
    print("  (tensorboard lib not found; run: tensorboard --logdir logs   to view curves)")
    raise SystemExit
for d in sorted(glob.glob("logs/*/")):
    evs = glob.glob(os.path.join(d, "events.out.tfevents.*"))
    if not evs:
        continue
    ev = max(evs, key=os.path.getmtime)
    try:
        ea = EventAccumulator(ev, size_guidance={'scalars': 0}); ea.Reload()
        if 'train/loss' not in ea.Tags().get('scalars', []):
            continue
        s = ea.Scalars('train/loss'); last = s[-1]
        recent = s[-20:]; avg = sum(x.value for x in recent)/len(recent)
        print(f"  {os.path.basename(d.rstrip('/')):40s} step={last.step:<7d} loss={last.value:.4f} (last20 avg {avg:.4f})")
    except Exception as e:
        print(f"  {os.path.basename(d.rstrip('/')):40s} (read error: {e})")
PYEOF

echo ""
echo "tip: live curves -> tensorboard --logdir logs --port 6006"
