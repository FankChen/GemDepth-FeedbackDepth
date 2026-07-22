#!/usr/bin/env bash
# Launch all pending multiscale-ablation experiments, one per GPU, backgrounded with per-run logs.
#
# Usage (from anywhere; it cd's to the repo root):
#     bash scripts/launch_all_ablation.sh
# Override the python binary if needed:  PY=/usr/local/bin/python3 bash scripts/launch_all_ablation.sh
#
# Each JOBS line is "config_name:gpu_index". Logs go to runlogs/<config>.log.
# Edit/comment JOBS lines to run a subset or re-pin GPUs.
#
# NOTE: B-ViT (scratch_ed_dinov3vits_ms_B_native_l2) is intentionally omitted -- it is the already
# completed AbsRel 0.170 run; reuse its checkpoint/NPYs instead of re-running.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || { echo "cannot cd to repo root"; exit 1; }
mkdir -p runlogs

PY="${PY:-python}"

# config_name : gpu_index
JOBS=(
  "scratch_ed_dinov3vits_ms_A_orig_l2:0"          # A  ViT-S+   all->GT      L2           (control)
  "scratch_ed_dinov3vits_ms_C_native_video:1"     # C  ViT-S+   native       video loss
  "scratch_ed_dinov3vits_ms_D_allgt_gamma08:2"    # D  ViT-S+   all->GT      L2 + gamma0.8 (IGEV wt)
  "scratch_ed_dinov3convnext_ms_A_orig_l2:3"      # A  ConvNeXt all->GT      L2
  "scratch_ed_dinov3convnext_ms_B_native_l2:4"    # B  ConvNeXt native       L2
  "scratch_ed_dinov3convnext_ms_C_native_video:5" # C  ConvNeXt native       video loss
  "scratch_ed_dinov3convnext_ms_D_allgt_gamma08:6" # D ConvNeXt all->GT      L2 + gamma0.8 (IGEV wt)
)

echo "==== GPU 空闲情况（启动前自查）===="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi 不可用)"
echo

launched=0
for job in "${JOBS[@]}"; do
  cfg="${job%%:*}"; gpu="${job##*:}"
  if [ ! -f "config/${cfg}.yaml" ]; then
    echo "[skip] 找不到 config/${cfg}.yaml"; continue
  fi
  if pgrep -f "config-name=${cfg}" >/dev/null 2>&1; then
    echo "[skip] ${cfg} 已在运行"; continue
  fi
  log="runlogs/${cfg}.log"
  echo "[launch] GPU ${gpu}  <-  ${cfg}   (log: ${log})"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "$PY" train.py --config-name="${cfg}" > "${log}" 2>&1 &
  echo "         PID=$!"
  launched=$((launched + 1))
  sleep 5
done

echo
echo ">> 已挂起 ${launched} 个。进程数=$(pgrep -fc 'train.py' 2>/dev/null || echo 0)"
echo "==== 30 秒后各日志尾巴（看是否进训练循环 / 有无 CUDA OOM）===="
sleep 30
for job in "${JOBS[@]}"; do
  cfg="${job%%:*}"; log="runlogs/${cfg}.log"
  echo "----- ${cfg} -----"
  tail -n 4 "${log}" 2>/dev/null || echo "(无日志)"
done
echo
echo "监控:  watch -n 30 'pgrep -fc train.py; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'"
echo "看某个:  tail -f runlogs/<config>.log     （应出现 [loss] multiscale_loss=... 和每200步 [loss] step N ...）"
