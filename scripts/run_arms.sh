#!/usr/bin/env bash
# =============================================================================
# 顺序/并行跑多个实验臂 (无需 LSF/bsub，纯 shell，适合阿里云单机)
# -----------------------------------------------------------------------------
# 用法:
#   bash scripts/run_arms.sh baseline errormap                 # 顺序跑这两个臂
#   bash scripts/run_arms.sh all                               # 顺序跑全部 7 个臂
#   GPUS="0 1" bash scripts/run_arms.sh all                    # 多卡: 轮流分配到 GPU 0/1 并行后台跑
#
# 说明:
#   - 默认逐个串行执行 (单 GPU 安全)。日志写到 logs/<arm>.log。
#   - 设 GPUS="0 1 2..." 时，按臂轮流绑定到这些 GPU 并后台并行，最后统一 wait。
#   - 每个臂调用 scripts/train_single_a100.sh <arm>，PY/ROOT/VKITTI_ROOT 等沿用其规则。
# =============================================================================
set -uo pipefail

ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"
mkdir -p logs jobs

ALL_ARMS=(baseline errormap em_rgb em_feat em_hog em_rgbfeat em_refine)

# 解析臂列表
ARMS=()
for a in "$@"; do
  if [ "$a" == "all" ]; then ARMS=("${ALL_ARMS[@]}"); else ARMS+=("$a"); fi
done
[ ${#ARMS[@]} -eq 0 ] && { echo "用法: bash scripts/run_arms.sh <arm...|all>  (arm: ${ALL_ARMS[*]})"; exit 1; }

echo "[run_arms] 待跑臂: ${ARMS[*]}"
echo "[run_arms] VKITTI_ROOT=${VKITTI_ROOT:-<未设置，将用 config 默认>}"

if [ -z "${GPUS:-}" ]; then
  # ---- 串行 ----
  for arm in "${ARMS[@]}"; do
    log="logs/${arm}.log"
    echo "[run_arms] >>> 串行启动 arm=${arm}  (日志: ${log})"
    bash scripts/train_single_a100.sh "$arm" 2>&1 | tee "$log"
    echo "[run_arms] <<< 完成 arm=${arm}"
  done
else
  # ---- 多卡并行: 按臂轮流绑定 GPU ----
  read -r -a GPU_ARR <<< "$GPUS"
  ng=${#GPU_ARR[@]}
  pids=()
  i=0
  for arm in "${ARMS[@]}"; do
    gpu=${GPU_ARR[$((i % ng))]}
    log="logs/${arm}.gpu${gpu}.log"
    echo "[run_arms] >>> 后台启动 arm=${arm} -> GPU ${gpu}  (日志: ${log})"
    CUDA_VISIBLE_DEVICES="$gpu" nohup bash scripts/train_single_a100.sh "$arm" > "$log" 2>&1 &
    pids+=($!)
    i=$((i+1))
    sleep 5   # 错开启动，避免同时抢初始化
  done
  echo "[run_arms] 已后台启动 ${#pids[@]} 个臂，PID: ${pids[*]}"
  echo "[run_arms] 用  tail -f logs/<arm>.gpu<n>.log  监控；等待全部结束..."
  wait
fi

echo "[run_arms] 全部任务结束。"
