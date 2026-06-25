#!/usr/bin/env bash
# =============================================================================
# GemDepth one-click environment bootstrap for Alibaba Cloud (or any Linux GPU box)
# -----------------------------------------------------------------------------
# Usage:
#   bash scripts/setup_aliyun.sh check       # 体检：探测 OS/Python/conda/GPU/CUDA/数据/权重 (不改动任何东西)
#   bash scripts/setup_aliyun.sh install      # 创建环境并 pip install -r requirements.txt
#   bash scripts/setup_aliyun.sh verify       # 验证 torch.cuda 可用 + 关键依赖可导入
#   bash scripts/setup_aliyun.sh all          # install 然后 verify 然后 check
#
# 可用环境变量覆盖：
#   ENV_NAME    conda/venv 环境名 (默认 gemdepth)
#   PY          直接指定解释器路径 (verify/install 时优先使用)
#   VKITTI_ROOT 指向解压后含 Scene01.. 的目录 (训练数据)
#   EVAL_ROOT   指向 KITTI 评测数据根 (仅评估需要, 默认 仓库/../../datasets/gemdepth_eval)
#   PIP_INDEX   pip 镜像 (中国大陆建议 https://pypi.tuna.tsinghua.edu.cn/simple)
# =============================================================================
set -uo pipefail

# ---- locate repo root (this script lives in <root>/scripts) ----
ROOT=${ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
cd "$ROOT"

ENV_NAME=${ENV_NAME:-gemdepth}
CKPT_PATH="$ROOT/checkpoint/gemdepth.pth"
REQ_FILE="$ROOT/requirements.txt"

# ---- pretty status helpers (work in plain terminals) ----
ok()   { printf '  [ OK ]  %s\n' "$*"; }
miss() { printf '  [MISS]  %s\n' "$*"; }
warn() { printf '  [WARN]  %s\n' "$*"; }
info() { printf '  [INFO]  %s\n' "$*"; }
hr()   { printf '%s\n' "------------------------------------------------------------"; }
sec()  { printf '\n== %s ==\n' "$*"; }

ISSUES=0
flag() { ISSUES=$((ISSUES+1)); }

# =============================================================================
# CHECK: diagnose the box, change nothing
# =============================================================================
do_check() {
  sec "1) 操作系统 / 基础工具"
  if [ -r /etc/os-release ]; then
    . /etc/os-release
    ok "OS: ${PRETTY_NAME:-unknown}  (kernel $(uname -r))"
  else
    warn "无法读取 /etc/os-release (kernel $(uname -r))"
  fi
  for t in git curl wget tar; do
    if command -v "$t" >/dev/null 2>&1; then ok "$t: $(command -v "$t")"; else miss "$t 未安装"; flag; fi
  done

  sec "2) Python / conda / venv"
  if command -v conda >/dev/null 2>&1; then
    ok "conda: $(conda --version 2>/dev/null) -> $(command -v conda)"
    if conda env list 2>/dev/null | grep -qE "^\s*${ENV_NAME}\s"; then
      ok "已存在 conda 环境 '${ENV_NAME}'"
    else
      info "尚无 conda 环境 '${ENV_NAME}' (install 步骤会创建)"
    fi
  else
    info "未检测到 conda (将回退到 python3 venv)"
  fi
  if command -v python3 >/dev/null 2>&1; then
    ok "python3: $(python3 --version 2>&1) -> $(command -v python3)"
  else
    miss "python3 未安装"; flag
  fi
  for pv in python3.10 python3.11 python3.12; do
    command -v "$pv" >/dev/null 2>&1 && info "可用解释器: $pv ($($pv --version 2>&1))"
  done

  sec "3) NVIDIA 驱动 / GPU / CUDA"
  if command -v nvidia-smi >/dev/null 2>&1; then
    local drv cuda gpus
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
    ok "驱动版本: ${drv:-未知} | 驱动支持的最高 CUDA: ${cuda:-未知}"
    if [ -n "${cuda:-}" ]; then
      # need CUDA >= 12.1 for torch 2.3.1+cu121
      if awk "BEGIN{exit !(${cuda} >= 12.1)}"; then
        ok "CUDA ${cuda} >= 12.1，满足 torch 2.3.1+cu121"
      else
        miss "CUDA ${cuda} < 12.1，torch 2.3.1+cu121 可能无法运行 GPU (需升级驱动)"; flag
      fi
    fi
    info "GPU 列表:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/        /'
    local n; n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    ok "可见 GPU 数量: ${n}  (多卡可并行跑多臂)"
  else
    miss "nvidia-smi 不存在 —— 没有 NVIDIA 驱动或不是 GPU 机器"; flag
  fi

  sec "4) 磁盘空间 (当前仓库所在盘)"
  df -h "$ROOT" | sed 's/^/  /'
  info "权重 2.84GB + VKITTI 解压 ~30GB + checkpoints，建议预留 >80GB"

  sec "5) 预训练权重 checkpoint/gemdepth.pth"
  if [ -f "$CKPT_PATH" ]; then
    local sz; sz=$(stat -c%s "$CKPT_PATH" 2>/dev/null || stat -f%z "$CKPT_PATH" 2>/dev/null)
    if [ "${sz:-0}" -ge 2800000000 ]; then
      ok "已就位: $CKPT_PATH ($(numfmt --to=iec "$sz" 2>/dev/null || echo "${sz}B"))"
    else
      warn "存在但大小异常 (${sz}B, 期望 ~2.84GB)，可能未传完"; flag
    fi
  else
    miss "缺失: $CKPT_PATH —— 训练前必须放好 (见 ALIYUN_REPRO.md 第4步)"; flag
  fi

  sec "6) VKITTI 训练数据 (VKITTI_ROOT)"
  if [ -n "${VKITTI_ROOT:-}" ]; then
    if [ -d "$VKITTI_ROOT" ]; then
      ok "VKITTI_ROOT=$VKITTI_ROOT 存在"
      local nscene nrgb
      nscene=$(find "$VKITTI_ROOT" -maxdepth 2 -type d -name 'Scene*' 2>/dev/null | wc -l | tr -d ' ')
      nrgb=$(find "$VKITTI_ROOT" -maxdepth 6 -type d -path '*frames/rgb/Camera_0' 2>/dev/null | head -1)
      if [ -n "$nrgb" ]; then
        ok "找到 frames/rgb/Camera_0 结构 (Scene 目录数: ${nscene})"
      else
        miss "未找到 .../frames/rgb/Camera_0 —— 数据可能未解压或路径不对"; flag
      fi
    else
      miss "VKITTI_ROOT=$VKITTI_ROOT 不存在"; flag
    fi
  else
    warn "未设置 VKITTI_ROOT 环境变量 (训练前需 export，见第5步)"
    flag
  fi

  sec "7) KITTI 评测数据 (EVAL_ROOT, 仅评估需要)"
  local er; er=${EVAL_ROOT:-"$ROOT/../../datasets/gemdepth_eval"}
  if [ -f "$er/kitti/kitti_video.json" ]; then
    ok "EVAL_ROOT=$er 含 kitti/kitti_video.json"
  else
    info "未找到 $er/kitti/kitti_video.json (现在只跑训练可忽略)"
  fi

  hr
  if [ "$ISSUES" -eq 0 ]; then
    printf '体检结论: 全部就绪 ✅  可以执行  bash scripts/setup_aliyun.sh install\n'
  else
    printf '体检结论: 发现 %d 项待处理 ⚠️  按上面 [MISS]/[WARN] 提示处理 (见 ALIYUN_REPRO.md)\n' "$ISSUES"
  fi
}

# =============================================================================
# INSTALL: create env + pip install
# =============================================================================
pip_install() {
  # $1 = pip executable
  local pip="$1"
  local extra=()
  [ -n "${PIP_INDEX:-}" ] && extra+=(--index-url "$PIP_INDEX")
  info "升级 pip ..."
  "$pip" install --upgrade pip "${extra[@]}"
  info "安装 requirements.txt (含 torch 2.3.1 / cu121 运行库) ..."
  # torch 2.3.1 默认 PyPI wheel 即 cu121 构建；如默认源拿不到，可加 PyTorch 官方索引兜底
  "$pip" install -r "$REQ_FILE" "${extra[@]}" \
    || "$pip" install -r "$REQ_FILE" "${extra[@]}" \
         --extra-index-url https://download.pytorch.org/whl/cu121
}

do_install() {
  sec "创建环境并安装依赖"
  if [ ! -f "$REQ_FILE" ]; then miss "找不到 $REQ_FILE"; exit 1; fi

  if command -v conda >/dev/null 2>&1; then
    info "使用 conda 创建环境 '${ENV_NAME}' (python 3.10)"
    if ! conda env list 2>/dev/null | grep -qE "^\s*${ENV_NAME}\s"; then
      conda create -y -n "$ENV_NAME" python=3.10
    else
      ok "conda 环境 '${ENV_NAME}' 已存在，复用"
    fi
    local cpip; cpip="$(conda run -n "$ENV_NAME" which pip)"
    pip_install "$cpip"
    ok "安装完成。激活方式:  conda activate ${ENV_NAME}"
    PY_HINT="$(conda run -n "$ENV_NAME" which python)"
  else
    # numpy 2.2.6 等依赖要求 Python >= 3.10：优先 3.10 > 3.11 > 3.12，最后才回退 python3
    local base_py=""
    for cand in python3.10 python3.11 python3.12; do
      command -v "$cand" >/dev/null 2>&1 && { base_py="$cand"; break; }
    done
    if [ -z "$base_py" ]; then
      base_py=python3
      local pyver; pyver=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
      if ! awk "BEGIN{exit !(${pyver:-0} >= 3.10)}"; then
        miss "默认 python3=${pyver} < 3.10，依赖(numpy 2.2.6 等)需要 >=3.10。请先安装 python3.10 (见 ALIYUN_REPRO.md 第3步) 或改用 conda。"
        exit 1
      fi
    fi
    local venv_dir="$ROOT/.venv-${ENV_NAME}"
    info "未发现 conda，使用 ${base_py} ($($base_py --version 2>&1)) 创建 venv: ${venv_dir}"
    "$base_py" -m venv "$venv_dir"
    pip_install "$venv_dir/bin/pip"
    ok "安装完成。激活方式:  source ${venv_dir}/bin/activate"
    PY_HINT="$venv_dir/bin/python"
  fi
  info "解释器: ${PY_HINT}"
  printf '%s\n' "$PY_HINT" > "$ROOT/.gemdepth_python_path"
}

# =============================================================================
# VERIFY: torch.cuda + key imports
# =============================================================================
resolve_py() {
  if [ -n "${PY:-}" ]; then echo "$PY"; return; fi
  if [ -f "$ROOT/.gemdepth_python_path" ]; then cat "$ROOT/.gemdepth_python_path"; return; fi
  if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -qE "^\s*${ENV_NAME}\s"; then
    conda run -n "$ENV_NAME" which python; return
  fi
  [ -x "$ROOT/.venv-${ENV_NAME}/bin/python" ] && { echo "$ROOT/.venv-${ENV_NAME}/bin/python"; return; }
  command -v python; 
}

do_verify() {
  sec "验证 PyTorch / CUDA / 关键依赖"
  local py; py="$(resolve_py)"
  info "使用解释器: ${py}"
  "$py" - <<'PYEOF'
import importlib, sys
print(f"  python    : {sys.version.split()[0]}  ({sys.executable})")
def chk(mod, attr="__version__"):
    try:
        m = importlib.import_module(mod)
        print(f"  [ OK ]  {mod:<14} {getattr(m, attr, '')}")
        return m
    except Exception as e:
        print(f"  [MISS]  {mod:<14} import 失败: {e}")
        return None
torch = chk("torch")
chk("torchvision"); chk("accelerate"); chk("hydra"); chk("omegaconf")
chk("cv2"); chk("numpy"); chk("kornia"); chk("decord"); chk("einops")
if torch is not None:
    print(f"  torch.version.cuda      = {torch.version.cuda}")
    avail = torch.cuda.is_available()
    print(f"  torch.cuda.is_available = {avail}")
    if avail:
        for i in range(torch.cuda.device_count()):
            print(f"      GPU{i}: {torch.cuda.get_device_name(i)}")
        try:
            x = torch.randn(8,8, device='cuda'); y = (x @ x).sum().item()
            print(f"  [ OK ]  GPU 矩阵乘法成功 (sum={y:.3f})")
        except Exception as e:
            print(f"  [MISS]  GPU 运算失败: {e}")
    else:
        print("  [WARN]  CUDA 不可用：检查驱动版本 >= 12.1 与 nvidia-smi")
        sys.exit(2)
else:
    sys.exit(2)
PYEOF
  local rc=$?
  hr
  [ $rc -eq 0 ] && ok "验证通过 ✅" || warn "验证发现问题 (退出码 $rc)，见上方 [MISS]/[WARN]"
  return $rc
}

# =============================================================================
# main
# =============================================================================
CMD=${1:-check}
case "$CMD" in
  check)   do_check ;;
  install) do_install ;;
  verify)  do_verify ;;
  all)     do_install && do_verify; echo; do_check ;;
  *) echo "用法: bash scripts/setup_aliyun.sh {check|install|verify|all}"; exit 1 ;;
esac
