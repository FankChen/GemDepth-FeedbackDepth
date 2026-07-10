#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create a conda env named "gemdepth" from GemDepth's official requirements.txt
#
# Target : Aliyun DSW (cn-shanghai, 8xH20).  Also works on any Linux x86_64 box
#          whose NVIDIA driver is CUDA >= 12.1 (H20 driver is 12.8 -> OK, the
#          cu121 wheels run under a newer driver by forward-compat).
#
# Why a conda env (not the system /usr/local/bin/python3)?
#   The image python ships torch 2.9.1+cu128; this env pins the *official*
#   GemDepth stack (torch 2.3.1 / cu121) for a faithful reproduction.
#
# Usage (on the DSW terminal):
#   cd /mnt/workspace/liren/GemDepth-FeedbackDepth
#   bash scripts/setup_conda_gemdepth.sh
#   conda activate gemdepth
#
# Overridable env vars:
#   ENV_NAME=gemdepth  PY_VER=3.10  CONDA_HOME=/mnt/workspace/miniconda3
#   PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
# ---------------------------------------------------------------------------
set -eo pipefail

ENV_NAME="${ENV_NAME:-gemdepth}"
PY_VER="${PY_VER:-3.10}"
# Persistent CPFS on the DSW -> a conda living here survives instance restarts.
CONDA_HOME="${CONDA_HOME:-/mnt/workspace/miniconda3}"
# Aliyun PyPI mirror (fast in China). The default linux wheel for torch==2.3.1
# on PyPI IS the cu121 build, and the nvidia-*-cu12==12.1.* pins match it.
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$ROOT/requirements.txt"
[ -f "$REQ" ] || { echo "ERROR: requirements.txt not found at $REQ"; exit 1; }

# --- 1) locate a *persistent* conda, or install Miniconda to $CONDA_HOME ------
pick_conda() {
  # already installed at the persistent location?
  [ -x "$CONDA_HOME/bin/conda" ] && { echo "$CONDA_HOME/bin/conda"; return; }
  # a system conda whose base is on persistent storage is also fine
  local c base
  for c in "$(command -v conda 2>/dev/null || true)" \
           /opt/conda/bin/conda "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
    [ -x "$c" ] || continue
    base="$("$c" info --base 2>/dev/null || true)"
    case "$base" in /mnt/workspace/*) echo "$c"; return;; esac
  done
  echo ""
}

CONDA_BIN="$(pick_conda)"
if [ -z "$CONDA_BIN" ]; then
  echo ">>> no persistent conda found -> installing Miniconda to $CONDA_HOME"
  mkdir -p "$(dirname "$CONDA_HOME")"
  TMP_SH="/tmp/miniconda_gemdepth.sh"
  URL_TUNA="https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  URL_OFF="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$URL_TUNA" -o "$TMP_SH" || curl -fsSL "$URL_OFF" -o "$TMP_SH"
  else
    wget -qO "$TMP_SH" "$URL_TUNA" || wget -qO "$TMP_SH" "$URL_OFF"
  fi
  bash "$TMP_SH" -b -p "$CONDA_HOME"
  rm -f "$TMP_SH"
  CONDA_BIN="$CONDA_HOME/bin/conda"
fi

CONDA_BASE="$("$CONDA_BIN" info --base)"
echo ">>> using conda: $CONDA_BIN   (base: $CONDA_BASE)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- 2) create (or reuse) the named env --------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">>> conda env '$ENV_NAME' already exists -> reusing it"
else
  echo ">>> creating conda env '$ENV_NAME' (python=$PY_VER)"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"
echo ">>> active python: $(which python)  ($(python --version 2>&1))"

# --- 3) install the official pinned dependencies -----------------------------
echo ">>> installing official requirements (this pulls torch 2.3.1 / cu121)"
python -m pip install --upgrade pip -i "$PIP_INDEX"
python -m pip install -r "$REQ" -i "$PIP_INDEX"

# --- 4) sanity check ----------------------------------------------------------
python - <<'PY'
import torch, torchvision, numpy
print("torch      :", torch.__version__)
print("torchvision:", torchvision.__version__)
print("numpy      :", numpy.__version__)
print("cuda build :", torch.version.cuda)
print("cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     :", torch.cuda.get_device_name(0))
PY

echo ""
echo ">>> DONE.  Activate the env with:"
echo "       source $CONDA_BASE/etc/profile.d/conda.sh   # once per shell if 'conda' not on PATH"
echo "       conda activate $ENV_NAME"
