#!/bin/bash
# Idempotent Jetson-specific PyTorch wheel installation.
# Installs PyTorch and torchvision from NVIDIA's PyPI index for Jetson.
# Usage: sudo bash scripts/install_jetson_pytorch.sh [VENV_DIR]
set -euo pipefail

VENV_DIR="${1:-/opt/mousedroid/venv}"
NVIDIA_INDEX="https://pypi.ngc.nvidia.com"

echo "=== Jetson PyTorch Installation ==="

# ---- Validate venv ----

if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    echo "       Create it first or pass the correct path as argument."
    exit 1
fi

PIP="${VENV_DIR}/bin/pip"
PYTHON="${VENV_DIR}/bin/python"

# ---- Detect JetPack version ----

echo "--- Detecting JetPack version ---"
if dpkg -l nvidia-jetpack 2>/dev/null | grep -q '^ii'; then
    JETPACK_VER=$(dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null)
    echo "    JetPack version: ${JETPACK_VER}"
else
    echo "    WARNING: nvidia-jetpack package not found"
    JETPACK_VER="unknown"
fi

# ---- Detect CUDA version ----

echo "--- Detecting CUDA version ---"
export CUDA_HOME=/usr/local/cuda

if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "    CUDA version: ${CUDA_VER}"
elif [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    CUDA_VER=$("${CUDA_HOME}/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "    CUDA version: ${CUDA_VER} (from ${CUDA_HOME})"
else
    echo "    WARNING: nvcc not found. CUDA may not be installed."
    CUDA_VER="unknown"
fi

# ---- Set environment for build ----

echo "--- Setting CUDA environment ---"
export CUDA_HOME=/usr/local/cuda
if [ -d "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    echo "    CUDA_HOME=${CUDA_HOME}"
else
    echo "    WARNING: ${CUDA_HOME} does not exist"
fi

# ---- Install PyTorch ----

echo "--- Installing PyTorch from NVIDIA index ---"
"${PIP}" install --quiet --upgrade pip

if "${PYTHON}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$("${PYTHON}" -c "import torch; print(torch.__version__)")
    echo "    PyTorch ${TORCH_VER} with CUDA support already installed"
else
    echo "    Installing torch and torchvision..."
    "${PIP}" install --quiet torch torchvision --index-url "${NVIDIA_INDEX}"
    echo "    PyTorch installation complete"
fi

# ---- Verify installation ----

echo "--- Verifying PyTorch installation ---"
if "${PYTHON}" -c "
import torch
print(f'    PyTorch version: {torch.__version__}')
print(f'    CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'    CUDA version: {torch.version.cuda}')
"; then
    echo "    PyTorch verification passed"
else
    echo "    WARNING: PyTorch verification failed"
    echo "    torch may need to be reinstalled for this JetPack version"
    exit 1
fi

echo "=== PyTorch installation complete ==="
