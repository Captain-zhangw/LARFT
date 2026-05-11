#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# LARFT Installation Script
#
# This script installs all dependencies required for LARFT training.
#
# Prerequisites:
#   - Python >= 3.9
#   - CUDA >= 11.8
#   - PyTorch >= 2.4.0 (installed separately)
#
# Usage:
#   bash install.sh           # Full installation
#   bash install.sh --no-vllm # Skip vLLM installation (if already installed)
# ============================================================================

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
FRAMEWORK_DIR="${SCRIPT_DIR}/verl_framework"

echo "============================================"
echo "  LARFT Installation"
echo "============================================"

# Parse arguments
INSTALL_VLLM=true
for arg in "$@"; do
    case $arg in
        --no-vllm) INSTALL_VLLM=false ;;
    esac
done

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: ${python_version}"

# Check PyTorch
if python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null; then
    echo "PyTorch is already installed."
else
    echo "WARNING: PyTorch not found. Please install PyTorch first:"
    echo "  pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
    echo ""
    echo "Visit https://pytorch.org/get-started/locally/ for installation instructions."
    exit 1
fi

# Step 1: Install the modified verl framework
echo ""
echo "[1/4] Installing modified verl framework..."
cd "${FRAMEWORK_DIR}"
pip install -e . 2>&1 | tail -5
echo "  verl framework installed."

# Step 2: Install vLLM (for rollout generation)
if [ "$INSTALL_VLLM" = true ]; then
    echo ""
    echo "[2/4] Installing vLLM..."
    pip install vllm==0.8.4 2>&1 | tail -5
    echo "  vLLM installed."
else
    echo ""
    echo "[2/4] Skipping vLLM installation (--no-vllm)."
fi

# Step 3: Install flash-attention
echo ""
echo "[3/4] Installing flash-attn..."
pip install flash-attn --no-build-isolation 2>&1 | tail -5
echo "  flash-attn installed."

# Step 4: Install additional dependencies
echo ""
echo "[4/4] Installing additional dependencies..."
pip install wandb liger-kernel 2>&1 | tail -3
echo "  Additional dependencies installed."

# Verify installation
echo ""
echo "============================================"
echo "  Verifying Installation"
echo "============================================"

python3 -c "
import verl
print(f'  verl: OK')
" 2>/dev/null && echo "" || echo "  WARNING: verl import failed"

python3 -c "
import vllm
print(f'  vllm: OK (version {vllm.__version__})')
" 2>/dev/null || echo "  WARNING: vllm not available"

python3 -c "
import flash_attn
print(f'  flash_attn: OK (version {flash_attn.__version__})')
" 2>/dev/null || echo "  WARNING: flash_attn not available"

python3 -c "
import torch
print(f'  torch: OK (version {torch.__version__}, CUDA {torch.version.cuda})')
print(f'  GPU count: {torch.cuda.device_count()}')
" 2>/dev/null || echo "  WARNING: torch not available"

echo ""
echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit configs/config_env.sh with your environment settings"
echo "  2. Prepare data:  python scripts/prepare_data.py --generate_sample"
echo "  3. Start training: bash scripts/train.sh"
echo ""
