#!/usr/bin/env bash
# ============================================================================
# LARFT Global Environment Configuration
#
# Edit this file to match your environment before running training.
# Usage: source configs/config_env.sh
# ============================================================================

# ==================== Base Model Path ====================
# Path to the pretrained model (HuggingFace format)
# Examples:
#   export BASE_MODEL_PATH="/path/to/Llama-3.2-1B-Instruct"
#   export BASE_MODEL_PATH="/path/to/Qwen2.5-3B-Instruct"
#   export BASE_MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"  # download from HF
export BASE_MODEL_PATH="/path/to/your/model"

# ==================== GPU Configuration ====================
# Comma-separated list of GPU IDs to use
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# ==================== Ray Configuration ====================
# Temporary directory for Ray distributed training
export RAY_TMPDIR="/tmp/larft_ray"
mkdir -p "${RAY_TMPDIR}" 2>/dev/null
export RAY_DEBUG_POST_MORTEM=0

# ==================== HuggingFace Cache ====================
# Set cache directory to avoid re-downloading models
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "${HF_HOME}" 2>/dev/null

# ==================== WandB Configuration ====================
# Set your WandB API key for experiment tracking
# You can also run `wandb login` instead
# export WANDB_API_KEY="your_wandb_api_key_here"

# To disable WandB logging, uncomment:
# export WANDB_MODE="disabled"

# ==================== Network Proxy (Optional) ====================
# Uncomment if behind a proxy
# export https_proxy="http://your.proxy:port"
# export http_proxy="http://your.proxy:port"

# ==================== Conda Environment (Optional) ====================
# Uncomment and edit if using conda
# export PATH="/path/to/miniconda3/condabin:$PATH"
# if [ -f /path/to/miniconda3/etc/profile.d/conda.sh ]; then
#     source /path/to/miniconda3/etc/profile.d/conda.sh
#     conda activate your_env_name
# fi

echo "[config_env.sh] Environment configured successfully"
echo "  BASE_MODEL_PATH: ${BASE_MODEL_PATH}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
