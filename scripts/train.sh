#!/usr/bin/env bash
set -euo pipefail
set -x

# ============================================================================
# LARFT Training Script: SFT + RL Fusion for Length-Following
# ============================================================================
#
# This script launches SFT+RL fusion training where the model simultaneously
# learns from:
#   1. RL (GRPO) policy gradient with length-based reward
#   2. SFT auxiliary loss (word-count perception) with dynamic scheduling
#
# Usage:
#   bash scripts/train.sh
#
# Before running, make sure to:
#   1. Edit configs/config_env.sh with your environment settings
#   2. Prepare data using: python scripts/prepare_data.py
#   3. Install dependencies using: bash install.sh
# ============================================================================

# ==================== Load environment config ====================
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${PROJECT_ROOT}/configs/config_env.sh"

# ==================== Training Hyperparameters ====================
# >> GPU Configuration
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}

# >> Model
MODEL_PATH=${MODEL_PATH:-"${BASE_MODEL_PATH}"}

# >> Batch sizes
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MICRO_BATCH_PER_GPU=${MICRO_BATCH_PER_GPU:-2}

# >> GRPO rollout
ROLLOUT_N=${ROLLOUT_N:-4}              # Number of responses per prompt
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.8}

# >> Sequence lengths
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8000}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-60000}

# >> Learning rate
LR=${LR:-1e-6}

# >> SFT+RL fusion
SFT_LAMBDA=${SFT_LAMBDA:-0.01}       # Max SFT loss weight (dynamic scheduling)

# >> Regularization
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}     # KL divergence loss coefficient
ENTROPY_COEFF=${ENTROPY_COEFF:-0.01}      # Entropy regularization coefficient

# >> Training schedule
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-5}

# >> Logging
PROJECT_NAME=${PROJECT_NAME:-"LARFT"}
EXP_NAME=${EXP_NAME:-"SFT-RL-Fusion-Linear-Epoch${TOTAL_EPOCHS}"}

# ==================== Paths ====================
DATA_DIR="${PROJECT_ROOT}/data"
RESULT_DIR="${PROJECT_ROOT}/results/${EXP_NAME}"
FRAMEWORK_DIR="${PROJECT_ROOT}/verl_framework"
mkdir -p "${RESULT_DIR}"

# ==================== Validation ====================
if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
    echo "Error: Training data not found at ${DATA_DIR}/train.parquet"
    echo "Please run: python scripts/prepare_data.py --generate_sample"
    exit 1
fi

if [[ ! -f "${DATA_DIR}/test.parquet" ]]; then
    echo "Error: Test data not found at ${DATA_DIR}/test.parquet"
    echo "Please run: python scripts/prepare_data.py --generate_sample"
    exit 1
fi

# ==================== Print Configuration ====================
echo "============================================"
echo "  LARFT: SFT + RL Fusion Training"
echo "============================================"
echo "Model:              ${MODEL_PATH}"
echo "GPUs:               ${N_GPUS_PER_NODE} x ${NNODES} node(s)"
echo "Train batch size:   ${TRAIN_BATCH_SIZE}"
echo "Rollout N:          ${ROLLOUT_N}"
echo "Max response len:   ${MAX_RESPONSE_LENGTH}"
echo "Learning rate:      ${LR}"
echo "SFT lambda (max):   ${SFT_LAMBDA}"
echo "KL coef:            ${KL_LOSS_COEF}"
echo "Entropy coef:       ${ENTROPY_COEFF}"
echo "Epochs:             ${TOTAL_EPOCHS}"
echo "Output dir:         ${RESULT_DIR}"
echo "============================================"

# ==================== Launch Training ====================
cd "${FRAMEWORK_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_PER_NODE} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.default_local_dir="${RESULT_DIR}" \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.log_val_generations=4 \
    trainer.val_before_train=True \
    +trainer.sft_lambda=${SFT_LAMBDA} \
    "$@" \
    2>&1 | tee "${RESULT_DIR}/${EXP_NAME}.log"
