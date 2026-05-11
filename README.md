# LARFT: Length-Aware Reinforcement Fine-Tuning

<p align="center">
  <a href="https://arxiv.org/abs/2603.19255"><img src="https://img.shields.io/badge/arXiv-2603.19255-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/your-username/LARFT/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

> Official implementation of *"LARFT: Closing the Cognition-Action Gap for Length Instruction Following in Large Language Models"*



## Quick Start

### Requirements

- Python >= 3.9, CUDA >= 11.8, PyTorch >= 2.4.0
- 4+ GPUs (tested on 8x A800 80GB)

### Installation

```bash
git clone https://github.com/your-username/LARFT.git
cd LARFT
bash install.sh
```

### Configure

Edit `configs/config_env.sh`:

```bash
export BASE_MODEL_PATH="/path/to/your/model"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WANDB_API_KEY="your_key"  # optional
```

### Prepare Data

```bash
# Generate sample data for a quick test
python scripts/prepare_data.py --generate_sample

# Or convert your own data
python scripts/prepare_data.py --input_file your_data.jsonl --output_dir data/
```

Each sample needs: a prompt with a length constraint, and a `ground_truth` target word count. See `scripts/prepare_data.py --help` for the expected JSONL format.

### Train

```bash
bash scripts/train.sh
```

Override any parameter via environment variables:

```bash
N_GPUS_PER_NODE=8 TRAIN_BATCH_SIZE=256 TOTAL_EPOCHS=5 bash scripts/train.sh
```

## Configuration

Key hyperparameters (defaults match the paper):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BATCH_SIZE` | 128 | Rollout batch size |
| `ROLLOUT_N` | 4 | Responses per prompt (G) |
| `LR` | 1e-6 | Actor learning rate |
| `SFT_LAMBDA` | 0.01 | $\lambda_{\max}$ for awareness loss |
| `KL_LOSS_COEF` | 0.001 | KL penalty coefficient |
| `ENTROPY_COEFF` | 0.01 | Entropy regularization |
| `TEMPERATURE` | 0.7 | Rollout sampling temperature |
| `TOP_P` | 0.8 | Nucleus sampling threshold |
| `MAX_RESPONSE_LENGTH` | 8000 | Max generation length (tokens) |
| `TOTAL_EPOCHS` | 3 | Training epochs |

## Project Structure

```
LARFT/
├── configs/config_env.sh         # Environment settings (edit this)
├── scripts/
│   ├── train.sh                  # Training launcher
│   └── prepare_data.py           # Data preparation
├── data/                         # Train/test parquet files
└── verl_framework/               # Modified verl (v0.4.x)
    └── verl/
        ├── trainer/ppo/
        │   └── ray_trainer.py        # [LARFT] SFT data preparation + pipeline ordering
        ├── workers/actor/
        │   └── dp_actor.py           # [LARFT] Unified loss + cosine schedule
        └── utils/reward_score/
            └── length_control.py     # [LARFT] Length reward function
```

## Citation

```bibtex
@article{zhang2026larft,
  title={LARFT: Closing the Cognition-Action Gap for Length Instruction Following in Large Language Models},
  author={Zhang, Wei and Du, Lintong and Zhang, Yuanhe and Zhou, Zhenhong and Wang, Kun and Sun, Li and Su, Sen},
  journal={arXiv preprint arXiv:2603.19255},
  year={2026}
}
```

## Acknowledgments

Built on [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLM). We thank the verl team for their excellent framework.

## License

[Apache License 2.0](verl_framework/LICENSE)
