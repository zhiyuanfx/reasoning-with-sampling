#!/bin/bash
#SBATCH --job-name=psamp_math
#SBATCH -t 0-23:59                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores (MB)
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --array=0-39               # 5 shards × 8 seeds = 40 tasks

# --- map array id -> (batch_idx, seed) ---
NUM_SHARDS=5
NUM_SEEDS=8
SEED=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
SOLVER="${SOLVER:-mcmc}"
NESTED_MODE="${NESTED_MODE:-block}"
NEIGHBOR_BLOCKS="${NEIGHBOR_BLOCKS:-2}"
W_MIN="${W_MIN:-0.0}"
W_MAX="${W_MAX:-1e9}"
SAVE_STR="${SAVE_STR:-results/}"

module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01

export HF_HOME={HUGGING_FACE_HOME}
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"

export PYTHONPATH="$PYTHONPATH:{/path/to/reasoning-with-sampling/llm_experiments}"
export HF_TOKEN={HF_TOKEN}

source activate psamp
cd /path/to/reasoning-with-sampling/llm_experiments

echo "Running shard BATCH_IDX=${BATCH_IDX} with SEED=${SEED} (task ${SLURM_ARRAY_TASK_ID})"
python power_samp_math.py \
  --batch_idx="${BATCH_IDX}" \
  --mcmc_steps=10 \
  --temperature=0.25 \
  --seed="${SEED}" \
  --model=qwen_math \
  --save_str="${SAVE_STR}" \
  --solver="${SOLVER}" \
  --nested_mode="${NESTED_MODE}" \
  --neighbor_blocks="${NEIGHBOR_BLOCKS}" \
  --w_min="${W_MIN}" \
  --w_max="${W_MAX}"
