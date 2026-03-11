#!/bin/bash

# Example usage:
#   bash seq_math.sh --num_shards 5 --num_seeds 2 --mcmc_steps 20 --temp 0.3 --model qwen_math --batch_size 10 --index_sample inverse_prob --neighbor_blocks 1

# Configuration
NUM_SHARDS=1    # Indices 0 to 4 covers all 500 questions
NUM_SEEDS=1     # Set to 8 to reproduce paper fully, or 1 to just test.
MCMC_STEPS=10
TEMP=0.25
MODEL=qwen_math
BATCH_SIZE=10
SAVE_STR="results/"
SAMPLE_IN_BLOCK=""
INDEX_SAMPLE="uniform"
SEMANTIC_BLOCK_TRUNCATE=""
SEMANTIC_BLOCK_ONE_STEP=""
SOLVER="mcmc"
NESTED_MODE="block"
NEIGHBOR_BLOCKS=""
W_MIN=0.0
W_MAX=1e9
usage() {
    echo "Usage: bash seq_math.sh [--num_shards N] [--num_seeds N] [--mcmc_steps N] [--temp T] [--model NAME] [--batch_size N] [--save_str PATH] [--index_sample NAME] [--sample_in_block] [--semantic_block_truncate] [--semantic_block_one_step] [--solver NAME] [--nested_mode NAME] [--neighbor_blocks N] [--w_min X] [--w_max X]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_shards) NUM_SHARDS="$2"; shift 2 ;;
        --num_seeds) NUM_SEEDS="$2"; shift 2 ;;
        --mcmc_steps) MCMC_STEPS="$2"; shift 2 ;;
        --temp) TEMP="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --save_str) SAVE_STR="$2"; shift 2 ;;
        --index_sample) INDEX_SAMPLE="$2"; shift 2 ;;
        --solver) SOLVER="$2"; shift 2 ;;
        --nested_mode) NESTED_MODE="$2"; shift 2 ;;
        --neighbor_blocks) NEIGHBOR_BLOCKS="$2"; shift 2 ;;
        --w_min) W_MIN="$2"; shift 2 ;;
        --w_max) W_MAX="$2"; shift 2 ;;
        --sample_in_block) SAMPLE_IN_BLOCK="--sample_in_block"; shift 1 ;;
        --semantic_block_truncate) SEMANTIC_BLOCK_TRUNCATE="--semantic_block_truncate"; shift 1 ;;
        --semantic_block_one_step) SEMANTIC_BLOCK_ONE_STEP="--semantic_block_one_step"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$NEIGHBOR_BLOCKS" ]]; then
    if [[ "$SOLVER" == "nested" ]]; then
        NEIGHBOR_BLOCKS=2
    else
        NEIGHBOR_BLOCKS=0
    fi
fi

# # Activate your environment
# source activate psamp
# # Ensure you are in the right directory
# cd llm_experiments

# Loop through seeds
for (( seed=0; seed<NUM_SEEDS; seed++ )); do
    # Loop through dataset shards
    for (( shard=0; shard<NUM_SHARDS; shard++ )); do
        
        echo "Running Shard: $shard with Seed: $seed"
        
        python power_samp_math.py \
            --batch_idx="$shard" \
            --batch_size="$BATCH_SIZE" \
            --mcmc_steps="$MCMC_STEPS" \
            --temperature="$TEMP" \
            --seed="$seed" \
            --model="$MODEL" \
            --save_str="$SAVE_STR" \
            --index_sample="$INDEX_SAMPLE" \
            --solver="$SOLVER" \
            --nested_mode="$NESTED_MODE" \
            --neighbor_blocks="$NEIGHBOR_BLOCKS" \
            --w_min="$W_MIN" \
            --w_max="$W_MAX" \
            $SAMPLE_IN_BLOCK \
            $SEMANTIC_BLOCK_TRUNCATE \
            $SEMANTIC_BLOCK_ONE_STEP \

    done
done
