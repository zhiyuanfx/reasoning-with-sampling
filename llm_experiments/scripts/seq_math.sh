#!/bin/bash

# Configuration
NUM_SHARDS=1  # Indices 0 to 4 covers all 500 questions
NUM_SEEDS=1   # Set to 8 to reproduce paper fully, or 1 to just test.

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
            --mcmc_steps=10 \
            --temp=0.25 \
            --seed="$seed" \
            --model=qwen_math \
            
    done
done