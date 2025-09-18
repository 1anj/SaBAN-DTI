#!/bin/bash

# Configuration
CHECKPOINT_PATH="checkpoint/best_model.ckpt"
OUTPUT_DIR="test_results"
BATCH_SIZE=8
GPU_ID=0

# Task selection: 'dude', 'pcba', or 'both'
TASK=${1:-both}

echo "Starting evaluation on $TASK dataset(s)..."
echo "Using checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"

# Run evaluation
CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py \
    --checkpoint $CHECKPOINT_PATH \
    --dataset $TASK \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --device cuda

echo "Evaluation complete! Check $OUTPUT_DIR for results."