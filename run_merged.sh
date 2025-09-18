#!/bin/bash

# Script to run training on MERGED dataset
# Usage: ./run_merged.sh [DUDE|PCBA]
# Example: ./run_merged.sh DUDE
#          ./run_merged.sh PCBA

# Get dataset selection from command line argument
DATASET_SELECTION=${1:-""}  # Default to empty if not provided

# Validate dataset selection
if [ -z "$DATASET_SELECTION" ]; then
    echo "Error: Please specify dataset selection (DUDE or PCBA)"
    echo "Usage: ./run_merged.sh [DUDE|PCBA]"
    exit 1
fi

if [ "$DATASET_SELECTION" != "DUDE" ] && [ "$DATASET_SELECTION" != "PCBA" ]; then
    echo "Error: Invalid dataset selection. Must be DUDE or PCBA"
    echo "Usage: ./run_merged.sh [DUDE|PCBA]"
    exit 1
fi

# Dataset configuration
DATA_PATH="./dataset/MERGED"
EXCLUSION_FILE="uniprots_excluded_at_90.txt"  # Exclude proteins for homology-based analysis
NEG_SAMPLE_RATIO=1  # Ratio of negative to positive samples (1 = balanced, -1 = use all negatives)

# Set protein filter file based on dataset selection
if [ "$DATASET_SELECTION" == "DUDE" ]; then
    PROTEIN_FILTER_FILE="./dataset/DUDE/dude_targets.fasta"
    PROTEIN_FILTER_TYPE="DUDE"
elif [ "$DATASET_SELECTION" == "PCBA" ]; then
    PROTEIN_FILTER_FILE="./dataset/LIT-PCBA/lit_pcba_sequence_dict.json"
    PROTEIN_FILTER_TYPE="PCBA"
fi

BATCH_SIZE=64
LR=5e-5
DROPOUT=0.05
NUM_EPOCHS=200
PATIENCE=20

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

PROT_ENCODER="westlake-repl/SaProt_650M_AF2"
DRUG_ENCODER="HUBioDataLab/SELFormer"

TOKEN_CACHE="dataset/processed_token"
SAVE_PATH="checkpoint/"

echo "================================"
echo "Training on MERGED Dataset"
echo "================================"
echo "Data: $DATA_PATH"
echo "Dataset selection: $DATASET_SELECTION"
echo "Protein filter file: $PROTEIN_FILTER_FILE"
echo "Exclusion file: $EXCLUSION_FILE"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Dropout: $DROPOUT"
echo "Max epochs: $NUM_EPOCHS"
echo "Patience: $PATIENCE"
echo "Protein encoder: $PROT_ENCODER"
echo "Drug encoder: $DRUG_ENCODER"
echo "================================"

# Execute command
python train.py \
    --data_path "$DATA_PATH" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --dropout $DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --patience $PATIENCE \
    --prot_encoder_path $PROT_ENCODER \
    --drug_encoder_path $DRUG_ENCODER \
    --token_cache $TOKEN_CACHE \
    --save_path $SAVE_PATH \
    --k_folds 1 \
    --exclusion_file "$EXCLUSION_FILE" \
    --neg_sample_ratio $NEG_SAMPLE_RATIO \
    --protein_filter_file "$PROTEIN_FILTER_FILE" \
    --protein_filter_type "$PROTEIN_FILTER_TYPE" \
    --similarity_threshold 0.9