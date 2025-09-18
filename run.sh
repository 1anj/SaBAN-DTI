#!/bin/bash

# Usage: ./run.sh [data_path] [k_folds]

DATA_PATH=${2:-"./dataset/BindingDB/BindingDB.csv"}
K_FOLDS=${4:-"5"}
BATCH_SIZE=64
LR=5e-5
DROPOUT=0.05
NUM_EPOCHS=200
PATIENCE=20
PROT_ENCODER="westlake-repl/SaProt_650M_AF2"
DRUG_ENCODER="HUBioDataLab/SELFormer"

echo "================================"
echo "Data: $DATA_PATH"
echo "K-folds: $K_FOLDS"
echo "================================"

python train.py \
    --data_path "$DATA_PATH" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --dropout $DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --patience $PATIENCE \
    --prot_encoder_path $PROT_ENCODER \
    --drug_encoder_path $DRUG_ENCODER \
    --token_cache dataset/processed_token \
    --save_path checkpoint/ \
    --k_folds $K_FOLDS
