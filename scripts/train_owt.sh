#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH --job-name=owt_train
#SBATCH --output=logs/owt_train_%j.out
#SBATCH --error=logs/owt_train_%j.err
#SBATCH --time=6:00:00                 
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source myenvironment/bin/activate
cd ~/cs336-assignment1
mkdir -p logs checkpoints

# --- CONFIGURATION (Based on PDF Page 41 & 45) ---
# Total Tokens Budget: ~327,680,000 [cite: 1125]
# Batch Size: 128 (Standard efficient size)
# Context: 256
# Tokens per Step: 128 * 256 = 32,768
# Total Iters: 327,680,000 / 32,768 ≈ 10,000

# NOTE: vocab-size is 32000 for OWT, unlike 10000 for TinyStories.

python -m cs336_basics.training.training_together \
    --train-dataset "tokenized/owt_train_tokens.npy" \
    --vocab-size 32000 \
    --context-length 256 \
    --d-model 512 \
    --d-ff 1344 \
    --num-layers 4 \
    --num-heads 16 \
    --rope-theta 10000.0 \
    --batch-size 128 \
    --num-iterations 10000 \
    --max-lr 5e-4 \
    --warmup-iters 200 \
    --checkpoint "checkpoints/owt_model.pt" \
    --device "cuda" \
    --eval-interval 500