#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=eval_all
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --error=logs/eval_all_%j.err

cd ~/cs336-assignment1
source myenvironment/bin/activate

echo "============================================"
echo "Evaluating all trained models"
echo "Started at: $(date)"
echo "============================================"

python scripts/evaluate_all_models.py \
    --experiments-dir experiments \
    --val-dataset tokenized/tinystories_valid.npy \
    --batch-size 32 \
    --num-batches 50 \
    --device cuda \
    --output eval_results.json

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"

