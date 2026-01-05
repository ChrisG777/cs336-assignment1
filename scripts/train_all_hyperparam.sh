#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ts_sweep
#SBATCH --output=logs/ts_sweep_%j.out
#SBATCH --error=logs/ts_sweep_%j.err

source myenvironment/bin/activate
cd ~/cs336-assignment1
mkdir -p logs experiments

echo "============================================"
echo "TinyStories Hyperparameter Sweep"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "============================================"

# Fixed params
TRAIN_DATASET="tokenized/tinystories_train.npy"
VAL_DATASET="tokenized/tinystories_valid.npy"
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
NUM_ITERATIONS=5000
EVAL_INTERVAL=500
DEVICE="cuda"

# Model configs: name d_model num_layers num_heads d_ff
declare -a MODELS=(
    "small 256 4 4 1024"
    "medium 384 6 6 1536"
)

# Hyperparameters to sweep
LEARNING_RATES=(1e-3 5e-4 3e-4)
BATCH_SIZES=(32 64)
WEIGHT_DECAYS=(0.1 0.01)

experiment_count=0

for model_config in "${MODELS[@]}"; do
    read -r size d_model num_layers num_heads d_ff <<< "$model_config"
    
    for lr in "${LEARNING_RATES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for wd in "${WEIGHT_DECAYS[@]}"; do
                
                exp_name="ts_${size}_lr${lr}_bs${bs}_wd${wd}"
                experiment_count=$((experiment_count + 1))
                
                # Calculate warmup (10% of iterations)
                warmup_iters=$((NUM_ITERATIONS / 10))
                
                echo ""
                echo "============================================"
                echo "[$experiment_count] Running: $exp_name"
                echo "  Model: d=$d_model, L=$num_layers, H=$num_heads, ff=$d_ff"
                echo "  LR=$lr, BS=$bs, WD=$wd"
                echo "  Started at: $(date)"
                echo "============================================"
                
                # Create experiment directory
                mkdir -p "experiments/${exp_name}"
                
                python -m cs336_basics.training.training_together \
                    --train-dataset "$TRAIN_DATASET" \
                    --val-dataset "$VAL_DATASET" \
                    --vocab-size $VOCAB_SIZE \
                    --context-length $CONTEXT_LENGTH \
                    --d-model $d_model \
                    --num-layers $num_layers \
                    --num-heads $num_heads \
                    --d-ff $d_ff \
                    --num-iterations $NUM_ITERATIONS \
                    --batch-size $bs \
                    --max-lr $lr \
                    --min-lr 1e-5 \
                    --warmup-iters $warmup_iters \
                    --cosine-cycle-iters $NUM_ITERATIONS \
                    --weight-decay $wd \
                    --max-grad-norm 1.0 \
                    --device $DEVICE \
                    --checkpoint "experiments/${exp_name}/checkpoint.pt" \
                    --eval-interval $EVAL_INTERVAL \
                    --eval-batches 20 \
                    --exp-name "$exp_name"
                
                echo "Finished $exp_name at: $(date)"
                
            done
        done
    done
done

echo ""
echo "============================================"
echo "All $experiment_count experiments complete!"
echo "Finished at: $(date)"
echo "============================================"

# Run evaluation on all models
echo ""
echo "Running final evaluation..."
python scripts/evaluate_all_models.py \
    --experiments-dir experiments \
    --val-dataset "$VAL_DATASET" \
    --device cuda \
    --output eval_results.json

echo "Done! Results in eval_results.json"

