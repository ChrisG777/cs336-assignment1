#!/bin/bash
# Hyperparameter search script for TinyStories training
# Submits multiple SLURM jobs with varying hyperparameters

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================
# Fixed hyperparameters (recommended baseline)
# ============================================
TRAIN_DATASET="tokenized/tinystories_train.npy"
VAL_DATASET="tokenized/tinystories_valid.npy"
VOCAB_SIZE=10000
CONTEXT_LENGTH=256
ROPE_THETA=10000.0
MIN_LR=1e-5
WARMUP_RATIO=0.1  # warmup_iters = WARMUP_RATIO * num_iterations
EVAL_INTERVAL=500
EVAL_BATCHES=20
DEVICE="cuda"

# ============================================
# Hyperparameter ranges to search
# ============================================

# Learning rates to try
LEARNING_RATES=(1e-3 5e-4 3e-4 1e-4)

# Model sizes: (d_model, num_layers, num_heads, d_ff)
# Small, Medium, Large configurations
declare -A MODEL_CONFIGS
MODEL_CONFIGS["small"]="256 4 4 1024"
MODEL_CONFIGS["medium"]="384 6 6 1536"
MODEL_CONFIGS["large"]="512 8 8 2048"

# Batch sizes
BATCH_SIZES=(32 64)

# Weight decay values
WEIGHT_DECAYS=(0.1 0.01)

# Number of iterations (adjust based on compute budget)
NUM_ITERATIONS=10000

# ============================================
# Submit jobs for each combination
# ============================================

echo "============================================"
echo "TinyStories Hyperparameter Search"
echo "============================================"

job_count=0

for model_size in "small" "medium"; do  # Start with small/medium to iterate faster
    # Parse model config
    read d_model num_layers num_heads d_ff <<< "${MODEL_CONFIGS[$model_size]}"
    
    for lr in "${LEARNING_RATES[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for weight_decay in "${WEIGHT_DECAYS[@]}"; do
                
                # Create experiment name
                exp_name="ts_${model_size}_lr${lr}_bs${batch_size}_wd${weight_decay}"
                
                # Calculate warmup iterations
                warmup_iters=$(python3 -c "print(int($NUM_ITERATIONS * $WARMUP_RATIO))")
                
                echo "Submitting: $exp_name"
                echo "  Model: d_model=$d_model, layers=$num_layers, heads=$num_heads, d_ff=$d_ff"
                echo "  LR: $lr, Batch: $batch_size, Weight Decay: $weight_decay"
                
                # Submit the job
                sbatch <<EOF
#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=${exp_name}
#SBATCH --output=logs/${exp_name}_%j.out
#SBATCH --error=logs/${exp_name}_%j.err

echo "============================================"
echo "Experiment: ${exp_name}"
echo "Started at: \$(date)"
echo "Node: \$(hostname)"
echo "============================================"

# Activate environment
source myenvironment/bin/activate
cd ~/cs336-assignment1

# Create checkpoint directory
mkdir -p experiments/${exp_name}

# Run training
python -m cs336_basics.training.training_together \
    --train-dataset "${TRAIN_DATASET}" \
    --val-dataset "${VAL_DATASET}" \
    --vocab-size ${VOCAB_SIZE} \
    --context-length ${CONTEXT_LENGTH} \
    --d-model ${d_model} \
    --num-layers ${num_layers} \
    --num-heads ${num_heads} \
    --d-ff ${d_ff} \
    --rope-theta ${ROPE_THETA} \
    --num-iterations ${NUM_ITERATIONS} \
    --batch-size ${batch_size} \
    --max-lr ${lr} \
    --min-lr ${MIN_LR} \
    --warmup-iters ${warmup_iters} \
    --cosine-cycle-iters ${NUM_ITERATIONS} \
    --weight-decay ${weight_decay} \
    --max-grad-norm 1.0 \
    --device ${DEVICE} \
    --checkpoint "experiments/${exp_name}/checkpoint.pt" \
    --eval-interval ${EVAL_INTERVAL} \
    --eval-batches ${EVAL_BATCHES} \
    --exp-name "${exp_name}"

echo "============================================"
echo "Finished at: \$(date)"
echo "============================================"
EOF
                
                job_count=$((job_count + 1))
                
                # Small delay to avoid overwhelming the scheduler
                sleep 0.5
            done
        done
    done
done

echo ""
echo "============================================"
echo "Submitted $job_count jobs"
echo "Monitor with: squeue -u \$USER"
echo "View logs in: logs/"
echo "Results in: experiments/"
echo "============================================"

