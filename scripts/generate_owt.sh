#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH --job-name=gen_owt
#SBATCH --output=logs/gen_owt_%j.out
#SBATCH --error=logs/gen_owt_%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# ============================================
# Generate Text with Trained OWT Model
# ============================================

REPO_DIR="$HOME/cs336-assignment1"
ENV_DIR="$REPO_DIR/myenvironment"

cd $REPO_DIR
source $ENV_DIR/bin/activate
mkdir -p logs

echo "============================================"
echo "Text Generation with OWT Model"
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "============================================"

# Default prompt - can be overridden by passing as argument to script
PROMPT="${1:-The future of artificial intelligence is}"

# Model architecture (must match training)
VOCAB_SIZE=32000
CONTEXT_LENGTH=256
D_MODEL=512
D_FF=1344
NUM_LAYERS=4
NUM_HEADS=16
ROPE_THETA=10000.0

# Generation parameters
TOP_P=0.9
TEMPERATURE=0.8
MAX_TOKENS=200

python -m cs336_basics.decoding.decoding \
    --checkpoint "checkpoints/owt_model.pt" \
    --vocab "tokenized/vocab_owt_train.json" \
    --merges "tokenized/merges_owt_train.txt" \
    --vocab-size $VOCAB_SIZE \
    --context-length $CONTEXT_LENGTH \
    --d-model $D_MODEL \
    --d-ff $D_FF \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --rope-theta $ROPE_THETA \
    --prompt "$PROMPT" \
    --top-p $TOP_P \
    --temperature $TEMPERATURE \
    --max-tokens $MAX_TOKENS \
    --special-tokens "<|endoftext|>" \
    --device "cuda"

echo ""
echo "============================================"
echo "Finished at: $(date)"
echo "============================================"

