#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -A mit_general
#SBATCH --gres=gpu:1
#SBATCH --time=1:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=eval_owt
#SBATCH --output=logs/eval_owt_%j.out
#SBATCH --error=logs/eval_owt_%j.err

source myenvironment/bin/activate
cd ~/cs336-assignment1

echo "============================================"
echo "Evaluating OWT Model on Validation Set"
echo "Started at: $(date)"
echo "============================================"

python -c "
import torch
import numpy as np
from einops import rearrange
from cs336_basics.model import Transformer_LM
from cs336_basics.optimizer import cross_entropy
from cs336_basics.training import get_batch

# Model config (from train_owt.sh)
VOCAB_SIZE = 32000
CONTEXT_LENGTH = 256
D_MODEL = 512
D_FF = 1344
NUM_LAYERS = 4
NUM_HEADS = 16
ROPE_THETA = 10000.0

# Evaluation config
CHECKPOINT_PATH = 'checkpoints/owt_model.pt'
VAL_DATASET_PATH = 'tokenized/owt_valid_tokens.npy'
BATCH_SIZE = 32
NUM_BATCHES = 100
DEVICE = 'cuda'

print(f'Loading model from: {CHECKPOINT_PATH}')
print(f'Validation dataset: {VAL_DATASET_PATH}')

# Load validation dataset
val_dataset = np.load(VAL_DATASET_PATH, mmap_mode='r')
print(f'Val dataset shape: {val_dataset.shape}, dtype: {val_dataset.dtype}')

# Create model
model = Transformer_LM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
    rope_theta=ROPE_THETA,
    device=DEVICE,
).to(DEVICE)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model'])
print(f'Loaded checkpoint from iteration: {checkpoint.get(\"iteration\", \"unknown\")}')

# Evaluate
model.eval()
total_loss = 0.0

print(f'Evaluating on {NUM_BATCHES} batches...')
with torch.no_grad():
    for i in range(NUM_BATCHES):
        inputs, labels = get_batch(val_dataset, BATCH_SIZE, CONTEXT_LENGTH, DEVICE)
        logits = model(inputs)
        
        logits_flat = rearrange(logits, 'b s v -> (b s) v')
        labels_flat = rearrange(labels, 'b s -> (b s)')
        
        loss = cross_entropy(logits_flat, labels_flat)
        total_loss += loss.item()
        
        if (i + 1) % 20 == 0:
            print(f'  Batch {i+1}/{NUM_BATCHES}: running avg loss = {total_loss / (i+1):.4f}')

avg_loss = total_loss / NUM_BATCHES
perplexity = np.exp(avg_loss)

print()
print('============================================')
print('RESULTS')
print('============================================')
print(f'Validation Loss: {avg_loss:.4f}')
print(f'Perplexity: {perplexity:.2f}')
print('============================================')
"

echo ""
echo "Finished at: $(date)"

