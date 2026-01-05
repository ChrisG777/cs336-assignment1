#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH -J encode_valid
#SBATCH -o logs/encode_valid_%j.out
#SBATCH -e logs/encode_valid_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load miniforge
source ~/cs336-assignment1/myenvironment/bin/activate

# 1. Move to Project Root
cd ~/cs336-assignment1/

# 2. Use scratch for fast I/O
TMP_DIR="/home/$USER/orcd/scratch/tok_shards_val_$$"

# 3. Run Encoder
# NOTE: We use the TRAINING vocab/merges to ensure token ID consistency
python -m cs336_basics.BPE_tokenizer.parallel_encode \
    --input ./data/owt_valid.txt \
    --vocab tokenized/vocab_owt_train.json \
    --merges tokenized/merges_owt_train.txt \
    --output tokenized/owt_valid_tokens.npy \
    --special-tokens "<|endoftext|>" \
    --num-workers $SLURM_CPUS_PER_TASK \
    --num-chunks $(($SLURM_CPUS_PER_TASK * 4)) \
    --tmp-dir "$TMP_DIR"

# 4. Cleanup
rm -rf "$TMP_DIR"

echo "Done! Created tokenized/owt_valid_tokens.npy"