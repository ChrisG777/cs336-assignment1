#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH -J encode_ts_val
#SBATCH -o logs/encode_ts_val_%j.out
#SBATCH -e logs/encode_ts_val_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

module load miniforge
source ~/cs336-assignment1/myenvironment/bin/activate

cd ~/cs336-assignment1/

# Use scratch for fast I/O
TMP_DIR="/home/$USER/orcd/scratch/tok_shards_ts_val_$$"

# --- CONFIGURATION ---
# Input: Assuming standard filename. CHECK THIS PATH!
INPUT_FILE="./data/TinyStoriesV2-GPT4-valid.txt"

# Vocab/Merges: MUST use the ones generated from TRAINING
VOCAB="tokenized/vocab_tinystories.json"
MERGES="tokenized/merges_tinystories.txt"

# Output
OUTPUT_FILE="tokenized/tinystories_valid.npy"

echo "Encoding $INPUT_FILE using vocab from $VOCAB..."

python -m cs336_basics.BPE_tokenizer.parallel_encode \
    --input "$INPUT_FILE" \
    --vocab "$VOCAB" \
    --merges "$MERGES" \
    --output "$OUTPUT_FILE" \
    --special-tokens "<|endoftext|>" \
    --num-workers $SLURM_CPUS_PER_TASK \
    --num-chunks $(($SLURM_CPUS_PER_TASK * 4)) \
    --tmp-dir "$TMP_DIR"

rm -rf "$TMP_DIR"

echo "Done! Output saved to $OUTPUT_FILE"