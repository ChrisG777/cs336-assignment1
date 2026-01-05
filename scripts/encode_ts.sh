#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH -J encode_ts
#SBATCH -o logs/encode_ts_%j.out
#SBATCH -e logs/encode_ts_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load miniforge
source ~/cs336-assignment1/myenvironment/bin/activate

cd ~/cs336-assignment1/

# Use scratch for fast I/O
TMP_DIR="/home/$USER/orcd/scratch/tok_shards_ts_$$"

# Note: pointing to tokenized/ folder for vocab and output
python -m cs336_basics.BPE_tokenizer.parallel_encode \
    --input ./data/TinyStoriesV2-GPT4-train.txt \
    --vocab tokenized/vocab_tinystories.json \
    --merges tokenized/merges_tinystories.txt \
    --output tokenized/tinystories_train.npy \
    --special-tokens "<|endoftext|>" \
    --num-workers $SLURM_CPUS_PER_TASK \
    --num-chunks $(($SLURM_CPUS_PER_TASK * 4)) \
    --tmp-dir "$TMP_DIR"

rm -rf "$TMP_DIR"

echo "Done! Created tokenized/tinystories_train.npy"