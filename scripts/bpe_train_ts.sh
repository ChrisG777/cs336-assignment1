#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH -J bpe_ts
#SBATCH -o logs/bpe_ts_%j.out
#SBATCH -e logs/bpe_ts_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16      # TinyStories is smaller, 16 cores is plenty
#SBATCH --mem=32G               # 32GB RAM is sufficient [cite: 253]
#SBATCH --time=02:00:00

module load miniforge
source ~/cs336-assignment1/myenvironment/bin/activate

# Move to project ROOT
cd ~/cs336-assignment1/

# Ensure output directory exists
mkdir -p tokenized

export BPE_NUM_PROCESSES=$SLURM_CPUS_PER_TASK

# Vocab size 10000 for TinyStories 
python -m cs336_basics.BPE_tokenizer.bpe \
    ./data/TinyStoriesV2-GPT4-train.txt \
    10000 \
    --vocab-output tokenized/vocab_tinystories.json \
    --merges-output tokenized/merges_tinystories.txt