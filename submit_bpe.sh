#!/bin/bash
#SBATCH -p mit_normal
#SBATCH -A mit_general
#SBATCH -J bpe_train
#SBATCH -o bpe_%j.out
#SBATCH -e bpe_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00

module load miniforge
source ~/cs336-assignment1/myenvironment/bin/activate

# IMPORTANT: Move to the project ROOT, not the subdirectory
cd ~/cs336-assignment1/

# Set parallelism
export BPE_NUM_PROCESSES=$SLURM_CPUS_PER_TASK

# Run module from the root
python -m cs336_basics.bpe ./data/owt-valid.txt 10000 --vocab-output vocab_owt_valid.json --merges-output merges_owt_valid.txt 