#!/bin/bash
#SBATCH --job-name=qa-retrieval
#SBATCH --partition=cpu_only
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1

uv run ptqa retrieval
