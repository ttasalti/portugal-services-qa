#!/bin/bash
#SBATCH --job-name=qa-fetch
#SBATCH --partition=cpu_only
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
uv run ptqa fetch
