#!/bin/bash
#SBATCH --job-name=qa-index
#SBATCH --partition=cpu_only
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=""

uv run ptqa index
uv run ptqa search -k 3 \
  "Que documentos preciso para a autorização de residência para investigadores?" \
  "Serbest çalışan olarak Segurança Social'a ne kadar katkı ödemem gerekir?" \
  "How can a foreigner get a NIF in Portugal?"
