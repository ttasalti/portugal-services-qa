#!/bin/bash
#SBATCH --job-name=qa-download
#SBATCH --partition=cpu_only
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"

for model in "$@"; do
  uv run python -c "import sys; from huggingface_hub import snapshot_download; \
print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt']))" \
    "$model"
done
