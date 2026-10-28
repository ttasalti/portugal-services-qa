#!/bin/bash
#SBATCH --job-name=qa-serve
#SBATCH --partition=slurm_queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_OFFLINE=1

VLLM_ENV=${VLLM_ENV:-$HOME/envs/pt-rag-serve}
MODEL=${MODEL:-Qwen/Qwen3.5-4B}
PORT=${PORT:-8010}
GPU_UTIL=${GPU_UTIL:-0.6}

"$VLLM_ENV/bin/vllm" serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enforce-eager \
  --gdn-prefill-backend triton \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

for _ in $(seq 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm exited"; exit 1; }
  sleep 5
done

export LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export LLM_MODEL="$MODEL"
uv run ptqa "$@"
