#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/model}"
FILE_NAME="${FILE_NAME:-model-name}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DECORD_EOF_RETRY_MAX=40960
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python ./src/eval_bench.py --model_path "${MODEL_PATH}" --file_name "${FILE_NAME}"
