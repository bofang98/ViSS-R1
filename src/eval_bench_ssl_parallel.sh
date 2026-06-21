#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/model}"
FILE_NAME="${FILE_NAME:-model-name}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
PORT_BASE="${PORT_BASE:-15100}"
PORT_STRIDE="${PORT_STRIDE:-10}"
export DECORD_EOF_RETRY_MAX=40960
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python ./src/eval_bench_ssl_parallel.py \
    --model_path "${MODEL_PATH}" \
    --file_name "${FILE_NAME}" \
    --port_base "${PORT_BASE}" \
    --port_stride "${PORT_STRIDE}"
