#!/usr/bin/env bash
set -euo pipefail

cd src/r1-v

# unset LD_LIBRARY_PATH   
# export CUDAHOSTCXX=/usr/bin/g++-11
# export CC=/usr/bin/gcc-11
# export CXX=/usr/bin/g++-11

 
export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${LOG_PATH:-./debug_log.txt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
DATASET_PATH="${DATASET_PATH:-../../preprocess_SFT/SSL-COT-153k.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./log/SFT/VideoRFT-7B-ssl-cot-lr1e6-32frame}"
RUN_NAME="${RUN_NAME:-Qwen2.5-VL-7B-Video-cot-sft}"
MASTER_PORT="${MASTER_PORT:-12349}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/sft_video.py \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_PATH}" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name "${RUN_NAME}" \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true
