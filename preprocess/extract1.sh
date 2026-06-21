#!/usr/bin/env bash
set -euo pipefail

VLM_ROOT="${VLM_ROOT:-/path/to/checkpoints}"
VLM_MODEL="${VLM_MODEL:-Qwen2.5-VL-7B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python mask.py \
--n_gpu 2 \
--num_processes 2 \
--num_masks 16 \
--ones_per_mask 16 \
--vl_model "${VLM_ROOT}/${VLM_MODEL}" \
--output_base_path output_mask_loss/
