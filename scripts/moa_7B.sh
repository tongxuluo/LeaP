#!/bin/bash

L=${1:-4}
A=${1:-3}
TASK=${2:-"aime"}

python ./moa.py \
    --model_path "../models/DeepSeek-R1-Distill-Qwen-7B" \
    --data_path "./data/${TASK}.json" \
    --save_path "./outputs/moa_7B_16k_${L}_${A}/${TASK}.json" \
    --temperature 0.6 \
    --top_p 0.95 \
    --min_p 0.05 \
    --max_tokens 16000 \
    --num_layers ${L} \
    --num_agents ${A} \
    --num_gpus 8 \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 1 \
    --batch_size 32
