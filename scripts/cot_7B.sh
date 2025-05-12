#!/bin/bash

N=${1:-32}
TASK=${2:-"aime"}

python ./cotsc.py \
    --model_path "../models/DeepSeek-R1-Distill-Qwen-7B" \
    --data_path "./data/${TASK}.json" \
    --save_path "./outputs/cot_7B_16k/${TASK}.json" \
    --temperature 0.6 \
    --top_p 0.95 \
    --min_p 0.05 \
    --max_tokens 16000 \
    --n ${N} \
    --num_gpus 8 \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 1 \
    --batch_size 32
