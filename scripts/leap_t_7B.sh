#!/bin/bash

MAX_TURNS=${1:-5}
MAX_TOKENS=${2:-2048}
N=${3:-4}
PEER_TOP_K=${4:-2}
ROUTER=${5:-"dissimilar"}
TASKS=${6:-"aime"}
BS=${7:-16}

python leap.py \
    --model_path "../models/LeaP-T-7B" \
    --data_dir "./data" \
    --save_dir "./outputs/leap_t_7B_${MAX_TURNS}_${MAX_TOKENS}_${N}_${ROUTER}_top${PEER_TOP_K}" \
    --tasks ${TASKS} \
    --max_turns ${MAX_TURNS} \
    --peer_top_k ${PEER_TOP_K} \
    --router ${ROUTER} \
    --temperature 0.6 \
    --top_p 0.95 \
    --min_p 0.05 \
    --max_tokens ${MAX_TOKENS} \
    --summarize_max_tokens 256 \
    --n ${N} \
    --num_gpus 8 \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 1 \
    --is_leap_t_model True \
    --batch_size ${BS}
