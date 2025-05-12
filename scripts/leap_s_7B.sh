#!/bin/bash

FIRST_TOKENS=${1:-2048}
MAX_TOKENS=${2:-4096}
NUM_MIX=${3:-4}
N=${4:-8}
PEER_TOP_K=${5:-2}
COM=${6:-1}
ROUTER=${7:-"dispersed"}
TASKS=${8:-"aime"}
BS=${9:-16}

python leap_s.py \
    --model_path "../models/DeepSeek-R1-Distill-Qwen-7B" \
    --data_dir "./data" \
    --save_dir "./outputs/leap_s_7B_${FIRST_TOKENS}_${MAX_TOKENS}_${COM}_${NUM_MIX}_${N}_${ROUTER}_top${PEER_TOP_K}" \
    --tasks ${TASKS} \
    --num_mix ${NUM_MIX} \
    --peer_top_k ${PEER_TOP_K} \
    --router ${ROUTER} \
    --temperature 0.6 \
    --top_p 0.95 \
    --min_p 0.05 \
    --first_tokens ${FIRST_TOKENS} \
    --max_tokens ${MAX_TOKENS} \
    --summarize_max_tokens 256 \
    --n ${N} \
    --num_gpus 8 \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 1 \
    --batch_size ${BS}
