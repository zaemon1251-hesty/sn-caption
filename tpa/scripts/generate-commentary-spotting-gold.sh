#!/bin/bash

# 2024/09/04 created by moriy
export MODEL_DIR="/raid/moriy/model/sn-caption/Benchmarks/TemporallyAwarePooling"

GPU_ID=1

export CUDA_VISIBLE_DEVICES=$GPU_ID

python src/utils.py --type commentary_gold