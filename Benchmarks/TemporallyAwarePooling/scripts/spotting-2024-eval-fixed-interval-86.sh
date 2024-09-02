#!/bin/bash

# 2024/08/19 created by moriy
# 目的 ナイーブな手法でのspottingの評価を行う
# 仮説 コメントの平均間隔をそのまま使う手法は、spottingのベースラインよりも性能が低いはず
export MODEL_DIR="/raid/moriy/model/sn-caption/Benchmarks/TemporallyAwarePooling"

GPU_ID=1

export CUDA_VISIBLE_DEVICES=$GPU_ID

python src/utils.py --fixed_interval 86
