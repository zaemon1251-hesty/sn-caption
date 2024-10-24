#!/bin/bash

# 2024/02/11 21:50 created by moriy

# WANDB_MODE=disabled

GPU_ID=1

EXP_NO=2

SOCCERNET_PATH=/raid_elmo/home/lr/moriy/SoccerNet

BATCH_SIZE=1
LR=0.0002

FRAME_RATE=1
POOL=NetVLAD
WINDOW_SIZE_CAPTION=45
WINDOW_SIZE_SPOTTING=7
NMS_WINDOW=30
NUM_LAYERS=4
FIRST_STAGE=spotting

MODEL_NAME=baidu-$POOL-caption-$EXP_NO

export CUDA_VISIBLE_DEVICES=$GPU_ID


python src/main.py \
    --SoccerNet_path $SOCCERNET_PATH \
    --model_name $MODEL_NAME \
    --features baidu_soccer_embeddings.npy \
    --framerate $FRAME_RATE \
    --pool $POOL \
    --window_size_caption $WINDOW_SIZE_CAPTION \
    --window_size_spotting $WINDOW_SIZE_SPOTTING \
    --NMS_window $NMS_WINDOW \
    --num_layers $NUM_LAYERS \
    --first_stage $FIRST_STAGE \
    --GPU $GPU_ID \
    --batch_size $BATCH_SIZE \
    --LR $LR \
    --split_test test \
    --fp16 \
    --evaluation_frequency 1000000 \
    --max_num_worker 1
