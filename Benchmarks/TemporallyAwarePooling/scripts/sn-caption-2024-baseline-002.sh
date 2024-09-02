#!/bin/bash

# 2024/02/07 21:50 created by moriy

GPU_ID=0

SOCCERNET_PATH=/raid_elmo/home/lr/moriy/SoccerNet

MODEL_NAME=sn-caption-2024-baseline-002

BATCH_SIZE=8
LR=0.0002

FRAME_RATE=1
POOL=NetVLAD
WINDOW_SIZE_CAPTION=15
WINDOW_SIZE_SPOTTING=15
NMS_WINDOW=30
NUM_LAYERS=4
FIRST_STAGE=spotting


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
    --pretrain \
    --GPU $GPU_ID \
    --batch_size $BATCH_SIZE \
    --LR $LR 
