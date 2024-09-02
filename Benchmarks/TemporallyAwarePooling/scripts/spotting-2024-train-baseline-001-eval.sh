#!/bin/bash

# 2024/07/16 created by moriy
# hypo: sn-caption論文の再現実験 from scratch
# with dropout_rate=0.4
export MODEL_DIR="/raid/moriy/model/sn-caption/Benchmarks/TemporallyAwarePooling"


GPU_ID=0

EXP_NO=1

SOCCERNET_PATH=/raid_elmo/home/lr/moriy/SoccerNet

EPOCHS=16
BATCH_SIZE=1
# LR=1e-02
LR=1e-3

FRAME_RATE=1
POOL=NetVLAD
WINDOW_SIZE_SPOTTING=15
NMS_WINDOW=30

MODEL_NAME=baidu-$POOL-spotting-$EXP_NO

export CUDA_VISIBLE_DEVICES=$GPU_ID


python src/spotting.py \
    --SoccerNet_path $SOCCERNET_PATH \
    --model_name $MODEL_NAME \
    --features baidu_soccer_embeddings.npy \
    --framerate $FRAME_RATE \
    --pool $POOL \
    --window_size $WINDOW_SIZE_SPOTTING \
    --NMS_window $NMS_WINDOW \
    --GPU $GPU_ID \
    --batch_size $BATCH_SIZE \
    --LR $LR \
    --split_test test \
    --evaluation_frequency 5 \
    --max_num_worker 4 \
    --max_epochs $EPOCHS \
    --NMS_threshold 0 \
    --test_only
    # --scheduler CosineAnnealingLR \
    # --T_max 10 \
