#!/bin/bash

# 2024/08/04 created by moriy
# 目的 sn-caption spottingモデルでcommentary spottingをやってみる
# 仮説 sn-caption バリデーションをみて止めるoptimizer(plateau)は動いてないし、シンプルなスケジューラの方が良いのでは
# stride=window_size で、window_size分だけ発話後の映像も入力する(sn-captionに合わせる)
# training 時、モデルの予測対象は window_size=15 秒間隔とする(sn-captionに合わせる)
# with dropout_rate=0.4
export MODEL_DIR="/raid/moriy/model/sn-caption/Benchmarks/TemporallyAwarePooling"


GPU_ID=1

EXP_NO=1

SOCCERNET_PATH=/raid_elmo/home/lr/moriy/SoccerNet

EPOCHS=50
BATCH_SIZE=256
LR=1e-3

FRAME_RATE=1
POOL=NetVLAD
WINDOW_SIZE_SPOTTING=15
NMS_WINDOW=30

MODEL_NAME=baidu-$POOL-commentary-spotting-$EXP_NO

export CUDA_VISIBLE_DEVICES=$GPU_ID


python src/spotting_commentary.py \
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
    --test_only \
    --NMS_threshold 0 \
    --scheduler CosineAnnealingLR \
    --T_max 10
