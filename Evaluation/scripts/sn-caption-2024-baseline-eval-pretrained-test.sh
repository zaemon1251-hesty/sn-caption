#!/bin/bash

# 2024/03/02 created by moriy

GPU_ID=0

SOCCERNET_PATH=/raid_elmo/home/lr/moriy/SoccerNet

MODEL_NAME=baidu-NetVlad-pretrain-caption-2

PREDICTIONS_PATH=/raid_elmo/home/lr/moriy/sn-caption
PREDICTIONS_PATH=$PREDICTIONS_PATH/Benchmarks/TemporallyAwarePooling/models
PREDICTIONS_PATH=$PREDICTIONS_PATH/$MODEL_NAME
PREDICTIONS_PATH=$PREDICTIONS_PATH/outputs/test


python EvaluateDenseVideoCaption.py \
    --SoccerNet_path $SOCCERNET_PATH \
    --Predictions_path $PREDICTIONS_PATH \
    --split test


# {'Bleu_1': 0.32476715564485165, 'Bleu_2': 0.2694118271733901, 'Bleu_3': 0.23656099476398026, 'Bleu_4': 0.21169028240152726, 'METEOR': 0.21993795872551086, 'ROUGE_L': 0.2700857289201959, 'CIDEr': 0.27629622129516734, 'Recall': 0.9317838049654555, 'Precision': 0.5601840945919081}

