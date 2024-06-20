#!/bin/bash
gpu=$1
ARGS=${@:2}

python train.py \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard \
  --gpu $gpu \
  --epochs 30 \
  --num-copies 1 \
  --num-workers 1 \
  $ARGS



  #--num-copies 1 \
  #--num-workers 1 \