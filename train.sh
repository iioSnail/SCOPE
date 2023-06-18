#!/usr/bin/env bash
# -*- coding: utf-8 -*-

mkdir -p outputs
python -u finetune/train.py \
--bert_path FPT \
--data_dir data \
--save_path outputs \
--max_epoch=30 \
--lr=5e-5 \
--warmup_proporation 0.1 \
--batch_size=32 \
--gpus=0, \
--accumulate_grad_batches=2  \
--reload_dataloaders_every_n_epochs 1 
sleep 1

# python -u finetune/train.py --bert_path FPT --data_dir data --save_path outputs --max_epoch=30 --lr=5e-5 --warmup_proporation 0.1 --batch_size=32 --gpus=0, --accumulate_grad_batches=2  --reload_dataloaders_every_n_epochs 1

# nohup bash train.sh 2>&1 >train.log &