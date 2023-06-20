#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=../SCOPE
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=outputs/checkpoint/epoch=29-df=80.2158-cf=78.5971.ckpt

OUTPUT_DIR=outputs/predict
mkdir -p $OUTPUT_DIR

python -u finetune/predict.py \
  --bert_path FPT \
  --ckpt_path $ckpt_path \
  --data_dir data \
  --save_path $OUTPUT_DIR \
  --label_file data/test.sighan15.lbl.tsv \
  --gpus=0,


# python finetune/predict.py --bert_path ./FPT --ckpt_path --data_dir outputs --save_path outputs
