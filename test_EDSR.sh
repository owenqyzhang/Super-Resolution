#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/EDSR/ \
    --summary_dir ./result/EDSR/log/ \
    --mode test \
    --is_training False \
    --task EDSR \
    --batch_size 16
    --input_dir_LR ./data/test_LR/ \
    --input_dir_HR ./data/test_HR/ \
    --num_resblocks 16 \
    --pre_trained_model True \
    --checkpoint_EDSR ./experiment_EDSR/model-1000000

