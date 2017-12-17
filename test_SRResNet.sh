#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --output_dir ./result/SRResNet/ \
    --summary_dir ./result/SRResNet/log/ \
    --mode test \
    --is_training False \
    --task SRResNet \
    --batch_size 16 \
    --input_dir_LR ./data/test_LR/ \
    --input_dir_HR ./data/test_HR/ \
    --num_resblocks 16 \
    --pre_trained_model True \
    --checkpoint_SRResNet ./experiment_SRResNet/model-1000000

