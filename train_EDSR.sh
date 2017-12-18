#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python3 main.py \
--output_dir ./experiment_EDSR/ \
--summary_dir ./experiment_EDSR/log/ \
--mode train \
--is_training True \
--task EDSR \
--batch_size 16 \
--flip True \
--random_crop True \
--crop_size 48 \
--input_dir_LR ./data/train_LR/ \
--input_dir_HR ./data/train_HR/ \
--num_resblock 32 \
--name_queue_capacity 2048 \
--image_queue_capacity 2048 \
--perceptual_mode L1 \
--queue_thread 32 \
--ratio 0.001 \
--learning_rate 0.0001 \
--decay_step 200000 \
--decay_rate 0.5 \
--stair True \
--beta 0.9 \
--max_iter 1000000 \
--save_freq 10000