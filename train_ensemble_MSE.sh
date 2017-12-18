#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 \
python3 main.py \
--output_dir ./experiment_ensemble_MSE/ \
--summary_dir ./experiment_ensemble_MSE/log/ \
--mode train \
--is_training True \
--task ensemble \
--batch_size 16 \
--flip True \
--random_crop True \
--crop_size 24 \
--input_dir_LR ./data/train_LR/ \
--input_dir_HR ./data/train_HR/ \
--num_resblock 16 \
--name_queue_capacity 4096 \
--image_queue_capacity 4096 \
--perceptual_mode MSE \
--queue_thread 16 \
--ratio 0.001 \
--learning_rate 0.0001 \
--decay_step 500000 \
--decay_rate 0.1 \
--stair True \
--beta 0.9 \
--max_iter 20000 \
--save_freq 20000 \
--checkpoint_EDSR ./experiment_EDSR/model-1000000 \
--checkpoint_SRResNet ./experiment_SRResNet/model-1000000