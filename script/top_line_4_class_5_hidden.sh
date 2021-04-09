#!/usr/bin/env bash

python main.py \
    --input_size 100 \
    --layer1_node 5 \
    --output_size 4 \
    --batch_size 128 \
    --epoch 300 \
    --popu_num 20 \
    --mu_p 0.8 \
    --running_mode train \
    --class_list 0 1 4 7 \
    --resume_dir log/88.70_4_100_5_4 \
