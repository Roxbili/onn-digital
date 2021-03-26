#!/usr/bin/env bash

python main.py \
    --input_size 100 \
    --layer1_node 20 \
    --output_size 4 \
    --batch_size 1000 \
    --epoch 100 \
    --popu_num 20 \
    --mu_p 0.8 \
    --resume_dir log/44.69_4 \
    --running_mode train \
    --class_list 0 1 4 7 \
