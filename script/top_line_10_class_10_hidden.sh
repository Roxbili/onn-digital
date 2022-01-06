#!/usr/bin/env bash

python main.py \
    --input_size 100 \
    --layer1_node 10 \
    --output_size 10 \
    --batch_size 1000 \
    --epoch 500 \
    --popu_num 20 \
    --mu_p 0.8 \
    --running_mode train \
    --class_list 0 1 2 3 4 5 6 7 8 9 \
    --resume_dir log/45.70_10_100_10_10 \
