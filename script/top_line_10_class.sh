#!/usr/bin/env bash

python main.py \
    --input_size 100 \
    --layer1_node 5 \
    --output_size 10 \
    --batch_size 1000 \
    --epoch 300 \
    --popu_num 20 \
    --mu_p 0.8 \
    --running_mode train \
    --class_list 0 1 2 3 4 5 6 7 8 9 \
    --resume_dir log/43.60_10_100_5_10 \
