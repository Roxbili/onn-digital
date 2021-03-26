#!/usr/bin/env bash

python main.py \
    --input_size 100 \
    --layer1_node 20 \
    --output_size 2 \
    --batch_size 1000 \
    --epoch 10 \
    --popu_num 20 \
    --mu_p 0.8 \
    --resume_dir log/96.69 \
    --running_mode train