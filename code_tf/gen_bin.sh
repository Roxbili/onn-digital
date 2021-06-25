#!/usr/bin/env bash

# running dir: onn-digital
# running command: bash code_tf/gen_bin.sh

# python code_tf/gen_bin.py \
/home/fanding/miniconda3/envs/onn/bin/python code_tf/gen_bin.py \
    --model_path log_tf/10_256_round_clamp_floor_e_noAdd3_genInputs_16x16_quant \
    --bin_name 1_2.bin \
    # --save_bin_num 4 \
    # --save_params \