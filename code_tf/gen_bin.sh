#!/usr/bin/env bash

# running dir: onn-digital
# running command: bash code_tf/gen_bin.sh

/home/chengzhen/miniconda3/envs/py37/bin/python code_tf/gen_bin.py \
    --model_path log_tf/10_64_round_clamp_floor_e_noAdd3_genInputs_quant \
    --bin_name 1_2.bin \
    --save_bin_num 4 \
    # --save_params \