#!/usr/bin/env bash

# 不往pl侧写入，算法识别
/root/berryconda3/envs/onn/bin/python demo/demo.py --debug --soct

# 往pl侧写入
# /root/berryconda3/envs/onn/bin/python demo/demo.py --soct

# 上位机显示+运行，不用zynq
# /root/berryconda3/envs/onn/bin/python demo/demo.py --debug