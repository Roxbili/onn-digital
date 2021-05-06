#-*- encoding: utf-8 -*-

import numpy as np
import os, sys

# sys.path.append('../onn-digital')

def save_bin(data, binary_path):
    print(binary_path)
    print('file shape: {}, dtype: {}'.format(data.shape, data.dtype))

    data.tofile(binary_path)
    # read bin file use fromfile
    print('Done\n')

def load_weight(w_path):
    w = np.load(w_path).astype(np.int8)
    print(w.shape, w.dtype)
    return w


model_path = 'log_tf/10_64_round_clamp_floor_e_noAdd3_genInputs_quant'
npy_path = os.path.join(model_path, 'npy')

w1_path = os.path.join(npy_path, 'w1.npy')
weight1 = load_weight(w1_path)

w2_path = os.path.join(npy_path, 'w2.npy')
weight2 = load_weight(w2_path)

e1_path = os.path.join(npy_path, 'e1.npy')
e1 = load_weight(e1_path)
print(e1)

print('start generating bin file...')

bin_path = os.path.join(model_path, 'bin')
if os.path.exists(bin_path) == False:
    os.mkdir(bin_path)

w1_bin_path = os.path.join(bin_path, 'w1.bin')
save_bin(weight1, w1_bin_path)

w2_bin_path = os.path.join(bin_path, 'w2.bin')
save_bin(weight2, w2_bin_path)
