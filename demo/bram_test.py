import os, sys
import numpy as np
import time
sys.path.append('../onn-digital')

from code_tf.onn_numpy_test import Net
from utils.bram import BRAM

model_path = 'log_tf/10_256_round_clamp_floor_e_noAdd3_genInputs_16x16_quant'
npy_path = os.path.join(model_path, 'npy')
data_path = os.path.join(model_path, 'bin', '0_7.bin')

data = np.fromfile(data_path, dtype=np.uint8)

bram = BRAM()

while True:
    bram.write(data, 'data')
    time.sleep(1e-6)