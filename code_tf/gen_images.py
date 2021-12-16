#-*- encoding: utf-8 -*-

import os
import numpy as np
import sys
sys.path.append('../onn-digital')

from dataset import MNIST

############### data pre-processing ###############

test_set = MNIST('mnist', 't10k', (28, 28))
test_set.save_img('images/28x28', save_num=100, worker_num=4, mode='img') # save_num parameters change

test_set = MNIST('mnist', 't10k', (16, 16))
test_set.save_img('images/16x16', save_num=100, worker_num=4, mode='img') # save_num parameters change