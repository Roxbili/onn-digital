#-*- encoding: utf-8 -*-

import os
import numpy as np
import sys

from utils.utils import rescale, softmax
from dataset import MNIST, Feature
from model import Net_1, Net_2, Optim, AccFunc, CrossEntropyLoss

import tensorflow as tf

############### network parameters ###############
input_size = 100
layer1_node = 512
# layer2_node = 128
output_size = 10

batch_size = 1000
epoch = 1000

learning_rate = 0.01
lr_end = 1e-4
lr_decay = (lr_end / learning_rate)**(1. / epoch)

train = True
load_model = True
load_model_path = 'log_torch/10_512_lr_decay/float.pt'

dir_path = 'log_torch/10_512_lr_decay'
float_path = os.path.join(dir_path, 'float.pt')
quant_path = os.path.join(dir_path, 'quant.pt')
def create_dir(dir_path):
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)

############### data pre-processing ###############

train_set = MNIST('mnist', 'train', (10, 10))
test_set = MNIST('mnist', 't10k', (10, 10))

train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
train_fv, train_label = train_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
train_fv = train_fv.reshape(-1, 100)
train_fv = rescale(train_fv, 50, 200, True)

test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test_fv = test_fv.reshape(-1, 100)
test_fv = rescale(test_fv, 50, 200, True)

############### Net ###############

sess = Session()

images = tf.placeholder(tf.float32, (None, 100))
labels = tf.placeholder(tf.int32, (None,))

def Linear(images, in_size, out_size):
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],mean=0, stddev=1))
    output = tf.matmul(inputs, )
