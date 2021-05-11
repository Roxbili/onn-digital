#-*- encoding: utf-8 -*-

import numpy as np
import os, sys
sys.path.append('../onn-digital')

from utils.utils import rescale, softmax, generate_frequency, maxPooling
from dataset import MNIST, Feature
from code_tf.onn_numpy_test import Net


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


def save_paramaeters(npy_path, bin_path):
    
    w1_path = os.path.join(npy_path, 'w1.npy')
    weight1 = load_weight(w1_path)

    w2_path = os.path.join(npy_path, 'w2.npy')
    weight2 = load_weight(w2_path)

    e1_path = os.path.join(npy_path, 'e1.npy')
    e1 = load_weight(e1_path)
    print(e1)

    print('start generating bin file...')
    w1_bin_path = os.path.join(bin_path, 'w1.bin')
    save_bin(weight1.T, w1_bin_path)    # 100个100个给，给64次

    w2_bin_path = os.path.join(bin_path, 'w2.bin')
    save_bin(weight2.T, w2_bin_path)

def save_inputs(bin_path, num):
    input_size = 100
    output_size = 10
    batch_size = 1  # 只能是1，不能改变

    weight1 = np.load(os.path.join(npy_path, 'w1.npy'))
    weight2 = np.load(os.path.join(npy_path, 'w2.npy'))
    e1 = np.load(os.path.join(npy_path, 'e1.npy'))

    test_set = MNIST('mnist', 't10k', (10, 10))

    test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
    test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # test_fv = maxPooling(test_fv, size=2, stride=2)
    test_fv_gen = generate_frequency(test_fv)
    test_fv_gen = test_fv_gen.reshape(-1, input_size)
    input_test_data = test_feature.cut_into_batch(batch_size=batch_size, vector=test_fv_gen, labels=test_label, num_class=output_size, one_hot=True)

    net = Net(weight1, weight2, e1)
    for i, (images, labels) in enumerate(input_test_data): 
        prediction = net(images)
        correct_prediction = np.equal(np.argmax(prediction, 1), np.argmax(labels, 1))
        if correct_prediction[0] == True:
            data = test_fv[i].astype(np.uint8)
            save_bin(data.flatten(), os.path.join(bin_path, str(np.argmax(labels, 1)[0]) + '.bin'))
            print(data)

            num -= 1
            if num == 0:
                break

if __name__ == "__main__":
    model_path = 'log_tf/10_64_round_clamp_floor_e_noAdd3_genInputs_quant'
    npy_path = os.path.join(model_path, 'npy')
    if os.path.exists(npy_path) == False:
        os.mkdir(npy_path)
    bin_path = os.path.join(model_path, 'bin')
    if os.path.exists(bin_path) == False:
        os.mkdir(bin_path)

    save_paramaeters(npy_path, bin_path)
    # save_inputs(bin_path, 2)