#-*- encoding: utf-8 -*-

from ast import parse
from functools import total_ordering
import numpy as np
import os, sys
import argparse

from numpy.__config__ import show
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
    input_size = 256
    output_size = 10
    batch_size = 1  # 只能是1，不能改变

    weight1 = np.load(os.path.join(npy_path, 'w1.npy'))
    weight2 = np.load(os.path.join(npy_path, 'w2.npy'))
    e1 = np.load(os.path.join(npy_path, 'e1.npy'))

    test_set = MNIST('mnist', 't10k', (16, 16))

    test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
    test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # test_fv = maxPooling(test_fv, size=2, stride=2)
    test_fv_gen = generate_frequency(test_fv)
    test_fv_gen = test_fv_gen.reshape(-1, input_size)
    input_test_data = test_feature.cut_into_batch(batch_size=batch_size, vector=test_fv_gen, labels=test_label, num_class=output_size, one_hot=True)

    cnt = 0
    net = Net(weight1, weight2, e1)
    for i, (images, labels) in enumerate(input_test_data): 

        # # 由于数据集是有序的，这里专门针对提供的图片2进行处理，这段代码没有泛用性
        # if num == 1 and np.argmax(labels, 1)[0] == 2:
        #     net.print_intermediate = True
        # else:
        #     net.print_intermediate = False
            
        prediction = net(images)

        correct_prediction = np.equal(np.argmax(prediction, 1), np.argmax(labels, 1))
        if correct_prediction[0] == True:
            data = test_fv[i].astype(np.uint8)
            save_bin(data.flatten(), os.path.join(bin_path, str(cnt) + '_' + str(np.argmax(labels, 1)[0]) + '.bin'))
            # print(np.argmax(labels, 1)[0])
            # print(data)

            cnt += 1
            if cnt >= num:
                break

def show_intermediate(npy_dir, bin_dir, bin_name):
    bin_file = os.path.join(bin_dir, bin_name)
    
    weight1 = np.load(os.path.join(npy_dir, 'w1.npy'))
    weight2 = np.load(os.path.join(npy_dir, 'w2.npy'))
    e1 = np.load(os.path.join(npy_dir, 'e1.npy'))

    net = Net(weight1, weight2, e1, print_intermediate=True)

    images = np.fromfile(bin_file, dtype=np.uint8)
    images = images.astype(np.int)
    images = generate_frequency(images)

    prediction = net(images)
    print('file %s' % bin_file)
    print('prediction result: {}'.format(np.argmax(prediction)))
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        help='Directory of the model')
    parser.add_argument(
        '--save_bin_num',
        type=int,
        default=0,
        help='The number of correct sample to be saved'
    )
    parser.add_argument(
        '--bin_name',
        default=None,
        help='The name of bin file which is ready to be inferenced. Bin file is stored in [model_path]/bin/'
    )
    parser.add_argument(
        '--save_params',
        action="store_true",
        help='Whether to save parameters'
    )

    args = parser.parse_args()

    # model_path = 'log_tf/10_64_round_clamp_floor_e_noAdd3_genInputs_quant'
    npy_path = os.path.join(args.model_path, 'npy')
    if os.path.exists(npy_path) == False:
        os.mkdir(npy_path)
    bin_path = os.path.join(args.model_path, 'bin')
    if os.path.exists(bin_path) == False:
        os.mkdir(bin_path)

    if args.save_params:
        save_paramaeters(npy_path, bin_path)
    if args.save_bin_num != 0:
        # print(args.save_bin_num)
        save_inputs(bin_path, args.save_bin_num)
    if args.bin_name != None:
        # print(args.bin_name)
        show_intermediate(npy_path, bin_path, args.bin_name)