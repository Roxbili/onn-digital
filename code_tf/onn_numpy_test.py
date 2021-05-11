#-*- encoding: utf-8 -*-

import os
import numpy as np
import sys
sys.path.append('../onn-digital')

from utils.utils import rescale, softmax, generate_frequency, maxPooling
from dataset import MNIST, Feature


############### Net ###############

class Net(object):
    def __init__(self, weight1, weight2, e1):
        self.weight1 = weight1
        self.weight2 = weight2
        self.e1 = e1

    def Linear(self, inputs, weight, activation_func=None):
        counters = np.floor(inputs / 10) - 1
        outputs = np.dot(counters, weight)

        if activation_func != None:
            outputs = activation_func(outputs)

        return outputs

    def relu(self, inputs):
        return np.maximum(0, inputs)

    def mapping(self, inputs, e):
        N = 2**e

        countersWdiv4n = np.round(inputs / N)
        clamp_countersWdiv4n = np.clip(countersWdiv4n, 0., 15.) 
        outputs = (clamp_countersWdiv4n + 5) * 10
        return outputs
    
    def __call__(self, inputs):
        outputs = self.Linear(inputs, self.weight1, activation_func=self.relu)
        outputs = self.mapping(outputs, self.e1)
        outputs = self.Linear(outputs, self.weight2)
        return outputs


if __name__ == "__main__":

    ############### network parameters ###############
    input_size = 100
    layer1_node = 64
    # layer2_node = 128
    output_size = 10

    batch_size = 1000

    npy_path = 'log_tf/10_64_round_clamp_floor_e_noAdd3_genInputs_quant/npy'

    ############### data pre-processing ###############

    train_set = MNIST('mnist', 'train', (10, 10))
    test_set = MNIST('mnist', 't10k', (10, 10))

    train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
    train_fv, train_label = train_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # train_fv = maxPooling(train_fv, size=2, stride=2)
    train_fv = generate_frequency(train_fv)
    train_fv = train_fv.reshape(-1, input_size)
    input_train_data = train_feature.cut_into_batch(batch_size=batch_size, vector=train_fv, labels=train_label, num_class=output_size, one_hot=True)


    test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
    test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # test_fv = maxPooling(test_fv, size=2, stride=2)
    test_fv = generate_frequency(test_fv)
    test_fv = test_fv.reshape(-1, input_size)
    input_test_data = test_feature.cut_into_batch(batch_size=batch_size, vector=test_fv, labels=test_label, num_class=output_size, one_hot=True)

    ############################# run #############################
    weight1 = np.load(os.path.join(npy_path, 'w1.npy'))
    weight2 = np.load(os.path.join(npy_path, 'w2.npy'))
    e1 = np.load(os.path.join(npy_path, 'e1.npy'))

    net = Net(weight1, weight2, e1)

    total_accuracy = 0.
    for i, (images, labels) in enumerate(input_test_data): 
        prediction = net(images)
        correct_prediction = np.equal(np.argmax(prediction, 1), np.argmax(labels, 1))
        accuracy = np.mean(correct_prediction)

        total_accuracy += accuracy * batch_size / len(test_fv)

    print('Accuracy of the network on the 10000 test images: %.4f' % total_accuracy)
