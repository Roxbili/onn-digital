#-*- encoding: utf-8 -*-

import os
import numpy as np

from dataset import MNIST, Feature
from model import Net

config = {
    'input_size': 9,
    'layer1_node': 5,
    'batch_size': 1000,
    'num_class': 10,
}

if __name__ == '__main__':
    
    ############### data pre-processing ###############

    train_set = MNIST('mnist', 'train', (10, 10))
    test_set = MNIST('mnist', 't10k', (10, 10))

    train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
    train_v, _ = train_feature.calc_feature_vector()
    # train_compress_v = train_feature.compress()
    train_fv, train_label = train_feature.encoding(train_v, is_round=True)
    # train_feature.hist(save_path='log/train_hist.png', data=train_fv)
    input_train_data = train_feature.cut_into_batch(batch_size=1000, vector=train_fv, labels=train_label)

    test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
    test_v, _ = test_feature.calc_feature_vector()
    # test_compress_v = test_feature.compress()
    test_fv, test_label = test_feature.encoding(test_v, is_round=True)
    # test_feature.hist(save_path='log/test_hist.png', data=test_fv)
    input_test_data = test_feature.cut_into_batch(batch_size=1000, vector=test_fv, labels=test_label)


    ############### model define ###############

    net = Net(config=config)