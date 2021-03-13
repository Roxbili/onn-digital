#-*- encoding: utf-8 -*-

import numpy as np

class Linear(object):
    """Linear layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.bias = np.array([160 for _ in range(out_features)])     # init value is the middle value of 20MHz to 300MHz

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        """Forward network

            Args:
                input: shape=(batch_size, in_features)
        """
        pass


class Net(object):
    def __init__(self, config):
        self.config = config

    def onn(self):
        pass