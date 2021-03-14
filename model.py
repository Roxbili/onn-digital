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
        img_sum = input.sum(axis=1, keepdims=True) # shape = (batch_size,)
        img_sum = img_sum.repeat(self.out_features, axis=1)
        # print(img_sum.shape)
        batch_size_bias = np.tile(self.bias, (input.shape[0], 1))
        # print(batch_size_bias.shape)

        output = img_sum + batch_size_bias
        # print(output.shape)     # output shape = (batch_size, out_features)
        return output


class LossFunc(object):
    """Loss function for this network"""
    def __init__(self, ):
        pass

class Net(object):
    def __init__(self, config):
        self.config = config

        self.layer1 = Linear(self.config['input_size'], self.config['layer1_node'])
        self.layer2 = Linear(self.config['layer1_node'], self.config['num_class'])

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return out2
        


if __name__ == '__main__':
    layer1 = Linear(9, 5)
    layer2 = Linear(5, 10)

    data = np.ones((1000, 9))
    out = layer1(data)
    layer2(out)