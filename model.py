#-*- encoding: utf-8 -*-

import numpy as np
import math
import os, sys
import random

from utils.utils import limit_scale, rescale


f = range(50, 210, 10)
c = range(4, 20, 1)
f2c = dict(zip(f, c))   # frequency to counter number

class Relu(object):
    """Relu layer
    
        Relu layer will not change input shape

        Args:
            in_features: the number of input neuron
            lower_bound: the lower bound of threshold
            upper_bound: the upper bound of threshold
    """
    def __init__(self):
        pass

    def __call__(self, inputs):
        """Forward network

            Return the result of one output neuron

            Args:
                inputs: shape=(batch_size, layer_shape)

            Return:
                output: shape=(batch_size,), type=np.array
        """
        return self.forward(inputs)

    def forward(self, inputs):
        output = np.copy(inputs)
        output[output < 0] = 0
        return output


class Mapping(object):
    def __init__(self, input_node):
        self.input_node = input_node

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        output = inputs // (4 * self.input_node)
        output = (output + 5) * 10  # return (50, 200)
        return output


class Linear(object):
    """Linear layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = 3

    def __call__(self, inputs, w):
        return self.forward(inputs, w)

    def forward(self, inputs, w):
        """Forward network

            Args:
                inputs: shape=(batch_size, in_features)
            
            Return:
                output: shape=(batch_size, out_features), type=np.array
        """
        # change frequency to counter number
        shape = inputs.shape
        output = inputs.reshape(-1)
        output = np.array([f2c[x] for x in output])
        output = output.reshape(shape)

        output = np.matmul(output, w)
        output += self.bias
        return output


class AccFunc(object):
    """Loss function for this network"""
    def __init__(self):
        pass

    def __call__(self, outputs, labels):
        """Calculate accuracy

            Args:
                outputs: shape=(batch_size, 1)
                labels: shape=(batch_size, num_class)
        """
        # self.outputs = outputs
        # self.labels = labels
        
        return self.calc(outputs, labels)

    def calc(self, outputs, labels):

        ########### for one output ###########
        # outputs = outputs.T[0]     # (batch_size,)
        # prediction = np.array(list(map(self._o2prediction, outputs)))
        # print('prediction: ', prediction[:10])
        # print('labels:     ', labels[:10])
        # print()
        # equal_array = np.equal(prediction, labels)

        ########### for num_class output ###########
        equal_array = np.equal(outputs.argmax(axis=1), labels.argmax(axis=1))

        acc = np.mean(equal_array)
        return acc

    def _o2prediction(self, x, num_class=2):
        if num_class == 10:
            if x <= 50:
                return 0
            elif x <= 70:
                return 1
            elif x <= 90:
                return 2
            elif x <= 110:
                return 3
            elif x <= 130:
                return 4
            elif x <= 150:
                return 5
            elif x <= 170:
                return 6
            elif x <= 190:
                return 7
            elif x <= 220:   # 30
                return 8
            else:
                return 9  
        elif num_class == 2:
            if x <= 140:
                return 0
            else:
                return 2


class CrossEntropyLoss(object):
    """CrossEntropyLoss function
    
        Return:
            loss: average batch images loss
    """
    def __init__(self):
        self.nx = None
        self.ny = None
        self.dnx = None

    def __call__(self, nx, ny):
        return self.loss(nx, ny)

    def loss(self, nx, ny):
        self.nx = nx
        self.ny = ny
        loss = np.sum(- ny * np.log(nx), axis=1)
        return loss.mean()

    def backward(self):
        self.dnx = -1 * self.ny / self.nx
        return self.dnx


class Optim(object):
    """Optimization operator, to updata parameters in model"""
    def __init__(self, net):
        self.net = net
    
    def step(self):
        pass


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 


###########################################################################
################################# Network #################################
###########################################################################

class Net_1(object):
    def __init__(self, input_size, layer1_node, output_size):
        self.w1 = self._init_weight((input_size, layer1_node))
        self.w2 = self._init_weight((layer1_node, output_size))

        self.layer1 = Linear(input_size, layer1_node)
        self.map1 = Mapping(input_size)
        self.layer2 = Linear(layer1_node, output_size)
        self.relu = Relu()

    def __call__(self, x):
        return self.forward(x)

    def _init_weight(self, shape):
        return np.random.randint(-3, 4, shape)

    def forward(self, x):
        out = self.layer1(x, self.w1)
        out = self.relu(out)
        out = self.map1(out)
        out = self.layer2(out, self.w2)
        # out = softmax(out)
        return out

    def get_parameters(self):
        """Return network parameters

            Return:
                np.array, which shape is (layer_num, out_features, in_features)
        """
        return [self.w1, self.w2]

    def set_parameters(self, threshold):
        self.layer1.set_parameters(threshold[0])
        self.layer2.set_parameters(threshold[1])


class Net_2(object):
    def __init__(self, input_size, layer1_node, layer2_node, output_size):

        self.layer1 = Linear(input_size, layer1_node)
        self.layer2 = Linear(layer1_node, layer2_node)
        self.layer3 = Linear(layer2_node, output_size)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

    def get_parameters(self):
        return [self.layer1.parameters(), self.layer2.parameters(), self.layer3.parameters()]

    def set_parameters(self, threshold):
        self.layer1.set_parameters(threshold[0])
        self.layer2.set_parameters(threshold[1])
        self.layer3.set_parameters(threshold[2])


if __name__ == '__main__':
    print(softmax(np.array([-1,2,3])))
    pass