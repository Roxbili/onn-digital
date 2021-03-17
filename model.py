#-*- encoding: utf-8 -*-

import numpy as np
import random

class Linear(object):
    """Linear layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.bias = np.array([random.randint(-30, 30) * 10 for _ in range(out_features)])
        # print(self.bias)

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        """Forward network

            Args:
                input: shape=(batch_size, in_features)
        """
        # img_sum = input.sum(axis=1, keepdims=True) # shape = (batch_size,)
        # img_sum = img_sum.repeat(self.out_features, axis=1)
        
        img_sum = self._sum_regular(input)
        # print(img_sum.shape)
        batch_size_bias = np.tile(self.bias, (input.shape[0], 1))
        # print(batch_size_bias.shape)

        output = img_sum + batch_size_bias
        # print(output.shape)     # output shape = (batch_size, out_features)
        return output

    def _sum_regular(self, input):
        """Change input scale
        
            (20*9, 300*9) -> (500/16, 500/2)
        """
        img_sum = input.sum(axis=1, keepdims=True) # shape = (batch_size,)
        img_sum /= 9.

        # Consider to reuse this block combined with dataset.py's encoding function
        k = (250 - 31.25) / (300 - 20)
        img_sum = 31.25 + k * (img_sum - 20)
        img_sum = img_sum // 10 * 10    # round to multiples of 10
        return img_sum

class LossFunc(object):
    """Loss function for this network"""
    def __init__(self):
        pass

    def __call__(self, outputs, labels):
        self.outputs = outputs
        self.labels = labels
        
        return self.calc()

    def calc(self):
        equal_array = np.equal(self.outputs.argmax(axis=1), self.labels)
        acc = np.mean(equal_array)
        return acc

class Optim(object):
    """Optimization operator, to updata parameters in model"""
    def __init__(self, net):
        self.net = net
        self.search_step = None

        self.max_acc = 0
        self.store_bias = None
        self.store_non_label_output_sum = None

    def update(self, acc, outputs, labels):
        if acc > self.max_acc:
            self.max_acc = acc
            self.store_bias = self.net.get_parameters()
            self.store_non_label_output_sum = self._calc_non_label_output_sum(outputs, labels)
        else:
            if self._accept(outputs, labels) == False:
                # give up this bias, restore last bias
                self.net.set_parameters(self.store_bias)
            else:
                # althought acc is smaller, still accept this bias
                self.store_bias = self.net.get_parameters()
        # all situation need to step for next bias
        self._step(self.search_step, self.net.get_parameters())

    def _accept(self, outputs, labels):
        """Judge to accept this bias or not

            if the sum of non_abel output is smaller than last epoch, accepy.
        """
        non_label_output_sum = self._calc_non_label_output_sum(outputs, labels)
        if non_label_output_sum < self.store_non_label_output_sum:
            self.store_non_label_output_sum = non_label_output_sum
            return True
        else:
            return False

    def _calc_non_label_output_sum(self, outputs, labels):
        """Calculate the sum of non-label output"""
        mask = np.ones(outputs.shape)
        for i in range(len(labels)):
            j = labels[i]
            mask[i][j] = 0
        non_label_output = outputs * mask
        # print(non_label_output)
        return non_label_output.sum()

    def _step(self, search_step, net_bias):
        """Update bias

            Args: the range of search area. If search_step = 300 means init position.
        """
        base_search_step = search_step // 10
        new_bias = []
        for layer_bias in net_bias:
            # print(layer_bias.shape)
            step_info = [random.randint(-base_search_step, base_search_step) * 10 for _ in range(layer_bias.shape[0])]
            layer_bias += step_info
            new_bias.append(layer_bias)
        self.net.set_parameters(new_bias)

class Net(object):
    def __init__(self, input_size, layer1_node, num_class):

        self.layer1 = Linear(input_size, layer1_node)
        self.layer2 = Linear(layer1_node, num_class)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return out2

    def get_parameters(self):
        return [self.layer1.bias, self.layer2.bias]

    def set_parameters(self, bias):
        self.layer1.bias = bias[0]
        self.layer2.bias = bias[1]


if __name__ == '__main__':
    # layer1 = Linear(9, 5)
    # layer2 = Linear(5, 10)

    # data = np.ones((1000, 9))
    # out = layer1(data)
    # layer2(out)

    optimizer = Optim(None)
    optimizer._calc_non_label_output_sum(np.random.rand(3,4), np.array([1, 0, 2]))