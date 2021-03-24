#-*- encoding: utf-8 -*-

import numpy as np
import math

from utils.utils import limit_scale, rescale

class Relu(object):
    """Relu layer
    
        Relu layer will not change input shape

        Args:
            in_features: the number of input neuron
            lower_bound: the lower bound of threshold
            upper_bound: the upper bound of threshold
    """
    def __init__(self, in_features, lower_bound=30, upper_bound=60):
        self.threshold = np.random.randint(lower_bound, upper_bound, in_features)   # shape = (in_features,)

    def __call__(self, input):
        """Forward network

            Return the result of one output neuron

            Args:
                input: shape=(batch_size, layer_shape)

            Return:
                output: shape=(batch_size,), type=np.array
        """
        return self.forward(input)

    def forward(self, input):
        x = input.copy()
        tmp = x > self.threshold
        x[~tmp] = 0
        output = x.sum(axis=1) // (tmp.sum(axis=1) + 0.001) # output precision

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

        self.relu = [Relu(in_features) for _ in range(out_features)]
        # print(len(self.relu))

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        """Forward network

            Args:
                input: shape=(batch_size, in_features)
            
            Return:
                output: shape=(batch_size, out_features), type=np.array
        """
        output = np.zeros((self.out_features, input.shape[0]), dtype=np.int)  # transpose when return, here is convenient for assignment
        for i in range(self.out_features):
            output[i] = self.relu[i](input)
        return output.T

    def parameters(self):
        """Get parameters

            Return:
                threshold: shape=(out_features, in_features), type=np.array
        """
        threshold = np.array([self.relu[i].threshold for i in range(self.out_features)])
        return threshold

    def set_parameters(self, threshold):
        """Set parameters"""
        for i in range(threshold.shape[0]):
            self.relu[i].threshold = threshold[i]


class BatchNorm(object):
    def __init__(self, momentum, eps, num_features):
        """Init parameters

            Args:
                momentum: 追踪样本整体均值和方差的动量
                eps: 防止数值计算错误
                num_features: 特征数量
        """
        # 对每个batch的mean和var进行追踪统计
        self._running_mean = 0
        self._running_var = 1
        # 更新self._running_xxx时的动量
        self._momentum = momentum
        # 防止分母计算为0
        self._eps = eps
        # 对应论文中需要更新的beta和gamma，采用pytorch文档中的初始化值
        self._beta = np.zeros(shape=(num_features, ))
        self._gamma = np.ones(shape=(num_features, ))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x_mean = x.mean(axis=0)
        x_var = x.var(axis=0)
        # 对应running_mean的更新公式
        self._running_mean = (1 - self._momentum) * x_mean + self._momentum * self._running_mean
        self._running_var = (1 - self._momentum) * x_var + self._momentum * self._running_var
        # 对应论文中计算BN的公式
        x_hat = (x - x_mean) / np.sqrt(x_var + self._eps)
        y = self._gamma * x_hat + self._beta
        print(y)
        return self.shift(y)

    def shift(self, x, lower_bound=30, upper_bound=250):
        """Shift x to (lower_bound, upper_bound)"""
        k = 220
        ret = lower_bound + k * x
        return ret


class LossFunc(object):
    """Loss function for this network"""
    def __init__(self):
        pass

    def __call__(self, outputs, labels):
        """Calculate accuracy

            Args:
                outputs: shape=(batch_size, 1)
                labels: shape=(batch_size,)
        """
        # self.outputs = outputs
        # self.labels = labels
        
        return self.calc(outputs, labels)

    def calc(self, outputs, labels):
        outputs = outputs.T[0]     # (batch_size,)
        prediction = np.array(list(map(self._o2prediction, outputs)))
        # print('prediction: ', prediction[:10])
        # print('labels:     ', labels[:10])
        # print()
        equal_array = np.equal(prediction, labels)
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


class Optim(object):
    """Optimization operator, to updata parameters in model"""
    def __init__(self, net):
        self.net = net
        self.search_step = None
        self.T = None

        self._acc = 0
        self.store_params = None
        self.max_acc = 0

    def update(self, acc, outputs, labels):
        # recode max accuracy
        if acc > self.max_acc:
            print(acc)
            self.max_acc = acc

        # update parameters
        delta_t = self._acc - acc
        if delta_t < 0 or self._accept(delta_t) == True:
            # print('lager acc', acc)
            self._acc = acc
            self.store_params = self.net.get_parameters()
        else:
            self.net.set_parameters(self.store_params)

        # all situation need to step for next bias
        self._step(self.search_step, self.net.get_parameters())

    def _accept(self, delta_t):
        p = math.exp(-delta_t / self.T)
        # print(p)
        rand = np.random.rand(1)
        if p > rand[0]:
            # accept this change
            return True
        else:
            return False

    def _step(self, search_step, net_params):
        """Update bias

            Args: the range of search area. If search_step = 300 means init position.
        """
        base_search_step = search_step // 10
        # print(base_search_step)
        new_params = []
        for layer_params in net_params:
            # print(layer_bias.shape)
            # step_info = [random.randint(-base_search_step, base_search_step) * 10 for _ in range(layer_bias.shape[0])]
            step_info = np.random.randint(-base_search_step, base_search_step, layer_params.shape)
            step_info *= 10
            # print(step_info[:3])
            layer_params += step_info
            # print(layer_params.shape)
            for i in range(layer_params.shape[0]):
                layer_params[i] = np.array(list(map(self._HT, layer_params[i])))

            new_params.append(layer_params)
        self.net.set_parameters(new_params)

    def _HT(self, x, lower_bound=30, upper_bound=250):
        """HT function

            HT(x) = {
                x, lower_bound <= x <= upper_bound
                upper_bound, x > upper_bound
                lower_bound, x < lower_bound
            }
        """
        # print(x.shape)
        if x < lower_bound:
            return lower_bound
        elif x > upper_bound:
            return upper_bound
        else:
            return x




class Net_1(object):
    def __init__(self, input_size, layer1_node, output_size):
        self.layer1 = Linear(input_size, layer1_node)
        self.layer2 = Linear(layer1_node, output_size)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

    def get_parameters(self):
        """Return network parameters

            Return:
                np.array, which shape is (layer_num, out_features, in_features)
        """
        return [self.layer1.parameters(), self.layer2.parameters()]

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
    # layer1 = Linear(9, 5)
    # layer2 = Linear(5, 10)

    # data = np.ones((1000, 9))
    # out = layer1(data)
    # layer2(out)

    # a = np.array([[40, 54], [83, 120], [110, 34]])
    # batch_norm = BatchNorm(momentum=0.01, eps=0.001, num_features=2)
    # print(batch_norm(a))
    pass