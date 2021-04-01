#-*- encoding: utf-8 -*-

import numpy as np
import math
import os, sys
import random

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

        # every output node has a relu class
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
        # print(threshold.shape[0])
        # print(len(self.relu))
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
        # print(y)
        return self.shift(y)

    def shift(self, x, lower_bound=30, upper_bound=250):
        """Shift x to (lower_bound, upper_bound)"""
        k = 220
        ret = lower_bound + k * x
        return ret


class AccFunc(object):
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


class Population(object):
    """Population
    
        Args:
            num: the number of individuals
    """
    def __init__(self, num, *layer_node):
        self.num = num
        self.layer_node = layer_node
        self.popu_params = self._init_param(num, layer_node)
        self.best_params = None

    def _init_param(self, num, layer_node, lower_bound=30, upper_bound=250):
        """Init population"""
        popu_params = []
        length = len(layer_node)
        for _ in range(num):    # the number of individuals
            layers_params = []
            for i in range(length):
                if i + 1 >= length:
                    break
                out_features = layer_node[i+1]
                in_features = layer_node[i]
                param = np.random.randint(lower_bound, upper_bound, (out_features, in_features))
                layers_params.append(param)
            popu_params.append(layers_params)
        return popu_params

    def update_popu(self, params):
        self.popu_params = params

    def get_all_params(self):
        return self.popu_params

    def _fitness(self, loss):
        """Fitness function"""
        return np.array(list(map(lambda x: 1 / x, loss)))

    def select(self, loss):
        fit_val = self._fitness(loss)
        num2next = self.num * fit_val / fit_val.sum()
        integer = num2next.astype(np.int)
        decimal = num2next - integer
        # print(integer)
        # print(decimal)

        new_popu_params = []
        
        for i in range(num2next.shape[0]):
            cp_num = integer[i]
            layers_params = self.popu_params[i]
            for _ in range(cp_num):
                new_popu_params.append(layers_params)
        
        # # use decimal order to fill population
        # sort_index = np.argsort(-decimal)
        # for i in range(self.num - len(new_popu_params)):    # fill population to M
        #     layers_params = self.popu_params[sort_index[i]]
        #     new_popu_params.append(layers_params)
        
        # use best individual to fill population
        for i in range(self.num - len(new_popu_params)):
            new_popu_params.append(self.best_params)

        # shuffle the populaltion
        random.shuffle(new_popu_params)
        # individuals to be saved
        self.popu_params = new_popu_params

    def crossover(self, x, y, mode='one_point', grained='fine'):
        """One-point Crossover
        
            Args:
                x, y: parentes
                mode: one_point | two_point
                grained: coarse | fine
        """

        assert len(x) == len(y)
        child_1 = []
        child_2 = []

        if mode == 'one_point':
            '''output node crossover'''
            for layer_num in range(len(x)):
                x_layer_params = x[layer_num]   # shape = (out_size, in_size)
                y_layer_params = y[layer_num]

                # exchange
                if grained == 'coarse':
                    cut_pos = random.randint(0, x_layer_params.shape[0])    # create cut position
                    child_1.append(np.concatenate((x_layer_params[:cut_pos], y_layer_params[cut_pos:]), axis=0))
                    child_2.append(np.concatenate((y_layer_params[:cut_pos], x_layer_params[cut_pos:]), axis=0))
                    # print(child_1.shape)
                    # print(child_2.shape)
                elif grained == 'fine':
                    # reshape
                    x_layer_params_shape = x_layer_params.shape
                    y_layer_params_shape = y_layer_params.shape
                    x_layer_params = x_layer_params.reshape(-1)
                    y_layer_params = y_layer_params.reshape(-1)

                    # exchange
                    cut_pos = random.randint(0, x_layer_params_shape[0])
                    x_child = np.concatenate((x_layer_params[:cut_pos], y_layer_params[cut_pos:]))
                    y_child = np.concatenate((y_layer_params[:cut_pos], x_layer_params[cut_pos:]))

                    # re-reshape
                    child_1.append(x_child.reshape(x_layer_params_shape))
                    child_2.append(y_child.reshape(y_layer_params_shape))

        elif mode == 'two_point':
            '''one layer node crossover'''
            for layer_num in range(len(x)):
                x_layer_params = x[layer_num]   # shape = (out_size, in_size)
                y_layer_params = y[layer_num]

                if grained == 'coarse':
                    cut_pos = np.random.randint(0, x_layer_params.shape[0], 2)
                    cut_pos.sort()
                    child_1.append(np.concatenate((x_layer_params[:cut_pos[0]], y_layer_params[cut_pos[0]:cut_pos[1]], x_layer_params[cut_pos[1]:]), axis=0))
                    child_2.append(np.concatenate((y_layer_params[:cut_pos[0]], x_layer_params[cut_pos[0]:cut_pos[1]], y_layer_params[cut_pos[1]:]), axis=0))

                elif grained == 'fine':
                    # reshape
                    x_layer_params_shape = x_layer_params.shape
                    y_layer_params_shape = y_layer_params.shape
                    x_layer_params = x_layer_params.reshape(-1)
                    y_layer_params = y_layer_params.reshape(-1)

                    # exchange
                    cut_pos = np.random.randint(0, x.layer_params.shape[0], 2)
                    cut_pos.sort()
                    x_child = np.concatenate((x_layer_params[:cut_pos[0]], y_layer_params[cut_pos[0]:cut_pos[1]], x_layer_params[cut_pos[1]:]))
                    y_child = np.concatenate((y_layer_params[:cut_pos[0]], x_layer_params[cut_pos[0]:cut_pos[1]], y_layer_params[cut_pos[1]:]))
                    
                    # re-reshape
                    child_1.append(x_child.reshape(x_layer_params_shape))
                    child_2.append(y_child.reshape(y_layer_params_shape))

        return (child_1, child_2)

    def mutation(self, x, lower_bound=30, upper_bound=250):
        """Simple mutation

            This function will change the x
        """
        
        ############# element mutation #############
        layer_param_num = []
        for layer_params in x:
            layer_param_num.append(layer_params.shape)
        # create mutation position and value
        pos_layer = random.randint(0, len(layer_param_num) - 1)
        mu_pos_x = random.randint(0, layer_param_num[pos_layer][0] - 1)
        mu_pos_y = random.randint(0, layer_param_num[pos_layer][1] - 1)
        mu_val = random.randint(lower_bound, upper_bound)

        # mutation begin, this will change the x array
        x[pos_layer][mu_pos_x][mu_pos_y] = mu_val

    def record_best(self, index):
        self.best_params = self.popu_params[index]

    def save_best(self, acc, num_class, dir='log'):
        """Save the parameters of all layers to each .npy format

            Args:
                acc: the training accuracy of the result of parameters
                dir: the diretory to save the parameters, default='log'
        """
        acc *= 100
        dir_path = os.path.join(dir, str('%.2f' % acc) + '_' + str(num_class))
        for layer_size in self.layer_node:
            dir_path += '_' + str(layer_size)
            
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        for i, layer_params in enumerate(self.best_params):
            np.save(os.path.join(dir_path, 'layer%d.npy' % i), layer_params)

        print('save completed')
    
    def get_best(self):
        return self.best_params

###########################################################################
################################# Network #################################
###########################################################################

class Net_1(object):
    def __init__(self, input_size, layer1_node, output_size):
        self.layer1 = Linear(input_size, layer1_node)
        self.layer2 = Linear(layer1_node, output_size)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # print(x.shape)
        # print(x[:3])
        out = self.layer1(x)
        # print(out[:3])
        # sys.exit(0)
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