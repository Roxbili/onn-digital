#-*- encoding: utf-8 -*-

import os
import numpy as np
import sys

from utils.utils import rescale, softmax
from dataset import MNIST, Feature
from model import Net_1, Net_2, Optim, AccFunc, CrossEntropyLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

############### network parameters ###############
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 100
layer1_node = 128
# layer2_node = 50
output_size = 10

batch_size = 1000
epoch = 100
learning_rate = 0.01

train = True
load_model = True
load_model_path = 'log_torch/10_128/float.pt'

float_path = 'log_torch/10_128_0.01/float.pt'
quant_path = 'log_torch/10_128_0.01/quant.pt'

############### data pre-processing ###############

train_set = MNIST('mnist', 'train', (10, 10))
test_set = MNIST('mnist', 't10k', (10, 10))

train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
train_fv, train_label = train_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
train_fv = train_fv.reshape(-1, 100)
train_fv = rescale(train_fv, 50, 200, True)

test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test_fv = test_fv.reshape(-1, 100)
test_fv = rescale(test_fv, 50, 200, True)

############## 创建pytoch使用的数据集 ##############

class MyDataset(Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, inputs, labels): #初始化一些需要传入的参数
        self.inputs = inputs
        # self.labels = F.one_hot(torch.from_numpy(labels))   # to one_hot
        self.labels = labels
 
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
 
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return self.labels.shape[0]

train_data = MyDataset(train_fv, train_label)
test_data = MyDataset(test_fv, test_label)

#然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

############## 网络定义 ##############

# f = range(50, 210, 10)
# c = range(4, 20, 1)
# f2c = dict(zip(f, c))   # frequency to counter number

class WinarizeF(Function):

    @staticmethod
    def forward(cxt, input):
        output = input.round()
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input

# aliases
winarize = WinarizeF.apply

class Linear(object):
    """Linear layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
    """
    def __init__(self, in_features, out_features, f2c_func):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = 3
        self.f2c_func = f2c_func

    def __call__(self, inputs, w, is_train):
        return self.forward(inputs, w, is_train)

    def forward(self, inputs, w, is_train):
        """Forward network

            Args:
                inputs: torch tensor, shape=(batch_size, in_features)
            
            Return:
                output: shape=(batch_size, out_features), type=np.array
        """
        # round w
        w = winarize(w)
        # change frequency to counter number
        shape = inputs.shape
        output = inputs.reshape(-1)
        # output = torch.Tensor([f2c[int(x/10)*10] for x in output])
        output = self.f2c_func(output, is_train=is_train)
        output = output.reshape(shape)

        output = torch.matmul(output, w)
        output += self.bias
        return output

class Mapping(object):
    def __init__(self, input_node):
        self.input_node = input_node

    def __call__(self, inputs, is_train):
        return self.forward(inputs, is_train)

    def forward(self, inputs, is_train):
        if is_train == True:
            output = inputs / (4 * self.input_node)
        else:
            output = inputs // (4 * self.input_node)
        output = (output + 5) * 10  # return (50, 200)
        return output

class F2C(object):
    def __init__(self):
        pass
    
    def __call__(self, frequency, is_train):
        return self.forward(frequency, is_train)
    
    def forward(self, frequency, is_train):
        if is_train == True:
            counter = frequency / 10 - 1
            # counter = (2/15) * frequency - (23/3)
        else:
            counter = frequency // 10 - 1
            # counter = (2/15) * frequency - (23/3)
            # counter = counter.floor()
        return counter

class Net(nn.Module):
    def __init__(self, input_size, layer1_node, output_size):
        super(Net, self).__init__()

        self.is_train = None
        self.w1 = torch.nn.Parameter(self._init_weight((input_size, layer1_node), requires_grad=True))
        self.w2 = torch.nn.Parameter(self._init_weight((layer1_node, output_size), requires_grad=True))

        self.f2c = F2C()
        self.layer1 = Linear(input_size, layer1_node, self.f2c)
        self.layer2 = Linear(layer1_node, output_size, self.f2c)

        self.mapping = Mapping(input_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def __call__(self, x):
        return self.forward(x)

    def _init_weight(self, shape, requires_grad):
        # w = torch.randint(-3, 4, shape).float()
        w = torch.empty(shape)
        w = torch.nn.init.uniform_(w, a=-3., b=3.)
        w.requires_grad=requires_grad
        return w

    def forward(self, x):
        out = self.layer1(x, self.w1, self.is_train)
        out = self.relu(out)
        out = self.mapping(out, self.is_train)
        out = self.dropout(out)
        out = self.layer2(out, self.w2, self.is_train)
        out = self.softmax(out)
        return out

    def train_flag(self, flag):
        self.is_train = flag


class Optim(object):
    """Optimization operator, to updata parameters in model"""
    def __init__(self, net):
        self.net = net
    
    def step(self):
        pass

    def zero_grad(self):
        self.net.zero_grad()


def clip_weight(parameters):
    for p in parameters:
        p = p.data
        p.clamp_(-3., 3.)


criterion = nn.CrossEntropyLoss()
net = Net(input_size, layer1_node, output_size)
net.to(device)

if train == True:

    ###################### Train ######################
    net.train()
    if load_model == True:
        net.load_state_dict(torch.load(load_model_path))

    net.train_flag(True)
    # optimizer = Optim(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(epoch):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.float().to(device)
            labels = labels.long().to(device)

            # Forward pass
            outputs = net(images)

            # batch accuracy
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = correct / total

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            clip_weight(net.parameters())

            # print(net.w1.grad)
            # print(net.w2.grad)

            if (i + 1) % 10 == 0:
                print('Epoch: %d' % epoch, acc, float(loss))
                # print(net.w2)
            
            # sys.exit(0)

    torch.save(net.state_dict(), float_path)

    ###################### quantization ######################
    def toscale(inputs, bottom_line, top_line):
        """map inputs value from bottom_line to top_line"""
        k = (top_line - bottom_line) / (inputs.max() - inputs.min())
        ret = bottom_line + k * (inputs - inputs.min())
        return torch.round(ret)

    state_dict = net.state_dict()
    # print(state_dict)
    # state_dict['w1'] = toscale(state_dict['w1'], -3, 3)
    # state_dict['w2'] = toscale(state_dict['w2'], -3, 3)
    state_dict['w1'] = state_dict['w1'].round()
    state_dict['w2'] = state_dict['w2'].round()
    print(state_dict['w1'].min(), state_dict['w1'].max())
    print(state_dict['w2'].min(), state_dict['w2'].max())
    net.load_state_dict(state_dict)
    # print(list(net.parameters()))

    torch.save(net.state_dict(), quant_path)



###################### Test ######################

net.train_flag(False)
net.eval()
net.load_state_dict(torch.load(quant_path))
state_dict = net.state_dict()
# print(state_dict)
# print(state_dict['w1'].min(), state_dict['w1'].max())
# print(state_dict['w2'].min(), state_dict['w2'].max())

# Test model
correct = 0
total = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.float().to(device)
        labels = labels.long().to(device)

        # Forward pass
        outputs = net(images)

        # batch accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))