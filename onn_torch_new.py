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
layer1_node = 512
layer2_node = 32
output_size = 10

batch_size = 1000
epoch = 500

learning_rate = 0.0001
lr_end = 1e-4
lr_decay = (lr_end / learning_rate)**(1. / epoch)

clip = False
w_round = False

train = True
load_model = False
load_model_path = 'log_torch/10_512_lr_decay/float.pt'

dir_path = 'log_torch/10_512_float'
float_path = os.path.join(dir_path, 'float.pt')
quant_path = os.path.join(dir_path, 'quant.pt')
def create_dir(dir_path):
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)

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


############## Net ##############

class Linear(nn.Linear):
    def forward(self, inputs):
        # binary_weight = binarize(self.weight)
        weight = self.weight
        outputs = inputs / 10 - 1
        if self.bias is None:
            return F.linear(outputs, weight) + 3
        else:
            return F.linear(outputs, weight, self.bias)

class Mapping(nn.Module):
    def __init__(self, in_size):
        super(Mapping, self).__init__()
        self.in_size = in_size

    def forward(self, inputs):
        outputs = inputs / (4 * self.in_size)
        outputs = (outputs + 5) * 10
        return outputs

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()

        self.layer = nn.Sequential(
            Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            Mapping(input_size),
            nn.Dropout(p=0.2))
        self.fc = Linear(hidden_size, num_classes, bias=False)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = self.fc(outputs)
        outputs = self.softmax(outputs)
        return outputs

def clip_weight(parameters):
    for p in parameters:
        p = p.data
        p.clamp_(-3., 3.)

# learning rate schedule
def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = lr * lr_decay
        param_group['lr'] = lr



net = Net(input_size, layer1_node, output_size).to(device)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

############## Train ##############
net.train()

for e in range(epoch):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.float().to(device)
        labels = labels.long().to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum()

        if (i+1) % 10 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Accuracy: %.4f, Loss: %.4f' 
                   %(e+1, epoch, i+1, len(train_data)//batch_size, correct * 1.0 / batch_size, loss.item()))

    adjust_learning_rate(optimizer)
    
    # Test the Model
    net.eval()
    correct = 0
    for images, labels in test_loader:
        images = images.float().to(device)
        labels = labels.long().to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / (len(test_loader) * batch_size)))

    net.train()