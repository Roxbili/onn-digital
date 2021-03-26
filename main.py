#-*- encoding: utf-8 -*-

import os
import numpy as np
import sys
import argparse
import time

from utils.utils import rescale, softmax, Message
from dataset import MNIST, Feature
from model import Net_1, Net_2, Optim, AccFunc, Population, CrossEntropyLoss


############### init ###############

# input_size = 100
# layer1_node = 20
# layer2_node = 50
# output_size = 2

# batch_size = 1000
# epoch = 1000

# popu_num = 20
# mu_p = 0.8

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', type=int, help='The input of network.')
parser.add_argument('--layer1_node', type=int, help='The first hidden layer size.')
parser.add_argument('--layer2_node', type=int, help='The second hidden layer size.')
parser.add_argument('--output_size', type=int, help='The number of class')
parser.add_argument('--batch_size', type=int, help='Batch size.')
parser.add_argument('--epoch', type=int, help='Trainging epoch.')
parser.add_argument('--popu_num', type=int, help='The number of population each epoch')
parser.add_argument('--mu_p', type=float, help='The mutation probability is 1-mu_p')
parser.add_argument('--resume_dir', type=str, default=None, help='Reload pre-training parameters, please input directory.')
parser.add_argument('--running_mode', type=str, default='train', help='Running mode, train | test. Default: train')
parser.add_argument('--class_list', type=int, nargs="+", default=[0,1], help='Class in dataset.')
args = parser.parse_args()

############### logger ###############
base = os.path.basename(__file__)
time_ = time.strftime("_%Y%m%d_%H%M%S", time.localtime()) 
log_name = os.path.splitext(base)[0] + time_ + '.log'
log_path = os.path.join('log', log_name)
logger = Message(log_path)

logger('Log file: %s' % log_path)

############### data pre-processing ###############

train_set = MNIST('mnist', 'train', (10, 10))
test_set = MNIST('mnist', 't10k', (10, 10))

'''10 class
train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
train_fv = train_feature._data['images'].reshape(-1, 100)         # 10 class, 100 input
rescale(train_fv, 30, 250, False)
train_label = train_feature._data['labels']
input_train_data = train_feature.cut_into_batch(batch_size=args.batch_size, vector=train_fv, labels=train_label)

test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
test_fv = test_feature._data['images'].reshape(-1, 100)           # 10 class, 100 input
rescale(test_fv, 30, 250, False)
test_label = test_feature._data['labels']
input_test_data = test_feature.cut_into_batch(batch_size=args.batch_size, vector=test_fv, labels=test_label)
'''

# any class
train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
train_fv, train_label = train_feature.extract_num_class(args.class_list)
# print(train_fv.shape)
# print(train_label.shape)
train_fv = train_fv.reshape(-1, 100)
rescale(train_fv, 30, 250, False)
input_train_data = train_feature.cut_into_batch(batch_size=args.batch_size, vector=train_fv, labels=train_label, num_class=args.output_size, one_hot=True)

test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
test_fv, test_label = test_feature.extract_num_class(args.class_list)
test_fv = test_fv.reshape(-1, 100)
rescale(test_fv, 30, 250, False)
input_test_data = test_feature.cut_into_batch(batch_size=args.batch_size, vector=test_fv, labels=test_label, num_class=args.output_size, one_hot=True)


############### model define ###############

net = Net_1(args.input_size, args.layer1_node, args.output_size)
# net = Net_2(args.input_size, args.layer1_node, layer2_node, args.output_size)

individuals = Population(args.popu_num, args.input_size, args.layer1_node, args.output_size)
acc_func = AccFunc()
criterion = CrossEntropyLoss()


############### Resume ###############

if args.resume_dir != None:
    npy_path = list(map(lambda x: os.path.join(args.resume_dir, x), os.listdir(args.resume_dir)))
    best_params = []
    for item_path in npy_path:
        best_params.append(np.load(item_path))

    if args.running_mode == 'train':
        popu_params = []
        for i in range(args.popu_num):  # all population have the same individual
            popu_params.append(best_params)
    elif args.running_mode == 'test':
        popu_params = [best_params]

    # update population
    individuals.update_popu(popu_params)
    individuals.record_best(0)


start_time = time.time()
############### train ###############
if args.running_mode == 'train':
    max_acc = 0.
    for _ in range(args.epoch):
        logger("Epoch: %d" % _)
        for i, (images, labels) in enumerate(input_train_data):
            individual_loss = []
            acc_collecter = []
            for param in individuals.get_all_params():
                # change different individual
                net.set_parameters(param)
                
                outputs = net(images)
                outputs = softmax(outputs)
                # print(outputs[:3])
                # print()
                # print(outputs.shape)    # shape=(batch_size, 1)

                loss = criterion(outputs, labels)
                # print(loss)
                individual_loss.append(loss)   # label should one-hot
                
                acc = acc_func(outputs, labels)
                acc_collecter.append(acc)

            ############ Top 5 acc and record #############
            acc_collecter = np.array(acc_collecter)
            acc_sorted = abs(np.sort(-acc_collecter))
            logger(acc_sorted[:5])
            if acc_sorted[0] > max_acc:
                individuals.record_best(np.argsort(-acc_collecter)[0])
                max_acc = acc_sorted[0]

            ############ GA #############
            individuals.select(individual_loss) # selection good individuals
            # get next generation
            popu_params = individuals.get_all_params()
            new_popu_params = []
            for i in range(0, len(popu_params), 2):
                child_1, child_2 = individuals.crossover(popu_params[i], popu_params[i+1])
                new_popu_params.append(child_1)
                new_popu_params.append(child_2)
            
            # mutation
            popu_p = np.random.rand(args.popu_num)
            for i in range(args.popu_num):
                if popu_p[i] > args.mu_p:
                    # print('Epoch: %d, individual: %d' % (_, i))
                    individuals.mutation(new_popu_params[i])

            # update population
            individuals.update_popu(new_popu_params)

    # Save best parameters 
    individuals.save_best(max_acc, num_class=args.output_size, dir='log')

logger('Training time: %f' % time.time() - start_time)

############### test ###############
'''Both training and testing will exacute this block'''

logger('Start testing...')

best_param = individuals.get_best()
net.set_parameters(best_param)
acc_collecter = []
for i, (images, labels) in enumerate(input_test_data):
    outputs = net(images)
    outputs = softmax(outputs)

    acc = acc_func(outputs, labels)
    acc_collecter.append(acc)

logger('Testing accuracy: %.2f' % (np.mean(acc_collecter) * 100))
logger('Testing time: %f' % time.time() - start_time)