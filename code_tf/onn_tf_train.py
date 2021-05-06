#-*- encoding: utf-8 -*-

import os
import numpy as np
import sys
import time
sys.path.append('../onn-digital')

from utils.utils import rescale, softmax, generate_frequency, maxPooling
from dataset import MNIST, Feature
from model import Net_1, Net_2, Optim, AccFunc, CrossEntropyLoss

import tensorflow as tf


############### network parameters ###############
input_size = 100
layer1_node = 64
# layer2_node = 64
# layer3_node = 32
output_size = 10

batch_size = 1000
epoch = 1000

param_low = -3.
param_high = 3.

learning_rate = 0.01
decay_rate = 0.96
decay_step = 1000

checkpoint_path = 'log_tf/10_64_round_clamp_floor_e_noAdd3_genInputs/limit'

############### data pre-processing ###############

pre_start = time.time()

'''
# 10类，28x28输入
train_set = MNIST('mnist', 'train', None)
test_set = MNIST('mnist', 't10k', None)

train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
train_fv, train_label = train_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# train_fv = rescale(train_fv, 50, 200, True)
train_fv = maxPooling(train_fv, size=4, stride=4)
train_fv = generate_frequency(train_fv)
train_fv = train_fv.reshape(-1, input_size)    # train_fv shape (batch_size, )
input_train_data = train_feature.cut_into_batch(batch_size=batch_size, vector=train_fv, labels=train_label, num_class=output_size, one_hot=True)


test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# test_fv = rescale(test_fv, 50, 200, True)
test_fv = maxPooling(test_fv, size=4, stride=4)
test_fv = generate_frequency(test_fv)
test_fv = test_fv.reshape(-1, input_size)
input_test_data = test_feature.cut_into_batch(batch_size=batch_size, vector=test_fv, labels=test_label, num_class=output_size, one_hot=True)
'''


# 10类，100输入
train_set = MNIST('mnist', 'train', (10, 10))
test_set = MNIST('mnist', 't10k', (10, 10))

train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
train_fv, train_label = train_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# train_fv = maxPooling(train_fv, size=2, stride=2)
train_fv = generate_frequency(train_fv)
train_fv = train_fv.reshape(-1, input_size)    # train_fv shape (batch_size, )
input_train_data = train_feature.cut_into_batch(batch_size=batch_size, vector=train_fv, labels=train_label, num_class=output_size, one_hot=True)


test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# test_fv = maxPooling(test_fv, size=2, stride=2)
test_fv = generate_frequency(test_fv)
test_fv = test_fv.reshape(-1, input_size)
input_test_data = test_feature.cut_into_batch(batch_size=batch_size, vector=test_fv, labels=test_label, num_class=output_size, one_hot=True)


'''
# 10类，9输入
train_set = MNIST('mnist', 'train', (10, 10))
test_set = MNIST('mnist', 't10k', (10, 10))

train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
train_v, train_label = train_feature.calc_feature_vector()
train_fv, train_label = train_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], train_v, train_label)
train_fv = train_fv.reshape(-1, 9)
train_fv = rescale(train_fv, 50, 200, True)
input_train_data = train_feature.cut_into_batch(batch_size=batch_size, vector=train_fv, labels=train_label, num_class=output_size, one_hot=True)


test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
test_v, test_label = test_feature.calc_feature_vector()
test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], test_v, test_label)
test_fv = test_fv.reshape(-1, 9)
test_fv = rescale(test_fv, 50, 200, True)
input_test_data = test_feature.cut_into_batch(batch_size=batch_size, vector=test_fv, labels=test_label, num_class=output_size, one_hot=True)
'''

pre_end = time.time()
print('=' * 50)
print('data pre-processing time: {}'.format(pre_end - pre_start))
print('=' * 50)

############### 网络组件 ###############

sess = tf.Session()

#使用修饰器，建立梯度反向传播函数。其中op.input包含输入值、输出值，grad包含上层传来的梯度
@tf.RegisterGradient("CopyGrad")
def round_grad(op, grad):
    new_grad = tf.identity(grad)
    return new_grad
 
#使用with上下文管理器覆盖原始的sign梯度函数
def round_(inputs):
    with tf.get_default_graph().gradient_override_map({"Round":'CopyGrad'}):
        outputs = tf.round(inputs)
    return outputs

def clamp(inputs, value_min, value_max):
    '''被截断的部分可以考虑梯度设置成0'''
    with tf.get_default_graph().gradient_override_map({"ClipByValue":'CopyGrad'}):
        outputs = tf.clip_by_value(inputs, value_min, value_max)
    return outputs

def floor(inputs):
    with tf.get_default_graph().gradient_override_map({"Floor":'CopyGrad'}):
        outputs = tf.floor(inputs)
    return outputs

def Linear(inputs, in_size, out_size, activation_func=None):
    # Weights = tf.Variable(tf.truncated_normal([in_size,out_size], mean=0, stddev=1))
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size], mean=0, stddev=3), name='weight')
    # Bias = tf.Variable(tf.zeros([out_size]))

    clamp_weights = clamp(Weights, param_low, param_high)
    # clamp_bias = clamp(Bias, -3., 3.)

    w = round_(clamp_weights)
    # b = round_(clamp_bias)

    counters = floor(inputs / 10) - 1
    outputs = tf.matmul(counters, w)

    if activation_func != None:
        outputs = activation_func(outputs)

    return outputs, clamp_weights

def mapping(inputs, in_size):
    # e = tf.Variable(tf.zeros([1, in_size]), dtype=tf.float32, name='e')
    e = tf.Variable(0, dtype=tf.float32, name='e')
    e = round_(e)
    N = 2**e

    countersWdiv4n = round_(inputs / N)
    clamp_countersWdiv4n = clamp(countersWdiv4n, 0., 15.)
    outputs = (clamp_countersWdiv4n + 5) * 10
    return outputs

def BatchNorm(inputs, is_train=False):
    # batch norm参数
    gamma = tf.Variable(tf.ones([num_units]), name='gamma')
    beta = tf.Variable(tf.zeros([num_units]), name='beta')
    pop_mean = tf.Variable(tf.zeros([num_units]), trainable=False, name='mean')
    pop_variance = tf.Variable(tf.ones([num_units]), trainable=False, name='variance')
    epsilon=1e-3

    if is_train == True:
        batch_mean, batch_variance = tf.nn.moments(inputs, [0])

        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))

        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_variance, beta, gamma, epsilon)
    else:
        return tf.nn.batch_normalization(layer, pop_mean, pop_variance, beta, gamma, epsilon)


############################### 占位符等基本参数设置 ###############################

x = tf.placeholder(tf.float32, (None, input_size))
y = tf.placeholder(tf.int32, (None, output_size))
dropout_rate = tf.placeholder(tf.float32)  # dropout rate
is_train = tf.placeholder(tf.bool)
batch_norm = tf.placeholder(tf.bool)

# 学习率衰减
global_step = tf.Variable(tf.constant(0))
lr_current = tf.train.exponential_decay(learning_rate, global_step, decay_step, decay_rate, staircase=False)

############################### Network ###############################

l1, weight1 = Linear(x, input_size, layer1_node, activation_func=tf.nn.relu)
l1_mapping = mapping(l1, layer1_node)
if batch_norm == True:
    l1_batchnorm = tf.layers.batch_normalization(l1_mapping, training=is_train)
    # l1_batchnorm = BatchNorm(l1_mapping, is_train=is_train)
    l1_relu = tf.nn.relu(l1_batchnorm)
else:
    l1_relu = tf.nn.relu(l1_mapping)
l1_dropout = tf.nn.dropout(l1_mapping, rate=dropout_rate)

# l2, weight2 = Linear(l1_mapping, layer1_node, layer2_node, activation_func=tf.nn.relu)
# l2_mapping = mapping(l2, layer2_node)
# if batch_norm == True:
#     l2_batchnorm = tf.layers.batch_normalization(l2_mapping, training=is_train)
#     # l2_batchnorm = BatchNorm(l2_mapping, is_train=is_train)
#     l2_relu = tf.nn.relu(l2_batchnorm)
# else:
#     l2_relu = tf.nn.relu(l2_mapping)
# l2_dropout = tf.nn.dropout(l2_mapping, rate=dropout_rate)

# l3, weight3 = Linear(l2_mapping, layer2_node, layer3_node, activation_func=tf.nn.relu)
# l3_mapping = mapping(l3, layer3_node)
# if batch_norm == True:
#     l3_batchnorm = tf.layers.batch_normalization(l3_mapping, training=is_train)
#     # l3_batchnorm = BatchNorm(l3_mapping, is_train=is_train)
#     l3_relu = tf.nn.relu(l3_batchnorm)
# else:
#     l3_relu = tf.nn.relu(l3_mapping)
# l3_dropout = tf.nn.dropout(l3_mapping, rate=dropout_rate)

prediction, weight2 = Linear(l1_dropout, layer1_node, output_size)
# prediction, weight2 = Linear(l2_dropout, layer2_node, output_size)
# prediction, weight2 = Linear(l3_dropout, layer3_node, output_size)

# tf.contrib.quantize.experimental_create_training_graph(sess.graph,
#                                                         weight_bits=3,
#                                                         activation_bits=8,
#                                                         quant_delay=200,
#                                                         symmetric=True)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
if batch_norm == True:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(lr_current).minimize(loss)
else:
    train_step = tf.train.AdamOptimizer(lr_current).minimize(loss)
    

# gradient
clamp1_grad = tf.gradients(loss, weight1)
clamp2_grad = tf.gradients(loss, weight2)


############################### run ###############################

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

for _ in range(epoch):
    for i, (images, labels) in enumerate(input_train_data):
        sess.run(train_step, feed_dict={x: images, y: labels, dropout_rate: 0.2, global_step: i, is_train: True, batch_norm: False})
        if i % 10 == 0:
            acc_, loss_ = sess.run([accuracy, loss], feed_dict={x: images, y: labels, dropout_rate: 0.2, is_train: True, batch_norm: False})
            print('Epoch %d, accuracy: %.5f, loss: %.6f' % (_, acc_, loss_))

            # print(sess.run([clamp1_grad, clamp2_grad], feed_dict={x: images, y: labels, dropout_rate: 0.2}))
    
    total_accuracy = 0.
    for i, (images, labels) in enumerate(input_test_data): 
        acc_ = sess.run(accuracy, feed_dict={x: images, y: labels, dropout_rate: 0, is_train: False, batch_norm: False})
        total_accuracy += acc_ * batch_size / len(test_fv)
    print('Accuracy of the network on the 10000 test images: %.4f' % total_accuracy)

saver.save(sess, checkpoint_path)