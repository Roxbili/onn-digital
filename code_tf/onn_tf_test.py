#-*- encoding: utf-8 -*-

import os
import numpy as np
import sys
sys.path.append('../onn-digital')

from utils.utils import rescale, softmax, generate_frequency, maxPooling
from dataset import MNIST, Feature
from model import Net_1, Net_2, Optim, AccFunc, CrossEntropyLoss

import tensorflow as tf


############### network parameters ###############
input_size = 100
layer1_node = 64
# layer2_node = 128
output_size = 10

batch_size = 1000

param_low = -3.
param_high = 3.

checkpoint_dir = 'log_tf/10_64_round_clamp_floor_e_noAdd3_genInputs/'
checkpoint_quant_path = 'log_tf/10_64_round_clamp_floor_e_noAdd3_genInputs_quant/quant'
quant = False
save = True

############### quantization ###############

if quant == True:
    with tf.Session() as sess:
        new_var_list=[] #新建一个空列表存储更新后的Variable变量
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir): #得到checkpoint文件中所有的参数（名字，形状）元组
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name) #得到上述参数的值

            #除了修改参数名称，还可以修改参数值（var）
            # print(var)
            if 'e' in var_name:
                var = tf.round(var)
            elif var_name == 'weight' or var_name == 'weight_1':
                print('quant', var_name)
                var = tf.clip_by_value(var, param_low, param_high)
                var = tf.round(var)
                # print(var)
            # print(var_name, var.max(), var.min())

            renamed_var = tf.Variable(var, name=var_name) #使用加入前缀的新名称重新构造了参数
            new_var_list.append(renamed_var) #把赋予新名称的参数加入空列表
        
        print('starting to write new checkpoint !')
        saver = tf.train.Saver(var_list=new_var_list) #构造一个保存器
        sess.run(tf.global_variables_initializer()) #初始化一下参数（这一步必做）
        saver.save(sess, checkpoint_quant_path) #直接进行保存
        print("done! Change quant flag to False and run again.")
    sys.exit(0)

############### data pre-processing ###############

train_set = MNIST('mnist', 'train', (10, 10))
test_set = MNIST('mnist', 't10k', (10, 10))

train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
train_fv, train_label = train_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# train_fv = maxPooling(train_fv, size=2, stride=2)
train_fv = generate_frequency(train_fv)
train_fv = train_fv.reshape(-1, input_size)
input_train_data = train_feature.cut_into_batch(batch_size=batch_size, vector=train_fv, labels=train_label, num_class=output_size, one_hot=True)


test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# test_fv = maxPooling(test_fv, size=2, stride=2)
test_fv = generate_frequency(test_fv)
test_fv = test_fv.reshape(-1, input_size)
input_test_data = test_feature.cut_into_batch(batch_size=batch_size, vector=test_fv, labels=test_label, num_class=output_size, one_hot=True)

############### Net ###############

sess = tf.Session()

def Linear(inputs, in_size, out_size, activation_func=None):
    # Weights = tf.Variable(tf.truncated_normal([in_size,out_size], mean=0, stddev=1))
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size], mean=0, stddev=3), name='weight')

    clamp_weights = tf.clip_by_value(Weights, param_low, param_high)

    w = tf.round(clamp_weights)
    # b = round_(clamp_bias)

    counters = tf.floor(inputs / 10) - 1
    outputs = tf.matmul(counters, w)

    if activation_func != None:
        outputs = activation_func(outputs)

    return outputs, w, counters

def mapping(inputs, in_size):
    e = tf.Variable(0, dtype=tf.float32, name='e')
    e = tf.round(e)
    N = 2**e

    countersWdiv4n = tf.round(inputs / N)
    clamp_countersWdiv4n = tf.clip_by_value(countersWdiv4n, 0., 15.) 
    outputs = (clamp_countersWdiv4n + 5) * 10
    return outputs, clamp_countersWdiv4n, e

x = tf.placeholder(tf.float32, (None, input_size))
y = tf.placeholder(tf.int32, (None, output_size))
dropout_rate = tf.placeholder(tf.float32)  # dropout rate

# Net
l1, weight1, l1_counters = Linear(x, input_size, layer1_node, activation_func=tf.nn.relu)
l1_mapping, l1_countersWdiv4n, l1_e = mapping(l1, input_size)
l1_relu = tf.nn.relu(l1_mapping)
l1_dropout = tf.nn.dropout(l1_mapping, rate=dropout_rate)

prediction, weight2, l2_counters = Linear(l1_dropout, layer1_node, output_size)

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(os.path.split(checkpoint_quant_path)[0]))

# sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    


############################# run #############################

if save == True:
    npy_path = os.path.join(os.path.split(checkpoint_quant_path)[0], 'npy')
    if os.path.exists(npy_path) == False:
        os.mkdir(npy_path)
        
    w1, w2, e1 = sess.run([weight1, weight2, l1_e])
    np.save(os.path.join(npy_path, 'w1.npy'), w1)
    np.save(os.path.join(npy_path, 'w2.npy'), w2)
    np.save(os.path.join(npy_path, 'e1.npy'), e1)

frequency = []
counters1 = []
counters2 = []
countersWdiv4n = []

total_accuracy = 0.
for i, (images, labels) in enumerate(input_test_data): 
    acc_ = sess.run(accuracy, feed_dict={x: images, y: labels, dropout_rate: 0})
    total_accuracy += acc_ * batch_size / len(test_fv)

    f, c1, c2, cwdiv4n = sess.run([l1_relu, l1_counters, l2_counters, l1_countersWdiv4n], feed_dict={x: images, y: labels, dropout_rate: 0})
    frequency.append(f)
    counters1.append(c1)
    counters2.append(c2)
    countersWdiv4n.append(cwdiv4n)

print('Accuracy of the network on the 10000 test images: %.4f' % total_accuracy)


if save == True:
    def toArraySave(lst, path):
        tmp = np.array(lst)
        np.save(path, tmp)
        print('save %s successfully' % path)

    toArraySave(frequency, os.path.join(npy_path, 'frequency.npy'))
    toArraySave(counters1, os.path.join(npy_path, 'counters1.npy'))
    toArraySave(counters2, os.path.join(npy_path, 'counters2.npy'))
    toArraySave(countersWdiv4n, os.path.join(npy_path, 'countersWdiv4n.npy'))