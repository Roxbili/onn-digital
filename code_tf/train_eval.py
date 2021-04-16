#-*- encoding: utf-8 -*-

import os
import numpy as np
import sys
sys.path.append('../onn-digital')

from utils.utils import rescale, softmax
from dataset import MNIST, Feature
from model import Net_1, Net_2, Optim, AccFunc, CrossEntropyLoss

import tensorflow as tf

sess = tf.InteractiveSession()

############### network parameters ###############
input_size = 100
layer1_node = 512
# layer2_node = 128
output_size = 10

batch_size = 1000
epoch = 300

learning_rate = 0.01

############### data pre-processing ###############

train_set = MNIST('mnist', 'train', (10, 10))
test_set = MNIST('mnist', 't10k', (10, 10))

train_feature = Feature(train_set.data, kernel_size=(4,4), stride=(3,3))
train_fv, train_label = train_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
train_fv = train_fv.reshape(-1, 100)
train_fv = rescale(train_fv, 50, 200, True)
input_train_data = train_feature.cut_into_batch(batch_size=batch_size, vector=train_fv, labels=train_label, num_class=output_size, one_hot=True)


test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
test_fv, test_label = test_feature.extract_num_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test_fv = test_fv.reshape(-1, 100)
test_fv = rescale(test_fv, 50, 200, True)
input_test_data = test_feature.cut_into_batch(batch_size=batch_size, vector=test_fv, labels=test_label, num_class=output_size, one_hot=True)

############### Net ###############

def Linear(inputs, in_size, out_size, activation_func=None):
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],mean=0, stddev=1))

    outputs = inputs / 10 - 1
    outputs = tf.matmul(outputs, Weights) + 3

    if activation_func != None:
        outputs = activation_func(outputs)

    return outputs

def mapping(inputs, in_size):
    outputs = inputs / (4 * in_size)
    outputs = (outputs + 5) * 10
    return outputs

x = tf.placeholder(tf.float32, (None, input_size), name='inputs')
y = tf.placeholder(tf.int32, (None, output_size))
dropout_rate = tf.placeholder(tf.float32)  # dropout rate

# Net
l1 = Linear(x, input_size, layer1_node, activation_func=tf.nn.relu)
l1_mapping = mapping(l1, input_size)
l1_dropout = tf.nn.dropout(l1_mapping, rate=dropout_rate)

prediction = Linear(l1_dropout, layer1_node, output_size)
logits = tf.identity(prediction, name='logits')

tf.contrib.quantize.experimental_create_eval_graph(sess.graph,
                                                    weight_bits=3,
                                                    activation_bits=8,
                                                    quant_delay=0,
                                                    symmetric=True)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))


saver = tf.train.Saver()
# saver=tf.train.import_meta_graph('./log_tf/10_512/float.meta')
saver.restore(sess, tf.train.latest_checkpoint('log_tf/10_512'))


total_accuracy = 0
for i, (images, labels) in enumerate(input_test_data):

    acc_ = sess.run(accuracy, feed_dict={x: images, y: labels, dropout_rate: 0.2})
    total_accuracy += (acc_ * batch_size) / len(test_label)

print('Test accuracy: %f' % total_accuracy)

saver.save(sess, 'log_tf/10_512_eval/final')