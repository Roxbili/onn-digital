#-*- encoding: utf-8 -*-

from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import numpy as np
import logging


######################### images #########################

def save_one_img(index, path, img, label):
    # fig = plt.figure()
    # plotwindow = fig.add_subplot(111)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(path, str(index) + '_' + str(label) + '.png'))
    plt.close()

def save_one_img_array(index, path, img, label):
    file_path = os.path.join(path, str(index) + '_' + str(label))
    np.save(file_path, img)

def save_images(path, images, labels, worker_num=4, mode='array'):
    save = {
        'img': save_one_img,
        'array': save_one_img_array
    }[mode]

    p = Pool(worker_num)
    for i in range(len(labels)):
        if i == 100:
            break
        p.apply_async(save, args=(i, path, images[i], labels[i]))
    p.close()
    p.join()


######################### tools #########################

def limit_scale(input, bottom_line, top_line):
    """Limit input scale

        el = input[i]
        if el < bottom_line:
            d = bottom_line - el
            el = el + randint(d, top_line - el)
        else:
            d = el - top_line
            el = el - ranint(d, el - bottom_line)
    """
    low_d = bottom_line - input[input < bottom_line]
    high_d = input[input > top_line] - top_line

    input[input < bottom_line] += np.random.randint(low_d, top_line - input[input < bottom_line])
    input[input > top_line] -= np.random.randint(high_d, input[input > top_line] - bottom_line)
    return input // 10 * 10

def rescale(input, bottom_line, top_line, is_round=True):
    """map input value from bottom_line to top_line"""
    k = (top_line - bottom_line) / (input.max() - input.min())
    ret = bottom_line + k * (input - input.min())
    if is_round == True:
        ret = ret // 10 * 10    # round to multiples of 10
    return ret.astype(np.int)

def maxPooling(feature_map, size=2, stride=2):
    channel=feature_map.shape[0]
    height=feature_map.shape[1]
    width=feature_map.shape[2]

    pool_out = np.zeros((channel, int((height - size) / stride + 1), int((width - size) / stride + 1)))
    
    for map_num in range(channel):  
        out_height = 0  
        for r in np.arange(0,height, stride):  
            out_width = 0  
            for c in np.arange(0, width, stride):  
                pool_out[map_num,out_height, out_width] = np.max(feature_map[map_num,r:r+size,c:c+size])  
                out_width=out_width+1
            out_height=out_height+1
    return pool_out

def generate_frequency(inputs):
    outputs = np.floor(inputs / 2**4)
    outputs = (outputs + 5) * 10

    return outputs

def softmax(x):
    """Compute the softmax function for each row of the input x.

        Args:
            x: A N dimensional vector or M x N dimensional numpy matrix.

        Return:
            x: You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax,1,x)
        denominator = np.apply_along_axis(denom,1,x) 
        
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0],1))
        
        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator =  1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
    
    assert x.shape == orig_shape
    return x


class Message(object):
    """Log message"""
    def __init__(self, log_path):
        LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
        logging.basicConfig(filename=log_path, level=logging.DEBUG, format=LOG_FORMAT)
        self.logger = logging.getLogger(__name__)

    def __call__(self, msg):
        print(msg)
        self.logger.info(msg)