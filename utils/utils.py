#-*- encoding: utf-8 -*-

from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import numpy as np


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