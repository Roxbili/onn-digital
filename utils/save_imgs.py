#-*- encoding: utf-8 -*-

from multiprocessing import Pool
import matplotlib.pyplot as plt
import os
import numpy as np

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
