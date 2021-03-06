#-*- encoding: utf-8 -*-

import numpy as np
import struct
import os
import cv2
import matplotlib.pyplot as plt

from utils.utils import save_images, rescale

class MNIST(object):
    def __init__(self, data_dir, kind, shape=None):
        self._data = {}

        self._load_mnist(data_dir, kind, shape)

    @property
    def data(self):
        return self._data

    def _load_mnist(self, data_dir, kind, shape):
        """Load MNIST data from `path`

            Args:
                data_dir: the directory where mnist data is stored
                kind: 'train' or 't10k'
                shape: the shape images should be resize to. default: None
        """

        labels_path = os.path.join(data_dir, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(data_dir, '%s-images-idx3-ubyte' % kind)

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            # images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28)

        if shape != None:
            images = np.array(list(map(lambda x: self._resize_array(x, shape), images)))
            # print(images.shape)

        self._data['images'] = images
        self._data['labels'] = labels

    def _resize_array(self, data, shape):
        """Resize numpy array
            
            Args:
                data: numpy array
                shape: the shape data should be resize to
        """
        if isinstance(shape, tuple) == False or len(shape) != 2:
            raise ValueError('shape should be tuple like (4,4)')
        ret = cv2.resize(data, dsize=shape, interpolation=cv2.INTER_AREA)
        return ret

    def save_img(self, path, save_num, worker_num, mode='array'):
        """Save images to path

            Args:
                path: where to save images
                worker_num: how many threads to use
                mode: 'img' or 'array'
        """
        save_images(path, self._data['images'], self._data['labels'], save_num, worker_num=worker_num, mode=mode)
        print('all images have been saved to %s' % path)


class Feature(object):
    """Get features object

        Args:
            data: array of images data
            kernel_size: size of the feature templete kernel, a single number or a tuple
            stride: controls the stride for the cross-correlation, a single number or a tuple
    """
    def __init__(self, data, kernel_size, stride):
        self._data = data
        self.kernel_size = kernel_size
        self.stride = stride

        self.integ_graph = None
        self.white_point = None
        self.black_point = None
        
        self._v = []
        self._fv = None
        self._compress_v = []

        self._integrogram()

    def calc_feature_vector(self):
        """Calculate features, which are vector array"""
        self._get_points()  # points are added to white_point and black point
        self._calc_features()
        return (self._v, self._data['labels'])

    def compress(self):
        """Compress feature vector to small vector

            This is an optional selection.
        """
        for single_v in self._v:
            compress_v = []
            for start_id in range(0, len(single_v), 3):
                compress_v.append(sum(single_v[start_id:start_id+3]))
            self._compress_v.append(compress_v)
        self._compress_v = np.array(self._compress_v)
        return (self._compress_v, self._data['labels'])

    def hist(self, save_path, data):
        """Develop function"""
        fig = plt.figure()
        plt.hist(data)
        # fig = plt.gcf()
        fig.savefig(save_path)

    def encoding(self, vector, is_round=False):
        # k = (300 - 20) * 1. / (vector.max() - vector.min())
        # self._fv = 20 + k * (vector - vector.min())
        # if is_round == True:
        #     self._fv = self._fv // 10 * 10
        self._fv = rescale(vector, 20, 300, is_round=is_round)
        return (self._fv, self._data['labels'])

    def cut_into_batch(self, batch_size, vector, labels, num_class, one_hot=False, shuffle=False):
        """Cut data into batch_data
            
            Return:
                input_data: a dict, use input_data['data'] and input_data['labels']
        """
        if shuffle == True:
            permutation = np.random.permutation(vector.shape[0])
            vector = vector[permutation]
            labels = labels[permutation]

        input_data = []
        for start_batch in range(0, vector.shape[0], batch_size):
            batch_imgs = vector[start_batch:start_batch+batch_size]
            batch_labels = labels[start_batch:start_batch+batch_size]
            if one_hot == True:
                batch_labels = np.eye(num_class)[batch_labels]  # subsequent numbers, which begins with 0
            input_data.append((batch_imgs, batch_labels))
        self.input_data = input_data
        return self.input_data

    def _calc_features(self):
        """Get freature vectory array according to point position"""
        for i in range(self.integ_graph.shape[0]):
            # if i % 1000 == 0:
            #     print(i)
            v = []
            for j in range(len(self.white_point)):
                black_val = self._area_val(i, self.black_point[j])
                white_val = self._area_val(i, self.white_point[j])
                # print(black_val, white_val)
                v.append(white_val - black_val)
            self._v.append(v)
        self._v = np.array(self._v)

    def _area_val(self, index, points):
        """Calculate the value of one area"""
        lt_line_id, lt_col_id, rb_line_id, rb_col_id = points
        sum = 0
        sum += self._get_integ_graph_val(index, lt_line_id - 1, lt_col_id - 1)
        sum += self._get_integ_graph_val(index, rb_line_id, rb_col_id)
        sum -= self._get_integ_graph_val(index, lt_line_id - 1, rb_col_id)
        sum -= self._get_integ_graph_val(index, rb_line_id, lt_col_id - 1)
        return sum

    def _get_integ_graph_val(self, i, line_id, col_id):
        if line_id < 0 or col_id < 0:
            return 0
        else:
            return self.integ_graph[i][line_id][col_id]    # coordinate (x, y) -> array[y][x]

    def _get_points(self):
        """Get feature areas points"""
        shape = self._data['images'].shape[1:]

        self.black_point = []
        self.white_point = []

        self.xkernet_size, self.ykernel_size = self._int_or_tuple(self.kernel_size)
        self.xstride, self.ystride = self._int_or_tuple(self.stride)

        for i in range(int((shape[0] - self.xkernet_size) / self.xstride + 1)):
            for j in range(int((shape[1] - self.ykernel_size) / self.ystride + 1)):
                self.black_point.append(tuple(map(int, self._calc_xy(0, i, j))))
                self.white_point.append(tuple(map(int, self._calc_xy(self.xkernet_size / 2, i, j))))    # error col

    def _calc_xy(self, base_pos, i, j):
        """Calculate left point and right point
        
            Return:
                lt_line_id, lt_col_id, rb_line_id, rb_col_id
        """
        lt_line_id = i * self.ystride
        lt_col_id = base_pos + j * self.xstride
        rb_line_id = lt_line_id + self.ykernel_size - 1
        rb_col_id = lt_col_id + self.xkernet_size / 2 - 1

        return(lt_line_id, lt_col_id, rb_line_id, rb_col_id)

    def _int_or_tuple(self, data):
        if isinstance(data, tuple) == True:
            return data[0], data[1]
        else:
            return data, data

    def _integrogram(self):
        """Get integrogram graph"""
        self.integ_graph = np.zeros((self._data['images'].shape[0], self._data['images'].shape[1], self._data['images'].shape[2]), dtype=np.int)
        for i in range(self._data['images'].shape[0]):
            graph = np.zeros((self._data['images'].shape[1], self._data['images'].shape[2]), dtype=np.int)
            for x in range(self._data['images'].shape[1]):
                sum_clo = 0
                for y in range(self._data['images'].shape[2]):
                    sum_clo = sum_clo + self._data['images'][i][x][y]
                    graph[x][y] = graph[x-1][y] + sum_clo
            self.integ_graph[i] = graph

    def extract_num_class(self, target, images=[], labels=[]):
        """Extract needed labels and images in target labels

            Args:
                target: the class to be saved
                images: the image data
                labels: the image label
        """
        assert type(target) == list
        if len(images) == 0:
            images = self._data['images']
        if len(labels) == 0:
            labels = self._data['labels']

        # make hash dict for 0-9, accelerate finding speed
        num = range(10)
        num_bool = [False for _ in num]
        hash_dict = dict(zip(num, num_bool))
        for i in target:
            hash_dict[i] = True
        label_encode = dict(zip(target, range(len(target))))    # {0: 0, 1: 1, 4: 2, 7: 3}

        new_images = []
        new_labels = []
        for i in range(labels.shape[0]):
            # if i % 1000 == 0:
            #     print(i)
            label = labels[i]
            if hash_dict[label] == True:
                new_labels.append(label_encode[label])
                new_images.append(images[i])
        new_images = np.array(new_images, dtype=np.int)
        new_labels = np.array(new_labels, dtype=np.int)
        return (new_images, new_labels)


if __name__ == '__main__':
    # train_set = MNIST('mnist', 'train', (10, 10))
    # test_set = MNIST('mnist', 't10k', (10, 10))
    # test_feature = Feature(test_set.data, kernel_size=(4,4), stride=(3,3))
    # test_fv, test_label = test_feature.extract_num_class(0, 1)
    # test_fv = test_fv.reshape(-1, 100)
    # rescale(test_fv, 30, 250, False)
    # input_test_data = test_feature.cut_into_batch(batch_size=1000, vector=test_fv, labels=test_label)
    pass