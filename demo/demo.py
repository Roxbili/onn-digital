import os, sys
import numpy as np
import argparse
import cv2
import math
sys.path.append('../onn-digital')

from code_tf.onn_numpy_test import Net
from utils.bram import BRAM
from tcp.send_zynq import MyVideoCapture, Soct, FrameRateMonitor

def crop_frame(frame, shape: tuple):
    '''裁减frame

        Args: 
            shape: 目标裁减大小
    '''
    assert shape[0] <= frame.shape[0] and shape[1] <= frame.shape[1], \
        'error, target shape is bigger than frame_shape'

    h_step = (frame.shape[0] - shape[0]) / 2.
    w_step = (frame.shape[1] - shape[1]) / 2.

    h_top = int(h_step)
    h_bottom = frame.shape[0] - math.ceil(h_step)
    w_top = int(w_step)
    w_bottom = frame.shape[1] - math.ceil(w_step)
    return frame[h_top:h_bottom, w_top:w_bottom,:]

def resize_array(data, shape):
        """Resize numpy array
            
            Args:
                data: numpy array
                shape: the shape data should be resize to
        """
        if isinstance(shape, tuple) == False or len(shape) != 2:
            raise ValueError('shape should be tuple like (4,4)')
        ret = cv2.resize(data, dsize=shape, interpolation=cv2.INTER_AREA)
        return ret

def gen_input_data(frame, threshold):
    '''生成输入数据，返回16x16的图像'''
    # load image
    # rsz_img = cv2.resize(frame, None, fx=0.25, fy=0.25) # resize since image is huge
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # convert to grayscale
    
    # 二值化图像，大于阈值的置为255，即白色
    ret, thresh_gray = cv2.threshold(gray, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY_INV)
    data = resize_array(thresh_gray, (16,16))
    return data

def init_network(model_path):
    '''初始化网络实例'''
    weight1 = np.load(os.path.join(model_path, 'npy', 'w1.npy'))
    weight2 = np.load(os.path.join(model_path, 'npy', 'w2.npy'))
    e1 = np.load(os.path.join(model_path, 'npy', 'e1.npy'))
    model = Net(weight1, weight2, e1)
    return model


def get_args():
    '''配置参数'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                    help='debug mode will use numpy network to inference')
    parser.add_argument('--soct', action='store_true',
                    help='whether use socket to send frame')
    parser.add_argument('--threshold', type=int, default=120,
                    help='binarization threshold')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # 获取参数
    args = get_args()

    #################### 初始化 #################### 
    ## scoket通信
    # address = ('127.0.0.1', 6887)  # 服务端地址和端口
    # address = ('192.168.2.151', 6887)  # 服务端地址和端口
    address = ('192.168.177.106', 6887)  # 服务端地址和端口
    # address = ('10.130.147.227', 6887)  # 服务端地址和端口

    if args.debug == True:
        model_path = 'log_tf/10_256_round_clamp_floor_e_noAdd3_genInputs_16x16_quant'
        model = init_network(model_path)
    else:
        ## bram配置
        bram = BRAM()

    ## 摄像头相关
    vid = MyVideoCapture(0)
    frame_rate_monitor = FrameRateMonitor()
    #################### 从摄像头读取数据并发送至上位机 #################### 
    while True:
        ret, frame = vid.get_frame()
        if ret == False:
            raise ValueError("Unable to get frame from camera")

        crop_img = crop_frame(frame, (200, 200))   # 裁减图像
        data = gen_input_data(crop_img, threshold=args.threshold)   # 生成输入，裁减为16x16的大小

        if args.soct == False:
            ############ 直接在PC显示 ############ 
            cv2.imshow("capture", data)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            ############ 传输至PC显示 ############ 
            # str_encode = Frame2JPEG.get_jpeg_data(frame)
            # str_encode = cv2.imencode('.jpg', frame)[1].tostring()
            str_encode = crop_img.tobytes()

            soct = Soct(address)
            soct.send(str_encode)
            # soct.send(pickle.dumps(data))
            # soct.send(str(result[0]))    # 转换成字符串再发送
            # soct.send(str(result[1]))    # 转换成字符串再发送
            # soct.send('Hello world')
            soct.disconnect()

            # fps = frame_rate_monitor.get_fps()
            # if fps != None:
            #     print(fps)
            
        if args.debug:
            ############ 推理 ############
            data = data.flatten()[np.newaxis, :]
            prediction = model(data)
            print(np.argmax(prediction, 1))
        else:
            ############# 写至pl侧 #############
            bram.write(data, 'data')