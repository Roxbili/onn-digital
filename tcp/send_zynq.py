#-*- encoding: utf-8 -*-

import socket
import cv2
import sys
import time

class Soct(object):
    def __init__(self, address):
        # buffer_size = 16
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_address = address
        self.connect()

    def connect(self):
        try:
            self.s.connect(self.recv_address)  # 绑定端口
        except Exception as e:
            # print('[!] Server not found or not open')
            print(e)
            sys.exit(0)
    
    def disconnect(self):
        self.s.close()

    def send(self, info):
        try:
            self.s.sendall(info)
        except IOError as e:
            pass


class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open vidio source, make sure /dev/video{} existed.".format(video_source))

        # 获得摄像头采集的高和宽(注意转成int，否则报错非法指令)
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # 返回获得视频帧成功标记位和RGB格式的frame(读取到的frame是BGR格式)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)


class Frame2JPEG:
    '''视频帧压缩类，压缩为JPEG格式'''

    # quality: 0 ~ 100
    jpeg_quality = 10
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    
    @staticmethod
    def get_jpeg_data(frame):
        '''该方法实现了视频帧的压缩

            Args:
                frame: 视频帧
            
            Return:
                str_encode: bytes类型的数据
        '''
        data_encode = cv2.imencode('.jpg', frame, Frame2JPEG.params)[1]
        str_encode = data_encode.tobytes()
        return str_encode


class FrameRateMonitor:
    '''使用平均采样法统计帧率'''
    def __init__(self):
        self.avg_time = 0.0
        self.alpha = 1 / 100.   # 采样率
        self.last_time = None   # 上一次统计开始时间

    def get_fps(self):
        if self.last_time is None:
            self.last_time = time.time()
            return None
        
        current_time = time.time()
        if self.avg_time == 0:
            self.avg_time = current_time - self.last_time
        else:
            self.avg_time = self.avg_time * (1 - self.alpha) + (current_time - self.last_time) * self.alpha
        self.last_time = current_time

        fps = round(1. / self.avg_time)
        return fps


if __name__ == "__main__":
    #################### 初始化 #################### 

    # address = ('127.0.0.1', 6887)  # 服务端地址和端口
    # address = ('192.168.2.151', 6887)  # 服务端地址和端口
    address = ('192.168.2.166', 6887)  # 服务端地址和端口
    # address = ('10.130.147.227', 6887)  # 服务端地址和端口

    vid = MyVideoCapture(0)

    frame_rate_monitor = FrameRateMonitor()
    #################### 从摄像头读取数据并发送至上位机 #################### 
    while True:
        ret, frame = vid.get_frame()
        if ret == False:
            raise ValueError("Unable to get frame from camera")

        # str_encode = Frame2JPEG.get_jpeg_data(frame)
        # str_encode = cv2.imencode('.jpg', frame)[1].tostring()
        str_encode = frame.tobytes()

        soct = Soct(address)
        soct.send(str_encode)
        # soct.send(pickle.dumps(data))
        # soct.send(str(result[0]))    # 转换成字符串再发送
        # soct.send(str(result[1]))    # 转换成字符串再发送
        # soct.send('Hello world')
        soct.disconnect()

        fps = frame_rate_monitor.get_fps()
        if fps != None:
            print(fps)