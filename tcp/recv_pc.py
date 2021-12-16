#-*- encoding: utf-8 -*-

import socket
import numpy as np
import cv2
import time
import sys

class Soct(object):
    def __init__(self, address):
        # buffer_size = 16
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.bind(address)  # 绑定端口
            self.s.listen(5)

        except Exception as e:
            # print('[!] Server not found or not open')
            print(e)
            sys.exit(0)

        self.buffer_size = 640 * 480 * 3

    def wait_for_connection(self):
        self.conn, addr = self.s.accept()
        print("...connected from", addr)
    
    def __del__(self):
        self.s.close()

    def recv(self):
        total_data = bytes()
        while True:
            data = self.conn.recv(self.buffer_size)
            if not data:
                break
            total_data += data
        return total_data


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
        

if __name__ == '__main__':
    HOST, PORT = "192.168.177.106", 6887     # 这里localhost务必换成本机ip地址，否则一般外部无法访问！！！！
    print('waiting for video...')

    soct = Soct((HOST, PORT))
    frame_rate_monitor = FrameRateMonitor()

    while True:
        soct.wait_for_connection()
        receive_data = soct.recv()

        image = np.frombuffer(receive_data, np.uint8)
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = image.reshape(200,200,3)
        # image = image.reshape(120,160)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        fps = frame_rate_monitor.get_fps()
        if fps != None:
            print(fps)

        cv2.imshow('video', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()