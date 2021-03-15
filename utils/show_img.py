#-*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

cnt = 0
up = 5

for root, dir, files in os.walk('images'):
    for filename in files:
        path = os.path.join(root, filename)
        img = np.load(path)

        plt.imshow(img, cmap='gray')
        plt.title('label: ' + filename[-5])
        plt.show()
        plt.close()
        
        cnt += 1
        if cnt == up:
            break
