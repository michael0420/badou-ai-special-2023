# -*- coding:utf-8 -*-
"""
实现灰度化和二值化
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#原图
img = cv2.imread('lenna.png',1)
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
plt.subplot(311)
plt.title('img')
plt.imshow(img)

#灰度图
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(312)
plt.title('img_gray')
plt.imshow(img_gray,cmap='gray')

#二值图
img_binary = np.where(img_gray>=128,1,0)
plt.subplot(313)
plt.title('img_binary')
plt.imshow(img_binary, cmap='gray')

#显示图像
plt.show()
