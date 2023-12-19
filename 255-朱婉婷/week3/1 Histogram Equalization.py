# -*- coding:utf8 -*-

""""
直方图均衡化：函数equalizeHist
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#灰度图像直方图均衡化
img_gray = cv2.imread('lenna.png',0)
dst_gray = cv2.equalizeHist(img_gray)

plt.figure()
plt.hist(dst_gray.ravel(),256)
plt.show()

cv2.imshow('Histogram Equalization', np.hstack([img_gray,dst_gray]))
cv2.waitKey(0)

# 彩色图像直方图均衡化
img = cv2.imread('lenna.png')
b,g,r = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

cv2.imshow('r_g_b_equlizeHist',np.hstack([rH,gH,bH]))
cv2.waitKey(0)

#合并通道
result = cv2.merge([rH,gH,bH])
cv2.imshow('rgb_equlizeHist',result)
cv2.waitKey(0)