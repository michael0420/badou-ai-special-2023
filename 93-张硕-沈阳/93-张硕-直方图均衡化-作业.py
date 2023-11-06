import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy import shape
import random

#直方图均衡化代码实现
img = cv2.imread('lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h = img.shape[0]
w = img.shape[1]

pixel_counts = {}

#获取像素值，数量 对应的字典
for i in range(h):
    for j in range(w):
        num = img[i, j]
        if num in pixel_counts:
            pixel_counts[num] += 1
        else:
            pixel_counts[num] = 1
#将字典按照像素值排序
pixel_counts1 = dict(sorted(pixel_counts.items()))

#获取原来像素值和计算后的像素值对应关系
xiangsu = {}
sumpi = 0
for num,count in pixel_counts1.items():
    #print(num, count)
    n1 = count / (h * w)
    sumpi += n1
    q = int(sumpi * 256 - 1)
    xiangsu[num] = q

#根据对应关系生成新图
img1 = np.zeros([h, w], dtype=np.uint8)
for i in range(h):
    for j in range(w):
        old = img[i, j]
        img1[i, j] = xiangsu[old]

cv2.imshow('test', np.hstack([img,img1]))
cv2.waitKey(0)

plt.Figure()
plt.hist(img1.ravel(), 256)
plt.show()

'''
#直方图均衡化，调用模块
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst_img = cv2.equalizeHist(gray)
'''

'''
#彩色直方图均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

result = cv2.merge((bH, gH, rH))
cv2.imshow('dst_rgb',np.hstack([result, img]))
cv2.waitKey(0)
'''