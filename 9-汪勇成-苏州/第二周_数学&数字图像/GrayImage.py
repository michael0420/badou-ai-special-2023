# -*-coding: utf-8 -*-
"""
彩色图像的灰度化，二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[0:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]   # 当前图像长宽的坐标
        img_gray[i, j] = int(m[0]*0.11+m[1]*0.59+m[3]*0.)   # 将BGR左边转化位gray坐标并赋值给新图片
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)

plt.subplots(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("------image lenna")
print(img)

# 灰度化方法2 调用skimage
img_gray = rgb2gray(img)
plt.subplots(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray---")
print(img_gray)

# 二值化方法1
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

#方法2
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("----image binary----")
print(img_binary)
print(img_binary.shape)

plt.subplots(223)
plt.imshow(img_binary, cmap='gray')
plt.show()

