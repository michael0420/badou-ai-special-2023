#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = [u'SimHei']

img_input = cv2.imread("lenna.png", 1)

# cv2的彩色转灰度图
img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
output = cv2.Canny(img_gray, 250, 300)

plt.subplot(121), plt.title("原始灰度图")
plt.imshow(img_gray, cmap='gray')
plt.subplot(122), plt.title("原始灰度图")
plt.imshow(output, cmap='gray')
plt.show()