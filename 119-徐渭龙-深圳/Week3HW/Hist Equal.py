'''
直方图均衡化
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('lenna.png',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray_equ = cv2.equalizeHist(gray)  # 均衡化


'''
函数模型： calcHist(images,channels,mask,histSize,ranges,hist=None,accumulate=None)
images: 图像矩阵，例如：[images]
channels:通道数,例如:0
mask:掩膜,一般为None
histSize:直方图大小，一般等于灰度级数
ranges:横轴范围. 
'''
hist = cv2.calcHist([gray_equ],[0],None,[256],[0,256])

plt.figure("gray_equ")
plt.hist(gray_equ.ravel(),256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray,gray_equ]))
cv2.waitKey(0)