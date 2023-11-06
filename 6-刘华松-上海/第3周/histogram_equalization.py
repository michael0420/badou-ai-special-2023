#实现直方图均衡化******************************************

import cv2
import numpy as np

# 读取图像
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 计算灰度直方图
histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))

# 计算累积直方图
cumulative_hist = np.cumsum(histogram)

# 计算灰度级别映射
cumulative_hist_normalized = (cumulative_hist * 255) / cumulative_hist[-1]
gray_levels_mapping = np.round(cumulative_hist_normalized).astype(np.uint8)

# 将灰度级别映射应用于图像
equalized = gray_levels_mapping[image]

# 显示原始图像和均衡化后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized)


#调用高级函数的方式***************************************


import cv2

# 读取图像
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 进行直方图均衡化
equalized = cv2.equalizeHist(image)

# 显示原始图像和均衡化后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized)

