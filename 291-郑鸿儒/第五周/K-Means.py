#!/usr/bin/env python
# encoding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna.png", 0)
rows, cols = img.shape
data = img.reshape((rows * cols, 1))
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

"""
cv2.kmeans() 求k均值聚类
Args
    data(float32 ndarray): 每行表示一个样本
    k(int): 聚类簇数
    bestLabels(ndarray | None): 表示输出的整数数组，用于存储每个样本的聚类标签索引， 默认None
    criteria(tuple): (type, max_iter, epsilon)
        type: 
            - cv2.TERM_CRITERIA_EPS: 精度满足epsilon停止
            - cv2.TERM_CRITERIA_MAX_ITER: 迭代次数满足max_iter停止
            - cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        max_iter: 迭代次数
        epsilon: 迭代精度
    attempts(int): 重复实验看means算法的次数
    flag(string):
        - cv2.KMEANS_RANDOM_CENTERS: 随机选取
        - cv2.KMEANS_PP_CENTERS: 使用Kmeans++ 算法初始化中心
        - cv2.KMEANS_USR_INITIAL_LABELS: 使用初始标签调整聚类中心
Returns
    ret(float32): 聚类的总体误差
    labels: 结果标记
    centers: 每个簇的中心坐标组成的数组
"""
ret, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, flags)
print(ret, labels, centers)
plt.figure()
# 使plt正常显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.subplot(1, 2, 1), plt.title("原始图像"), plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.title("聚类图像"), plt.imshow(labels.reshape(img.shape), cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

