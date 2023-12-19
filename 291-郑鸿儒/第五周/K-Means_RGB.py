#!/usr/bin/env python
# encoding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("./img/lenna.png", 1)
# 保留三通道
data = img.reshape((-1, 3))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

ret2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
re4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
re8, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
re16, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
re32, labels32, centers32 = cv2.kmeans(data, 32, None, criteria, 10, flags)
re64, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# centers2[labels2.flatten()] 按聚类的标签，取中心值代替聚类所有值
img_clust_2 = np.uint8(centers2[labels2.flatten()].reshape(img.shape))
img_clust_4 = np.uint8(centers4[labels4.flatten()].reshape(img.shape))
img_clust_8 = np.uint8(centers8[labels8.flatten()].reshape(img.shape))
img_clust_16 = np.uint8(centers16[labels16.flatten()].reshape(img.shape))
img_clust_32 = np.uint8(centers32[labels32.flatten()].reshape(img.shape))
img_clust_64 = np.uint8(centers64[labels64.flatten()].reshape(img.shape))

titles = ["原始图像", "聚类图像k=2", "聚类图像k=4", "聚类图像k=8", "聚类图像k=16", "聚类图像k=32", "聚类图像k=64"]
images = [img, img_clust_2, img_clust_4, img_clust_8, img_clust_16, img_clust_32, img_clust_64]
plt.figure(figsize=(12, 9))
plt.rcParams["font.sans-serif"] = ["SimHei"]
for i in range(7):
    plt.subplot(3, 3, i + 1)
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))

plt.show()
