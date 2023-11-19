# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 随机生成数据
X = np.random.uniform(0, 35, (50, 2))
X2 = np.random.uniform(30, 50, (50, 2))
X = np.vstack((X, X2))
X = np.float32(np.array(X))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 1])
ax.set_title('data')
plt.show()

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 设置起始中心
flags = cv2.KMEANS_RANDOM_CENTERS
# K-Means聚类 聚集成K类
K = 3
compactness, labels, centers = cv2.kmeans(X, K, None, criteria, 10, flags)
# 画聚类后的图像
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(100):
    ax.scatter(X[i][0], X[i][1], color=colors[labels[i][0]])
ax.set_title('data')
plt.show()
