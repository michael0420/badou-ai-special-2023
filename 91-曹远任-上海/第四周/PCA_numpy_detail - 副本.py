# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PCA_detail(X, K):
    # 中心化
    X_centre = X - np.mean(X, axis=0)
    # 求协方差矩阵
    X_cov = np.dot(X_centre.T, X_centre) / (X_centre.shape[0] - 1)
    # 求协方差矩阵的特征值和特征向量
    a, b = np.linalg.eig(X_cov)
    # 使用最大的K个特征值对应的特征向量作为列向量，组成变换矩阵
    a_sorti = np.argsort(-1 * a)
    global W
    W = np.array([b[:, a_sorti[i]] for i in range(K)]).T
    sum1 = np.sum([a[a_sorti[i]] for i in range(K)])
    sum2 = np.sum(a)
    print("主成分占比：", sum1 / sum2)
    # 原数据乘变换矩阵
    return np.dot(X, W)


# 随机生成样本
X = np.random.uniform(0, 50, (10, 3))
K = np.shape(X)[1] - 1
X_pca = PCA_detail(X, K)
print("原数据：\n", X, "\nPCA后数据：\n", X_pca)

# 以下是画图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.set_title('original data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.quiver(np.mean(X, axis=0)[0], np.mean(X, axis=0)[1], np.mean(X, axis=0)[2], W[0][0] * 10, W[1][0] * 10, W[2][0] * 10, color=(1, 0, 0, 0.5))
ax.quiver(np.mean(X, axis=0)[0], np.mean(X, axis=0)[1], np.mean(X, axis=0)[2], W[0][1] * 10, W[1][1] * 10, W[2][1] * 10, color=(1, 0, 0, 0.5))

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.scatter(X_pca[:, 0], X_pca[:, 1])
ax.set_title('PCA data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()
