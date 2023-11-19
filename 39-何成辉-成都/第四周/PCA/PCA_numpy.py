# coding=utf-8


"""
@author: BraHitYQ
Implementation of Principal Component Analysis (PCA)主成分分析（PCA）的实现
"""

import numpy as np


class PCA():
    # 用于设置主成分的数量
    def __init__(self,n_components):
        self.n_components = n_components

    # 用于对输入的数据进行降维处理。
    def fit_transform(self, X):
        # 我们首先获取输入数据的维度self.n_features_.
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        # 我们对输入数据进行中心化处理，即减去每个特征的均值。然后，我们计算数据的协方差矩阵self.covariance
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X)/X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        # 我们使用NumPy的线性代数模块np.linalg中的eig函数来计算协方差矩阵的特征值和特征向量。这些特征值和特征向量将用于确定主成分。
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 我们根据特征值的大小对特征值进行排序，并获取排序后的索引。
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        # 我们选择前self.n_components个最大的特征值对应的特征向量作为主成分。这些主成分将用于降维。
        self.components_ = eig_vectors[:,idx[:self.n_components]]
        # 对X进行降维
        # 我们将输入数据与主成分相乘，得到降维后的数据。这个降维后的数据将作为方法的返回值。
        return np.dot(X, self.components_)
 
# 调用


'''
    创建了一个PCA类的实例pca，并设置了主成分的数量为2。然后，我们定义了一个输入数据矩阵X，并调用fit_transform方法对其进行降维处理。最后，我们打印出降维后的数据newX。
'''


pca = PCA(n_components=2)
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
newX = pca.fit_transform(X)
print(newX)                  # 输出降维后的数据
