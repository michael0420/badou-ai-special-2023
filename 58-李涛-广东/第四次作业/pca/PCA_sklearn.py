# coding=utf-8

import numpy as np
from sklearn.decomposition import PCA

"""
pca.fit_transform和pca.fit方法的区别
fit：方法用于计算PCA模型的主成分，它仅对数据进行拟合，不进行转换。在调用fit方法后，PCA模型会根据输入数据的特征进行计算，并存储计算得到的主成分信息.
fit_transform：方法则是将数据同时进行拟合和转换的方法。它首先根据输入数据进行拟合，然后将输入数据转换为降维后的表示
"""

X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4

pca = PCA(n_components=2)  # 降到2维
pca.fit(X)  # 训练
newX = pca.fit_transform(X)  # 降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  # 输出贡献率
print(newX)  # 输出降维后的数据
