# coding=utf-8

import numpy as np
from sklearn.decomposition import PCA

X = np.random.uniform(0, 50, (10, 3))
pca = PCA(n_components=2)
pca.fit(X)
newX = pca.fit_transform(X)
# 输出贡献率
print(pca.explained_variance_ratio_)
# 输出降维后的数据
print(newX)
