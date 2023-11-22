import numpy as np
from sklearn.decomposition import PCA


X = np.array([
    [10, 15, 29, 33],
    [15, 46, 13, 66],
    [23, 21, 30, 44],
    [11, 9, 35, 66],
    [42, 45, 11, 22],
    [9, 48, 5, 17],
    [11, 21, 14, 55],
    [8, 5, 15, 88],
    [11, 12, 21, 99],
    [21, 20, 25, 25]
])
pca = PCA(n_components=2)
# pca.fit(X)
# 用X拟合模型并对X应用降维。
newX = pca.fit_transform(X)
print(newX)
print(pca.explained_variance_ratio_)# 贡献率
