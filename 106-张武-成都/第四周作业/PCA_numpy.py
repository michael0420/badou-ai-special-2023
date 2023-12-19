import numpy as np

"""
numpy使用方式
"""
class PCA:
    def __init__(self, K):
        # 降维矩阵维度K
        self.K = K

    def fit_transform(self, X):
        # 样本矩阵X中心化,使均值为0 公式: 原始样本集的每一纬度的值 - 对应纬度的均值  axis对0纬求均值
        centerX = X - X.mean(axis=0)  # X.shape = (10, 4)
        # 中心化后的协方差矩阵
        covariance = np.dot(centerX.T, centerX)/centerX.shape[0]  # covariance.shape = (4, 4)
        # 特征向量和特征值
        eig_values, eig_vectors = np.linalg.eig(covariance)
        print(eig_values, eig_vectors)
        # 降维转换矩阵 对特征值降序排序后,取前K个特征值对应的特征向量
        idx = np.argsort(-eig_values)
        matrix = eig_vectors[:, idx[:self.K]]
        # 原始矩阵 * 降维转换矩阵
        return np.dot(X, matrix)


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
pca = PCA(K=2)
newX = pca.fit_transform(X)
print(newX)

