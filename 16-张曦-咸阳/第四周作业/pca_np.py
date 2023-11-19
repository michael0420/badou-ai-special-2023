import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # 1、去中心化
        X = X - np.mean(X, axis=0)
        # 2.求协方差矩阵
        self._cov = np.dot(X.T, X) / (X.shape[0]-1)
        # 3. 求特征值和特征矩阵
        eig_v, eig_t = np.linalg.eig(self._cov)
        # 4. 得到降维特征矩阵
        index = np.argsort(-1 * eig_v)
        self._t = eig_t[:, index[:self.n_components]]
        # 5. 对X求降维矩阵
        return np.dot(X, self._t)


pca = PCA(n_components=2)
X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
result = pca.fit_transform(X)
print("custom \n", result)


import sklearn.decomposition as dp

pca=dp.PCA(n_components=2) #加载pca算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(X) #对原始数据进行降维，保存在reduced_x中
print("sklearn \n",reduced_x)