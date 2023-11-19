import numpy as np

class NUMPYPCA():
    def __init__(self,n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        self.n_features_ = X.shape[1]
        #求协方差矩阵
        X = X - X.mean(axis=0)
        self.cov = np.dot(X.T,X) / X.shape[0]
        #求协方差矩阵的特征值和特征向量
        eig_vals,eig_vectors = np.linalg.eig(self.cov)
        # 获得降序排序特征值的序号
        idx = np.argsort(-eig_vals)
        #降维矩阵
        self.components_ = eig_vectors[:,idx[:self.n_components]]
        #对X进行降维
        return np.dot(X,self.components_)

#调用PCA方法
pca_value = NUMPYPCA(n_components=3)
#创建待降维矩阵，维度为4
X = np.array([[10, 15, 29, 2],
              [15, 46, 13, 10],
              [23, 21, 30, 34],
              [11, 9, 35, 12],
              [42, 45, 11, 45],
              [9, 48, 5, 21],
              [11, 21, 14, 21],
              [8, 5, 15, 19],
              [11, 12, 21, 23],
              [21, 20, 25, 14]])
Z = pca_value.fit_transform(X)
print(Z)