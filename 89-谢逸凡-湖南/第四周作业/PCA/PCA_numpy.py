import numpy as np
class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        #求协方差矩阵
        X = X - X.mean(axis = 0)
        self.covariance = np.dot(X.T, X)/X.shape[0]
        #求协方差矩阵对应的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        #获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        #降维矩阵
        self.n_components_ = eig_vectors[:, idx[:self.n_components]]
        #对X进行降维
        return np.dot(X, self.n_components_)
    
pca = PCA(n_components=2)
X = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6],[7,8,9,10]])
newX = pca.fit_transform(X)
print(newX)