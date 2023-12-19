import numpy as np
from sklearn.datasets import load_iris

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        X = X - X.mean(axis = 0)
        self.covariance = np.dot(X.T, X)/X.shape[0]
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        idx = np.argsort(-1*eig_vals)
        self.components = eig_vectors[:, idx[:self.n_components]]
        # idx[:self.n_components]: 这部分是一个切片操作，它从 idx 中选择前 self.n_components 个元素。 [:self.n_components] 部分将选择前 self.n_components 个元素的子数组。
        # 将 eig_vectors 中的特征向量的一个子集（前 self.n_components 个特征向量）赋值给 self.components_。
        return np.dot(X, self.components)

pca = PCA(n_components=2)
iris = load_iris()
X = iris.data
newX = pca.fit_transform(X)
print(newX)