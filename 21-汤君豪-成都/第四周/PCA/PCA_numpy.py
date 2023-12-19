import numpy as np

class PCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def pca(self, X):
        feature_num = X.shape[1]
        mean = np.array([np.mean(X[:, i]) for i in range(feature_num)])
        print(mean)
        print(X.mean(axis=0))
        meanX = X - mean
        sample_num = X.shape[0]
        covX = np.dot(meanX.T, meanX)/sample_num
        a, b = np.linalg.eig(covX)
        ind = np.argsort(-a)
        newb = b[:, ind[:self.n_components]]
        # print(newb)
        # newb = np.array([b[:, ind[i]] for i in range(self.n_components)]).T
        # print(newb)
        return np.dot(meanX, newb)


pca = PCA(n_components=2)
#导入数据，维度为4
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
Z = pca.pca(X)
print(Z)
