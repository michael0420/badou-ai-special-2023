import numpy as np
from sklearn.decomposition import PCA


X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
'''
pca = PCA(n_components=2) #降到2维
pca.fit(X)
newX = pca.fit_transform(X) #降维
print(pca.explained_variance_ratio_) #输出贡献率
print(newX)
'''

pca = PCA(n_components=2)

X = X - X.mean(axis=0)   #X的每一列减去该列的均值,中心化

convariance = np.dot(X.T, X)/X.shape[0] #协方差矩阵

eig_vals, eig_vectors = np.linalg.eig(convariance) #eig_vals: 特征值  eig_vectors：特征向量

idx = np.argsort(-eig_vals)

components = eig_vectors[:, idx[: 2]]

result = np.dot(X, components)

print(result)