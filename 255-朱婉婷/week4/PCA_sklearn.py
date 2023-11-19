
""""
PCA使用接口实现
"""
import numpy as np
from sklearn.decomposition import PCA 

X = np.array([[-1,2,66,-1], 
              [-2,6,58,-1], 
              [-3,8,45,-2], 
              [1,9,36,1], 
              [2,10,62,1], 
              [3,5,83,2]])
pca = PCA(n_components=2)
pca.fit(X)
new_X = pca.fit_transform(X)
print('原矩阵X为：\n',X)
print('降维后矩阵new_X为：\n',new_X)