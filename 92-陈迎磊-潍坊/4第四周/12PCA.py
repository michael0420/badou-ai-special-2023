import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
pca=PCA(n_components=3)    #设置参数，降到3维
pca.fit(X)                 #执行PCA
newX=pca.fit_transform(X)  #降维后的数据
print(pca.explained_variance_ratio_)  #输出贡献率
print(newX)                  #输出降维后的数据






