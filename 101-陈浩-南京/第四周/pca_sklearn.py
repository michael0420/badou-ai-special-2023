import  numpy as np
from sklearn.decomposition import  PCA

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
pca = PCA(n_components=1) #设置降维维度
pca.fit(X) #训练
Z = pca.fit_transform(X) #降维后的数据
print(pca.explained_variance_ratio_) #输出贡献率
print(Z) #输出降维后的数据
