import numpy as np
from sklearn.decomposition import PCA

X = np.array([[21, 36, 17, 19],
                  [51, 6, 26, 12],
                  [37, 14, 36, 61],
                  [2, 89, 41, 70],
                  [33, 59, 10, 20],
                  [52, 21, 89, 27],
                  [9, 46, 29, 38],
                  [9, 2, 39, 66],
                  [81, 29, 40, 128],
                  [33, 56, 6, 96]])
k = X.shape[1]-2
pca = PCA(n_components=k)
pca.fit(X)
NewX = pca.fit_transform(X)  # 降维后的数据
print(NewX)
