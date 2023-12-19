import cv2
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])
# 创建PCA对象,指定主成分的数量
pca=PCA(n_components=2)
# 降维
res_Data = pca.fit_transform(X)
print(res_Data)
