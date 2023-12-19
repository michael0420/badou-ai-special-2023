#!/usr/bin/env python
# encoding=utf-8
import cv2
from sklearn.decomposition import PCA
import numpy as np

# X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
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
pca = PCA(n_components=2)
pca.fit(X)
res = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
print(res)
