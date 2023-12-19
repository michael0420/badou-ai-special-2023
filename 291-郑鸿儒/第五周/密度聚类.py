#!/usr/bin/env python
# encoding=utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris


iris = load_iris()
# 只取四维，但其实只有四维
data = iris.data[:, :4]
# print(iris.data)
dbscan = DBSCAN(eps=0.4, min_samples=9)
# print(dbscan)
dbscan.fit(data)
label_pred = dbscan.labels_
# print(label_pred)
# 这里的分类数量应该是不知道的
labels_arr = [x for i, x in enumerate(label_pred) if np.where(label_pred == x)[0][0] == i and x >= 0]
# print(labels_arr)
# print(np.argsort(labels_arr))
markers = ["x", "*", "o"]
color = ["red", "green", "blue"]
for i in np.argsort(labels_arr):
    x = data[label_pred == labels_arr[i]]
    plt.scatter(x[:, 0], x[:, 1], c=color[i], marker=markers[i], label="label" + str(i))
plt.xlabel("speal length")
plt.ylabel("speal width")
plt.legend(loc="upper right")
    # print(x[:, 0])
plt.show()
