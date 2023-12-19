#!/usr/bin/env python
# encoding=gbk
# from pca_np import PCA
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的属性数据，y表示数据标签

pca = dp.PCA(n_components=2)
reduce_x = pca.fit_transform(x)


red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduce_x)):
    if y[i] == 0:
        red_x.append(reduce_x[i][0])
        red_y.append(reduce_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduce_x[i][0])
        blue_y.append(reduce_x[i][1])
    else:
        green_x.append(reduce_x[i][0])
        green_y.append(reduce_x[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
