#!/usr/bin/env python
# encoding=utf-8
import cv2
import sklearn.decomposition as dp
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# return_X_y=True 表示返回元组（x, y）False则返回Bunch对象
x, y = load_iris(return_X_y=True)
pca = dp.PCA(n_components=2)
pca.fit(x)
res = pca.fit_transform(x)
red_x, red_y = [], []
green_x, green_y = [], []
blue_x, blue_y = [], []
print(res)

for i in range(len(res)):
    # print(x[i])
    if y[i] == 0:
        red_x.append(res[i][0])
        red_y.append(res[i][1])
    elif y[i] == 1:
        green_x.append(res[i][0])
        green_y.append(res[i][1])
    elif y[i] == 2:
        blue_x.append(res[i][0])
        blue_y.append(res[i][1])

plt.scatter(red_x, red_y, c='r', marker='o')
plt.scatter(green_x, green_y, c='g', marker='s')
plt.scatter(blue_x, blue_y, c='b', marker='d')
plt.show()
