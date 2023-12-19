#!/usr/bin/env python
# encoding=gbk

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

"""
PCA是一种常用的降维技术，它可以对高维数据进行降维，并将其转换为低维空间，以便于可视化和分析
"""

# 加载经典的鸢尾花数据集，并且返回输入特征矩阵X和目标变量y
x, y = load_iris(return_X_y=True)
# 加载pca算法，设置降维后主成分数目为2
pca = dp.PCA(n_components=2)
# 对原始数据进行降维，保存在reduced_x中
reduced_x = pca.fit_transform(x)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

# 按鸢尾花的类别将降维后的数据点保存在不同的表中
for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

# 绘制散点图
# marker='x'是用来设置散点图中数据点的标记形状的参数
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
