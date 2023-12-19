# coding=utf-8
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']

# 1、创建数据集
# 随机创建一个随机数据集，一共20个点，在一张100X100的坐标系上
input_points = np.random.randint(0, 100, size=(30, 2))

# 2、聚类
# 3个类别，clf相当于设定好聚类的中心的model
# model_kmeans.fit_predict(input)对input数据进行聚类应用，
# 并用fit_predict()返回每个数的分类标签
model_kmeans = KMeans(n_clusters=3)
predict = model_kmeans.fit_predict(input_points)
 
# 3、可视化
# 把每个点的横纵坐标值记录下来
x_list = []
y_list = []
for i in input_points:
     x_list.append(i[0])
     y_list.append(i[1])
# 按照坐标绘图
plt.scatter(x_list, y_list, c=predict, marker='x')

plt.title("聚类图")
plt.xlabel("x 轴")
plt.ylabel("y 轴")
plt.legend(["A", "B", "C"])
plt.show()
