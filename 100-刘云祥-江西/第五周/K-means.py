import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 计算欧氏距离
def CalcDis(DataSet, Centroids, k):
    CalList = []
    for data in DataSet:
        diff = np.tile(data, (k, 1)) - Centroids
        SquaredDiff = diff**2
        SquareDist = np.sum(SquaredDiff, axis=1)  #按列相加
        Distance = SquareDist*0.5
        CalList.append(Distance)
    CalList = np.array(CalList)
    return CalList
# 计算质心
def Classify(DataSet, Centroids, k):
    CalList = CalcDis(DataSet, Centroids, k)
    MinDistIndices = np.argmin(CalList, axis=1)  # 按列的方式得到最小值的序号
    A = pd.DataFrame(DataSet)
    NewCentroids = pd.DataFrame(DataSet).groupby(MinDistIndices).mean()
    NewCentroids = NewCentroids.values
    Changed = NewCentroids - Centroids
    return Changed, NewCentroids
# k-means分类
def Kmeans(DataSet, k):
    Centroids = random.sample(DataSet, k)
    Changed, NewCentroid = Classify(DataSet, Centroids, k)
    while np.any(Changed != 0):  # 终止条件  质心不变
        Changed, NewCentroid = Classify(DataSet, NewCentroid, k)
    Centroids = sorted(NewCentroid.tolist())

    Cluster = []
    CalList = CalcDis(DataSet, Centroids, k)
    MinDistIndices = np.argmin(CalList, axis=1)
    for i in range(k):
        Cluster.append([])
    for i, j in enumerate(MinDistIndices):
        Cluster[j].append(DataSet[i])
    return Centroids, Cluster

if __name__=='__main__':
    DataSet = [[1, 1], [1, 2], [2, 1], [6, 4], [6, 3], [5, 4]]
    Centroids, Cluster = Kmeans(DataSet, 2)
    for i in range(len(DataSet)):
        plt.scatter(DataSet[i][0], DataSet[i][1], marker='o', color='red', s=30, label='原始点')

    for j in range(len(Centroids)):
        plt.scatter(Centroids[j][0], Centroids[j][1], marker='x', color='green', s=60, label='质心')
    plt.show()







