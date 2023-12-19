import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 创建数据集
X = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])

# 定义聚类数量k
k = 2

# 创建kMeans对象，拟合数据
"""
通过实例化KMeans类并传入n_clusters参数的值为k,我们创建了一个KMeans对象kmeans，这个对象将用于执行k-means聚类算法。
fit方法用于拟合数据集X，执行K-means算法的主要步骤。
它采用输入数据集X，根据指定的聚类数量K，并使用迭代优化的方法，不断寻找最优的聚类中心。
"""
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)


# 获取聚类中心和数据点的分配
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

print("聚类中心点：")
print(cluster_centers)


print("数据点的分配：")
print(labels)




# 可视化数据点和聚类中心点
plt.scatter(X[:,0],X[:,1],c = labels,cmap='viridis',alpha=0.5)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],marker='x',color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering')
plt.show()

