'''
K-Means聚类
1、	确定K值，即将数据聚集成K个类簇或小组
2、	从数据集中随机选择K个数据点作为质心（Centroid）或数据中心
3、	分别计算每个点到每个质心之间的距离，并将每个点划分到离最近质心的小组。
4、	当每个质心都聚集了一些点后，重新定义算法选出新的质心。（对于每个簇，计算其均值，即得到新的K个质心点）
5、	迭代执行第三步到第四步，直到迭代终止条件满足为止（聚类结果不再变化）
'''

from sklearn.cluster import KMeans

"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示玩家每局击杀数：assault_per_minute
第二列表示玩家每局死亡数：dead_per_minute
第三列表示玩家每局助攻数：assists_per_minute
"""
X = [[6,11,13],
     [10,11,10],
     [4,8,19],
     [9,11,10],
     [0,11,17],
     [6,4,24],
     [16,9,19],
     [8,6,29],
     [14,5,21],
     [7,5,24]
     ]

print(X)

#第二部分：KMeans聚类
clf = KMeans(n_clusters=3,n_init='auto')                                           # clf即赋值为KMeans；n_clusters=3) 表示类簇数为3，聚成3类数据；n_init='auto'是将KMeans改为自动运行模式，随着KM1.4版本更新，这样就能避免掉Futurewarning的报错
y_pred = clf.fit_predict(X)                                           # 载入数据集X，并且将聚类的结果赋值给y_pred

print(clf)                                                            # 输出完整Kmeans函数

print("y_pred = ", y_pred)                                            # 输出聚类预测结果


#第三部分：可视化绘图

import numpy as np
import matplotlib.pyplot as plt

x = [n[0] for n in X]                                                # 获取数据集的第一列//第三列 使用for循环获取 n[0]表示X第一列
print(x)
y = [n[2] for n in X]                                                # n[0]表示X的第三列
print(y)

''' 
绘制散点图 
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
plt.scatter(x, y ,c=y_pred, marker='x')

plt.title("Kmeans-Gameplayer Data")                                      # 绘制标题

plt.xlabel("assault_per_minute")                                         # 绘制x轴和y轴坐标
plt.ylabel("assists_per_minute")

plt.legend(["A", "B", "C"])                                              # 设置右上角图例

plt.show()                                                               # 显示图形

from warnings import simplefilter

simplefilter("ignore", category=FutureWarning)