from sklearn.cluster import  KMeans

"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""

X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]

#打印数据集
print (X)

# Kmeans 聚类，聚合成3类
clf = KMeans(n_clusters=3) #设置类簇数
y_predict = clf.fit_predict(X) #执行聚类操作

print(clf) #输出Kmeans 函数
print('y_predict =',y_predict) #输出聚类预测结果

#将聚类结果可视化展示
import numpy as np
import matplotlib.pyplot as plt

#获取数据集的第一列和第二列 使用for 循环获取 n[0]表示X第一列
x = [n[0] for n in X]
print(x)
y = [n[1] for n in X]
print(y)

#绘制图像，散点图
plt.scatter(x, y, c=y_predict, marker='x')

#绘制标题
plt.title("Kmeans-Basketball Data")

#绘制x和y坐标轴
plt.xlabel('asist')
plt.ylabel('points')

#设置右上角图例
plt.legend(["A","B","C"])

plt.show()