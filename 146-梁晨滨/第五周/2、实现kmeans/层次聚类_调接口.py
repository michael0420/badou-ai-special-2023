###cluster.py
#导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
import numpy as np

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''
# 1、创建数据集
# 随机创建一个随机数据集，一共20个点，在一张100X100的坐标系上
input_points = np.random.randint(0, 100, size=(10, 2))

# 2、linkage聚类出所有层次聚类， ward是离差平方和距离
# linkage聚类方法d(x, y) = sqrt[ (|y| + |s|/T)d(y, s)^2 + (|y| + |t|/T)d(y, t)^2 + (|y| + |s|/T)d(s, t)^2) ]
Z = linkage(input_points, 'ward')
print(Z)
# fcluster聚类,提取出linkage聚类中阈值设定为6的图
f = fcluster(Z, 6, 'distance')
# dendrogram()是绘制聚类树图函数
dn = dendrogram(Z)
plt.show()
