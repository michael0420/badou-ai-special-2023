# 导入相应的包,从scipy库的cluster.hierarchy模块中导入dendrogram、linkage和fcluster函数。
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
    1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。 若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
    2. method是指计算类间距离的方法。

fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
    1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
    2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, 'ward')

# 根据Z中的聚类距离，将聚类结果分为4个簇，将结果存储在变量f中。
f = fcluster(Z,4,'distance')

# 创建一个大小为5x3的图形对象，并将其存储在变量fig中。
fig = plt.figure(figsize=(5, 3))

# 绘制Z中的聚类树状图，并将结果存储在变量dn中。
# dendrogram(),这个函数是用来绘制树状图的，它接受一个链接矩阵Z作为输入，并返回一个matplotlib的Axes对象。
dn = dendrogram(Z)
print(Z)
plt.show()
