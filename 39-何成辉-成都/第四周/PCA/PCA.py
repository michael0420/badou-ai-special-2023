# encoding=gbk

"""
@author: BraHitYQ
Use PCA (Principal Component Analysis) to reduce the dimensionality of the Iris dataset and visualize the results.(使用PCA（主成分分析）对鸢尾花数据集进行降维，并将结果可视化)
"""

"""
scatter函数的参数如下：
    x：x轴上的数据点。
    y：y轴上的数据点。
    s：散点的大小，默认为None。
    c：散点的颜色，默认为None。
    marker：散点的标记样式，默认为None。
    cmap：颜色映射，默认为None。
    norm：归一化对象，默认为None。
    vmin：颜色映射的最小值，默认为None。
    vmax：颜色映射的最大值，默认为None。
    alpha：散点的透明度，默认为None。
    linewidths：散点的线宽，默认为None。
    edgecolors：散点的边缘颜色，默认为None。
    plotnonfinite：是否绘制非有限值的数据，默认为False。
    data：数据源，默认为None。
    **kwargs：其他关键字参数。
"""

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris  # 从sklearn库的datasets模块中导入load_iris函数，用于加载鸢尾花数据集。

x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的属性数据，y表示数据标签,调用load_iris函数加载鸢尾花数据集，并将特征矩阵x和目标向量y分别赋值给变量x和y。
pca = dp.PCA(n_components=2)  # 加载pca算法,创建一个PCA对象,设置降维后主成分数目为2
reduced_x = pca.fit_transform(x)  # 使用PCA对象对特征矩阵x进行降维，将降维后的结果赋值给变量reduced_x。
red_x, red_y = [], []  # 创建2个空列表，用于存储不同类别的降维后的特征值
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_x)):  # 遍历降维后的特征矩阵reduced_x的长度.（按鸢尾花的类别将降维后的数据点保存在不同的表中）
    if y[i] == 0:  # 判断目标向量y的第i个元素是否等于0。
        red_x.append(reduced_x[i][0])  # 如果等于0，则将降维后的特征值的第一个分量添加到red_x列表中。
        red_y.append(reduced_x[i][1])  # 如果等于0，则将降维后的特征值的第二个分量添加到red_y列表中。
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])  # 如果等于1，则将降维后的特征值的第一个分量添加到blue_x列表中。
        blue_y.append(reduced_x[i][1])  # 如果等于1，则将降维后的特征值的第二个分量添加到blue_y列表中。
    else:
        green_x.append(reduced_x[i][0])  # 将降维后的特征值的第一个分量添加到green_x列表中。
        green_y.append(reduced_x[i][1])  # 将降维后的特征值的第二个分量添加到green_y列表中。
plt.scatter(red_x, red_y, c='r', marker='x')  # 使用matplotlib的scatter函数绘制红色x标记的散点图，表示类别0的数据。
plt.scatter(blue_x, blue_y, c='b', marker='D')  # 使用matplotlib的scatter函数绘制蓝色D标记的散点图，表示类别1的数据。
plt.scatter(green_x, green_y, c='g', marker='.')  # 使用matplotlib的scatter函数绘制绿色点标记的散点图，表示类别2的数据。
plt.show()
