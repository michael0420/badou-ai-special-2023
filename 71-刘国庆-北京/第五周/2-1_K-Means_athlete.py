# 导入KMeans类和matplotlib.pyplot库
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 定义包含二维数据点的列表X
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
     [0.1956, 0.4280]]
# 输出数据集X
print(X)
# 创建K均值聚类对象clf，指定聚类数目为3
clf = KMeans(n_clusters=3)
# 输出K均值聚类的完整信息，包括许多省略的参数
print(f"K均值聚类的完整信息:\n{clf}")
# 使用fit_predict方法对数据集X进行拟合和预测，将聚类标签存储在y_pred中
y_pred = clf.fit_predict(X)
# 输出聚类预测结果
print(f"聚类预测结果:\n{y_pred}")
# 通过列表推导式获取数据集的第一列和第二列数据，分别存储在x和y中
x = [n[0] for n in X]
print(f"第一列数据:\n{x}")
y = [n[1] for n in X]
print(f"第二列数据:\n{y}")
# 使用matplotlib.pyplot库的scatter函数绘制散点图，其中c参数指定颜色，marker参数指定标记类型
plt.scatter(x, y, c=y_pred, marker='x')
# 设置图表标题、x轴标签和y轴标签
plt.title("Kmeans-Basketball Data")
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")
# 设置右上角的图例，标记为"A"、"B"、"C"
plt.legend(["A", "B", "C"])
# 显示图形
plt.show()
