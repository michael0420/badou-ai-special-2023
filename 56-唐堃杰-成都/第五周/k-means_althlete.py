# 使用k-mean对运动员数据进行分类
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 共20行，两列分别是球员分钟助攻数和分钟的分数
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

if __name__ == '__main__':
    # 设置k-means对象，数据分3类进行聚类
    k_obj = KMeans(n_clusters=3)
    y_predict = k_obj.fit_predict(X)
    print(k_obj)
    print("y_predict = ", y_predict)

    # 分别获取第一二列的数据
    x = [n[0] for n in X]
    y = [n[1] for n in X]

    # 通过预测的聚类与数据绘制散点图
    # 显示中文标签（不设置下面使用的中文title会报错）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 这里marker：o表示圆点，*表示星型，x表示点
    plt.scatter(x, y, c=y_predict, marker='*')
    plt.title("athlete")
    # 绘制x轴与y轴
    plt.xlabel("分钟助攻数")
    plt.ylabel("分钟得分数")
    # 设置右上角图例(如果有多个数据可以填充)
    plt.legend(["A"])
    # 显示图像
    plt.show()