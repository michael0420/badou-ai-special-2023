import numpy as np
import matplotlib.pyplot as plt


def Normalization1(x):
    # 归一化（0~1）
    # x_=(x−x_min)/(x_max−x_min)
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def Normalization2(x):
    # 归一化（-1~1）
    # x_=(x−x_mean)/(x_max−x_min)
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]


# 标准化
def z_score(x):
    # x=(x−μ)/σ
    return [(i - np.mean(x)) / np.std(x) for i in x]


data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
        11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
counts = [data.count(d) for d in data]
n1 = Normalization1(data)
n2 = Normalization2(data)
z = z_score(data)

plt.plot(data, counts)
plt.plot(n1, counts)
plt.plot(n2, counts)
plt.plot(z, counts)
plt.legend(("original data", "0~1", "-1~1", "z_score"))
plt.show()
