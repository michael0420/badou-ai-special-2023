import numpy as np  # 导入NumPy库，用于科学计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图


# 归一化的两种方式
def Normalization1(x):
    """归一化（0~1）"""
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def Normalization2(x):
    """归一化（-1~1）"""
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(j) - np.mean(x)) / (max(x) - min(x)) for j in x]


# 标准化
def z_score(x):
    """x∗=(x−μ)/σ"""
    x_mean = np.mean(x)
    s2 = sum([(k - x_mean) * (k - x_mean) for k in x]) / len(x)
    return [(float(k) - x_mean) / s2 for k in x]


# 数据
data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
        11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]  # 定义数据列表

# 数据处理
cs = []
for i in data:
    # 计算元素在列表中出现的次数
    c = data.count(i)
    cs.append(c)
print(f"元素在列表中出现的次数{cs}")

# 归一化和标准化
# 使用Normalization2函数进行归一化
n = Normalization2(data)
# 使用z_score函数进行标准化
z = z_score(data)
print(n)
print(z)

# 绘制原始数据图（蓝线）
plt.plot(data, cs)
# 绘制经过标准化的数据图（橙线）
plt.plot(z, cs)
# 显示图形
plt.show()
