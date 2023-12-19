import numpy as np
import matplotlib.pyplot as plt

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

# 归一化0到1
'''x_=(x−x_min)/(x_max−x_min)'''
def normalization_0_1(data):
    min = np.min(data)
    max = np.max(data)
    new_l = []
    for i in range(len(l)):
        new_l.append((float(l[i])-min)/(float(max - min)))
    return new_l

# 归一化-1到1
'''x_=(x−x_mean)/(x_max−x_min)'''
def normalization__1_1(data):
    mean = np.mean(data)
    min = np.min(data)
    max = np.max(data)
    new_l = []
    for i in range(len(l)):
        new_l.append((float(l[i])-mean)/(float(max - min)))
    return new_l

# 标准化
'''x∗=(x−μ)/σ'''
def z_score(data):
    mean = np.mean(data)
    sum = 0
    for i in range(len(l)):
        sum += float(l[i] - mean) * float(l[i] - mean)
    sum = sum / len(l)
    s = np.sqrt(sum)
    new_l = []
    for i in range(len(l)):
        new_l.append((float(l[i])-mean)/s)
    return new_l

count = []
for i in range(len(l)):
    count.append(l.count(l[i]))

plt.plot(l, count)
plt.plot(normalization_0_1(l), count)
plt.plot(normalization__1_1(l), count)
plt.plot(z_score(l), count)
plt.show()

