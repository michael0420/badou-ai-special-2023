import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 读取表格
sales = pd.read_csv('train_data.csv', sep='\s*,\s*', engine='python')  # 读取CSV
X = sales['X'].values  # 存csv的第一列
Y = sales['Y'].values  # 存csv的第二列

img = mpimg.imread('zuixiaoercheng.jpg')

x_array = np.array(X)
y_array = np.array(Y)


def linear_regression_fit(x, y):
    """
    根据最小二乘法的公示进行代码编写
    :param x: x的坐标
    :param y: y坐标
    :return: 根据最小二乘算出来的k,b
    """
    xmean = x.mean()
    ymean = y.mean()
    n = x.shape[0]
    xy = np.sum(x * y)
    nxy = n * xmean * ymean
    xx = np.sum(x * x)
    nx2 = n * xmean ** 2

    k = (xy - nxy) / (xx - nx2)
    b = ymean - k * xmean
    return k, b


k, b = linear_regression_fit(x_array, y_array)
print("k=", k)
print("b=", b)
y_hat = k * x_array + b

f1 = plt.figure(figsize=(15, 5))
ax1 = f1.add_subplot(1, 2, 1)
ax1.scatter(x_array, y_array, marker="*")
ax1.plot(x_array, y_hat, color='r')
ax1.set_title("最小二乘")
plt.xlabel("x")
plt.ylabel("y")

ax2 = f1.add_subplot(1, 2, 2)
ax2.axis("off")
ax2.imshow(img)

plt.show()

# #初始化赋值
# s1 = 0
# s2 = 0
# s3 = 0
# s4 = 0
# n = 4       ####你需要根据的数据量进行修改
#
# #循环累加
# for i in range(n):
#     s1 = s1 + X[i]*Y[i]     #X*Y，求和
#     s2 = s2 + X[i]          #X的和
#     s3 = s3 + Y[i]          #Y的和
#     s4 = s4 + X[i]*X[i]     #X**2，求和
#
# #计算斜率和截距
# k = (s2*s3-n*s1)/(s2*s2-s4*n)
# b = (s3 - k*s2)/n
# print("Coeff: {} Intercept: {}".format(k, b))
# #y=1.4x+3.5
