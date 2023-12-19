'''
k-Means

设置K值
随机选择k个质心
计算每个点到质心的距离，划分到最近的质心组
重新选择质心，每个簇计算均值，得到新的质心
重复以上两步，直到达到结束条件或聚类结果不再变化
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 准备数据
# 读取灰度图
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE
                 )
# 获取宽高
# row, col = img.shape
# 图像二维转一维
data = img.reshape(-1, 1)
# 转为float32
data = np.float32(data)
# 指定K值
K = 4

# 最大迭代次数
max_iter = 10
# 精度
epsilon = 1.0
# 停止条件
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon
)
# 设置初始中心选择方式
flag = cv2.KMEANS_RANDOM_CENTERS
# 应用K-均值
compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flag)
print(compactness, labels, centers)
# 聚类结果
dst = labels.reshape((img.shape))

# 显示图像
titles = [u'原始图像', u'最终图像']

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.subplot(1, 2, 1)
plt.imshow(img, 'gray')
plt.title(u'原始图像')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(dst, 'gray')
plt.title(u'最终图像')
plt.xticks([])
plt.yticks([])

plt.show()

