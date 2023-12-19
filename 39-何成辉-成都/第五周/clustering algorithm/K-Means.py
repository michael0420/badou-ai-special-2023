# coding: utf-8

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('lenna.png', 0)  # 读取名为'lenna.png'的图像，并将其转换为灰度图像。
print (img.shape)  # 打印图像的形状，即高度、宽度和通道数。

#获取图像高度、宽度
rows, cols = img.shape[:]

# 图像二维像素转换为一维，将图像数据重塑为一维数组，每个元素对应一个像素值。
data = img.reshape((rows * cols, 1))
data = np.float32(data)  # 将数据类型转换为浮点数，以便进行后续的计算。

# 停止条件 (type,max_iter,epsilon)设置K-means算法的终止条件，包括最大迭代次数和误差阈值。
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签，设置K-means算法的初始中心选择方式为随机选择。
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 生成最终图像，将聚类结果重塑为与原图像相同的形状。
dst = labels.reshape((img.shape[0], img.shape[1]))

# 用来正常显示中文标签，设置Matplotlib的字体为SimHei，以便正确显示中文字符。
plt.rcParams['font.sans-serif']=['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像']  
images = [img, dst]  
for i in range(2):  
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  # 隐藏子图的坐标轴刻度。
plt.show()
