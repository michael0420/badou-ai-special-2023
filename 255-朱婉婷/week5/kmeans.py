#coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
对RGB图像进行K—Means聚类
"""
#读取图像
img = cv2.imread('lenna.png')

#将三维形状（高*宽*通道数）->（像素数*通道数）
data = img.reshape((-1,3))
data = np.float32(data)

#设置停止条件和标签
#达到指定精度(epsilon，0.1)和最大迭代次数（10）
criteria = (cv2.TERM_CRITERIA_EPS+
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#随机选择初始点
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类，聚集成4类
#None:不使用预设的初始中心点；criteria:停止条件；10:重复运行的算法次数
#输出：
# 1. 各数据点到对应簇中心的距离平方和
# 2. 数据点的标签（表示所属的簇）
# 3. 每个簇的中心点
k = 4
compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

#将像素值映射到对应的聚类中心,恢复三维数据
centers = np.uint8(centers)
res = centers[labels.flatten()]
dst = res.reshape((img.shape))

#图像转换为RGB显示
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst,cv2.COLOR_BGR2RGB)

#显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

#显示图像
titles = [u'原始图像',u'聚类图像 k=4']
images = [img, dst]
for i in range(len(images)):
    plt.subplot(2,1,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    #不显示x和y轴标签
    plt.xticks([])
    plt.yticks([])
plt.show()