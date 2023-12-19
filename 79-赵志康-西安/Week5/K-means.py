'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    compactness：表示每个样本到其所属类别中心的距离之和。
    labels：表示每个样本所属的类别标签。
    centers：表示每个类别的中心点坐标。
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

#读取原始图像的灰度颜色
img = cv2.imread('lenna.png', 0)
print(img.shape)

#获取图像的高度宽度
h,w = img.shape[:2]
print(h,w)

#图像二维像素转一维像素
data = img.reshape(h*w,1)#将矩阵a1转变成（x,y,z,…）---->一维长度x，二维长度y，三维长度z，…的矩阵。
data = np.float32(data)
print(data.shape)
print(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#精确度（误差）满足epsilon=1.0停止，迭代次数超过max_iter停止=10。

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#使用K-means进行聚类，聚集为4类
compactness, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, flags)
print(labels)

#生成最终的图像
dst = labels.reshape(img.shape[0], img.shape[1])

#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
"""
是用于设置matplotlib库中字体的代码。
在Python的matplotlib库中，所有的文本都是通过字体来显示的。这里的 'font.sans-serif' 是一个字典的键，表示我们要修改的是字体设置。
而 ['SimHei'] 是一个列表，表示我们要将字体设置为 'SimHei'。
"""

titles = [u'原始图像', u'聚类图像']#u前缀表示这是一个Unicode字符串，可以包含各种字符，如中文、英文等
images = [img, dst]
for i in range(2):  #i = 0, 1
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()