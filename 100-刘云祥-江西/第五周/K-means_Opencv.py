import cv2
import numpy as np
import matplotlib.pyplot as plt

SrcImg = cv2.imread('F:/LCE_Test/AI_Homework/AI_Homework/HomeworkImg.jpg', 0)
cols, rows = SrcImg.shape[:]  # 获取行列
ImgData = SrcImg.reshape((cols*rows), 1)
ImgData = np.float32(ImgData)
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
Compactness, labels, Centers = cv2.kmeans(ImgData, 4, None, criteria, 10, flags)  # 调用opencv中kmeans函数
DstImg = labels.reshape(cols, rows)

# 转为二维图像
Centers = np.uint8(Centers)
res = Centers[labels.flatten()]
res2 = res.reshape((SrcImg.shape))
cv2.imshow('聚类图像', res2)
cv2.waitKey(0)

Imgs = [SrcImg, DstImg]
#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'聚类图像']

# 函数原型 subplot(nrows, ncols, index, **kwargs)，一般我们只用到前三个参数，将整个绘图区域分成 nrows 行和 ncols 列，而 index 用于对子图进行编号
# plt.xticks()和plt.yticks()函数用于设置坐标轴的步长和刻度。其中，第一个参数表示设置的步长大小；第二个参数表示显示的坐标轴刻度，默认为坐标的值
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(Imgs[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
