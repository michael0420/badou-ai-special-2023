# k-means聚类

import cv2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 获取原始图片的灰度图
    img = cv2.imread("../lenna.png", 0)
    # 获取灰度图的高与宽
    rows,cols = img.shape[:]
    # 将图片转换为一维
    data = img.reshape(rows*cols, 1)
    # 将数据转换为浮点数
    data = np.float32(data)
    # 设置停止条件(格式：停止类型，循环次数，误差)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    # 初始中心点
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 使用k-means设置k值为60
    conpactness, labels, centers = cv2.kmeans(data, 60,
                                              None, criteria, 10, flags)
    # 生成图片
    dstImg = labels.reshape(rows, cols)
    # 显示中文标签（不设置下面使用的中文title会报错）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = [u'原始', u'聚类']
    imges = [img, dstImg]
    for i in range(2):
        plt.subplot(1, 2, i+1), plt.imshow(imges[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

