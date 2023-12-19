# -*- coding: utf-8 -*-

# @FileName: equalizeHist.py
# @Time    : 2023/11/2 3:44 PM
# @Author  : lijiaojiao
# @Software: PyCharm

import cv2
from matplotlib import pyplot as plt

"""
equalizeHist-直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵（单通道图像）
dst：默认即可
"""

if __name__ == '__main__':
    # 灰度图像获取
    img = cv2.imread("../images/lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)

    # 直方图均衡化
    dst = cv2.equalizeHist(gray)
    # 显示直方图均衡化后的图像
    cv2.imshow("Histogram Equalization", dst)

    # 直方图
    plt.figure()
    plt.hist(dst.ravel(), 256)
    plt.show()

    cv2.waitKey(0)




