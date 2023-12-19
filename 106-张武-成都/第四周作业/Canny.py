"""
Canny步骤
1.对图像进行灰度化(优化项)
2.对图像进行高斯滤波(降噪)
3.sobel边缘检测
4.对梯度负值进行非极大值抑制
5.双阈值算法检测和连接边缘
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

# 读取图像
# img = plt.imread('lenna.png')
# print(img)

# 灰度化
img1 = cv.imread('lenna.png')
gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
print(gray)

# 创建窗口并定义窗口名称
cv.namedWindow('Canny demo')

low_threshold = 0
max_low_threshold = 100
ratio = 3
kernel_size = 3


def CannyThreshold(low_threshold):
    # Canny边缘检测
    edges = cv.Canny(gray, low_threshold, low_threshold * ratio, kernel_size)
    cv.imshow("Canny demo", edges)


"""
创建一个轨迹栏

第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
"""

cv.createTrackbar('阈值', 'Canny demo', low_threshold, max_low_threshold, CannyThreshold)

CannyThreshold(0)
if cv.waitKey(0) == 27:
    cv.destroyAllWindows()
