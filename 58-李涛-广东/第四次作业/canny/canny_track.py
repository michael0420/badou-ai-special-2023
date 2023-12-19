#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np

'''
Canny边缘检测：优化的程序
'''

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3


def CannyThreshold(lowThreshold):
    """
    高斯滤波，滤去理想图像中叠加的高频噪声
    第一个参数是输入的图像，这里是gray，表示对灰度图像进行高斯模糊处理。
    第二个参数是高斯核的大小，这里是(3, 3)，表示高斯核的宽和高都为3。
    第三个参数是X和Y方向上的标准差，这里是0，表示在X和Y方向上的标准差都为0，表示自动根据高斯核的大小计算标准差。
    """
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)

    """
    cv2.bitwise_and:
    第一个和第二个参数都是输入的图像，这里都是img，表示对原始图像img进行按位与操作。
    mask参数指定了要应用的掩码，即detected_edges，表示使用detected_edges作为掩码进行按位与操作。
    """
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    cv2.imshow('canny demo', dst)


img = cv2.imread('b.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('canny demo')

'''
createTrackbar方法参数解释如下：
trackbarName：滑动条的名称，是一个字符串。
windowName：包含滑动条的窗口名称，是一个字符串。
value：滑动条的初始值，通常是一个整数。
count：滑动条的最大值，通常是一个整数。
onChange：回调函数，当滑动条的值发生变化时会调用该函数。
通过使用createTrackbar函数，可以在OpenCV的窗口中创建滑动条，并在滑动条数值变化时执行相关操作，如调整图像处理的参数、更新显示等。
'''
cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(0)  # initialization  
if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
    cv2.destroyAllWindows()
