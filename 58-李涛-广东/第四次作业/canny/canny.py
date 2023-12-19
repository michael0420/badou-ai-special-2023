#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是滞后阈值1；
第三个参数是滞后阈值2。
'''

img = cv2.imread("b.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imshow第二个参数是要显示的图像,对灰度图像gray进行Canny边缘检测，并将结果作为要显示的图像
cv2.imshow("canny", cv2.Canny(gray, 200, 300))
cv2.waitKey()
cv2.destroyAllWindows()
