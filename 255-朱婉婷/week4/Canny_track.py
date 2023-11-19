
""""
使用调节杠观察Canny算法
"""

import numpy as np
import cv2

def CannyTrack(lowThreshold):
    #(3,3)的高斯滤波
    detected_edges = cv2.GaussianBlur(img,(3,3),0)
    #sobel边缘检测
    detected_edges = cv2.Canny(detected_edges,
                               lowThreshold,
                               lowThreshold*ratio,
                               apertureSize=kernel_size)
    #将原图像放到检测边缘上
    dst = cv2.bitwise_and(img,img,mask=detected_edges)
    cv2.imshow('canny lena',dst)

#最低阈值变化范围
lowThreshold = 0
max_lowThreshold = 100

ratio = 3
kernel_size = 3
img = cv2.imread('lenna.png',0)

#设置调节杠,需要先设置窗口才能显现
cv2.namedWindow('canny lenna')
cv2.createTrackbar('Min threshold','canny lenna',lowThreshold,max_lowThreshold,CannyTrack)

#初始化，最低阈值为0
CannyTrack(0)
#等待ESC键退出cv2
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()