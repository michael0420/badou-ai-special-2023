import cv2
import numpy as np


# 读取Lenna图
img = cv2.imread("C:/Users/15082/Desktop/lenna.png")
# 将图片转化为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 均衡化
equal_img = cv2.equalizeHist(gray_img)

# 打印原图、灰度图、均衡后的灰度图
cv2.imshow('img', img)
cv2.imshow('gray_img', gray_img)
cv2.imshow("equal_img", equal_img)
cv2.waitKey(0)