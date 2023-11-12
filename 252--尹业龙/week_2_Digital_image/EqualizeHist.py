"""
<2023.9.11> version:1.0
实现直方图均衡化算法
reference:
opencv中转化灰度图技巧:https://blog.csdn.net/weixin_35755562/article/details/129511612
直方图与直方图均衡化:https://zhuanlan.zhihu.com/p/589716894?utm_id=0
直方图及其绘制:https://blog.csdn.net/juzicode00/article/details/121259545
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
# 读取彩色图像
img = cv2.imread("D:\\subject_learning\\cv_learn\\project\\lenna.jpg")
'''灰度图像的直方图均衡化和直方图展示'''
# 获取灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
# 直方图计算calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
# fig1 = plt.figure()
# # 绘制直方图时入参x要求为一维数组，所以使用ravel()方法将图像展开；bin=256表示绘制的柱子的数量
# plt.hist(dst.ravel(), 256)
# plt.title('equalizeHist')
# fig2 = plt.figure()
# plt.hist(gray.ravel(), 256)
# plt.title('gray')
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

'''彩色图像均衡化,需要分解通道 对每一个通道均衡化'''
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
# cv2.waitKey(0)