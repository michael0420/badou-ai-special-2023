import cv2
import matplotlib.pyplot as plt
import numpy as np


# 灰度图
# imread params: @filename pass
# @flags -1 8bit origin
#         0 8 bit 1 channel
#         1 8 bit 3 channels
#         2 origin 1 channel
#         3 origin 3 channel
img = cv2.imread('lenna.png')
# 直接读取灰度图也可以
img_gray = cv2.imread('lenna.png', 0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)
dst1 = cv2.equalizeHist(img_gray)
# calcHist params: @images 图片数据 []包裹，@channels 选用通道 []包裹
# @mask 掩码 图像范围 全图为None @histsize: 灰度级数量 []包裹 @ranges：颜色范围 [] 包裹
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
hist1 = cv2.calcHist([dst1], [0], None, [256], [0, 256])
plt.figure()
plt.subplot(221)
plt.hist(dst.ravel(), 256, [0, 256], color='r')
plt.subplot(222)
plt.hist(dst1.ravel(), 256, [0, 256], color='r')
# plot 折线图
# plt.plot(hist1, 'r')
plt.show()

cv2.imshow('Hist', np.hstack([dst, dst1]))
cv2.waitKey()

# 彩色
img_bgr = cv2.imread('lenna.png', 3)
b, g, r = cv2.split(img_bgr)
dst_b = cv2.equalizeHist(b)
dst_g = cv2.equalizeHist(g)
dst_r = cv2.equalizeHist(r)

dst = cv2.merge((dst_b, dst_g, dst_r))
cv2.imshow('rgb img', dst)
cv2.waitKey()


