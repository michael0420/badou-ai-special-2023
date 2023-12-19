import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
equalizeHist 直方图均衡化
函数原型：equalizeHist(src, dst=None)
src：图像矩阵（单通道图像）
dst：默认即可
"""

# 获取灰度图像
img = cv2.imread("lenna.png", 1)  # flag默认值为1，图像的通道和色彩信息
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
plt.hist(dst.ravel(), 256)  # 绘制直方图,参数bins表示直方图的条形数
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
