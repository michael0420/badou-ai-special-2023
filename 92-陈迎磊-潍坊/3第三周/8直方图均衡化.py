import numpy as np
import cv2
from matplotlib import pyplot as plt

# 读取图像，转化为灰度图
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 原图直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.subplot(211)
plt.hist(gray.ravel(), 256)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 均衡化后直方图
hist1 = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.subplot(212)
plt.hist(dst.ravel(), 256)
plt.show()

# 对比原图与直方图均衡化后的图
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
