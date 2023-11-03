import cv2
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#灰度图像直方图均衡化
equalization = cv2.equalizeHist(gray)
#直方图
hist = cv2.calcHist([equalization], [0], None, [256], [0, 256])
import matplotlib.pyplot as plt
plt.figure()
plt.hist(equalization.ravel(), 256)
# plt.plot(hist)
plt.show()
import numpy as np
cv2.imshow('Histogram Equalization', np.hstack([gray, equalization]))
cv2.waitKey()

#彩色图像直方图均衡化
colors = cv2.split(img)
equalization_b = cv2.equalizeHist(colors[0])
equalization_g = cv2.equalizeHist(colors[1])
equalization_r = cv2.equalizeHist(colors[2])
result = cv2.merge((equalization_b, equalization_g, equalization_r))
cv2.imshow('Histogram Equalization', np.hstack([img, result]))
cv2.waitKey()