import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# 展示灰度图
plt.subplot(221)
plt.imshow(gray, "gray")

# 展示灰度直方图图
plt.subplot(222)
plt.plot(gray_hist)

# 展示均衡化的灰度图
plt.subplot(223)
gray_equalizeHist = cv2.equalizeHist(gray)
plt.imshow(gray_equalizeHist, "gray")

# 展示均衡化的灰度直方图
gray_equalizeHist_hist = cv2.calcHist([gray_equalizeHist], [0], None, [256], [0, 256])
plt.subplot(224)
plt.plot(gray_equalizeHist_hist)
plt.show()
