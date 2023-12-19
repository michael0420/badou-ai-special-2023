import cv2
import matplotlib.pyplot as plt
import numpy as np

# 单通道直方图均衡化
img_gray = cv2.imread("lenna.png", 0)
# cv2.imshow("image",img)
# cv2.waitKey()

dst = cv2.equalizeHist(img_gray)

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([img_gray, dst]))
cv2.waitKey()
