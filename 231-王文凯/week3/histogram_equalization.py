import cv2
import numpy as np

img = cv2.imread("../Images/lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img_gray_hist_self = histogram_equalization(img_gray)
img_gray_hist = cv2.equalizeHist(img_gray)

# img_hist_self = histogram_equalization(img)
b, g, r = cv2.split(img)
b_hist, g_hist, r_hist = cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)
img_hist = cv2.merge((b_hist, g_hist, r_hist))

# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
#  图像 通道 图像掩膜(与图像大小相同的8bit灰度图像矩阵) 每个图像维度参与直方图计算的直方图组数 参与直方图计算的每个维度的数值范围
# hist_self = cv2.calcHist(img_gray_hist_self, [0], None, [256], [0, 256])
hist = cv2.calcHist(img_gray_hist, [0], None, [256], [0, 256])

cv2.imshow("color Histogram Equalization", np.hstack([img, img_hist]))
cv2.imshow("gray Histogram Equalization", np.hstack([img_gray, img_gray_hist]))
cv2.waitKey(0)