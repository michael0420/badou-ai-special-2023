import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img = cv2.imread("lenna.png")

# detail灰度化
h, w = img.shape[: 2]
# img.dtype 必要，浮点数不能正常显示图片，不加类型即使赋予整数也会以浮点数的形式存储
img_gray = np.zeros((h, w), img.dtype)
for i in range(h):
    for j in range(w):
        img_gray[i, j] = int(img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3)
print(img_gray)
cv2.imshow('lenna', img_gray)
cv2.waitKey()

# # 接口灰度化1
# img_gray = cv2.imread("lenna.png", 0)
# cv2.imshow('lenna', img_gray)
# cv2.waitKey()
#
# # 接口灰度化2
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('lenna', img_gray)
# cv2.waitKey()
#
# # 接口灰度化3
# img = plt.imread("lenna.png")
# img_gray = rgb2gray(img)
# cv2.imshow('lenna', img_gray)
# cv2.waitKey()
