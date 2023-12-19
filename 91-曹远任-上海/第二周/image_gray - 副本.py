from skimage.color import rgb2gray
import numpy as np
import cv2

img = cv2.imread("lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
# 灰度化
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
cv2.imshow("gray_float", img_gray)

img_gray = np.mean(img, axis=2).astype(img.dtype)
cv2.imshow("gray_average", img_gray)

for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 28 + m[1] * 151 + m[2] * 76) >> 8
cv2.imshow("gray_shift", img_gray)

img_gray = rgb2gray(img)
cv2.imshow("gray_rgb2gray", img_gray)

# 二值化
img_binary = np.where(img_gray >= 0.5, 255, 0).astype(img.dtype)
cv2.imshow("binary", img_binary)
cv2.waitKey()
