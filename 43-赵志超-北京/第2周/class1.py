from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度图
img = cv2.imread("lenna.png")
h, w = img.shape[:2]
img_ary = np.zeros((h, w), img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_ary[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
plt.subplot(221)
plt.imshow(img_ary, cmap='gray')

# 二值图
img2 = plt.imread("lenna.png")
img_gray2 = rgb2gray(img2)
img_binary = np.where(img_gray2 >= 0.5, 1, 0)
plt.subplot(222)
plt.imshow(img_binary, cmap='gray')

# 最邻近插值
scale = 1.5
img3 = cv2.imread("lenna.png")
cv2.imshow("1X", img3)
h3, w3, channels = img3.shape
h_target = int(h3 * scale)
w_target = int(w3 * scale)
img_ary3 = np.zeros((h_target, w_target, channels), np.uint8)
for i in range(h_target):
    for j in range(w_target):
        x = int(i / scale + 0.5)
        y = int(j / scale + 0.5)
        x = min(x, h3 - 1)
        y = min(y, w3 - 1)
        val = img3[x, y]
        img_ary3[i, j] = val
cv2.imshow(f"nearest {scale}X", img_ary3)

# 双线性插值
img4 = cv2.imread("lenna.png")
h4, w4, channels4 = img4.shape
h_target4 = int(h4 * scale)
w_target4 = int(w4 * scale)
img_ary4 = np.zeros((h_target4, w_target4, channels4), np.uint8)
for i in range(h_target4):
    for j in range(w_target4):
        x = (i + 0.5) / scale - 0.5
        y = (j + 0.5) / scale - 0.5

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))

        x1 = min(x0 + 1, h4 - 1)
        y1 = min(y0 + 1, w4 - 1)

        tmp1 = (x1 - x) * img4[x0, y0] + (x - x0) * img4[x1, y0]
        tmp2 = (x1 - x) * img4[x0, y1] + (x - x0) * img4[x1, y1]
        val = (y1 - y) * tmp1 + (y - y0) * tmp2
        img_ary4[i, j] = val

cv2.imshow(f"bilinear {scale}X", img_ary4)

plt.show()
