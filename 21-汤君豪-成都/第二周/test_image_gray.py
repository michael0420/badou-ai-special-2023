#灰度化
import cv2
img = cv2.imread('lenna.png')
h, w = img.shape[:2]
import numpy as np
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        img_gray[i, j] = int(img[i, j, 0]*0.11 + img[i, j, 1]*0.59 + img[i, j, 2]*0.3)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray", img_gray)
#或者直接用函数
# from skimage.color import rgb2gray
# img_gray = rgb2gray(img) # 里面的值已经是0-1的范围了
# print(img_gray)
# cv2.imshow("image show gray", img_gray)

#二值化
img_binary = np.zeros([h, w])
for i in range(h):
    for j in range(w):
        data = img_gray[i, j]/255
        if data > 0.5:
            img_binary[i, j] = 1
print("image show binary: %s"%img_binary)
cv2.imshow("image show binary", img_binary)

import matplotlib.pyplot as plt
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()

