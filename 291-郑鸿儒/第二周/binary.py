import cv2
import numpy as np


# detail 二值化
img_gray = cv2.imread("lenna.png", 0)
h, w = img_gray.shape[: 2]
img_binary = np.zeros([h, w], img_gray.dtype)
print(img_gray)
for i in range(h):
    for j in range(w):
        if img_gray[i, j] >= 128:
            img_binary[i, j] = 255
        else:
            img_binary[i, j] = 0
cv2.imshow("lenna", img_binary)
cv2.waitKey()
