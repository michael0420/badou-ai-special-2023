# 实现最邻近插值算法

import cv2
import numpy as np


def function(input_img):
    height, width, channels = input_img.shape
    des_img = np.zeros((1000, 1000, channels), np.uint8)
    sh = 1000 / height
    sw = 1000 / width
    for i in range(1000):
        for j in range(1000):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            des_img[i, j] = input_img[x, y]
    return des_img


img = cv2.imread("tiger.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
