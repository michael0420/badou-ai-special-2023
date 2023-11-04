import numpy as np
import cv2
# 灰度化
img_gray = cv2.imread("lenna.png",0) #读取灰度图
rows, cols = img_gray.shape
img_threshold=img_gray
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 128):
            img_threshold[i, j] = 0
        else:
            img_threshold[i, j] = 250

cv2.imshow("image show threshold", img_threshold)
cv2.waitKey()