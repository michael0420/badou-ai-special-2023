# 作业1：实现二值化
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
cv2.imshow("origin", img)

print("---twovalue image lenna----")
h, w = img.shape[:2]
# 1、创建一个单通道新图
img_dst = np.zeros((h, w, 1), dtype=np.uint8)
# 2、取一个中间值 255/2 = 128
middleValue = 128
for degree in range(1):
    for i in range(h):
        for j in range(w):
            # 3、要么0要么255
            if (img[i, j, degree] <= middleValue):
                img_dst[i, j, degree] = 0
            else:
                img_dst[i, j, degree] = 255
cv2.imshow("after two values", img_dst)
cv2.waitKey(0)  # 等待键盘输入， cv2.imshow需要这一行，不知道为什么

