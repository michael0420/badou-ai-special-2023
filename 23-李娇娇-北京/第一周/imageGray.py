import cv2
import numpy as np

img = cv2.imread("lenna.png")
# 读取原始图像的宽和高
height, weight = img.shape[: 2]
# 根据原始图像大小创建一个大小为[height,weight]的矩阵
img2Grap = np.zeros((height, weight), img.dtype)

# 灰度化
for i in range(height):
    for j in range(weight):
        # 将BGR图像转换成灰度图像
        img2Grap[i, j] = int(img[i, j][0] * 0.11 + (img[i, j][1] * 0.59) + (img[i, j][2] * 0.3))

print(img2Grap)
cv2.imshow('img', img)
cv2.imshow('img2Grap', img2Grap)

# 二值化
for i in range(height):
    for j in range(weight):
        if img2Grap[i, j] < 128:
            img2Grap[i, j] = 100
        else:
            img2Grap[i, j] = 200

print(img2Grap)
cv2.imshow('binary', img2Grap)





