import cv2
import numpy as np

img = cv2.imread('lenna.png')
# 获取原始图像的高和宽，CV里面高在前宽在后。
srcHeight, srcWidth = img.shape[:2]
img2gray = np.zeros((srcHeight, srcWidth), img.dtype)
for i in range(srcHeight):
    for j in range(srcWidth):
        temp = img[i][j]
        # cv里面的图像是BGR
        img2gray[i][j] = int(temp[0]*0.11+temp[1]*0.59+temp[2]*0.3)


# 二值化操作
img_Binary = img2gray.copy()  # 创建一个二值化后的图像副本
Binary_H, Binary_W = img_Binary.shape

for i in range(Binary_H):
    for j in range(Binary_W):
        if img_Binary[i][j] <= 128:
            img_Binary[i][j] = 0
        else:
            img_Binary[i][j] = 255

cv2.imshow('img', img)
cv2.imshow('gray', img2gray)
cv2.imshow('Binary', img_Binary)
cv2.waitKey(0)
cv2.destroyAllWindows()