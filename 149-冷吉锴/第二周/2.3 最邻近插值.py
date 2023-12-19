import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)  # 建立一个空白的图像
    sh = 800 / height  # 等比缩放
    sw = 800 / width  # 等比缩放
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)  # int(),转为整形,使用向下取整 原图的像素点坐标
            y = int(j / sw + 0.5)  # 原图的像素点坐标
            emptyImage[i, j] = img[x, y]
    return emptyImage


# 读取图片
img = cv2.imread("lenna.png")
zoom = function(img)  # 放大或缩小
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
