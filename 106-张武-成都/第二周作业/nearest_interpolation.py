import cv2
import numpy as np


def nearest(img):
    height, width, channels = img.shape

    currImage = np.zeros((800, 800, channels), dtype=np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            # 计算当前点在原图上的坐标
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)

            currImage[i, j] = img[x, y]

    return currImage


# 读取图片
img = cv2.imread('lenna.png')

# 最邻近插值
nearest_img = nearest(img)
cv2.imwrite('nearest.png', nearest_img)

# opencv 最邻近插值
nearest_img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
cv2.imwrite('nearest_cv.png', nearest_img)
