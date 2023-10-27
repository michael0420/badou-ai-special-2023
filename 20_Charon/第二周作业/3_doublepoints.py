import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

# 二值图
# 读取基础图片信息
src_img = cv2.imread('lenna.png')
img_gray = rgb2gray(src_img)

rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

plt.imshow(img_gray, cmap='gray')
plt.show()
