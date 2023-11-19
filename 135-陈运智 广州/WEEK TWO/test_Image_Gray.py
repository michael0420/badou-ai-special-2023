from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("lenna.png") #图片加载
h, w = img.shape[:2]#获取高度和宽度，[:2]通过切片操作提取元组中的两个元素
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
# 二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j]/255 <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 255
print(img_gray)
cv2.imwrite("hello1.png",img_gray)
