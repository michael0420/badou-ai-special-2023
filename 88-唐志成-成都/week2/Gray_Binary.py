"""
1.实现灰度化和二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# 灰度化
# --------------------------原理--------------------------------------------------#
# 1.读入图片
img = cv2.imread("lenna.png")
# 2.预处理
h, w = img.shape[:2]  # img.shape = (512, 512, 3) (高度，宽度，三通道)
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片，初值为0，类型img.dtype
# 如果不使用img.dtype,默认为float
# 2.灰度化 0.11*B+0.59*G+0.3*R，将3通道将为单通道
for i in range(h):
    for j in range(w):
        img_gray[i, j] = int(img[i, j, 0] * 0.11 + img[i, j, 1] * 0.59 + img[i, j, 2] * 0.3)

print("---image gray----")
print(img_gray)
print(img_gray.dtype)
# --------------------------原理--------------------------------------------------#

# 灰度化
# --------------------------掉包--------------------------------------------------#
# 法一：正常处理，映射到（0， 1）
img = cv2.imread("lenna.png")
img_gray = rgb2gray(img)
print("---image gray----")
print(img_gray)
print(img_gray.dtype)

# 法二：和原理结果相同，没有映射到（0， 1）区间内，会影响后面的二值化
# img = cv2.imread("lenna.png")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print("---image gray----")
# print(img_gray)
# print(img_gray.shape)
# --------------------------掉包--------------------------------------------------#

# 二值化
# --------------------------原理------------------------------------------------#
rows, cols = img_gray.shape
img_binary = np.zeros([rows, cols], img_gray.dtype)
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] > 0.5:
            img_binary[i, j] = 1
        else:
            img_binary[i, j] = 0

print(" -----image binary------")
print(img_binary)
print(img_binary.shape)
# --------------------------二值化------------------------------------------------#

# --------------------------调包------------------------------------------------#
# img_binary = np.where(img_gray >= 0.5, 1, 0)

# print("-----image binary------")
# print(img_binary)
# print(img_binary.shape)
# --------------------------调包------------------------------------------------#

# --------------------------------------------显示图片--------------------------------- #
plt.subplot(221)
# plt 重新读入原图片，采用cv2.imread读入会使原图片呈现蓝色效果
img = plt.imread("lenna.png")
plt.imshow(img)  # 显示原图片

# 显示灰度图
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

# 显示二值化图
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')

plt.show()
# --------------------------------------------显示图片--------------------------------- #
